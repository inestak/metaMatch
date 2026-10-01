#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


SEED = 2026

PAIRS = ["NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT"]
EXPERTS = [
    "Global403",
    "Lexical",
    "Syntaxique",
    "Semantic",
    "ROUGE",
    "Retrieval",
    "Structural",
    "Spectral",
    "TDA",
]

# Three safety tiers are exported. BALANCED is the primary one, but we keep
# SAFE and AGGRESSIVE so the later 9->1 consensus can be inspected without
# retraining anything.
TIERS = {
    "SAFE": 0.995,
    "BALANCED": 0.980,
    "AGGRESSIVE": 0.950,
}

Q_GRID = [
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
]

ROOT = Path(os.environ.get(
    "BIOML_NINE_EXPERT_ROOT",
    Path(__file__).resolve().parent / "bioml2026_work" /
    "outputs_bioml2026_union403" / "NINE_EXPERT_HPO_V3_CONSENSUS",
))


def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True).astype(float)


def safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if int(y.sum()) == 0:
        return 0.0
    return float(average_precision_score(y, p))


def finite(a):
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def entropy_from_values(x, temperature=1.0):
    x = finite(x)
    if len(x) <= 1:
        return 0.0, 0.0

    z = (x - np.max(x)) / max(float(temperature), 1e-8)
    ez = np.exp(np.clip(z, -60, 60))
    p = ez / max(float(ez.sum()), 1e-15)

    h = float(-(p * np.log(np.maximum(p, 1e-15))).sum())
    hn = h / math.log(len(p)) if len(p) > 1 else 0.0
    return h, hn


def entity_features_from_group(scores, threshold):
    x = np.sort(finite(scores))[::-1]
    n = len(x)

    if n == 0:
        raise ValueError("empty entity group")

    top = np.zeros(10, dtype=float)
    top[: min(10, n)] = x[:10]

    top1 = float(top[0])
    top2 = float(top[1]) if n >= 2 else 0.0
    top3 = float(top[2]) if n >= 3 else 0.0
    top5 = float(top[4]) if n >= 5 else 0.0
    top10 = float(top[9]) if n >= 10 else 0.0

    mean = float(np.mean(x))
    std = float(np.std(x))
    med = float(np.median(x))
    mn = float(np.min(x))
    mx = float(np.max(x))

    q25, q50, q75, q90, q95, q99 = [
        float(v) for v in np.quantile(
            x, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
    ]

    eps = 1e-12
    gap12 = top1 - top2
    gap13 = top1 - top3
    gap15 = top1 - top5
    gap110 = top1 - top10

    # z-score of the dominant candidate relative to the local score cloud.
    top_z = (top1 - mean) / max(std, eps)

    # Positive score mass concentration.
    xp = np.maximum(x, 0.0)
    total = float(xp.sum())
    if total <= eps:
        prob = np.full(n, 1.0 / n)
    else:
        prob = xp / total

    h_mass = float(
        -(prob * np.log(np.maximum(prob, eps))).sum()
    )
    h_mass_norm = (
        h_mass / math.log(n)
        if n > 1
        else 0.0
    )
    hhi = float(np.square(prob).sum())
    effective_n = 1.0 / max(hhi, eps)

    e005, en005 = entropy_from_values(x, 0.05)
    e010, en010 = entropy_from_values(x, 0.10)
    e020, en020 = entropy_from_values(x, 0.20)
    e050, en050 = entropy_from_values(x, 0.50)
    e100, en100 = entropy_from_values(x, 1.00)

    d = {
        "feat__count": float(n),
        "feat__max": mx,
        "feat__min": mn,
        "feat__mean": mean,
        "feat__median": med,
        "feat__std": std,
        "feat__range": mx - mn,
        "feat__cv": std / max(abs(mean), eps),
        "feat__q25": q25,
        "feat__q50": q50,
        "feat__q75": q75,
        "feat__q90": q90,
        "feat__q95": q95,
        "feat__q99": q99,
        "feat__top1": top1,
        "feat__top2": top2,
        "feat__top3": top3,
        "feat__top5": top5,
        "feat__top10": top10,
        "feat__gap12": gap12,
        "feat__gap13": gap13,
        "feat__gap15": gap15,
        "feat__gap110": gap110,
        "feat__relative_gap12": gap12 / max(abs(top1), eps),
        "feat__relative_gap13": gap13 / max(abs(top1), eps),
        "feat__relative_gap15": gap15 / max(abs(top1), eps),
        "feat__top2_ratio": top2 / max(abs(top1), eps),
        "feat__top3_ratio": top3 / max(abs(top1), eps),
        "feat__top5_ratio": top5 / max(abs(top1), eps),
        "feat__top_z": top_z,
        "feat__mass_entropy": h_mass,
        "feat__mass_entropy_norm": h_mass_norm,
        "feat__hhi": hhi,
        "feat__effective_n": effective_n,
        "feat__soft_entropy_t005": e005,
        "feat__soft_entropy_norm_t005": en005,
        "feat__soft_entropy_t010": e010,
        "feat__soft_entropy_norm_t010": en010,
        "feat__soft_entropy_t020": e020,
        "feat__soft_entropy_norm_t020": en020,
        "feat__soft_entropy_t050": e050,
        "feat__soft_entropy_norm_t050": en050,
        "feat__soft_entropy_t100": e100,
        "feat__soft_entropy_norm_t100": en100,
        "feat__top1_mass": float(prob[0]),
        "feat__top2_mass": float(prob[: min(2, n)].sum()),
        "feat__top3_mass": float(prob[: min(3, n)].sum()),
        "feat__top5_mass": float(prob[: min(5, n)].sum()),
        "feat__count_ge_090top": float((x >= 0.90 * top1).sum()),
        "feat__count_ge_075top": float((x >= 0.75 * top1).sum()),
        "feat__count_ge_050top": float((x >= 0.50 * top1).sum()),
        "feat__max_minus_thr": top1 - float(threshold),
        "feat__mean_minus_thr": mean - float(threshold),
        "feat__min_minus_thr": mn - float(threshold),
    }

    return d


def build_real_entity_table(
    base_pred: pd.DataFrame,
    entity_col: str,
    threshold: float,
    with_labels: bool,
):
    rows = []
    for entity, g in base_pred.groupby(entity_col, sort=False):
        f = entity_features_from_group(
            g["score"].to_numpy(float),
            threshold,
        )
        f["entity"] = str(entity)
        f["group_id"] = str(entity)
        f["is_counterfactual"] = 0
        f["sample_weight"] = 1.0

        if with_labels:
            f["label"] = int(g["label"].astype(int).max())

        rows.append(f)

    out = pd.DataFrame(rows)

    # Rank-invariant copy of every raw entity feature.
    raw = [
        c for c in out.columns
        if c.startswith("feat__")
    ]
    for c in raw:
        out["rank__" + c] = pct_rank(out[c])

    return out


def build_counterfactual_rows(
    base_pred: pd.DataFrame,
    entity_col: str,
    threshold: float,
    max_ratio: float = 0.50,
):
    """
    Auxiliary NIL examples: for supported entities, remove every true-positive
    candidate and describe the remaining all-FP score cloud.

    They are intentionally down-weighted. Historical work showed synthetic NIL
    can be too easy, so it is an auxiliary regularizer, never the sole label
    source.
    """
    candidates = []

    for entity, g in base_pred.groupby(entity_col, sort=False):
        y = g["label"].astype(int).to_numpy()
        if int(y.max()) != 1:
            continue

        neg = g.loc[y == 0]
        if len(neg) == 0:
            continue

        f = entity_features_from_group(
            neg["score"].to_numpy(float),
            threshold,
        )
        f["entity"] = "CF::" + str(entity)
        f["group_id"] = str(entity)
        f["is_counterfactual"] = 1
        f["sample_weight"] = 0.25
        f["label"] = 0
        candidates.append(f)

    if not candidates:
        return pd.DataFrame()

    cf = pd.DataFrame(candidates)

    # Limit the counterfactual population so it cannot dominate real labels.
    n_positive_real = (
        base_pred.groupby(entity_col)["label"]
        .max()
        .astype(int)
        .sum()
    )
    max_cf = int(max(20, round(max_ratio * n_positive_real)))

    if len(cf) > max_cf:
        cf = cf.sample(
            n=max_cf,
            random_state=SEED,
        ).reset_index(drop=True)

    return cf


def align_feature_columns(real_table: pd.DataFrame):
    return [
        c for c in real_table.columns
        if c.startswith("feat__") or c.startswith("rank__feat__")
    ]


def append_cf_with_ranks(real_table, cf_table):
    if cf_table is None or cf_table.empty:
        return real_table.copy()

    raw_cols = [
        c for c in real_table.columns
        if c.startswith("feat__")
    ]

    real = real_table[
        ["entity", "group_id", "label", "is_counterfactual", "sample_weight"]
        + raw_cols
        + ["rank__" + c for c in raw_cols]
    ].copy()

    cf = cf_table[
        ["entity", "group_id", "label", "is_counterfactual", "sample_weight"]
        + raw_cols
    ].copy()

    # Counterfactual rank features are mapped through the REAL public empirical
    # CDF. Real rows keep their original real-domain ranks; synthetic rows do
    # not redefine the percentile geometry.
    for c in raw_cols:
        ref = np.sort(real[c].to_numpy(float))
        vals = cf[c].to_numpy(float)
        cf["rank__" + c] = (
            np.searchsorted(ref, vals, side="right")
            / max(len(ref), 1)
        )

    return pd.concat([real, cf], ignore_index=True)


def candidate_models(y, seed):
    pos = float((np.asarray(y) == 1).sum())
    neg = float((np.asarray(y) == 0).sum())
    spw = neg / max(pos, 1.0)

    models = {}

    for C in [0.1, 1.0, 10.0]:
        models[f"logistic_C{C:g}"] = make_pipeline(
            RobustScaler(),
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=4000,
                random_state=seed,
            ),
        )

    for leaf in [1, 2, 4]:
        for depth in [None, 10, 20]:
            name = f"extra_trees_leaf{leaf}_depth{depth}"
            models[name] = ExtraTreesClassifier(
                n_estimators=700,
                min_samples_leaf=leaf,
                max_depth=depth,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=seed + leaf + (0 if depth is None else depth),
            )

    xgb_cfgs = [
        (350, 3, 0.03, 1, 0.90, 0.90, 0.0, 0.0, 1.0),
        (500, 4, 0.03, 1, 0.85, 0.90, 0.0, 0.0, 3.0),
        (700, 5, 0.02, 2, 0.80, 0.85, 0.05, 0.01, 5.0),
        (450, 6, 0.025, 4, 0.90, 0.75, 0.10, 0.10, 5.0),
        (800, 4, 0.015, 1, 0.75, 1.00, 0.10, 0.00, 10.0),
        (600, 3, 0.02, 2, 1.00, 0.80, 0.00, 0.10, 3.0),
    ]
    for i, (
        n_est, depth, lr, child, sub, col, gamma, alpha, lam
    ) in enumerate(xgb_cfgs):
        models[f"xgboost_{i:02d}"] = XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            min_child_weight=child,
            subsample=sub,
            colsample_bytree=col,
            gamma=gamma,
            reg_alpha=alpha,
            reg_lambda=lam,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed + 100 + i,
        )

    return models


def fit_model(model, X, y, w):
    if hasattr(model, "steps"):
        # Pipeline ending in LogisticRegression.
        model.fit(
            X,
            y,
            logisticregression__sample_weight=w,
        )
    else:
        model.fit(X, y, sample_weight=w)
    return model


def proba_model(model, X):
    return model.predict_proba(X)[:, 1]


def rank_ensemble_fit(train, feature_cols):
    y = train["label"].astype(int).to_numpy()
    weights = {}
    orientation = {}

    for c in feature_cols:
        x = train[c].to_numpy(float)
        auc = safe_auc(y, x)
        if auc >= 0.5:
            orientation[c] = 1.0
            sep = auc
        else:
            orientation[c] = -1.0
            sep = 1.0 - auc

        # Ignore weak/noisy entity-shape features.
        weights[c] = max(sep - 0.50, 0.0)

    keep = [
        c for c in feature_cols
        if weights[c] >= 0.05
    ]

    if not keep:
        keep = sorted(
            feature_cols,
            key=lambda c: weights[c],
            reverse=True,
        )[:10]

    return {
        "features": keep,
        "weights": {c: weights[c] for c in keep},
        "orientation": {c: orientation[c] for c in keep},
    }


def rank_ensemble_predict(cfg, df):
    vals = []
    ws = []

    for c in cfg["features"]:
        s = df[c].astype(float)
        r = pct_rank(s)
        if cfg["orientation"][c] < 0:
            r = 1.0 - r
        vals.append(r.to_numpy(float))
        ws.append(cfg["weights"][c])

    M = np.column_stack(vals)
    w = np.asarray(ws, dtype=float)
    if float(w.sum()) <= 0:
        w = np.ones_like(w)
    w /= w.sum()
    return M @ w


def gmm2_fit(train, feature_cols):
    # Use only rank features to make this branch distribution-scale invariant.
    cols = [
        c for c in feature_cols
        if c.startswith("rank__")
    ]
    if len(cols) > 30:
        # Keep the 30 most label-separating rank features.
        y = train["label"].astype(int).to_numpy()
        scored = []
        for c in cols:
            a = safe_auc(y, train[c].to_numpy(float))
            scored.append((max(a, 1.0 - a), c))
        cols = [
            c for _, c in sorted(
                scored,
                reverse=True,
            )[:30]
        ]

    X = finite(train[cols].to_numpy(float))

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        reg_covar=1e-5,
        n_init=5,
        random_state=SEED,
    )
    gmm.fit(X)

    p = gmm.predict_proba(X)
    y = train["label"].astype(int).to_numpy()

    means = []
    for k in range(2):
        denom = max(float(p[:, k].sum()), 1e-12)
        means.append(float((p[:, k] * y).sum() / denom))

    positive_component = int(np.argmax(means))

    return {
        "cols": cols,
        "gmm": gmm,
        "positive_component": positive_component,
    }


def gmm2_predict(cfg, df):
    X = finite(df[cfg["cols"]].to_numpy(float))
    return cfg["gmm"].predict_proba(X)[:, cfg["positive_component"]]


def crossfit_entity_models(train_table, real_mask, feature_cols, seed):
    y = train_table["label"].astype(int).to_numpy()
    groups = train_table["group_id"].astype(str).to_numpy()
    w = train_table["sample_weight"].astype(float).to_numpy()

    # Number of real positives/negatives drives split safety.
    real_y = y[real_mask]
    n_pos = int((real_y == 1).sum())
    n_neg = int((real_y == 0).sum())

    # Counterfactual rows ensure a second class can exist even when real
    # negatives are scarce, but downstream scoring is evaluated only on real.
    all_pos = int((y == 1).sum())
    all_neg = int((y == 0).sum())

    if all_pos < 5 or all_neg < 5:
        raise RuntimeError(
            f"Entity learner has too few classes: pos={all_pos} neg={all_neg}"
        )

    n_splits = min(5, max(2, min(all_pos, all_neg)))
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    model_defs = candidate_models(y, seed)
    oof = {
        name: np.zeros(len(train_table), dtype=float)
        for name in model_defs
    }
    oof["rank_ensemble"] = np.zeros(len(train_table), dtype=float)
    oof["gmm2_rank_vector"] = np.zeros(len(train_table), dtype=float)

    for fold, (tr, va) in enumerate(
        cv.split(
            np.zeros((len(train_table), 1)),
            y,
            groups,
        ),
        1,
    ):
        Xtr = finite(train_table.iloc[tr][feature_cols].to_numpy(float))
        Xva = finite(train_table.iloc[va][feature_cols].to_numpy(float))

        for name, model in candidate_models(y[tr], seed + 1000 * fold).items():
            fit_model(model, Xtr, y[tr], w[tr])
            oof[name][va] = proba_model(model, Xva)

        r_cfg = rank_ensemble_fit(
            train_table.iloc[tr],
            feature_cols,
        )
        oof["rank_ensemble"][va] = rank_ensemble_predict(
            r_cfg,
            train_table.iloc[va],
        )

        g_cfg = gmm2_fit(
            train_table.iloc[tr],
            feature_cols,
        )
        oof["gmm2_rank_vector"][va] = gmm2_predict(
            g_cfg,
            train_table.iloc[va],
        )

        print(
            f"    entity CV fold {fold}/{n_splits} "
            f"train={len(tr)} valid={len(va)}",
            flush=True,
        )

    # Soft ensemble of the strongest supervised + rank/GMM OOF scores.
    real_idx = np.where(real_mask)[0]
    scores = []
    for name, p in oof.items():
        ap = safe_ap(y[real_idx], p[real_idx])
        scores.append((ap, name))

    top = [
        name for _, name in sorted(
            scores,
            reverse=True,
        )[:5]
    ]

    M = np.column_stack([
        pd.Series(oof[name]).rank(pct=True).to_numpy(float)
        for name in top
    ])
    weights = np.asarray([
        max(safe_ap(y[real_idx], oof[name][real_idx]), 1e-6)
        for name in top
    ])
    weights /= weights.sum()
    oof["soft_ensemble"] = M @ weights

    perf = []
    for name, p in oof.items():
        perf.append({
            "model": name,
            "real_auc": safe_auc(
                y[real_idx],
                p[real_idx],
            ),
            "real_ap": safe_ap(
                y[real_idx],
                p[real_idx],
            ),
            "n_real": int(len(real_idx)),
            "n_real_positive": int(y[real_idx].sum()),
            "n_real_negative": int(
                len(real_idx) - y[real_idx].sum()
            ),
        })

    return oof, pd.DataFrame(perf), top, weights


def refit_predict_selected(
    train_table,
    hidden_real,
    feature_cols,
    model_name,
    ensemble_top,
    ensemble_weights,
    seed,
):
    y = train_table["label"].astype(int).to_numpy()
    w = train_table["sample_weight"].astype(float).to_numpy()
    Xtr = finite(train_table[feature_cols].to_numpy(float))
    Xh = finite(hidden_real[feature_cols].to_numpy(float))

    if model_name == "rank_ensemble":
        cfg = rank_ensemble_fit(train_table, feature_cols)
        return rank_ensemble_predict(cfg, hidden_real)

    if model_name == "gmm2_rank_vector":
        cfg = gmm2_fit(train_table, feature_cols)
        return gmm2_predict(cfg, hidden_real)

    if model_name == "soft_ensemble":
        preds = []
        for sub in ensemble_top:
            preds.append(
                refit_predict_selected(
                    train_table,
                    hidden_real,
                    feature_cols,
                    sub,
                    ensemble_top,
                    ensemble_weights,
                    seed + 777,
                )
            )
        M = np.column_stack([
            pd.Series(p).rank(pct=True).to_numpy(float)
            for p in preds
        ])
        return M @ ensemble_weights

    defs = candidate_models(y, seed)
    if model_name not in defs:
        raise KeyError(model_name)

    model = defs[model_name]
    fit_model(model, Xtr, y, w)
    return proba_model(model, Xh)


def pair_metrics(base, keep, total_gold):
    y = base["label"].astype(int).to_numpy()
    keep = np.asarray(keep, dtype=bool)

    tp = int(((y == 1) & keep).sum())
    fp = int(((y == 0) & keep).sum())
    fn = int(total_gold - tp)

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / total_gold if total_gold else 0.0
    f1 = (
        2 * p * r / (p + r)
        if p + r
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": int(keep.sum()),
    }


def apply_gate(src, tgt, gate, qs, qt):
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)

    if gate == "src_only":
        return src >= qs
    if gate == "tgt_only":
        return tgt >= qt
    if gate == "hard_and":
        return (src >= qs) & (tgt >= qt)
    if gate == "low_low_veto":
        return (src >= qs) | (tgt >= qt)
    if gate == "min_rank":
        return np.minimum(src, tgt) >= qs
    if gate == "geomean_rank":
        return np.sqrt(
            np.clip(src, 0, 1) * np.clip(tgt, 0, 1)
        ) >= qs

    raise ValueError(gate)


def select_downstream_gates(
    base,
    total_gold,
    src_real,
    tgt_real,
    src_oof,
    tgt_oof,
    src_perf,
    tgt_perf,
):
    base_m = pair_metrics(
        base,
        np.ones(len(base), dtype=bool),
        total_gold,
    )

    # Keep the strongest 6 entity learners on each side by real AP.
    src_models = (
        src_perf.sort_values(
            ["real_ap", "real_auc"],
            ascending=False,
        )
        .head(6)["model"]
        .tolist()
    )
    tgt_models = (
        tgt_perf.sort_values(
            ["real_ap", "real_auc"],
            ascending=False,
        )
        .head(6)["model"]
        .tolist()
    )

    src_entity_to_pos = {
        e: i
        for i, e in enumerate(
            src_real["entity"].astype(str)
        )
    }
    tgt_entity_to_pos = {
        e: i
        for i, e in enumerate(
            tgt_real["entity"].astype(str)
        )
    }

    rows = []

    for sm in src_models:
        src_rank = pd.Series(
            src_oof[sm][src_real.index.to_numpy()]
        ).rank(pct=True).to_numpy(float)

        src_map = {
            e: float(src_rank[i])
            for i, e in enumerate(
                src_real["entity"].astype(str)
            )
        }

        s_pair = (
            base["src_iri"].astype(str)
            .map(src_map)
            .fillna(0.0)
            .to_numpy(float)
        )

        for tm in tgt_models:
            tgt_rank = pd.Series(
                tgt_oof[tm][tgt_real.index.to_numpy()]
            ).rank(pct=True).to_numpy(float)

            tgt_map = {
                e: float(tgt_rank[i])
                for i, e in enumerate(
                    tgt_real["entity"].astype(str)
                )
            }

            t_pair = (
                base["tgt_iri"].astype(str)
                .map(tgt_map)
                .fillna(0.0)
                .to_numpy(float)
            )

            # Single-side.
            for q in Q_GRID:
                for gate in ["src_only", "tgt_only"]:
                    keep = apply_gate(
                        s_pair, t_pair, gate, q, q
                    )
                    m = pair_metrics(base, keep, total_gold)
                    rows.append({
                        "src_model": sm,
                        "tgt_model": tm,
                        "gate": gate,
                        "q_src": q,
                        "q_tgt": q,
                        **m,
                        "base_recall": base_m["recall"],
                        "recall_ratio_vs_base": (
                            m["recall"] / base_m["recall"]
                            if base_m["recall"] else 1.0
                        ),
                    })

                for gate in ["min_rank", "geomean_rank"]:
                    keep = apply_gate(
                        s_pair, t_pair, gate, q, q
                    )
                    m = pair_metrics(base, keep, total_gold)
                    rows.append({
                        "src_model": sm,
                        "tgt_model": tm,
                        "gate": gate,
                        "q_src": q,
                        "q_tgt": q,
                        **m,
                        "base_recall": base_m["recall"],
                        "recall_ratio_vs_base": (
                            m["recall"] / base_m["recall"]
                            if base_m["recall"] else 1.0
                        ),
                    })

            # Asymmetric bidirectional gates.
            for qs in Q_GRID:
                for qt in Q_GRID:
                    for gate in ["hard_and", "low_low_veto"]:
                        keep = apply_gate(
                            s_pair,
                            t_pair,
                            gate,
                            qs,
                            qt,
                        )
                        m = pair_metrics(base, keep, total_gold)
                        rows.append({
                            "src_model": sm,
                            "tgt_model": tm,
                            "gate": gate,
                            "q_src": qs,
                            "q_tgt": qt,
                            **m,
                            "base_recall": base_m["recall"],
                            "recall_ratio_vs_base": (
                                m["recall"] / base_m["recall"]
                                if base_m["recall"] else 1.0
                            ),
                        })

    sweep = pd.DataFrame(rows)

    selected = {}
    for tier, floor in TIERS.items():
        e = sweep[
            sweep["recall_ratio_vs_base"] >= floor - 1e-12
        ].copy()

        if e.empty:
            e = sweep.sort_values(
                "recall_ratio_vs_base",
                ascending=False,
            ).head(1)

        best = (
            e.sort_values(
                [
                    "f1",
                    "precision",
                    "recall",
                    "n",
                    "q_src",
                    "q_tgt",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
            )
            .iloc[0]
            .to_dict()
        )
        selected[tier] = best

    return base_m, selected, sweep


def build_hidden_filter(
    hidden_base,
    hidden_src_real,
    hidden_tgt_real,
    src_train,
    tgt_train,
    src_feature_cols,
    tgt_feature_cols,
    selected,
    src_ensemble_top,
    src_ensemble_weights,
    tgt_ensemble_top,
    tgt_ensemble_weights,
    seed,
):
    outputs = {}

    # Cache predictions for every entity learner required by selected tiers.
    src_needed = sorted(
        set(str(cfg["src_model"]) for cfg in selected.values())
    )
    tgt_needed = sorted(
        set(str(cfg["tgt_model"]) for cfg in selected.values())
    )

    src_scores = {}
    for name in src_needed:
        p = refit_predict_selected(
            src_train,
            hidden_src_real,
            src_feature_cols,
            name,
            src_ensemble_top,
            src_ensemble_weights,
            seed + 100,
        )
        src_scores[name] = pd.Series(p).rank(pct=True).to_numpy(float)

    tgt_scores = {}
    for name in tgt_needed:
        p = refit_predict_selected(
            tgt_train,
            hidden_tgt_real,
            tgt_feature_cols,
            name,
            tgt_ensemble_top,
            tgt_ensemble_weights,
            seed + 200,
        )
        tgt_scores[name] = pd.Series(p).rank(pct=True).to_numpy(float)

    for tier, cfg in selected.items():
        sm = str(cfg["src_model"])
        tm = str(cfg["tgt_model"])

        src_map = dict(
            zip(
                hidden_src_real["entity"].astype(str),
                src_scores[sm],
            )
        )
        tgt_map = dict(
            zip(
                hidden_tgt_real["entity"].astype(str),
                tgt_scores[tm],
            )
        )

        s = (
            hidden_base["src_iri"].astype(str)
            .map(src_map)
            .fillna(0.0)
            .to_numpy(float)
        )
        t = (
            hidden_base["tgt_iri"].astype(str)
            .map(tgt_map)
            .fillna(0.0)
            .to_numpy(float)
        )

        keep = apply_gate(
            s,
            t,
            str(cfg["gate"]),
            float(cfg["q_src"]),
            float(cfg["q_tgt"]),
        )

        out = hidden_base.loc[keep].copy()
        out["src_alignability_rank"] = s[keep]
        out["tgt_alignability_rank"] = t[keep]
        outputs[tier] = out

    return outputs


def prepare_side(
    public_base,
    hidden_base,
    entity_col,
    threshold,
    seed,
):
    real = build_real_entity_table(
        public_base,
        entity_col,
        threshold,
        with_labels=True,
    )

    cf = build_counterfactual_rows(
        public_base,
        entity_col,
        threshold,
    )

    train = append_cf_with_ranks(real, cf)
    feature_cols = align_feature_columns(train)

    # Preserve real-row positions inside the combined table.
    real_mask = (
        train["is_counterfactual"].astype(int).to_numpy() == 0
    )

    oof, perf, ens_top, ens_weights = crossfit_entity_models(
        train,
        real_mask,
        feature_cols,
        seed,
    )

    # Public real table must be indexed with positions in the combined table
    # so downstream model OOF arrays can be addressed correctly.
    real_positions = np.where(real_mask)[0]
    real_for_gate = train.iloc[real_positions].copy()
    real_for_gate.index = real_positions

    hidden_real = build_real_entity_table(
        hidden_base,
        entity_col,
        threshold,
        with_labels=False,
    )

    # Hidden ranks must be computed in hidden-domain entity space. They are
    # already included by build_real_entity_table.
    hidden_real = hidden_real.reset_index(drop=True)

    return {
        "real": real_for_gate,
        "train": train,
        "hidden": hidden_real,
        "features": feature_cols,
        "oof": oof,
        "perf": perf,
        "ensemble_top": ens_top,
        "ensemble_weights": ens_weights,
        "n_cf": 0 if cf is None else len(cf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=PAIRS)
    ap.add_argument("--expert", required=True, choices=EXPERTS)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    pair = args.pair
    expert = args.expert

    edir = ROOT / pair / "experts" / expert
    report_path = edir / "FINAL_REPORT.json"
    oof_path = edir / "OOF_FROZEN_SCORES.csv"
    hidden_path = edir / "HIDDEN_BASE_PREDICTIONS.tsv"

    for p in [report_path, oof_path, hidden_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    report = json.loads(report_path.read_text())
    threshold = float(
        report["frozen_oof_threshold"]["threshold"]
    )

    public = pd.read_csv(oof_path)
    hidden = pd.read_csv(hidden_path, sep="\t")

    public["src_iri"] = public["src_iri"].astype(str)
    public["tgt_iri"] = public["tgt_iri"].astype(str)
    public["label"] = pd.to_numeric(
        public["label"],
        errors="raise",
    ).astype(np.int8)
    public["score"] = pd.to_numeric(
        public["score"],
        errors="coerce",
    ).fillna(0.0)

    hidden["src_iri"] = hidden["src_iri"].astype(str)
    hidden["tgt_iri"] = hidden["tgt_iri"].astype(str)
    hidden["score"] = pd.to_numeric(
        hidden["score"],
        errors="coerce",
    ).fillna(0.0)

    public_base = public[
        public["score"] >= threshold
    ].copy()

    if len(public_base) == 0:
        raise RuntimeError("empty public base predictions")

    seed = (
        SEED
        + 10000 * PAIRS.index(pair)
        + 100 * EXPERTS.index(expert)
    )

    outdir = (
        Path(args.output_root)
        / pair
        / "experts"
        / expert
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("FULL ENTITY ALIGNABILITY FILTER")
    print("PAIR   =", pair)
    print("EXPERT =", expert)
    print("base threshold =", threshold)
    print("public base predictions =", len(public_base))
    print("hidden base predictions =", len(hidden))
    print("=" * 120, flush=True)

    print("\n[1/5] SOURCE ENTITY METASPACE", flush=True)
    src = prepare_side(
        public_base,
        hidden,
        "src_iri",
        threshold,
        seed + 1,
    )
    src["perf"].to_csv(
        outdir / "SOURCE_ALIGNABILITY_MODELS.csv",
        index=False,
    )

    print(
        f"  source real={len(src['real'])} "
        f"counterfactual={src['n_cf']} "
        f"features={len(src['features'])}",
        flush=True,
    )

    print("\n[2/5] TARGET ENTITY METASPACE", flush=True)
    tgt = prepare_side(
        public_base,
        hidden,
        "tgt_iri",
        threshold,
        seed + 2,
    )
    tgt["perf"].to_csv(
        outdir / "TARGET_ALIGNABILITY_MODELS.csv",
        index=False,
    )

    print(
        f"  target real={len(tgt['real'])} "
        f"counterfactual={tgt['n_cf']} "
        f"features={len(tgt['features'])}",
        flush=True,
    )

    src["real"].to_csv(
        outdir / "PUBLIC_SOURCE_ENTITY_METASPACE.csv",
        index=False,
    )
    tgt["real"].to_csv(
        outdir / "PUBLIC_TARGET_ENTITY_METASPACE.csv",
        index=False,
    )
    src["hidden"].to_csv(
        outdir / "HIDDEN_SOURCE_ENTITY_METASPACE.csv",
        index=False,
    )
    tgt["hidden"].to_csv(
        outdir / "HIDDEN_TARGET_ENTITY_METASPACE.csv",
        index=False,
    )

    print("\n[3/5] DOWNSTREAM OOF GATE SWEEP", flush=True)
    total_public_gold = int(public["label"].sum())

    base_m, selected, sweep = select_downstream_gates(
        public_base,
        total_public_gold,
        src["real"],
        tgt["real"],
        src["oof"],
        tgt["oof"],
        src["perf"],
        tgt["perf"],
    )
    sweep.to_csv(
        outdir / "ALIGNABILITY_GATE_SWEEP.csv",
        index=False,
    )

    (outdir / "SELECTED_ALIGNABILITY_FILTERS.json").write_text(
        json.dumps(
            {
                "base_public_metrics": base_m,
                "selected": selected,
            },
            indent=2,
        )
    )

    print("BASE PUBLIC")
    print(json.dumps(base_m, indent=2), flush=True)
    print("SELECTED FILTERS")
    print(json.dumps(selected, indent=2), flush=True)

    print("\n[4/5] REFIT ENTITY LEARNERS + HIDDEN FILTER", flush=True)
    hidden_outputs = build_hidden_filter(
        hidden,
        src["hidden"],
        tgt["hidden"],
        src["train"],
        tgt["train"],
        src["features"],
        tgt["features"],
        selected,
        src["ensemble_top"],
        src["ensemble_weights"],
        tgt["ensemble_top"],
        tgt["ensemble_weights"],
        seed,
    )

    hidden_counts = {}
    for tier, d in hidden_outputs.items():
        p = outdir / f"HIDDEN_FILTERED_{tier}.tsv"
        d.to_csv(p, sep="\t", index=False)
        hidden_counts[tier] = int(len(d))
        print(
            f"  {tier:10s}: {len(hidden)} -> {len(d)}",
            flush=True,
        )

    print("\n[5/5] REPORT", flush=True)
    final_report = {
        "pair": pair,
        "expert": expert,
        "protocol": (
            "OOF candidate scores -> rich source/target EntityMetaSpace -> "
            "cross-fitted alignability learners -> downstream pair-F1 gate "
            "selection with recall floors -> rank-invariant hidden transfer"
        ),
        "public_base_predictions": int(len(public_base)),
        "hidden_base_predictions": int(len(hidden)),
        "source": {
            "n_real_entities": int(len(src["real"])),
            "n_counterfactual": int(src["n_cf"]),
            "n_features": int(len(src["features"])),
            "model_performance": src["perf"].to_dict(orient="records"),
        },
        "target": {
            "n_real_entities": int(len(tgt["real"])),
            "n_counterfactual": int(tgt["n_cf"]),
            "n_features": int(len(tgt["features"])),
            "model_performance": tgt["perf"].to_dict(orient="records"),
        },
        "base_public_metrics": base_m,
        "selected_filters": selected,
        "hidden_counts": hidden_counts,
        "hidden_gold_used": False,
        "hidden_gold_cardinality_used": False,
        "leaderboard_used_for_selection": False,
    }

    (outdir / "FINAL_REPORT.json").write_text(
        json.dumps(final_report, indent=2)
    )

    print(json.dumps(final_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
