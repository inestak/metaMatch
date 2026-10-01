#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import zipfile
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from xgboost import XGBClassifier


SEED = 2026
N_JOBS = int(os.environ.get("BIOML_XGB_N_JOBS", "1"))
N_HPO_RANDOM = 28
CHUNK_SIZE = 50_000
V3_MIN_BASE_RECALL_RATIO = 0.95
V3_Q_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

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

PROJECT_ROOT = Path(os.environ.get("BIOML_PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = Path(os.environ.get("BIOML_DATA_DIR", PROJECT_ROOT / "bioml_2026"))
ARTIFACT_ROOT = Path(os.environ.get("BIOML_ARTIFACTS_DIR", PROJECT_ROOT))
ZIPS = {
    "NCIT-DOID": ARTIFACT_ROOT / "NCIT_DOID_Union403_for_Ines.zip",
    "SNOMED-FMA": ARTIFACT_ROOT / "SNOMED_FMA_Union403_for_Ines.zip",
    "SNOMED-NCIT": ARTIFACT_ROOT / "SNOMED_NCIT_Union403_for_Ines.zip",
}
GOLD_PATHS = {
    pair: {
        "train": DATA_ROOT / pair / "refs_equiv" / "train.tsv",
        "valid": DATA_ROOT / pair / "refs_equiv" / "valid.tsv",
    }
    for pair in PAIRS
}
PUBLIC_MEMBER = "train_valid_metaspace_403.csv"
TEST_MEMBER = "test_hidden_metaspace_403.csv"

NON_FEATURE = {
    "src_iri", "tgt_iri", "label", "model_score", "Score",
    "split", "fold", "pair",
}


def read_gold(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", dtype=str)
    low = {c.lower().replace("_", ""): c for c in d.columns}
    src = low.get("srcentity") or low.get("srciri")
    tgt = low.get("tgtentity") or low.get("tgtiri")
    if src is None or tgt is None:
        raise RuntimeError(f"Cannot identify gold columns in {path}: {list(d.columns)}")
    return (
        pd.DataFrame({
            "src_iri": d[src].astype(str),
            "tgt_iri": d[tgt].astype(str),
        })
        .drop_duplicates(["src_iri", "tgt_iri"])
        .reset_index(drop=True)
    )


def member_header(zip_path: Path, member: str) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as z:
        if member not in z.namelist():
            raise FileNotFoundError(f"{zip_path}: missing member {member}")
        with z.open(member, "r") as f:
            return list(pd.read_csv(f, nrows=0).columns)


def read_zip_csv(zip_path: Path, member: str, usecols=None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member, "r") as f:
            return pd.read_csv(f, usecols=usecols)


def build_spaces(columns: list[str]):
    features = [c for c in columns if c not in NON_FEATURE]

    syntaxique = [
        c for c in features
        if c.startswith("meta__syn_")
        or c.startswith("deep__label_")
        or c in {
            "deep__acronym_exact", "deep__digit_exact",
            "deep__tag_count_src", "deep__tag_count_tgt",
            "deep__tag_count_difference", "deep__tag_common_count",
            "deep__tag_jaccard",
        }
    ]
    lexical = [
        c for c in features
        if any(c.startswith(prefix) for prefix in [
            "deep__char_ngram_", "deep__word_ngram_",
            "deep__all_labels_", "deep__all_literals_", "deep__role_",
        ])
    ]
    semantic = [c for c in features if c.startswith("meta__cls_")]
    rouge = [c for c in features if c.startswith("meta__nlp_rouge")]
    retrieval = [
        c for c in features
        if c.startswith("meta__graph06_")
        or c in {"meta__from_metamatch", "meta__from_graph06", "meta__from_both"}
    ]
    structural = [
        c for c in features
        if c.startswith("deep__")
        and any(term in c for term in (
            "ancestor", "descendant", "siblings", "cousins",
            "neighbors", "cross_", "train_support",
        ))
    ]
    spectral = [c for c in features if c.startswith("meta__spc_")]
    tda = [c for c in features if c.startswith("meta__tda_")]

    spaces = {
        "Lexical": list(dict.fromkeys(lexical)),
        "Syntaxique": list(dict.fromkeys(syntaxique)),
        "Semantic": list(dict.fromkeys(semantic)),
        "ROUGE": list(dict.fromkeys(rouge)),
        "Retrieval": list(dict.fromkeys(retrieval)),
        "Structural": list(dict.fromkeys(structural)),
        "Spectral": list(dict.fromkeys(spectral)),
        "TDA": list(dict.fromkeys(tda)),
    }

    flat = [c for cols in spaces.values() for c in cols]
    counts = pd.Series(flat).value_counts()
    dup = counts[counts > 1]
    if len(dup):
        raise RuntimeError(f"Features overlap across MetaSpaces: {list(dup.index[:20])}")

    if len(flat) != 403:
        missing = sorted(set(features) - set(flat))
        raise RuntimeError(
            f"Expected exact 403-feature partition, got {len(flat)}. "
            f"Unassigned examples={missing[:30]}"
        )

    spaces["Global403"] = flat
    return spaces


def numeric_matrix(df: pd.DataFrame, features: list[str], medians=None):
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if medians is None:
        medians = X.median(axis=0).fillna(0.0)
    return X.fillna(medians), medians


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return {
            "threshold": 0.5, "precision": 0.0,
            "recall": 0.0, "f1": 0.0,
        }
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    )
    i = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
    }


def make_xgb(y, params, seed):
    y = np.asarray(y, dtype=int)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=seed,
        scale_pos_weight=neg / max(pos, 1.0),
        **params,
    )


def hpo_candidates(seed):
    # Two strong deterministic anchors + randomized search.
    anchors = [
        {
            "n_estimators": 250, "max_depth": 5, "learning_rate": 0.05,
            "min_child_weight": 1, "subsample": 0.90, "colsample_bytree": 0.90,
            "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
        },
        {
            "n_estimators": 900, "max_depth": 5, "learning_rate": 0.015,
            "min_child_weight": 1, "subsample": 0.70, "colsample_bytree": 1.00,
            "gamma": 0.10, "reg_alpha": 0.0, "reg_lambda": 5.0,
        },
    ]

    dist = {
        "n_estimators": [300, 450, 600, 800, 1000, 1200],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.015, 0.02, 0.03, 0.05, 0.08],
        "min_child_weight": [1, 2, 4, 8],
        "subsample": [0.65, 0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.60, 0.75, 0.90, 1.0],
        "gamma": [0.0, 0.05, 0.10, 0.25, 0.50],
        "reg_alpha": [0.0, 0.01, 0.10, 0.50, 1.0],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0],
    }

    sampled = list(
        ParameterSampler(
            dist,
            n_iter=N_HPO_RANDOM,
            random_state=seed,
        )
    )

    seen = set()
    out = []
    for p in anchors + sampled:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def tune_on_train_valid(train_df, valid_df, features, seed):
    Xtr, med = numeric_matrix(train_df, features)
    Xva, _ = numeric_matrix(valid_df, features, med)

    ytr = train_df["label"].astype(int).to_numpy()
    yva = valid_df["label"].astype(int).to_numpy()

    rows = []
    best = None

    for trial, params in enumerate(hpo_candidates(seed), 1):
        model = make_xgb(ytr, params, seed + trial)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]

        th = best_f1_threshold(yva, p)
        ap = float(average_precision_score(yva, p))

        row = {
            "trial": trial,
            **params,
            "threshold": th["threshold"],
            "precision": th["precision"],
            "recall": th["recall"],
            "f1": th["f1"],
            "average_precision": ap,
        }
        rows.append(row)

        key = (
            row["f1"],
            row["average_precision"],
            row["recall"],
            row["precision"],
            -row["n_estimators"],
        )
        if best is None or key > best[0]:
            best = (key, params.copy(), row.copy())

        if trial == 1 or trial % 5 == 0:
            print(
                f"  HPO {trial:02d}/{len(hpo_candidates(seed))}: "
                f"F1={row['f1']:.5f} AP={ap:.5f} "
                f"best={best[2]['f1']:.5f}",
                flush=True,
            )

    return best[1], best[2], pd.DataFrame(rows)


def public_split(public, train_gold, valid_gold):
    train_pairs = set(zip(train_gold["src_iri"], train_gold["tgt_iri"]))
    valid_pairs = set(zip(valid_gold["src_iri"], valid_gold["tgt_iri"]))

    train_sources = set(train_gold["src_iri"])
    valid_sources = set(valid_gold["src_iri"])

    overlap = train_sources & valid_sources
    if overlap:
        # Prevent source leakage: validation ownership wins.
        train_sources = train_sources - overlap

    train_df = public[public["src_iri"].isin(train_sources)].copy()
    valid_df = public[public["src_iri"].isin(valid_sources)].copy()

    train_df["label"] = np.fromiter(
        ((s, t) in train_pairs for s, t in zip(train_df["src_iri"], train_df["tgt_iri"])),
        dtype=np.int8,
        count=len(train_df),
    )
    valid_df["label"] = np.fromiter(
        ((s, t) in valid_pairs for s, t in zip(valid_df["src_iri"], valid_df["tgt_iri"])),
        dtype=np.int8,
        count=len(valid_df),
    )

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def oof_frozen_params(public, features, params, seed):
    y = public["label"].astype(int).to_numpy()
    groups = public["src_iri"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    oof = np.zeros(len(public), dtype=float)

    for fold, (tr, va) in enumerate(
        cv.split(np.zeros((len(public), 1)), y, groups),
        1,
    ):
        Xtr, med = numeric_matrix(public.iloc[tr], features)
        Xva, _ = numeric_matrix(public.iloc[va], features, med)

        model = make_xgb(y[tr], params, seed + 100 + fold)
        model.fit(Xtr, y[tr])
        oof[va] = model.predict_proba(Xva)[:, 1]

        print(f"  frozen-param OOF fold {fold}/5", flush=True)

    return oof


def pct_rank(x):
    return (
        pd.Series(np.asarray(x, float))
        .rank(method="average", pct=True)
        .to_numpy(float)
    )


def entity_confidence(pred: pd.DataFrame, entity_col: str):
    d = pred[[entity_col, "score"]].copy()
    d[entity_col] = d[entity_col].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce").fillna(0.0)

    s = d.sort_values([entity_col, "score"], ascending=[True, False], kind="mergesort")
    g = s.groupby(entity_col, sort=False)["score"]

    top1 = g.first()
    top2 = g.nth(1).reindex(top1.index).fillna(0.0)
    mean = g.mean().reindex(top1.index)
    std = g.std(ddof=0).reindex(top1.index).fillna(0.0)

    prof = pd.DataFrame({
        "top1": top1,
        "gap": top1 - top2,
        "rel_gap": (top1 - top2) / np.maximum(np.abs(top1), 1e-12),
        "top_z": ((top1 - mean) / np.maximum(std, 1e-12))
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0),
    })

    ranks = pd.DataFrame(
        {c: pct_rank(prof[c].to_numpy(float)) for c in prof.columns},
        index=prof.index,
    )
    return ranks.mean(axis=1)


def attach_v3_conf(pred):
    src_conf = entity_confidence(pred, "src_iri")
    tgt_conf = entity_confidence(pred, "tgt_iri")

    out = pred.copy()
    out["src_v3"] = out["src_iri"].astype(str).map(src_conf).fillna(0.0)
    out["tgt_v3"] = out["tgt_iri"].astype(str).map(tgt_conf).fillna(0.0)
    return out


def apply_gate(src, tgt, gate, q):
    src = np.asarray(src, float)
    tgt = np.asarray(tgt, float)

    if gate == "none":
        return np.ones(len(src), dtype=bool)
    if gate == "low_low_veto":
        return (src >= q) | (tgt >= q)
    if gate == "min_rank":
        return np.minimum(src, tgt) >= q
    if gate == "geomean_rank":
        return np.sqrt(np.clip(src, 0, 1) * np.clip(tgt, 0, 1)) >= q
    if gate == "src_only":
        return src >= q
    if gate == "tgt_only":
        return tgt >= q
    raise ValueError(gate)


def select_v3(oof_df, threshold):
    y_all = oof_df["label"].astype(int).to_numpy()
    base_mask = oof_df["score"].to_numpy(float) >= threshold
    base = oof_df.loc[base_mask].copy()

    if base.empty:
        raise RuntimeError("Frozen OOF threshold produced no predictions")

    base = attach_v3_conf(base)
    base_y = base["label"].astype(int).to_numpy()
    total_gold = int(y_all.sum())
    base_tp = int(base_y.sum())

    bp = base_tp / len(base)
    br = base_tp / total_gold if total_gold else 0.0
    bf1 = 2 * bp * br / (bp + br) if bp + br else 0.0

    rows = []

    for gate in ["none", "low_low_veto", "min_rank", "geomean_rank", "src_only", "tgt_only"]:
        qs = [0.0] if gate == "none" else V3_Q_GRID

        for q in qs:
            keep = apply_gate(
                base["src_v3"].to_numpy(float),
                base["tgt_v3"].to_numpy(float),
                gate,
                q,
            )

            tp = int(base_y[keep].sum())
            n = int(keep.sum())
            fp = n - tp

            p = tp / n if n else 0.0
            r = tp / total_gold if total_gold else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0

            rows.append({
                "gate": gate,
                "q": float(q),
                "TP": tp,
                "FP": fp,
                "precision": p,
                "recall": r,
                "f1": f1,
                "n": n,
                "base_precision": bp,
                "base_recall": br,
                "base_f1": bf1,
                "recall_ratio_vs_base": r / br if br else 1.0,
            })

    sweep = pd.DataFrame(rows)
    eligible = sweep[
        sweep["recall_ratio_vs_base"] >= V3_MIN_BASE_RECALL_RATIO - 1e-12
    ].copy()

    if eligible.empty:
        eligible = sweep[sweep["gate"] == "none"].copy()

    best = (
        eligible.sort_values(
            ["f1", "precision", "recall", "n", "q"],
            ascending=[False, False, False, True, False],
        )
        .iloc[0]
        .to_dict()
    )

    return best, sweep


def score_hidden(zip_path, features, model, medians, threshold):
    kept = []
    seen = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(TEST_MEMBER, "r") as f:
            reader = pd.read_csv(
                f,
                usecols=["src_iri", "tgt_iri"] + features,
                chunksize=CHUNK_SIZE,
            )

            for chunk_id, chunk in enumerate(reader, 1):
                X, _ = numeric_matrix(chunk, features, medians)
                p = model.predict_proba(X)[:, 1]
                m = p >= threshold

                if np.any(m):
                    kept.append(pd.DataFrame({
                        "src_iri": chunk.loc[m, "src_iri"].astype(str).to_numpy(),
                        "tgt_iri": chunk.loc[m, "tgt_iri"].astype(str).to_numpy(),
                        "score": p[m].astype(float),
                    }))

                seen += len(chunk)

                if chunk_id == 1 or chunk_id % 10 == 0:
                    print(
                        f"  hidden chunks={chunk_id} rows={seen} "
                        f"base_kept={sum(len(x) for x in kept)}",
                        flush=True,
                    )

    if not kept:
        return pd.DataFrame(columns=["src_iri", "tgt_iri", "score"]), seen

    out = (
        pd.concat(kept, ignore_index=True)
        .sort_values("score", ascending=False, kind="mergesort")
        .drop_duplicates(["src_iri", "tgt_iri"])
        .reset_index(drop=True)
    )
    return out, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=PAIRS)
    ap.add_argument("--expert", required=True, choices=EXPERTS)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    pair = args.pair
    expert = args.expert
    pidx = PAIRS.index(pair)
    eidx = EXPERTS.index(expert)
    seed = SEED + 10000 * pidx + 100 * eidx

    outroot = Path(args.output_root)
    outdir = outroot / pair / "experts" / expert
    outdir.mkdir(parents=True, exist_ok=True)

    zip_path = ZIPS[pair]
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    header_public = member_header(zip_path, PUBLIC_MEMBER)
    header_test = member_header(zip_path, TEST_MEMBER)
    spaces = build_spaces(header_public)
    features = spaces[expert]

    missing_test = [c for c in features if c not in header_test]
    if missing_test:
        raise RuntimeError(
            f"{pair}/{expert}: missing hidden features, examples={missing_test[:20]}"
        )

    print("=" * 110)
    print("9 EXPERT HPO + V3")
    print("PAIR   =", pair)
    print("EXPERT =", expert)
    print("features =", len(features))
    print("HPO = TRAIN -> VALID")
    print("final model = TRAIN + VALID")
    print("=" * 110, flush=True)

    usecols = ["src_iri", "tgt_iri"] + features
    public = read_zip_csv(zip_path, PUBLIC_MEMBER, usecols=usecols)
    public["src_iri"] = public["src_iri"].astype(str)
    public["tgt_iri"] = public["tgt_iri"].astype(str)
    public = public.drop_duplicates(["src_iri", "tgt_iri"]).reset_index(drop=True)

    train_gold = read_gold(GOLD_PATHS[pair]["train"])
    valid_gold = read_gold(GOLD_PATHS[pair]["valid"])
    train_df, valid_df = public_split(public, train_gold, valid_gold)

    print(
        f"train candidates={len(train_df)} positives={int(train_df.label.sum())} | "
        f"valid candidates={len(valid_df)} positives={int(valid_df.label.sum())}",
        flush=True,
    )

    print("\n[1/5] HPO XGBOOST PER EXPERT", flush=True)
    best_params, hpo_best_row, hpo_table = tune_on_train_valid(
        train_df, valid_df, features, seed
    )
    hpo_table.to_csv(outdir / "HPO_TRIALS.csv", index=False)
    (outdir / "BEST_HYPERPARAMETERS.json").write_text(
        json.dumps(
            {
                "pair": pair,
                "expert": expert,
                "n_features": len(features),
                "best_params": best_params,
                "validation": hpo_best_row,
            },
            indent=2,
        )
    )

    print("BEST HPO")
    print(json.dumps({
        "params": best_params,
        "validation": hpo_best_row,
    }, indent=2), flush=True)

    print("\n[2/5] OOF WITH FROZEN HYPERPARAMETERS", flush=True)
    all_gold = pd.concat([train_gold, valid_gold], ignore_index=True).drop_duplicates(
        ["src_iri", "tgt_iri"]
    )
    all_pairs = set(zip(all_gold["src_iri"], all_gold["tgt_iri"]))

    public_all = public.copy()
    public_all["label"] = np.fromiter(
        (
            (s, t) in all_pairs
            for s, t in zip(public_all["src_iri"], public_all["tgt_iri"])
        ),
        dtype=np.int8,
        count=len(public_all),
    )

    oof = oof_frozen_params(public_all, features, best_params, seed + 50000)
    yall = public_all["label"].astype(int).to_numpy()
    frozen_thr = best_f1_threshold(yall, oof)

    oof_df = public_all[["src_iri", "tgt_iri", "label"]].copy()
    oof_df["score"] = oof
    oof_df.to_csv(outdir / "OOF_FROZEN_SCORES.csv", index=False)

    print("FROZEN OOF THRESHOLD")
    print(json.dumps(frozen_thr, indent=2), flush=True)

    print("\n[3/5] V3 SELECTION ON FROZEN OOF", flush=True)
    v3_cfg, v3_sweep = select_v3(oof_df, frozen_thr["threshold"])
    v3_sweep.to_csv(outdir / "V3_OOF_SWEEP.csv", index=False)

    print("V3 FROZEN")
    print(json.dumps(v3_cfg, indent=2), flush=True)

    print("\n[4/5] REFIT BEST XGB ON TRAIN+VALID", flush=True)
    Xall, med_all = numeric_matrix(public_all, features)
    model_all = make_xgb(yall, best_params, seed + 90000)
    model_all.fit(Xall, yall)

    print("\n[5/5] FULL HIDDEN + V3", flush=True)
    hidden_base, hidden_rows = score_hidden(
        zip_path,
        features,
        model_all,
        med_all,
        float(frozen_thr["threshold"]),
    )

    hidden_base.to_csv(
        outdir / "HIDDEN_BASE_PREDICTIONS.tsv",
        sep="\t",
        index=False,
    )

    if hidden_base.empty:
        hidden_v3 = hidden_base.copy()
    else:
        h = attach_v3_conf(hidden_base)
        keep = apply_gate(
            h["src_v3"].to_numpy(float),
            h["tgt_v3"].to_numpy(float),
            str(v3_cfg["gate"]),
            float(v3_cfg["q"]),
        )
        hidden_v3 = h.loc[keep].copy()

    if len(hidden_v3):
        hidden_v3["score_rank"] = pct_rank(hidden_v3["score"].to_numpy(float))
    else:
        hidden_v3["score_rank"] = np.array([], dtype=float)

    hidden_v3.to_csv(
        outdir / "HIDDEN_V3_PREDICTIONS.tsv",
        sep="\t",
        index=False,
    )

    report = {
        "pair": pair,
        "expert": expert,
        "n_features": len(features),
        "best_hyperparameters": best_params,
        "hpo_validation": hpo_best_row,
        "frozen_oof_threshold": frozen_thr,
        "v3_config": v3_cfg,
        "hidden_candidate_rows": int(hidden_rows),
        "hidden_base_predictions": int(len(hidden_base)),
        "hidden_v3_predictions": int(len(hidden_v3)),
        "hidden_gold_used": False,
        "hidden_gold_cardinality_used": False,
        "output_tsv": str(outdir / "HIDDEN_V3_PREDICTIONS.tsv"),
    }

    (outdir / "FINAL_REPORT.json").write_text(
        json.dumps(report, indent=2)
    )

    print("FINAL EXPERT REPORT")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
