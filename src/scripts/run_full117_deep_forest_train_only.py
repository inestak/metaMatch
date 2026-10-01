#!/usr/bin/env python3
"""Train a leakage-free forest on MetaSpace-117 plus deep ontology features.

The script deliberately keeps the protocol strict:
* labels and model selection use ``train.tsv`` only;
* the threshold and 1:1 post-filter are selected from grouped train OOF scores;
* test predictions are written before ``test.tsv`` is optionally opened for one
  final diagnostic evaluation.

It expects the *same candidate universe* to have already been materialised for
train and test.  The 117 MetaSpace columns and the deep lexical/structural
columns are joined by candidate order / (src_iri, tgt_iri), respectively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.evaluate import compute_alignment_metrics

Pair = Tuple[str, str]
PAIR_COLUMNS = ["src_iri", "tgt_iri"]


def build_matches_with_postfilter(pred_df: pd.DataFrame, threshold: float, mode: str) -> pd.DataFrame:
    """Local 1:1 post-filters; kept here to avoid importing the OWL pipeline."""
    work = pred_df.copy()
    work["rank_src"] = work.groupby("src_iri")["score"].rank(method="first", ascending=False).astype(int)
    work["rank_tgt"] = work.groupby("tgt_iri")["score"].rank(method="first", ascending=False).astype(int)
    work = work[work.score >= threshold].sort_values("score", ascending=False, kind="mergesort")
    if mode == "none":
        chosen = work
    elif mode == "mutual_best":
        chosen = work[(work.rank_src == 1) & (work.rank_tgt == 1)]
    elif mode in {"greedy_1to1", "mutual_then_greedy"}:
        used_src: Set[str] = set()
        used_tgt: Set[str] = set()
        rows = []
        if mode == "mutual_then_greedy":
            anchors = work[(work.rank_src == 1) & (work.rank_tgt == 1)]
            for row in anchors.itertuples(index=False):
                used_src.add(row.src_iri); used_tgt.add(row.tgt_iri)
                rows.append((row.src_iri, row.tgt_iri, float(row.score)))
        for row in work.itertuples(index=False):
            if row.src_iri in used_src or row.tgt_iri in used_tgt:
                continue
            used_src.add(row.src_iri); used_tgt.add(row.tgt_iri)
            rows.append((row.src_iri, row.tgt_iri, float(row.score)))
        chosen = pd.DataFrame(rows, columns=["src_iri", "tgt_iri", "score"])
    else:
        raise ValueError(f"Unknown post-filter: {mode}")
    return chosen.rename(columns={"src_iri": "SrcEntity", "tgt_iri": "TgtEntity", "score": "Score"})[
        ["SrcEntity", "TgtEntity", "Score"]
    ].reset_index(drop=True)


def _read_pairs(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    raw = pd.read_csv(path, sep=sep, dtype=str)
    aliases = {raw.columns[0]: "src_iri", raw.columns[1]: "tgt_iri"}
    frame = raw.rename(columns=aliases)
    if not set(PAIR_COLUMNS).issubset(frame):
        raise ValueError(f"{path}: expected two pair columns")
    return frame[PAIR_COLUMNS].astype(str).reset_index(drop=True)


def _read_gold(path: Path) -> Set[Pair]:
    frame = _read_pairs(path)
    return set(zip(frame.src_iri, frame.tgt_iri))


def _load_matrix(path: Path, expected_rows: int, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(f"{name}: {len(frame)} rows, expected {expected_rows}")
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError(f"{name}: no numeric features")
    return numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_deep(path: Path, pairs: pd.DataFrame, name: str) -> pd.DataFrame:
    deep = pd.read_csv(path)
    if not set(PAIR_COLUMNS).issubset(deep):
        raise ValueError(f"{name}: missing src_iri/tgt_iri")
    if deep.duplicated(PAIR_COLUMNS).any():
        raise ValueError(f"{name}: duplicate candidate pairs")
    if "label" not in deep.columns:
        raise ValueError(f"{name}: missing label boundary column")
    # analyze_oracle_lexical_structural_2025 preserves candidate metadata on
    # the left, then writes `label`, followed by the freshly computed deep
    # feature contract.  Candidate metadata can legitimately differ between
    # train (for example an OOF score) and test.  It must not become a model
    # feature; only columns created after the label boundary are deep signals.
    feature_start = deep.columns.get_loc("label") + 1
    fresh_columns = set(deep.columns[feature_start:])
    numeric = [
        c for c in deep.select_dtypes(include=[np.number]).columns
        if c in fresh_columns and not c.startswith("gold_support_")
    ]
    selected = deep[PAIR_COLUMNS + numeric].copy()
    merged = pairs.merge(selected, on=PAIR_COLUMNS, how="left", validate="one_to_one")
    missing = merged[numeric].isna().all(axis=1).sum()
    if missing:
        raise ValueError(f"{name}: {missing} candidates have no deep features")
    result = merged[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result.columns = [f"deep__{c}" for c in result.columns]
    return result


def _matrix(meta_path: Path, deep_path: Path, pairs: pd.DataFrame, expected_meta: int, name: str):
    meta = _load_matrix(meta_path, len(pairs), f"{name} MetaSpace")
    if expected_meta and len(meta.columns) != expected_meta:
        raise ValueError(
            f"{name} MetaSpace: {len(meta.columns)} numeric features; "
            f"expected exactly {expected_meta}"
        )
    meta.columns = [f"meta117__{c}" for c in meta.columns]
    deep = _load_deep(deep_path, pairs, f"{name} deep")
    return pd.concat([meta.reset_index(drop=True), deep.reset_index(drop=True)], axis=1)


def _alignment_metrics(matches: pd.DataFrame, gold: Set[Pair]) -> dict:
    predicted = set(zip(matches.SrcEntity.astype(str), matches.TgtEntity.astype(str)))
    return compute_alignment_metrics(predicted, gold)


def _candidate_protocols(scores: np.ndarray, pairs: pd.DataFrame, gold: Set[Pair], min_recall: float):
    # Quantile grid keeps this tractable even for a large graph-augmented pool.
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 0.999, 151)))
    scored = pairs.copy()
    scored["score"] = scores
    rows = []
    for mode in ("none", "mutual_best", "greedy_1to1", "mutual_then_greedy"):
        for threshold in thresholds:
            matches = build_matches_with_postfilter(scored, float(threshold), mode=mode)
            metrics = _alignment_metrics(matches, gold)
            rows.append({"threshold": float(threshold), "post_filter_mode": mode, **metrics})
    grid = pd.DataFrame(rows)
    eligible = grid[grid.recall >= min_recall]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0].to_dict()
    return best, grid


def _forest(kind: str, trees: int, leaf: int, workers: int, seed: int):
    common = dict(
        n_estimators=trees, min_samples_leaf=leaf, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=workers,
    )
    if kind == "extra_trees":
        return ExtraTreesClassifier(**common)
    return RandomForestClassifier(**common)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-pairs", type=Path, required=True)
    p.add_argument("--test-pairs", type=Path, required=True)
    p.add_argument("--train-gold", type=Path, required=True)
    p.add_argument("--train-metaspace117", type=Path, required=True)
    p.add_argument("--test-metaspace117", type=Path, required=True)
    p.add_argument("--train-deep", type=Path, required=True)
    p.add_argument("--test-deep", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--test-gold", type=Path, default=None,
                   help="Optional, opened only after final predictions are saved.")
    p.add_argument("--forest", choices=("extra_trees", "random_forest"), default="extra_trees")
    p.add_argument("--trees", type=int, default=800)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--expected-metaspace-features", type=int, default=117)
    p.add_argument("--min-oof-recall", type=float, default=0.0,
                   help="Optional recall floor for train-only OOF protocol selection.")
    args = p.parse_args()

    train_pairs, test_pairs = _read_pairs(args.train_pairs), _read_pairs(args.test_pairs)
    if train_pairs.duplicated(PAIR_COLUMNS).any() or test_pairs.duplicated(PAIR_COLUMNS).any():
        p.error("Candidate pairs must be deduplicated before this script")
    train_gold = _read_gold(args.train_gold)
    y = np.fromiter((int(x in train_gold) for x in zip(train_pairs.src_iri, train_pairs.tgt_iri)), dtype=np.int8)
    if not y.any():
        p.error("No train gold pair occurs in --train-pairs")

    train_x = _matrix(args.train_metaspace117, args.train_deep, train_pairs,
                      args.expected_metaspace_features, "train")
    test_x = _matrix(args.test_metaspace117, args.test_deep, test_pairs,
                     args.expected_metaspace_features, "test")
    if list(train_x.columns) != list(test_x.columns):
        p.error("Train/test feature contracts differ")
    X = train_x.to_numpy(dtype=np.float32)
    X_test = test_x.to_numpy(dtype=np.float32)
    groups = train_pairs.src_iri.to_numpy()
    folds = min(args.folds, len(np.unique(groups)))
    if folds < 2:
        p.error("Need candidates from at least two source entities")

    oof = np.zeros(len(train_pairs), dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit, valid) in enumerate(splitter.split(X, y, groups), 1):
        model = _forest(args.forest, args.trees, args.min_samples_leaf, args.workers, 42 + fold)
        model.fit(X[fit], y[fit])
        oof[valid] = model.predict_proba(X[valid])[:, 1]
        print(f"fold {fold}/{folds}: {len(valid)} candidates")

    selected, grid = _candidate_protocols(oof, train_pairs, train_gold, args.min_oof_recall)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "train_oof_protocol_grid.csv", index=False)
    train_oof = train_pairs.copy(); train_oof["label"] = y; train_oof["oof_score"] = oof
    train_oof.to_csv(args.output_dir / "train_oof_predictions.csv", index=False)

    final_model = _forest(args.forest, args.trees, args.min_samples_leaf, args.workers, 777)
    final_model.fit(X, y)
    test_score = final_model.predict_proba(X_test)[:, 1]
    test_pred = test_pairs.copy(); test_pred["score"] = test_score
    test_pred.to_csv(args.output_dir / "test_predictions_pre_gold.csv", index=False)
    final = build_matches_with_postfilter(
        test_pred, float(selected["threshold"]), mode=str(selected["post_filter_mode"])
    )
    final.to_csv(args.output_dir / "matches.tsv", sep="\t", index=False)
    pd.DataFrame({"feature": train_x.columns, "importance": final_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(args.output_dir / "feature_importance.csv", index=False)

    summary = {
        "protocol": "train-only full117 + deep forest",
        "test_gold_used_for_training_or_selection": False,
        "forest": args.forest,
        "feature_count": int(train_x.shape[1]),
        "metaspace_feature_count": int(args.expected_metaspace_features),
        "deep_feature_count": int(train_x.shape[1] - args.expected_metaspace_features),
        "train_candidate_count": int(len(train_pairs)),
        "test_candidate_count": int(len(test_pairs)),
        "train_gold_in_candidates": int(y.sum()),
        "selected_from_train_oof": selected,
        "test_alignment_path": str(args.output_dir / "matches.tsv"),
        "test_alignment_count": int(len(final)),
    }
    # This happens last by design, so a blind run can omit --test-gold entirely.
    if args.test_gold is not None:
        metrics = _alignment_metrics(final, _read_gold(args.test_gold))
        summary["test_diagnostic_after_freeze"] = metrics
        print("TEST diagnostic (opened after freeze):", metrics)
    (args.output_dir / "final_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
