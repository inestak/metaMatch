#!/usr/bin/env python3
"""Fast train-only experiments on a previously saved leakage-free MetaSpace."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.model.evaluate import compute_alignment_metrics
from src.model.train import MetaMatchTrainer, prepare_training_data
from src.scripts.run_frozen_train_only_protocol import (
    _oof_scores_grouped,
    _select_train_threshold,
)
from src.scripts.run_token_overlap_pipeline import build_matches_with_postfilter


Pair = Tuple[str, str]
POST_FILTERS = (
    "none",
    "top1_src",
    "mutual_best",
    "greedy_1to1",
    "mutual_then_greedy",
)
RAW_RANK_FEATURES = (
    "rank_src_tgt",
    "rank_tgt_src",
    "min_rank",
    "max_rank",
    "abs_rank_difference",
)

# Canonical 79-feature set from
# metaMatch/src/scripts/run_token_overlap_pipeline_xg.py:
# 22 syntactic + 8 classical + 15 spectral + 34 topological/TDA.
BASE79_PREFIXES = ("syn_", "cls_", "spc_", "tda_")


def _parse_float_grid(raw: str) -> List[float]:
    values = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    if not values:
        raise ValueError("Empty numeric grid")
    return values


def _augment_rank_features(
    features: pd.DataFrame,
    pairs: pd.DataFrame,
    mode: str,
) -> Tuple[pd.DataFrame, List[str]]:
    frame = features.copy()
    names = list(frame.columns)
    if mode == "original":
        return frame, names
    if mode == "base79":
        selected = [name for name in names if name.startswith(BASE79_PREFIXES)]
        if len(selected) != 79:
            raise ValueError(
                "Invalid base79 decomposition: expected exactly 79 retained "
                f"features from {len(names)} saved features, got {len(selected)}"
            )
        return frame[selected].copy(), selected

    src_count = pairs.groupby("src_iri")["tgt_iri"].transform("size").clip(lower=1)
    tgt_count = pairs.groupby("tgt_iri")["src_iri"].transform("size").clip(lower=1)
    rs = frame["rank_src_tgt"].astype(float)
    rt = frame["rank_tgt_src"].astype(float)
    frame["rank_src_percentile"] = rs / src_count.to_numpy()
    frame["rank_tgt_percentile"] = rt / tgt_count.to_numpy()
    frame["reciprocal_rank_src"] = 1.0 / rs.clip(lower=1.0)
    frame["reciprocal_rank_tgt"] = 1.0 / rt.clip(lower=1.0)
    frame["log_rank_src"] = np.log1p(rs)
    frame["log_rank_tgt"] = np.log1p(rt)

    cosine = frame["sapbert_cosine"].astype(float)
    src_best = cosine.groupby(pairs["src_iri"]).transform("max")
    tgt_best = cosine.groupby(pairs["tgt_iri"]).transform("max")
    frame["sapbert_gap_from_src_best"] = src_best - cosine
    frame["sapbert_gap_from_tgt_best"] = tgt_best - cosine

    def group_margin(keys: pd.Series) -> np.ndarray:
        work = pd.DataFrame({"key": keys.to_numpy(), "score": cosine.to_numpy()})
        unique = work.drop_duplicates(["key", "score"])
        top = unique.groupby("key")["score"].nlargest(2).reset_index(level=0)
        margins: Dict[str, float] = {}
        for key, values in top.groupby("key")["score"]:
            scores = values.to_numpy()
            margins[key] = float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0])
        return keys.map(margins).fillna(0.0).to_numpy()

    frame["sapbert_margin_src"] = group_margin(pairs["src_iri"])
    frame["sapbert_margin_tgt"] = group_margin(pairs["tgt_iri"])

    if mode == "robust":
        frame = frame.drop(columns=[c for c in RAW_RANK_FEATURES if c in frame], errors="ignore")
    elif mode == "no_ranks":
        drop = list(RAW_RANK_FEATURES) + [
            "rrf_bidirectional", "mutual_top1", "mutual_top5", "mutual_top10"
        ]
        frame = frame.drop(columns=[c for c in drop if c in frame], errors="ignore")
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    return frame, list(frame.columns)


def _best_decision_protocol(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    gold: Set[Pair],
    post_filters: Sequence[str] = POST_FILTERS,
) -> Tuple[str, float, dict]:
    best = None
    for mode in post_filters:
        threshold, metrics = _select_train_threshold(pairs, scores, gold, mode)
        key = (metrics["f1"], metrics["recall"], metrics["precision"])
        if best is None or key > best[0]:
            best = (key, mode, threshold, metrics)
    assert best is not None
    return best[1], best[2], best[3]


def _fixed_decision_protocol(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    gold: Set[Pair],
    mode: str,
    threshold: float | None,
) -> Tuple[str, float, dict]:
    if threshold is None:
        selected_threshold, metrics = _select_train_threshold(pairs, scores, gold, mode)
        return mode, selected_threshold, metrics
    pred = pairs[["src_iri", "tgt_iri"]].copy()
    pred["score"] = scores
    matches = build_matches_with_postfilter(pred, threshold, mode)
    predicted = set(zip(matches["SrcEntity"].astype(str), matches["TgtEntity"].astype(str)))
    return mode, threshold, compute_alignment_metrics(predicted, gold)


def _train_one(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str],
    folds: int,
) -> Tuple[np.ndarray, MetaMatchTrainer]:
    oof = _oof_scores_grouped(X, y, groups, names, family, folds)
    trainer = MetaMatchTrainer(model_type=family)
    trainer.train(X, y, feature_names=list(names), verbose=False)
    return oof, trainer


def _prepare_recovery(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    reverse_ranks: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred = pairs[["src_iri", "tgt_iri"]].copy()
    pred["score"] = scores
    pred["rank_tgt_src"] = np.asarray(reverse_ranks)
    ranked = pred.sort_values(["src_iri", "score"], ascending=[True, False]).copy()
    ranked["src_order"] = ranked.groupby("src_iri").cumcount()
    seconds = ranked.loc[ranked["src_order"] == 1].set_index("src_iri")["score"]
    top = ranked.loc[ranked["src_order"] == 0].copy()
    top["second_score"] = top["src_iri"].map(seconds).fillna(0.0)
    top["margin"] = top["score"] - top["second_score"]
    return pred, top


def _recovery_matches(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    reverse_ranks: np.ndarray,
    core_threshold: float,
    rescue_threshold: float,
    min_margin: float,
    max_reverse_rank: int,
    prepared: Tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    pred, top = prepared or _prepare_recovery(pairs, scores, reverse_ranks)
    core = build_matches_with_postfilter(pred, core_threshold, "greedy_1to1")
    used_src = set(core["SrcEntity"].astype(str))
    used_tgt = set(core["TgtEntity"].astype(str))
    rescue = top[
        (top["score"] >= rescue_threshold)
        & (top["margin"] >= min_margin)
        & (top["rank_tgt_src"] <= max_reverse_rank)
        & (~top["src_iri"].isin(used_src))
    ].sort_values("score", ascending=False)

    extra = []
    for row in rescue.itertuples(index=False):
        if row.tgt_iri in used_tgt:
            continue
        used_src.add(row.src_iri)
        used_tgt.add(row.tgt_iri)
        extra.append({"SrcEntity": row.src_iri, "TgtEntity": row.tgt_iri, "Score": row.score})
    if extra:
        core = pd.concat([core, pd.DataFrame(extra)], ignore_index=True)
    return core


def _select_recovery(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    reverse_ranks: np.ndarray,
    gold: Set[Pair],
    core_grid: Sequence[float],
    rescue_grid: Sequence[float],
    margin_grid: Sequence[float],
    reverse_rank_grid: Sequence[int],
) -> Tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    prepared = _prepare_recovery(pairs, scores, reverse_ranks)
    for core in core_grid:
        for rescue in rescue_grid:
            if rescue > core:
                continue
            for margin in margin_grid:
                for reverse_rank in reverse_rank_grid:
                    matches = _recovery_matches(
                        pairs,
                        scores,
                        reverse_ranks,
                        core,
                        rescue,
                        margin,
                        reverse_rank,
                        prepared=prepared,
                    )
                    predicted = set(zip(matches["SrcEntity"], matches["TgtEntity"]))
                    metrics = compute_alignment_metrics(predicted, gold)
                    row = {
                        "core_threshold": core,
                        "rescue_threshold": rescue,
                        "min_margin": margin,
                        "max_reverse_rank": reverse_rank,
                        **metrics,
                    }
                    rows.append(row)
                    key = (metrics["f1"], metrics["recall"], metrics["precision"])
                    if best is None or key > best[0]:
                        best = (key, row)
    assert best is not None
    return best[1], pd.DataFrame(rows).sort_values(
        ["f1", "recall", "precision"], ascending=[False, False, False]
    )


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    source_dir = OUTPUTS_DIR / args.pair / args.input_subdir
    output_dir = OUTPUTS_DIR / args.pair / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_result = json.loads((source_dir / "final_results.json").read_text())

    train_pairs = pd.read_csv(source_dir / "train_oof_predictions.csv").drop(
        columns=["oof_score"], errors="ignore"
    )
    test_pairs = pd.read_csv(source_dir / "test_candidates_pre_gold.csv")
    train_raw = pd.read_csv(source_dir / "train_features.csv")[source_result["features"]]
    test_raw = pd.read_csv(source_dir / "test_features.csv")[source_result["features"]]
    train_features, feature_names = _augment_rank_features(train_raw, train_pairs, args.feature_mode)
    test_features, test_names = _augment_rank_features(test_raw, test_pairs, args.feature_mode)
    if feature_names != test_names:
        raise ValueError("Train/test feature mismatch")
    if args.feature_columns:
        requested = [name.strip() for name in args.feature_columns.split(",") if name.strip()]
        missing = [name for name in requested if name not in train_features.columns]
        if missing:
            raise ValueError(f"Requested feature columns are absent: {missing}")
        train_features = train_features[requested].copy()
        test_features = test_features[requested].copy()
        feature_names = requested
        print(f"Custom feature subset: {feature_names}")
    X_train, y_train, feature_names = prepare_training_data(
        train_features, train_pairs, feature_cols=feature_names
    )
    X_test, _, _ = prepare_training_data(test_features, test_pairs, feature_cols=feature_names)
    groups = train_pairs["src_iri"].to_numpy()
    post_filters = tuple(
        part.strip() for part in args.post_filter_grid.split(",") if part.strip()
    )
    if not post_filters:
        raise ValueError("--post-filter-grid cannot be empty")
    invalid_post_filters = [mode for mode in post_filters if mode not in POST_FILTERS]
    if invalid_post_filters:
        raise ValueError(f"Unknown post filters: {invalid_post_filters}")
    train_gold_df = pd.read_csv(
        BIOML_DIR / args.pair / f"refs_{args.task_type}" / "train.tsv", sep="\t"
    )
    train_gold = set(zip(train_gold_df["SrcEntity"].astype(str), train_gold_df["TgtEntity"].astype(str)))

    print(f"Experiment={args.experiment} features={args.feature_mode}")
    print(f"Saved MetaSpace train/test={len(train_pairs)}/{len(test_pairs)}")
    trainers: Dict[str, MetaMatchTrainer] = {}
    if args.experiment in {"ensemble", "recovery"}:
        xgb_oof, trainers["xgboost"] = _train_one(
            "xgboost", X_train, y_train, groups, feature_names, args.folds
        )
        rf_oof, trainers["random_forest"] = _train_one(
            "random_forest", X_train, y_train, groups, feature_names, args.folds
        )
        best = None
        for alpha in _parse_float_grid(args.alpha_grid):
            scores = alpha * xgb_oof + (1.0 - alpha) * rf_oof
            mode, threshold, metrics = _best_decision_protocol(
                train_pairs, scores, train_gold, post_filters
            )
            key = (metrics["f1"], metrics["recall"], metrics["precision"])
            if best is None or key > best[0]:
                best = (key, alpha, scores, mode, threshold, metrics)
        assert best is not None
        _, alpha, oof_scores, mode, threshold, oof_metrics = best
        test_scores = alpha * trainers["xgboost"].predict_proba(X_test) + (
            1.0 - alpha
        ) * trainers["random_forest"].predict_proba(X_test)
        selected = {"alpha_xgboost": alpha, "post_filter_mode": mode, "threshold": threshold}
    else:
        oof_scores, trainer = _train_one(
            args.model_family, X_train, y_train, groups, feature_names, args.folds
        )
        trainers[args.model_family] = trainer
        if args.fixed_post_filter_mode:
            mode, threshold, oof_metrics = _fixed_decision_protocol(
                train_pairs,
                oof_scores,
                train_gold,
                args.fixed_post_filter_mode,
                args.fixed_threshold,
            )
        else:
            mode, threshold, oof_metrics = _best_decision_protocol(
                train_pairs, oof_scores, train_gold, post_filters
            )
        test_scores = trainer.predict_proba(X_test)
        selected = {"post_filter_mode": mode, "threshold": threshold}

    if args.experiment == "recovery":
        recovery, grid = _select_recovery(
            train_pairs,
            oof_scores,
            train_raw["rank_tgt_src"].to_numpy(),
            train_gold,
            _parse_float_grid(args.core_threshold_grid),
            _parse_float_grid(args.rescue_threshold_grid),
            _parse_float_grid(args.margin_grid),
            [int(x) for x in _parse_float_grid(args.reverse_rank_grid)],
        )
        grid.to_csv(output_dir / "recovery_grid_train_only.csv", index=False)
        matches = _recovery_matches(
            test_pairs,
            test_scores,
            test_raw["rank_tgt_src"].to_numpy(),
            recovery["core_threshold"],
            recovery["rescue_threshold"],
            recovery["min_margin"],
            int(recovery["max_reverse_rank"]),
        )
        selected.update(recovery)
        oof_metrics = {k: recovery[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}
    else:
        pred = test_pairs[["src_iri", "tgt_iri"]].copy()
        pred["score"] = test_scores
        matches = build_matches_with_postfilter(pred, threshold, mode)

    if args.train_only:
        result = {
            "pair": args.pair,
            "experiment": args.experiment,
            "model_family": args.model_family,
            "feature_mode": args.feature_mode,
            "selected_train_only": selected,
            "oof_train_metrics": oof_metrics,
            "feature_count": len(feature_names),
            "features": list(feature_names),
            "elapsed_seconds": time.time() - started,
            "source_metaspace": str(source_dir),
            "test_tsv_access": "never",
        }
        (output_dir / "train_only_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"TRAIN-ONLY F1={oof_metrics['f1']:.4f} "
            f"P={oof_metrics['precision']:.4f} R={oof_metrics['recall']:.4f}"
        )
        print("test.tsv was never loaded")
        return result

    predictions = test_pairs[["src_iri", "tgt_iri", "src_label", "tgt_label"]].copy()
    predictions["score"] = test_scores
    predictions.to_csv(output_dir / "test_predictions_pre_gold.csv", index=False)
    matches.to_csv(output_dir / "matches_pre_gold.tsv", sep="\t", index=False)
    train_pairs.assign(oof_score=oof_scores).to_csv(
        output_dir / "train_oof_predictions.csv", index=False
    )

    if args.no_test_evaluation:
        result = {
            "pair": args.pair,
            "experiment": args.experiment,
            "model_family": args.model_family,
            "feature_mode": args.feature_mode,
            "selected_train_only": selected,
            "oof_train_metrics": oof_metrics,
            "feature_count": len(feature_names),
            "features": list(feature_names),
            "alignment_count": len(matches),
            "alignment_path": str(output_dir / "matches_pre_gold.tsv"),
            "elapsed_seconds": time.time() - started,
            "source_metaspace": str(source_dir),
            "prediction_status": "complete_without_test_evaluation",
            "test_tsv_access": "never",
        }
        (output_dir / "prediction_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Selected train-only:", selected)
        print(f"ALIGNEMENTS SANS GOLD: {len(matches)} -> {output_dir / 'matches_pre_gold.tsv'}")
        print("GUARD: test.tsv n'a jamais été chargé")
        return result

    # Gold is intentionally opened only after predictions and matches exist.
    test_gold_df = pd.read_csv(
        BIOML_DIR / args.pair / f"refs_{args.task_type}" / "test.tsv", sep="\t"
    )
    test_gold = set(zip(test_gold_df["SrcEntity"].astype(str), test_gold_df["TgtEntity"].astype(str)))
    predicted = set(zip(matches["SrcEntity"].astype(str), matches["TgtEntity"].astype(str)))
    metrics = compute_alignment_metrics(predicted, test_gold)
    result = {
        **metrics,
        "pair": args.pair,
        "experiment": args.experiment,
        "model_family": args.model_family,
        "feature_mode": args.feature_mode,
        "selected_train_only": selected,
        "oof_train_metrics": oof_metrics,
        "feature_count": len(feature_names),
        "features": list(feature_names),
        "elapsed_seconds": time.time() - started,
        "source_metaspace": str(source_dir),
        "test_data_access": "after matches_pre_gold.tsv was written",
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Selected train-only:", selected)
    print(
        f"FINAL precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} elapsed={result['elapsed_seconds']:.1f}s"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--task-type", default="equiv", choices=["equiv", "subs"])
    parser.add_argument("--input-subdir", default="frozen_protocol_final")
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--experiment", default="model", choices=["model", "ensemble", "recovery"])
    parser.add_argument(
        "--model-family",
        default="xgboost",
        choices=["xgboost", "random_forest", "extra_trees", "stacking"],
    )
    parser.add_argument(
        "--feature-mode",
        default="original",
        choices=["original", "base79", "robust", "no_ranks"],
        help=(
            "original=all saved features; base79=the exact 22 syn + 8 cls + "
            "15 spc + 34 tda features from run_token_overlap_pipeline_xg.py"
        ),
    )
    parser.add_argument(
        "--feature-columns",
        default="",
        help="Optional comma-separated exact feature subset, e.g. overlap_cos,sapbert_cosine",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--alpha-grid", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--fixed-post-filter-mode", choices=POST_FILTERS)
    parser.add_argument(
        "--post-filter-grid",
        default=",".join(POST_FILTERS),
        help="Post-filters compared on train OOF when no fixed mode is supplied",
    )
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--core-threshold-grid", default="0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--rescue-threshold-grid", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--margin-grid", default="0,0.02,0.05,0.1")
    parser.add_argument("--reverse-rank-grid", default="1,3,5,10")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Select and report from train OOF only; never open test.tsv",
    )
    parser.add_argument(
        "--no-test-evaluation",
        action="store_true",
        help="Save frozen test predictions and alignments without opening test.tsv",
    )
    args = parser.parse_args()
    if args.fixed_threshold is not None and not args.fixed_post_filter_mode:
        parser.error("--fixed-threshold requires --fixed-post-filter-mode")
    if args.fixed_post_filter_mode and args.experiment != "model":
        parser.error("--fixed-post-filter-mode is currently supported with --experiment model")
    if args.train_only and args.no_test_evaluation:
        parser.error("Choose either --train-only or --no-test-evaluation")
    run(args)


if __name__ == "__main__":
    main()
