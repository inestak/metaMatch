#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate a frozen candidate protocol without test leakage.

Required order:
1. load ontologies, train.tsv and a protocol tuned on train.tsv;
2. build train candidates and train the classifier;
3. derive unseen test entities from ontology classes minus train entities;
4. generate candidates, predict and save final matches;
5. only then load test.tsv and compute the final metrics.

The compact MetaSpace contains syntactic, classical embedding and optional NLP
features, plus the bidirectional SapBERT rank features from candidate tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.data.ontology_loader import OntologyLoader
from src.features.pipeline import FeaturePipeline
from src.model.evaluate import compute_alignment_metrics
from src.model.train import MetaMatchTrainer, prepare_training_data
from src.scripts.run_token_overlap_pipeline import (
    build_all_literal_token_maps,
    build_label_token_maps,
    build_matches_with_postfilter,
    infer_src_tgt_files,
)
from src.scripts.tune_train_candidate_protocol import (
    Pair,
    _add_sapbert_features,
    _build_neighborhood_documents,
    _build_retrieval_cache,
    _candidate_pairs,
    _load_or_encode_sapbert,
)


SAPBERT_FEATURES = [
    "sapbert_cosine",
    "rank_src_tgt",
    "rank_tgt_src",
    "min_rank",
    "max_rank",
    "abs_rank_difference",
    "rrf_bidirectional",
    "mutual_top1",
    "mutual_top5",
    "mutual_top10",
]


def _subset_resources(
    iris: Set[str],
    label_map: Dict[str, str],
    token_map: Dict[str, Set[str]],
) -> Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    labels = {iri: label_map.get(iri, "") for iri in iris if iri in label_map}
    tokens = {iri: token_map.get(iri, set()) for iri in labels}
    inverted: Dict[str, Set[str]] = {}
    for iri, values in tokens.items():
        for token in values:
            inverted.setdefault(token, set()).add(iri)
    return labels, tokens, inverted


def _apply_sapbert_filter(features: pd.DataFrame, rule: str, k: int) -> pd.DataFrame:
    if rule == "none":
        return features.copy()
    if rule == "and":
        mask = (features["rank_src_tgt"] <= k) & (features["rank_tgt_src"] <= k)
    elif rule == "or":
        mask = (features["rank_src_tgt"] <= k) | (features["rank_tgt_src"] <= k)
    else:
        raise ValueError(f"Unknown SapBERT filter rule: {rule}")
    return features.loc[mask].reset_index(drop=True)


def _pair_frame(
    features: pd.DataFrame,
    src_labels: Dict[str, str],
    tgt_labels: Dict[str, str],
) -> pd.DataFrame:
    pairs = features[["src_iri", "tgt_iri", "label"]].copy()
    pairs["src_label"] = pairs["src_iri"].map(src_labels).fillna("")
    pairs["tgt_label"] = pairs["tgt_iri"].map(tgt_labels).fillna("")
    return pairs


def _embedding_arrays(
    pairs: pd.DataFrame,
    src_embeddings: pd.DataFrame,
    tgt_embeddings: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    src = src_embeddings.reindex(pairs["src_iri"]).fillna(0.0).to_numpy(dtype=np.float32)
    tgt = tgt_embeddings.reindex(pairs["tgt_iri"]).fillna(0.0).to_numpy(dtype=np.float32)
    return src, tgt


def _compute_metaspace(
    pairs: pd.DataFrame,
    sapbert_features: pd.DataFrame,
    src_embeddings: pd.DataFrame,
    tgt_embeddings: pd.DataFrame,
    use_nlp: bool,
    include_sapbert_features: bool = True,
    batch_size: int = 50000,
) -> pd.DataFrame:
    pipeline = FeaturePipeline(
        use_syntax=True,
        use_classical=True,
        use_spectral=False,
        use_topological=False,
        use_nlp=use_nlp,
    )
    parts: List[pd.DataFrame] = []
    for start in range(0, len(pairs), batch_size):
        stop = min(start + batch_size, len(pairs))
        pair_batch = pairs.iloc[start:stop].reset_index(drop=True)
        src_vec, tgt_vec = _embedding_arrays(
            pair_batch, src_embeddings, tgt_embeddings
        )
        parts.append(
            pipeline.compute_features_batch(
                pair_batch,
                src_vec,
                tgt_vec,
                show_progress=len(pairs) <= batch_size,
            ).reset_index(drop=True)
        )
        if len(pairs) > batch_size:
            print(f"  Métaspace: {stop}/{len(pairs)} paires")
    base = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if include_sapbert_features:
        for name in SAPBERT_FEATURES:
            base[name] = sapbert_features[name].to_numpy()
    return base


def _oof_scores_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: Sequence[str],
    model_family: str,
    folds: int,
    xgb_params: Dict[str, Any] | None = None,
) -> np.ndarray:
    splitter = GroupKFold(n_splits=folds)
    scores = np.zeros(len(y), dtype=np.float64)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        trainer = MetaMatchTrainer(model_type=model_family, xgb_params=xgb_params)
        trainer.train(
            X[train_idx],
            y[train_idx],
            feature_names=list(feature_names),
            verbose=False,
        )
        scores[valid_idx] = trainer.predict_proba(X[valid_idx])
        print(f"  Fold classifieur {fold}/{folds}: {len(valid_idx)} candidats")
    return scores


def _parse_number_grid(raw: str, cast, name: str) -> List[Any]:
    try:
        values = sorted({cast(part.strip()) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated numeric list"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return values


def _select_train_model_protocol(
    X: np.ndarray,
    y: np.ndarray,
    pairs: pd.DataFrame,
    gold: Set[Pair],
    feature_names: Sequence[str],
    groups: np.ndarray,
    folds: int,
    scale_pos_weights: Sequence[float],
    max_depths: Sequence[int],
    n_estimators_grid: Sequence[int],
    post_filter_modes: Sequence[str],
    max_seconds: float = 0.0,
) -> Tuple[Dict[str, Any], str, float, dict, np.ndarray, pd.DataFrame]:
    """Jointly select XGBoost and decision protocol from train OOF scores only."""
    rows: List[dict] = []
    best_key: Tuple[float, float, float, float, int, int] | None = None
    best_payload = None
    search_started = time.time()
    total = len(scale_pos_weights) * len(max_depths) * len(n_estimators_grid)
    current = 0
    for weight in scale_pos_weights:
        for depth in max_depths:
            for n_estimators in n_estimators_grid:
                if current > 0 and max_seconds > 0 and time.time() - search_started >= max_seconds:
                    print(
                        f"  Budget recherche modèle atteint ({max_seconds / 60:.1f} min); "
                        "meilleure configuration courante conservée."
                    )
                    break
                current += 1
                params: Dict[str, Any] = {
                    "scale_pos_weight": float(weight),
                    "max_depth": int(depth),
                    "n_estimators": int(n_estimators),
                }
                print(
                    f"  Recherche XGBoost {current}/{total}: "
                    f"weight={weight:g}, depth={depth}, trees={n_estimators}"
                )
                scores = _oof_scores_grouped(
                    X,
                    y,
                    groups,
                    feature_names,
                    "xgboost",
                    folds,
                    xgb_params=params,
                )
                for mode in post_filter_modes:
                    threshold, metrics = _select_train_threshold(
                        pairs, scores, gold, mode
                    )
                    row = {
                        "scale_pos_weight": float(weight),
                        "max_depth": int(depth),
                        "n_estimators": int(n_estimators),
                        "post_filter_mode": mode,
                        **metrics,
                    }
                    rows.append(row)
                    # F1 is the objective. Recall and precision break genuine ties;
                    # the remaining terms prefer the cheaper/simpler model.
                    key = (
                        float(metrics["f1"]),
                        float(metrics["recall"]),
                        float(metrics["precision"]),
                        -float(weight),
                        -int(depth),
                        -int(n_estimators),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_payload = (
                            params.copy(),
                            mode,
                            threshold,
                            metrics.copy(),
                            scores.copy(),
                        )
            if max_seconds > 0 and time.time() - search_started >= max_seconds:
                break
        if max_seconds > 0 and time.time() - search_started >= max_seconds:
            break
    assert best_payload is not None
    search = pd.DataFrame(rows).sort_values(
        ["f1", "recall", "precision"], ascending=[False, False, False]
    )
    params, mode, threshold, metrics, scores = best_payload
    return params, mode, threshold, metrics, scores, search


def _select_train_threshold(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    train_gold: Set[Pair],
    post_filter_mode: str,
) -> Tuple[float, dict]:
    pred = pairs[["src_iri", "tgt_iri"]].copy()
    pred["score"] = scores
    best: dict | None = None
    for threshold in np.arange(0.01, 1.0, 0.01):
        matches = build_matches_with_postfilter(
            pred,
            threshold=float(threshold),
            mode=post_filter_mode,
        )
        predicted = set(zip(matches["SrcEntity"], matches["TgtEntity"]))
        metrics = compute_alignment_metrics(predicted, train_gold)
        row = {**metrics, "threshold": float(threshold)}
        if best is None or (row["f1"], row["recall"], row["precision"]) > (
            best["f1"], best["recall"], best["precision"]
        ):
            best = row
    assert best is not None
    return float(best["threshold"]), best


def _build_candidate_features(
    query_src: Sequence[str],
    query_tgt: Sequence[str],
    candidate_src_labels: Dict[str, str],
    candidate_src_tokens: Dict[str, Set[str]],
    candidate_src_inv: Dict[str, Set[str]],
    candidate_tgt_labels: Dict[str, str],
    candidate_tgt_tokens: Dict[str, Set[str]],
    candidate_tgt_inv: Dict[str, Set[str]],
    src_neighborhood_docs: Dict[str, str],
    tgt_neighborhood_docs: Dict[str, str],
    lexical: dict,
    sapbert_filter: dict,
    src_embeddings: pd.DataFrame,
    tgt_embeddings: pd.DataFrame,
    gold: Set[Pair],
    rrf_constant: float,
    retrieval_direction: str = "bidirectional",
) -> pd.DataFrame:
    dummy = pd.DataFrame(columns=["SrcEntity", "TgtEntity"])
    minimum = int(lexical["min_common_tokens"])
    k_overlap = int(lexical["k_overlap"])
    k_tfidf = int(lexical["k_tfidf"])
    k_neighborhood = int(lexical.get("k_neighborhood_tfidf", 0))
    k_word_tfidf = int(lexical.get("k_word_tfidf", 0))
    cache = _build_retrieval_cache(
        dummy,
        candidate_src_labels,
        candidate_src_tokens,
        candidate_src_inv,
        candidate_tgt_labels,
        candidate_tgt_tokens,
        candidate_tgt_inv,
        [minimum],
        k_overlap,
        k_tfidf,
        k_neighborhood,
        src_neighborhood_docs,
        tgt_neighborhood_docs,
        max_k_word_tfidf=k_word_tfidf,
        word_ngram_range=(
            int(lexical.get("word_ngram_min", 1)),
            int(lexical.get("word_ngram_max", 3)),
        ),
        src_queries_override=(
            [] if retrieval_direction == "tgt_to_src" else query_src
        ),
        tgt_queries_override=(
            [] if retrieval_direction == "src_to_tgt" else query_tgt
        ),
    )
    candidates = _candidate_pairs(
        cache, minimum, k_overlap, k_tfidf, k_neighborhood, k_word_tfidf
    )
    features = _add_sapbert_features(
        candidates,
        gold,
        src_embeddings,
        tgt_embeddings,
        rrf_constant,
    )
    return _apply_sapbert_filter(
        features,
        str(sapbert_filter["filter_rule"]),
        int(sapbert_filter["k_sapbert"]),
    )


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    pair_dir = BIOML_DIR / args.pair
    tuning_dir = OUTPUTS_DIR / args.pair / args.tuning_subdir
    protocol_path = tuning_dir / args.protocol_filename
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"Frozen protocol not found: {protocol_path}. Run tune_train_candidate_protocol first."
        )
    with open(protocol_path, encoding="utf-8") as handle:
        protocol = json.load(handle)
    lexical = protocol["selected_lexical"]
    sapbert_filter = protocol["selected_sapbert_filter"]
    if not sapbert_filter:
        raise ValueError("The frozen protocol has no SapBERT filter")
    frozen_recall = float(sapbert_filter.get("candidate_recall", 0.0))
    if frozen_recall < args.required_candidate_recall:
        raise ValueError(
            f"Frozen protocol candidate_recall={frozen_recall:.4f} is below the required "
            f"minimum {args.required_candidate_recall:.4f}. Retune candidate generation first."
        )

    output_dir = OUTPUTS_DIR / args.pair / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = pair_dir / f"refs_{args.task_type}"
    train_path = refs_dir / "train.tsv"
    train_df = pd.read_csv(train_path, sep="\t").drop_duplicates(
        ["SrcEntity", "TgtEntity"]
    )
    train_gold = set(zip(train_df["SrcEntity"].astype(str), train_df["TgtEntity"].astype(str)))

    print("=" * 78)
    print(f"Frozen train-only protocol + final evaluation: {args.pair}")
    print("GUARD: test.tsv will be loaded only after predictions are saved")
    print("=" * 78)
    print("Protocol lexical:", lexical)
    print("Protocol SapBERT:", sapbert_filter)

    src_file, tgt_file = infer_src_tgt_files(pair_dir, args.pair)
    src_onto = OntologyLoader(src_file).load()
    tgt_onto = OntologyLoader(tgt_file).load()
    token_source = protocol.get("token_source", "all_literals")
    retrieval_direction = protocol.get("retrieval_direction", "bidirectional")
    if retrieval_direction not in {"bidirectional", "src_to_tgt", "tgt_to_src"}:
        raise ValueError(f"Invalid frozen retrieval_direction={retrieval_direction!r}")
    if token_source == "all_literals":
        src_labels, src_tokens, src_inv = build_all_literal_token_maps(src_onto)
        tgt_labels, tgt_tokens, tgt_inv = build_all_literal_token_maps(tgt_onto)
    else:
        src_labels, src_tokens, src_inv = build_label_token_maps(src_onto)
        tgt_labels, tgt_tokens, tgt_inv = build_label_token_maps(tgt_onto)
    max_neighbors = int(protocol.get("neighborhood_max_neighbors", 20))
    src_neighborhood = _build_neighborhood_documents(src_onto, src_labels, max_neighbors)
    tgt_neighborhood = _build_neighborhood_documents(tgt_onto, tgt_labels, max_neighbors)

    src_emb, tgt_emb = _load_or_encode_sapbert(
        src_onto,
        tgt_onto,
        args.pair,
        tuning_dir / "embeddings_cache",
        args.model,
        args.device,
        args.use_synonyms,
        args.embedding_batch_size,
        args.max_length,
    )

    print("\n[1/5] Candidats d'entraînement...")
    train_candidates = _build_candidate_features(
        sorted(set(train_df["SrcEntity"].astype(str))),
        sorted(set(train_df["TgtEntity"].astype(str))),
        src_labels,
        src_tokens,
        src_inv,
        tgt_labels,
        tgt_tokens,
        tgt_inv,
        src_neighborhood,
        tgt_neighborhood,
        lexical,
        sapbert_filter,
        src_emb,
        tgt_emb,
        train_gold,
        float(protocol.get("rrf_constant", 10.0)),
        retrieval_direction,
    )
    train_pairs = _pair_frame(train_candidates, src_labels, tgt_labels)
    print(f"  Train candidates: {len(train_pairs)} | positifs couverts: {int(train_pairs.label.sum())}/{len(train_gold)}")

    print("\n[2/5] Métaspace compact d'entraînement...")
    train_features = _compute_metaspace(
        train_pairs,
        train_candidates,
        src_emb,
        tgt_emb,
        args.use_nlp,
        args.include_sapbert_features,
    )
    X_train, y_train, feature_names = prepare_training_data(train_features, train_pairs)
    selected_xgb_params: Dict[str, Any] = {}
    selected_post_filter = args.post_filter_mode
    if args.auto_xgb_search:
        if args.model_family != "xgboost":
            raise ValueError("--auto-xgb-search requires --model-family xgboost")
        weights = _parse_number_grid(
            args.xgb_scale_pos_weight_grid, float, "xgb-scale-pos-weight-grid"
        )
        depths = _parse_number_grid(args.xgb_max_depth_grid, int, "xgb-max-depth-grid")
        trees = _parse_number_grid(
            args.xgb_n_estimators_grid, int, "xgb-n-estimators-grid"
        )
        post_modes = [part.strip() for part in args.post_filter_grid.split(",") if part.strip()]
        valid_modes = {"none", "top1_src", "mutual_best", "greedy_1to1"}
        if not post_modes or any(mode not in valid_modes for mode in post_modes):
            raise ValueError(f"Invalid --post-filter-grid; allowed={sorted(valid_modes)}")
        (
            selected_xgb_params,
            selected_post_filter,
            threshold,
            oof_metrics,
            oof_scores,
            model_search,
        ) = _select_train_model_protocol(
            X_train,
            y_train,
            train_pairs,
            train_gold,
            feature_names,
            train_pairs["src_iri"].to_numpy(),
            args.folds,
            weights,
            depths,
            trees,
            post_modes,
            args.model_search_max_minutes * 60.0,
        )
        model_search.to_csv(output_dir / "model_protocol_grid_train_only.csv", index=False)
        print(
            "  Protocole modèle OOF sélectionné:",
            {**selected_xgb_params, "post_filter_mode": selected_post_filter,
             "threshold": threshold, "f1": oof_metrics["f1"]},
        )
    else:
        oof_scores = _oof_scores_grouped(
            X_train,
            y_train,
            train_pairs["src_iri"].to_numpy(),
            feature_names,
            args.model_family,
            args.folds,
        )
        threshold, oof_metrics = _select_train_threshold(
            train_pairs, oof_scores, train_gold, selected_post_filter
        )
    print(f"  Seuil OOF figé: {threshold:.2f} | F1 OOF={oof_metrics['f1']:.4f}")
    trainer = MetaMatchTrainer(
        model_type=args.model_family,
        xgb_params=selected_xgb_params or None,
    )
    trainer.threshold = threshold
    trainer.train(X_train, y_train, feature_names=feature_names, verbose=False)
    trainer.save(str(output_dir / "model"))
    train_features.to_csv(output_dir / "train_features.csv", index=False)
    train_pairs.assign(oof_score=oof_scores).to_csv(
        output_dir / "train_oof_predictions.csv", index=False
    )

    print("\n[3/5] Construction du test depuis les entités non vues du train...")
    seen_src = set(train_df["SrcEntity"].astype(str))
    seen_tgt = set(train_df["TgtEntity"].astype(str))
    unseen_src = set(src_labels) - seen_src
    unseen_tgt = set(tgt_labels) - seen_tgt
    test_src_labels, test_src_tokens, test_src_inv = _subset_resources(
        unseen_src, src_labels, src_tokens
    )
    test_tgt_labels, test_tgt_tokens, test_tgt_inv = _subset_resources(
        unseen_tgt, tgt_labels, tgt_tokens
    )
    test_src_neighborhood = {iri: src_neighborhood[iri] for iri in test_src_labels}
    test_tgt_neighborhood = {iri: tgt_neighborhood[iri] for iri in test_tgt_labels}
    print(f"  Entités non vues source/cible: {len(unseen_src)}/{len(unseen_tgt)}")
    test_candidates = _build_candidate_features(
        sorted(unseen_src),
        sorted(unseen_tgt),
        test_src_labels,
        test_src_tokens,
        test_src_inv,
        test_tgt_labels,
        test_tgt_tokens,
        test_tgt_inv,
        test_src_neighborhood,
        test_tgt_neighborhood,
        lexical,
        sapbert_filter,
        src_emb,
        tgt_emb,
        set(),
        float(protocol.get("rrf_constant", 10.0)),
        retrieval_direction,
    )
    test_pairs = _pair_frame(test_candidates, src_labels, tgt_labels)
    test_pairs.to_csv(output_dir / "test_candidates_pre_gold.csv", index=False)
    print(f"  Test candidates après filtre SapBERT: {len(test_pairs)}")

    print("\n[4/5] Métaspace et prédictions test (toujours sans test.tsv)...")
    test_features = _compute_metaspace(
        test_pairs,
        test_candidates,
        src_emb,
        tgt_emb,
        args.use_nlp,
        args.include_sapbert_features,
    )
    X_test, _, _ = prepare_training_data(
        test_features,
        test_pairs,
        feature_cols=feature_names,
    )
    scores = trainer.predict_proba(X_test)
    prediction_frame = test_pairs[["src_iri", "tgt_iri", "src_label", "tgt_label"]].copy()
    prediction_frame["score"] = scores
    prediction_frame.to_csv(output_dir / "test_predictions_pre_gold.csv", index=False)
    matches = build_matches_with_postfilter(
        prediction_frame,
        threshold=threshold,
        mode=selected_post_filter,
    )
    matches.to_csv(output_dir / "matches_pre_gold.tsv", sep="\t", index=False)
    test_features.to_csv(output_dir / "test_features.csv", index=False)
    print(f"  Alignements finaux sauvegardés avant lecture du gold: {len(matches)}")

    if args.defer_test_evaluation:
        result = {
            "pair": args.pair,
            "threshold": threshold,
            "post_filter_mode": selected_post_filter,
            "selected_xgb_params": selected_xgb_params,
            "candidate_count": len(test_pairs),
            "retrieval_direction": retrieval_direction,
            "oof_train_metrics": oof_metrics,
            "feature_count": len(feature_names),
            "features": feature_names,
            "elapsed_seconds": time.time() - started,
            "evaluation_status": "deferred_to_final_ranker",
            "test_data_access": "never",
        }
        with open(output_dir / "final_results.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print("\n[5/5] Évaluation différée: test.tsv n'a jamais été chargé")
        print(f"Métaspace sauvegardé: {output_dir}")
        return result

    print("\n[5/5] Lecture unique de test.tsv et évaluation finale...")
    test_path = refs_dir / "test.tsv"
    test_gold_df = pd.read_csv(test_path, sep="\t")
    test_gold = set(zip(test_gold_df["SrcEntity"].astype(str), test_gold_df["TgtEntity"].astype(str)))
    predicted = set(zip(matches["SrcEntity"].astype(str), matches["TgtEntity"].astype(str)))
    metrics = compute_alignment_metrics(predicted, test_gold)
    candidate_pairs = set(zip(test_pairs["src_iri"], test_pairs["tgt_iri"]))
    result = {
        **metrics,
        "pair": args.pair,
        "threshold": threshold,
        "post_filter_mode": selected_post_filter,
        "selected_xgb_params": selected_xgb_params,
        "candidate_count": len(test_pairs),
        "retrieval_direction": retrieval_direction,
        "candidate_gold_covered": len(candidate_pairs & test_gold),
        "candidate_gold_total": len(test_gold),
        "candidate_gold_recall": len(candidate_pairs & test_gold) / max(len(test_gold), 1),
        "oof_train_metrics": oof_metrics,
        "feature_count": len(feature_names),
        "features": feature_names,
        "elapsed_seconds": time.time() - started,
        "test_data_access": "after matches_pre_gold.tsv was written",
    }
    with open(output_dir / "final_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(
        output_dir / "final_results.csv", index=False
    )
    print(
        f"FINAL precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )
    print(f"Résultats: {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a frozen train-only protocol and evaluate test.tsv only at the end"
    )
    parser.add_argument("--pair", required=True)
    parser.add_argument("--task-type", default="equiv", choices=["equiv", "subs"])
    parser.add_argument("--tuning-subdir", default="train_only_protocol_tuning")
    parser.add_argument(
        "--protocol-filename",
        default="selected_protocol_train_only.json",
        help="Frozen train-only protocol JSON inside --tuning-subdir",
    )
    parser.add_argument("--output-subdir", default="frozen_protocol_final")
    parser.add_argument("--model", default="sapbert")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-synonyms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--model-family",
        default="xgboost",
        choices=["xgboost", "extra_trees", "random_forest", "stacking"],
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--post-filter-mode",
        default="greedy_1to1",
        choices=["none", "top1_src", "mutual_best", "greedy_1to1"],
    )
    parser.add_argument("--use-nlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-sapbert-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expose SapBERT cosine/rank features to the classifier. Disabling "
            "this does not disable the upstream SapBERT candidate filter."
        ),
    )
    parser.add_argument(
        "--required-candidate-recall",
        type=float,
        default=0.90,
        help="Refuse a frozen protocol below this train candidate recall (default: 0.90)",
    )
    parser.add_argument(
        "--auto-xgb-search",
        action="store_true",
        help="Joint train-OOF search over XGBoost, threshold and post-filter",
    )
    parser.add_argument("--xgb-scale-pos-weight-grid", default="1,2,4,8")
    parser.add_argument("--xgb-max-depth-grid", default="3,5,6")
    parser.add_argument("--xgb-n-estimators-grid", default="100,200")
    parser.add_argument(
        "--post-filter-grid",
        default="top1_src,mutual_best,greedy_1to1",
    )
    parser.add_argument(
        "--model-search-max-minutes",
        type=float,
        default=0.0,
        help="Stop the train-only model grid after this many minutes; 0 disables",
    )
    parser.add_argument(
        "--defer-test-evaluation",
        action="store_true",
        help=(
            "Build and save the complete train/test candidate MetaSpace without ever "
            "opening test.tsv; a downstream final ranker performs the sole evaluation"
        ),
    )
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be >= 2")
    if not 0.0 <= args.required_candidate_recall <= 1.0:
        parser.error("--required-candidate-recall must be in [0, 1]")
    if args.model_search_max_minutes < 0:
        parser.error("--model-search-max-minutes must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
