#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tune the lexical and SapBERT candidate filters using train.tsv only.

This script deliberately never loads test.tsv or test.cands.tsv.  The gold
alignments in train.tsv are split into entity-disjoint connected components for
reporting fold recall. Candidate retrieval itself is unsupervised: it uses only
ontology labels/literals and SapBERT embeddings.

The protocol is selected lexicographically:
1. stay within ``recall_tolerance`` of the best candidate recall;
2. among eligible configurations, keep the one with the fewest candidates.

Usage:
    python -m src.scripts.tune_train_candidate_protocol --pair omim-ordo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.data.ontology_loader import OntologyLoader
from src.scripts.run_token_overlap_pipeline import (
    build_all_literal_token_maps,
    build_label_token_maps,
    build_tfidf_index,
    build_word_tfidf_index,
    infer_src_tgt_files,
    topk_tfidf_candidates,
    topk_token_overlap_candidates,
)


Pair = Tuple[str, str]


def _parse_int_grid(raw: str, name: str, allow_zero: bool = False) -> List[int]:
    try:
        values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from exc
    minimum = 0 if allow_zero else 1
    if not values or any(value < minimum for value in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"{name} values must all be {qualifier}")
    return values


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _component_folds(gold_df: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Assign whole bipartite alignment components to balanced folds."""
    uf = UnionFind()
    for row in gold_df.itertuples(index=False):
        uf.union(f"src::{row.SrcEntity}", f"tgt::{row.TgtEntity}")

    component_rows: Dict[str, List[int]] = {}
    for idx, row in enumerate(gold_df.itertuples(index=False)):
        root = uf.find(f"src::{row.SrcEntity}")
        component_rows.setdefault(root, []).append(idx)

    fold_sizes = [0] * n_folds
    assignments = np.zeros(len(gold_df), dtype=np.int32)
    components = sorted(component_rows.values(), key=lambda rows: (-len(rows), rows[0]))
    for rows in components:
        fold = int(np.argmin(fold_sizes))
        assignments[rows] = fold
        fold_sizes[fold] += len(rows)
    return assignments


def _ordered_union(*parts: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for part in parts:
        for iri in part:
            if iri not in seen:
                seen.add(iri)
                result.append(iri)
    return result


@dataclass
class RetrievalCache:
    src_overlap: Dict[Tuple[str, int], List[str]]
    tgt_overlap: Dict[Tuple[str, int], List[str]]
    src_tfidf: Dict[str, List[str]]
    tgt_tfidf: Dict[str, List[str]]
    src_neighborhood_tfidf: Dict[str, List[str]]
    tgt_neighborhood_tfidf: Dict[str, List[str]]
    src_word_tfidf: Dict[str, List[str]]
    tgt_word_tfidf: Dict[str, List[str]]


def _build_neighborhood_documents(
    onto: OntologyLoader,
    label_map: Dict[str, str],
    max_neighbors: int,
) -> Dict[str, str]:
    """Build one-hop textual documents: entity + parents + children.

    Neighbors are sorted before the optional cap so the result is deterministic.
    Very high-degree ontology nodes are capped to avoid huge, generic documents.
    """
    documents: Dict[str, str] = {}
    for iri in tqdm(label_map, desc="Documents TF-IDF voisinage"):
        neighbors = sorted(set(onto.get_parents(iri)) | set(onto.get_children(iri)))
        if max_neighbors > 0:
            neighbors = neighbors[:max_neighbors]
        parts = [label_map.get(iri, "")]
        parts.extend(label_map.get(neighbor, "") for neighbor in neighbors)
        documents[iri] = " | ".join(part for part in parts if part)
    return documents


def _build_retrieval_cache(
    train_df: pd.DataFrame,
    src_label_map: Dict[str, str],
    src_token_map: Dict[str, Set[str]],
    src_inv: Dict[str, Set[str]],
    tgt_label_map: Dict[str, str],
    tgt_token_map: Dict[str, Set[str]],
    tgt_inv: Dict[str, Set[str]],
    min_common_grid: Sequence[int],
    max_k_overlap: int,
    max_k_tfidf: int,
    max_k_neighborhood_tfidf: int,
    src_neighborhood_documents: Dict[str, str],
    tgt_neighborhood_documents: Dict[str, str],
    max_k_word_tfidf: int = 0,
    word_ngram_range: Tuple[int, int] = (1, 3),
    src_queries_override: Sequence[str] | None = None,
    tgt_queries_override: Sequence[str] | None = None,
) -> RetrievalCache:
    """Precompute maximum rankings once, then grids only slice cached lists."""
    src_queries = (
        sorted(set(map(str, src_queries_override)))
        if src_queries_override is not None
        else sorted(set(train_df["SrcEntity"].astype(str)))
    )
    tgt_queries = (
        sorted(set(map(str, tgt_queries_override)))
        if tgt_queries_override is not None
        else sorted(set(train_df["TgtEntity"].astype(str)))
    )

    tgt_iris, tgt_vectorizer, tgt_matrix = build_tfidf_index(
        tgt_label_map, list(src_label_map.values())
    )
    src_iris, src_vectorizer, src_matrix = build_tfidf_index(
        src_label_map, list(tgt_label_map.values())
    )
    tgt_neigh_iris: List[str] = []
    tgt_neigh_vectorizer = None
    tgt_neigh_matrix = None
    src_neigh_iris: List[str] = []
    src_neigh_vectorizer = None
    src_neigh_matrix = None
    tgt_word_iris: List[str] = []
    tgt_word_vectorizer = None
    tgt_word_matrix = None
    src_word_iris: List[str] = []
    src_word_vectorizer = None
    src_word_matrix = None
    if max_k_neighborhood_tfidf > 0:
        tgt_neigh_iris, tgt_neigh_vectorizer, tgt_neigh_matrix = build_tfidf_index(
            tgt_neighborhood_documents, list(src_neighborhood_documents.values())
        )
        src_neigh_iris, src_neigh_vectorizer, src_neigh_matrix = build_tfidf_index(
            src_neighborhood_documents, list(tgt_neighborhood_documents.values())
        )
    if max_k_word_tfidf > 0:
        tgt_word_iris, tgt_word_vectorizer, tgt_word_matrix = build_word_tfidf_index(
            tgt_label_map, list(src_label_map.values()), word_ngram_range
        )
        src_word_iris, src_word_vectorizer, src_word_matrix = build_word_tfidf_index(
            src_label_map, list(tgt_label_map.values()), word_ngram_range
        )

    src_overlap: Dict[Tuple[str, int], List[str]] = {}
    tgt_overlap: Dict[Tuple[str, int], List[str]] = {}
    src_tfidf: Dict[str, List[str]] = {}
    tgt_tfidf: Dict[str, List[str]] = {}
    src_neighborhood_tfidf: Dict[str, List[str]] = {}
    tgt_neighborhood_tfidf: Dict[str, List[str]] = {}
    src_word_tfidf: Dict[str, List[str]] = {}
    tgt_word_tfidf: Dict[str, List[str]] = {}

    for src in tqdm(src_queries, desc="Cache lexical source -> target"):
        for minimum in min_common_grid:
            src_overlap[(src, minimum)] = topk_token_overlap_candidates(
                query_tokens=src_token_map.get(src, set()),
                query_label=src_label_map.get(src, ""),
                cand_token_map=tgt_token_map,
                cand_label_map=tgt_label_map,
                inv_index=tgt_inv,
                k=max_k_overlap,
                min_common_tokens=minimum,
            )
        src_tfidf[src] = topk_tfidf_candidates(
            query_label=src_label_map.get(src, ""),
            cand_iris=tgt_iris,
            vectorizer=tgt_vectorizer,
            cand_matrix=tgt_matrix,
            k=max_k_tfidf,
        )
        src_neighborhood_tfidf[src] = topk_tfidf_candidates(
            query_label=src_neighborhood_documents.get(src, ""),
            cand_iris=tgt_neigh_iris,
            vectorizer=tgt_neigh_vectorizer,
            cand_matrix=tgt_neigh_matrix,
            k=max_k_neighborhood_tfidf,
        ) if max_k_neighborhood_tfidf > 0 else []
        src_word_tfidf[src] = topk_tfidf_candidates(
            query_label=src_label_map.get(src, ""),
            cand_iris=tgt_word_iris,
            vectorizer=tgt_word_vectorizer,
            cand_matrix=tgt_word_matrix,
            k=max_k_word_tfidf,
            min_similarity=0.0,
        ) if max_k_word_tfidf > 0 else []

    for tgt in tqdm(tgt_queries, desc="Cache lexical target -> source"):
        for minimum in min_common_grid:
            tgt_overlap[(tgt, minimum)] = topk_token_overlap_candidates(
                query_tokens=tgt_token_map.get(tgt, set()),
                query_label=tgt_label_map.get(tgt, ""),
                cand_token_map=src_token_map,
                cand_label_map=src_label_map,
                inv_index=src_inv,
                k=max_k_overlap,
                min_common_tokens=minimum,
            )
        tgt_tfidf[tgt] = topk_tfidf_candidates(
            query_label=tgt_label_map.get(tgt, ""),
            cand_iris=src_iris,
            vectorizer=src_vectorizer,
            cand_matrix=src_matrix,
            k=max_k_tfidf,
        )
        tgt_neighborhood_tfidf[tgt] = topk_tfidf_candidates(
            query_label=tgt_neighborhood_documents.get(tgt, ""),
            cand_iris=src_neigh_iris,
            vectorizer=src_neigh_vectorizer,
            cand_matrix=src_neigh_matrix,
            k=max_k_neighborhood_tfidf,
        ) if max_k_neighborhood_tfidf > 0 else []
        tgt_word_tfidf[tgt] = topk_tfidf_candidates(
            query_label=tgt_label_map.get(tgt, ""),
            cand_iris=src_word_iris,
            vectorizer=src_word_vectorizer,
            cand_matrix=src_word_matrix,
            k=max_k_word_tfidf,
            min_similarity=0.0,
        ) if max_k_word_tfidf > 0 else []

    return RetrievalCache(
        src_overlap,
        tgt_overlap,
        src_tfidf,
        tgt_tfidf,
        src_neighborhood_tfidf,
        tgt_neighborhood_tfidf,
        src_word_tfidf,
        tgt_word_tfidf,
    )


def _candidate_pairs(
    cache: RetrievalCache,
    min_common_tokens: int,
    k_overlap: int,
    k_tfidf: int,
    k_neighborhood_tfidf: int,
    k_word_tfidf: int = 0,
) -> Set[Pair]:
    pairs: Set[Pair] = set()
    src_queries = sorted({src for src, _ in cache.src_overlap})
    tgt_queries = sorted({tgt for tgt, _ in cache.tgt_overlap})

    for src in src_queries:
        candidates = _ordered_union(
            cache.src_overlap[(src, min_common_tokens)][:k_overlap],
            cache.src_tfidf[src][:k_tfidf],
            cache.src_neighborhood_tfidf[src][:k_neighborhood_tfidf],
            cache.src_word_tfidf[src][:k_word_tfidf],
        )
        pairs.update((src, tgt) for tgt in candidates)

    for tgt in tgt_queries:
        candidates = _ordered_union(
            cache.tgt_overlap[(tgt, min_common_tokens)][:k_overlap],
            cache.tgt_tfidf[tgt][:k_tfidf],
            cache.tgt_neighborhood_tfidf[tgt][:k_neighborhood_tfidf],
            cache.tgt_word_tfidf[tgt][:k_word_tfidf],
        )
        pairs.update((src, tgt) for src in candidates)
    return pairs


def _recall_metrics(
    candidate_pairs: Set[Pair],
    gold_pairs: Set[Pair],
    fold_gold: Sequence[Set[Pair]],
) -> Dict[str, float]:
    covered = len(candidate_pairs & gold_pairs)
    row: Dict[str, float] = {
        "candidate_count": len(candidate_pairs),
        "gold_covered": covered,
        "gold_total": len(gold_pairs),
        "candidate_recall": covered / max(len(gold_pairs), 1),
        "negative_candidate_count": len(candidate_pairs - gold_pairs),
        "negative_per_covered_positive": len(candidate_pairs - gold_pairs) / max(covered, 1),
    }
    fold_recalls: List[float] = []
    for fold_idx, gold in enumerate(fold_gold):
        recall = len(candidate_pairs & gold) / max(len(gold), 1)
        row[f"recall_fold_{fold_idx + 1}"] = recall
        fold_recalls.append(recall)
    row["macro_fold_recall"] = float(np.mean(fold_recalls)) if fold_recalls else 0.0
    row["min_fold_recall"] = float(np.min(fold_recalls)) if fold_recalls else 0.0
    return row


def _select_smallest_near_best(results: pd.DataFrame, tolerance: float) -> pd.Series:
    best_recall = float(results["candidate_recall"].max())
    eligible = results[results["candidate_recall"] >= best_recall - tolerance].copy()
    eligible = eligible.sort_values(
        ["candidate_count", "candidate_recall", "min_fold_recall"],
        ascending=[True, False, False],
    )
    return eligible.iloc[0]


def _select_smallest_above_recall(
    results: pd.DataFrame,
    minimum_recall: float,
) -> pd.Series:
    eligible = results[results["candidate_recall"] >= minimum_recall].copy()
    if eligible.empty:
        best = float(results["candidate_recall"].max())
        raise ValueError(
            f"Aucune configuration n'atteint candidate_recall >= {minimum_recall:.4f}; "
            f"meilleur recall disponible={best:.4f}. Élargir la génération de candidats."
        )
    return eligible.sort_values(
        ["candidate_count", "candidate_recall", "min_fold_recall"],
        ascending=[True, False, False],
    ).iloc[0]


def _select_sapbert_with_budget(
    results: pd.DataFrame,
    tolerance: float,
    max_negative_ratio: float,
    max_recall_loss: float,
    minimum_recall: float,
) -> pd.Series:
    """Choose maximum recall under a candidate budget, then smallest space.

    With no budget, retain the original near-best-recall policy. With a budget,
    the recall is maximized among configurations respecting that budget; ties
    within ``tolerance`` are resolved by candidate count.
    """
    baseline_rows = results[results["filter_rule"] == "none"]
    baseline_recall = float(
        baseline_rows.iloc[0]["candidate_recall"]
        if not baseline_rows.empty
        else results["candidate_recall"].max()
    )
    if baseline_recall < minimum_recall:
        raise ValueError(
            f"Le générateur avant SapBERT atteint seulement {baseline_recall:.4f}, "
            f"sous le minimum requis {minimum_recall:.4f}."
        )
    recall_floor = max(minimum_recall, baseline_recall - max_recall_loss)

    recall_safe = results[results["candidate_recall"] >= recall_floor].copy()
    if recall_safe.empty:
        recall_safe = results[results["candidate_recall"] == results["candidate_recall"].max()].copy()

    if max_negative_ratio <= 0:
        return _select_smallest_near_best(recall_safe, tolerance)

    eligible = recall_safe[
        recall_safe["negative_per_covered_positive"] <= max_negative_ratio
    ].copy()
    if eligible.empty:
        print(
            f"WARNING: aucun filtre ne respecte à la fois ratio <= {max_negative_ratio:.2f} "
            f"et perte de recall <= {max_recall_loss:.4f}; le recall est prioritaire."
        )
        return recall_safe.sort_values(
            ["candidate_count", "candidate_recall", "min_fold_recall"],
            ascending=[True, False, False],
        ).iloc[0]
    best_budget_recall = float(eligible["candidate_recall"].max())
    near_best = eligible[
        eligible["candidate_recall"] >= best_budget_recall - tolerance
    ].copy()
    return near_best.sort_values(
        ["candidate_count", "candidate_recall", "min_fold_recall"],
        ascending=[True, False, False],
    ).iloc[0]


def _load_or_encode_sapbert(
    src_onto: OntologyLoader,
    tgt_onto: OntologyLoader,
    pair_name: str,
    cache_dir: Path,
    model_name: str,
    device: str,
    use_synonyms: bool,
    batch_size: int,
    max_length: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Import lazily so lexical-only tuning does not initialize the transformer
    # stack (or its transitive plotting/ML imports).
    from src.scripts.run_sapbert_bidir_baseline import SapBertVectorEncoder

    model_path = (
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        if model_name == "sapbert"
        else model_name
    )
    safe_model = model_path.replace("/", "_").replace("\\", "_")
    synonym_tag = "syn" if use_synonyms else "label"
    src_path = cache_dir / f"{pair_name}_src_{safe_model}_{synonym_tag}_vec.pkl"
    tgt_path = cache_dir / f"{pair_name}_tgt_{safe_model}_{synonym_tag}_vec.pkl"

    if src_path.exists() and tgt_path.exists():
        print(f"Chargement caches SapBERT: {src_path.name}, {tgt_path.name}")
        return pd.read_pickle(src_path), pd.read_pickle(tgt_path)

    encoder = SapBertVectorEncoder(model_name=model_name, device=device)
    src_emb = encoder.encode_ontology(
        src_onto, src_path, use_synonyms, batch_size, max_length
    )
    tgt_emb = encoder.encode_ontology(
        tgt_onto, tgt_path, use_synonyms, batch_size, max_length
    )
    return src_emb, tgt_emb


def _add_sapbert_features(
    candidate_pairs: Set[Pair],
    gold_pairs: Set[Pair],
    src_embeddings: pd.DataFrame,
    tgt_embeddings: pd.DataFrame,
    rrf_constant: float,
    vector_batch_size: int = 50000,
) -> pd.DataFrame:
    rows = sorted(candidate_pairs)
    src_dim = src_embeddings.shape[1]
    tgt_dim = tgt_embeddings.shape[1]
    if src_dim != tgt_dim:
        raise ValueError(f"Embedding dimensions differ: source={src_dim}, target={tgt_dim}")

    cosine_parts: List[np.ndarray] = []
    for start in range(0, len(rows), vector_batch_size):
        batch = rows[start : start + vector_batch_size]
        src_vectors = np.vstack([
            src_embeddings.loc[src].to_numpy(dtype=np.float32, copy=False)
            if src in src_embeddings.index else np.zeros(src_dim, dtype=np.float32)
            for src, _ in batch
        ])
        tgt_vectors = np.vstack([
            tgt_embeddings.loc[tgt].to_numpy(dtype=np.float32, copy=False)
            if tgt in tgt_embeddings.index else np.zeros(tgt_dim, dtype=np.float32)
            for _, tgt in batch
        ])
        src_vectors /= np.maximum(np.linalg.norm(src_vectors, axis=1, keepdims=True), 1e-12)
        tgt_vectors /= np.maximum(np.linalg.norm(tgt_vectors, axis=1, keepdims=True), 1e-12)
        cosine_parts.append(np.sum(src_vectors * tgt_vectors, axis=1))
    cosine = np.concatenate(cosine_parts) if cosine_parts else np.array([], dtype=np.float32)

    frame = pd.DataFrame(rows, columns=["src_iri", "tgt_iri"])
    frame["label"] = [int(pair in gold_pairs) for pair in rows]
    frame["sapbert_cosine"] = cosine
    frame["rank_src_tgt"] = (
        frame.groupby("src_iri")["sapbert_cosine"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    frame["rank_tgt_src"] = (
        frame.groupby("tgt_iri")["sapbert_cosine"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    frame["min_rank"] = frame[["rank_src_tgt", "rank_tgt_src"]].min(axis=1)
    frame["max_rank"] = frame[["rank_src_tgt", "rank_tgt_src"]].max(axis=1)
    frame["abs_rank_difference"] = (
        frame["rank_src_tgt"] - frame["rank_tgt_src"]
    ).abs()
    frame["rrf_bidirectional"] = (
        1.0 / (rrf_constant + frame["rank_src_tgt"])
        + 1.0 / (rrf_constant + frame["rank_tgt_src"])
    )
    for k in (1, 5, 10):
        frame[f"mutual_top{k}"] = (
            (frame["rank_src_tgt"] <= k) & (frame["rank_tgt_src"] <= k)
        ).astype(int)
    return frame


def _tune_sapbert_filters(
    features: pd.DataFrame,
    gold_pairs: Set[Pair],
    fold_gold: Sequence[Set[Pair]],
    k_grid: Sequence[int],
) -> pd.DataFrame:
    all_pairs = set(zip(features["src_iri"], features["tgt_iri"]))
    no_filter = {"filter_rule": "none", "k_sapbert": 0}
    no_filter.update(_recall_metrics(all_pairs, gold_pairs, fold_gold))
    no_filter["reduction_ratio"] = 0.0
    rows: List[dict] = [no_filter]
    for rule in ("and", "or"):
        for k in k_grid:
            if rule == "and":
                mask = (features["rank_src_tgt"] <= k) & (features["rank_tgt_src"] <= k)
            else:
                mask = (features["rank_src_tgt"] <= k) | (features["rank_tgt_src"] <= k)
            kept = set(zip(features.loc[mask, "src_iri"], features.loc[mask, "tgt_iri"]))
            row = {"filter_rule": rule, "k_sapbert": k}
            row.update(_recall_metrics(kept, gold_pairs, fold_gold))
            row["reduction_ratio"] = 1.0 - len(kept) / max(len(features), 1)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["candidate_recall", "candidate_count"], ascending=[False, True]
    ).reset_index(drop=True)


def _jsonable_row(row: pd.Series) -> dict:
    result = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    pair_dir = BIOML_DIR / args.pair
    refs_dir = pair_dir / f"refs_{args.task_type}"
    train_path = refs_dir / "train.tsv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    output_dir = OUTPUTS_DIR / args.pair / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    min_common_grid = _parse_int_grid(args.min_common_grid, "min-common-grid")
    k_overlap_grid = _parse_int_grid(args.k_overlap_grid, "k-overlap-grid")
    k_tfidf_grid = _parse_int_grid(args.k_tfidf_grid, "k-tfidf-grid")
    k_neighborhood_tfidf_grid = _parse_int_grid(
        args.k_neighborhood_tfidf_grid,
        "k-neighborhood-tfidf-grid",
        allow_zero=True,
    )
    k_word_tfidf_grid = _parse_int_grid(
        args.k_word_tfidf_grid,
        "k-word-tfidf-grid",
        allow_zero=True,
    )
    k_sapbert_grid = _parse_int_grid(args.k_sapbert_grid, "k-sapbert-grid")

    print("=" * 76)
    print(f"Train-only candidate protocol tuning: {args.pair}")
    print("GUARD: only train.tsv is read; test.tsv and test.cands.tsv are never loaded")
    print("=" * 76)

    train_df = pd.read_csv(train_path, sep="\t")
    required = {"SrcEntity", "TgtEntity"}
    if not required.issubset(train_df.columns):
        raise ValueError(f"{train_path} must contain columns {sorted(required)}")
    train_df = train_df.drop_duplicates(["SrcEntity", "TgtEntity"]).reset_index(drop=True)
    if args.max_train_mappings > 0:
        train_df = train_df.iloc[: args.max_train_mappings].copy()
        print(f"SMOKE MODE: first {len(train_df)} train mappings only")

    train_df["fold"] = _component_folds(train_df, args.folds)
    gold_pairs = set(zip(train_df["SrcEntity"].astype(str), train_df["TgtEntity"].astype(str)))
    fold_gold = [
        set(zip(part["SrcEntity"].astype(str), part["TgtEntity"].astype(str)))
        for _, part in train_df.groupby("fold", sort=True)
    ]
    print(f"Train gold: {len(gold_pairs)} | folds: {[len(fold) for fold in fold_gold]}")

    src_file, tgt_file = infer_src_tgt_files(pair_dir, args.pair)
    src_onto = OntologyLoader(src_file).load()
    tgt_onto = OntologyLoader(tgt_file).load()
    if args.token_source == "all_literals":
        src_label_map, src_token_map, src_inv = build_all_literal_token_maps(src_onto)
        tgt_label_map, tgt_token_map, tgt_inv = build_all_literal_token_maps(tgt_onto)
    else:
        src_label_map, src_token_map, src_inv = build_label_token_maps(src_onto)
        tgt_label_map, tgt_token_map, tgt_inv = build_label_token_maps(tgt_onto)

    if max(k_neighborhood_tfidf_grid) > 0:
        print("Construction des documents TF-IDF de voisinage (distance 1)...")
        src_neighborhood_documents = _build_neighborhood_documents(
            src_onto, src_label_map, args.neighborhood_max_neighbors
        )
        tgt_neighborhood_documents = _build_neighborhood_documents(
            tgt_onto, tgt_label_map, args.neighborhood_max_neighbors
        )
    else:
        src_neighborhood_documents = src_label_map
        tgt_neighborhood_documents = tgt_label_map

    print("\n[1/3] Pré-calcul des classements overlap et TF-IDF...")
    cache = _build_retrieval_cache(
        train_df,
        src_label_map,
        src_token_map,
        src_inv,
        tgt_label_map,
        tgt_token_map,
        tgt_inv,
        min_common_grid,
        max(k_overlap_grid),
        max(k_tfidf_grid),
        max(k_neighborhood_tfidf_grid),
        src_neighborhood_documents,
        tgt_neighborhood_documents,
        max_k_word_tfidf=max(k_word_tfidf_grid),
        word_ngram_range=(args.word_ngram_min, args.word_ngram_max),
        src_queries_override=(
            [] if args.retrieval_direction == "tgt_to_src" else None
        ),
        tgt_queries_override=(
            [] if args.retrieval_direction == "src_to_tgt" else None
        ),
    )

    print("\n[2/3] Grille lexicale train-only...")
    lexical_rows: List[dict] = []
    pair_cache: Dict[Tuple[int, int, int, int, int], Set[Pair]] = {}
    total_configs = (
        len(min_common_grid)
        * len(k_overlap_grid)
        * len(k_tfidf_grid)
        * len(k_neighborhood_tfidf_grid)
        * len(k_word_tfidf_grid)
    )
    iterator = tqdm(total=total_configs, desc="Configurations lexicales")
    for minimum in min_common_grid:
        for k_overlap in k_overlap_grid:
            for k_tfidf in k_tfidf_grid:
                for k_neighborhood_tfidf in k_neighborhood_tfidf_grid:
                    for k_word_tfidf in k_word_tfidf_grid:
                        key = (
                            minimum, k_overlap, k_tfidf,
                            k_neighborhood_tfidf, k_word_tfidf,
                        )
                        candidates = _candidate_pairs(
                            cache,
                            minimum,
                            k_overlap,
                            k_tfidf,
                            k_neighborhood_tfidf,
                            k_word_tfidf,
                        )
                        pair_cache[key] = candidates
                        row = {
                            "min_common_tokens": minimum,
                            "k_overlap": k_overlap,
                            "k_tfidf": k_tfidf,
                            "k_neighborhood_tfidf": k_neighborhood_tfidf,
                            "k_word_tfidf": k_word_tfidf,
                            "word_ngram_min": args.word_ngram_min,
                            "word_ngram_max": args.word_ngram_max,
                        }
                        row.update(_recall_metrics(candidates, gold_pairs, fold_gold))
                        lexical_rows.append(row)
                        iterator.update(1)
    iterator.close()

    lexical_results = pd.DataFrame(lexical_rows).sort_values(
        ["candidate_recall", "candidate_count"], ascending=[False, True]
    ).reset_index(drop=True)
    lexical_results.to_csv(output_dir / "lexical_grid_train_only.csv", index=False)
    if args.min_candidate_recall > 0:
        selected_lexical = _select_smallest_above_recall(
            lexical_results, args.min_candidate_recall
        )
    else:
        selected_lexical = _select_smallest_near_best(
            lexical_results, args.recall_tolerance
        )
    lexical_key = (
        int(selected_lexical["min_common_tokens"]),
        int(selected_lexical["k_overlap"]),
        int(selected_lexical["k_tfidf"]),
        int(selected_lexical["k_neighborhood_tfidf"]),
        int(selected_lexical["k_word_tfidf"]),
    )
    selected_pairs = pair_cache[lexical_key]
    print("Sélection lexicale:", _jsonable_row(selected_lexical))

    selected_sapbert = None
    sapbert_results = None
    candidate_features = None
    if not args.skip_sapbert:
        print("\n[3/3] Filtre SapBERT bidirectionnel sur les candidats lexicaux...")
        src_emb, tgt_emb = _load_or_encode_sapbert(
            src_onto=src_onto,
            tgt_onto=tgt_onto,
            pair_name=args.pair,
            cache_dir=output_dir / "embeddings_cache",
            model_name=args.model,
            device=args.device,
            use_synonyms=args.use_synonyms,
            batch_size=args.embedding_batch_size,
            max_length=args.max_length,
        )
        candidate_features = _add_sapbert_features(
            selected_pairs, gold_pairs, src_emb, tgt_emb, args.rrf_constant
        )
        sapbert_results = _tune_sapbert_filters(
            candidate_features, gold_pairs, fold_gold, k_sapbert_grid
        )
        sapbert_results.to_csv(output_dir / "sapbert_filter_grid_train_only.csv", index=False)
        selected_sapbert = _select_sapbert_with_budget(
            sapbert_results,
            args.recall_tolerance,
            args.max_negative_ratio,
            args.sapbert_max_recall_loss,
            args.min_candidate_recall,
        )
        rule = str(selected_sapbert["filter_rule"])
        k = int(selected_sapbert["k_sapbert"])
        if rule == "none":
            keep = pd.Series(True, index=candidate_features.index)
        elif rule == "and":
            keep = (candidate_features["rank_src_tgt"] <= k) & (
                candidate_features["rank_tgt_src"] <= k
            )
        else:
            keep = (candidate_features["rank_src_tgt"] <= k) | (
                candidate_features["rank_tgt_src"] <= k
            )
        candidate_features["selected_by_sapbert_filter"] = keep.astype(int)
        candidate_features.to_csv(
            output_dir / "selected_lexical_candidates_with_sapbert_features.csv",
            index=False,
        )
        print("Sélection SapBERT:", _jsonable_row(selected_sapbert))
    else:
        print("\n[3/3] SapBERT ignoré (--skip-sapbert).")

    protocol = {
        "pair": args.pair,
        "task_type": args.task_type,
        "data_boundary": "train.tsv only",
        "train_path": str(train_path),
        "train_mapping_count": len(gold_pairs),
        "folds": args.folds,
        "token_source": args.token_source,
        "retrieval_direction": args.retrieval_direction,
        "recall_tolerance": args.recall_tolerance,
        "selected_lexical": _jsonable_row(selected_lexical),
        "selected_sapbert_filter": (
            _jsonable_row(selected_sapbert) if selected_sapbert is not None else None
        ),
        "sapbert_metric": "cosine_on_l2_normalized_embeddings",
        "sapbert_rank_scope": "selected_lexical_candidate_space",
        "rrf_constant": args.rrf_constant,
        "neighborhood_definition": "entity + parents + children at distance 1",
        "neighborhood_max_neighbors": args.neighborhood_max_neighbors,
        "word_ngram_range": [args.word_ngram_min, args.word_ngram_max],
        "max_negative_ratio": args.max_negative_ratio,
        "sapbert_max_recall_loss": args.sapbert_max_recall_loss,
        "min_candidate_recall": args.min_candidate_recall,
        "elapsed_seconds": time.time() - started,
    }
    with open(output_dir / "selected_protocol_train_only.json", "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)

    print(f"\nRésultats: {output_dir}")
    print(f"Protocole figé: {output_dir / 'selected_protocol_train_only.json'}")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune lexical + bidirectional SapBERT filters using train.tsv only"
    )
    parser.add_argument("--pair", required=True)
    parser.add_argument("--task-type", default="equiv", choices=["equiv", "subs"])
    parser.add_argument("--token-source", default="all_literals", choices=["label", "all_literals"])
    parser.add_argument(
        "--retrieval-direction",
        default="bidirectional",
        choices=["bidirectional", "src_to_tgt", "tgt_to_src"],
        help="Lexical generation direction; use src_to_tgt for very large target ontologies",
    )
    parser.add_argument("--min-common-grid", default="1,2,3,5")
    parser.add_argument("--k-overlap-grid", default="10,20,30,40,50")
    parser.add_argument("--k-tfidf-grid", default="5,10,15")
    parser.add_argument(
        "--k-word-tfidf-grid",
        default="0",
        help="Top-K TF-IDF word n-grams; 0 disables the rescue channel",
    )
    parser.add_argument("--word-ngram-min", type=int, default=1)
    parser.add_argument("--word-ngram-max", type=int, default=3)
    parser.add_argument(
        "--k-neighborhood-tfidf-grid",
        default="0,3,5,10",
        help="Top-K TF-IDF on entity+parent+child documents; 0 disables the branch",
    )
    parser.add_argument("--neighborhood-max-neighbors", type=int, default=20)
    parser.add_argument("--k-sapbert-grid", default="1,3,5,7,10,15,20,30,50")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--recall-tolerance",
        type=float,
        default=0.002,
        help="Maximum recall loss from the best grid result (default: 0.002)",
    )
    parser.add_argument("--model", default="sapbert")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-synonyms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--rrf-constant", type=float, default=10.0)
    parser.add_argument(
        "--max-negative-ratio",
        type=float,
        default=0.0,
        help="Select best SapBERT recall with negatives/covered-positive <= this value; 0 disables budget",
    )
    parser.add_argument(
        "--sapbert-max-recall-loss",
        type=float,
        default=0.04,
        help="Maximum absolute candidate-recall loss caused by SapBERT (default: 0.04)",
    )
    parser.add_argument(
        "--min-candidate-recall",
        type=float,
        default=0.0,
        help="Hard minimum recall before and after SapBERT; 0 disables the constraint",
    )
    parser.add_argument("--skip-sapbert", action="store_true")
    parser.add_argument(
        "--max-train-mappings",
        type=int,
        default=0,
        help="Smoke-test limit; 0 uses every train mapping",
    )
    parser.add_argument("--output-subdir", default="train_only_protocol_tuning")
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not 0.0 <= args.recall_tolerance < 1.0:
        parser.error("--recall-tolerance must be in [0, 1)")
    if args.neighborhood_max_neighbors < 0:
        parser.error("--neighborhood-max-neighbors must be >= 0")
    if args.word_ngram_min < 1 or args.word_ngram_max < args.word_ngram_min:
        parser.error("word n-gram bounds must satisfy 1 <= min <= max")
    if args.max_negative_ratio < 0:
        parser.error("--max-negative-ratio must be >= 0")
    if not 0.0 <= args.sapbert_max_recall_loss < 1.0:
        parser.error("--sapbert-max-recall-loss must be in [0, 1)")
    if not 0.0 <= args.min_candidate_recall <= 1.0:
        parser.error("--min-candidate-recall must be in [0, 1]")
    run(args)


if __name__ == "__main__":
    main()
