#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline ontology matching BioGITOM + briques + MetaSpace.

Idée:
- réduire l'espace de recherche avec trois briques candidates:
  1) overlap tokens sur labels/toutes balises littérales,
  2) TF-IDF char n-grams,
  3) SapBERT + recherche nearest-neighbor L2 façon BioGITOM.
- entraîner MetaMatch/MetaSpace sur les features choisies.
- produire des matches one-to-one via greedy_1to1.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.data.ontology_loader import OntologyLoader
from src.data.tsv_parser import load_bioml_task
from src.embeddings.encoder import EmbeddingCache, LabelEncoder
from src.features.pipeline import FeaturePipeline
from src.model.evaluate import compute_alignment_metrics
from src.model.train import MetaMatchTrainer, find_optimal_threshold, prepare_training_data
from src.scripts.run_token_overlap_pipeline import (
    build_all_literal_token_maps,
    build_label_token_maps,
    build_tfidf_index,
    build_token_freq,
    build_matches_with_postfilter,
    deduplicate_pairs,
    heuristic_filter_candidates,
    infer_src_tgt_files,
    merge_candidates,
    topk_tfidf_candidates,
    topk_token_overlap_candidates,
)


DEFAULT_FEATURES = [
    "syn_cosine_bigrams",
    "syn_jaccard_trigrams",
    "syn_jaccard_tokens",
    "syn_jaro_winkler",
    "cls_euclidean",
    "syn_lcs_ratio",
    "cls_chebyshev",
    "syn_len_b",
    "syn_common_suffix_ratio",
    "tda_h0_entropy_combined",
    "overlap_sym",
]

BRICK_FEATURES = [
    "bio_l2_sim",
    "bio_l2_rank_src",
    "bio_l2_rank_tgt",
    "brick_embedding_src",
    "brick_embedding_tgt",
    "brick_overlap",
    "brick_tfidf",
    "brick_lex_exact_norm",
    "brick_lex_high",
    "brick_consensus",
]

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class DirectTransformerLabelEncoder:
    """Encodeur token+vecteur compatible avec le SapBERT brut de BioGITOM."""

    def __init__(self, model_name: str = "sapbert", device: str = "cpu"):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("SapBERT direct requiert transformers et torch.") from exc

        self.torch = torch
        self.model_name = model_name
        self.model_path = _clean_model_name(model_name)
        self.device = device
        print(f"Chargement du modèle transformers {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)
        print(f"  Dimension: {self.embedding_dim}")

    @staticmethod
    def _pool(arr: np.ndarray, pooling: str) -> np.ndarray:
        if arr.size == 0:
            return np.zeros(1, dtype=np.float32)
        if pooling == "max":
            return arr.max(axis=0).astype(np.float32)
        return arr.mean(axis=0).astype(np.float32)

    def encode_token_and_vector(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        pooling: str = "mean",
        max_tokens: int = 64,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Batches")

        token_matrices: List[np.ndarray] = []
        pooled_vectors: List[np.ndarray] = []
        with self.torch.no_grad():
            for start in iterator:
                batch_texts = [str(x or "") for x in texts[start : start + batch_size]]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                special_mask = encoded.pop("special_tokens_mask")
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state.detach().cpu().numpy().astype(np.float32)
                attn = encoded["attention_mask"].detach().cpu().numpy().astype(bool)
                special = special_mask.detach().cpu().numpy().astype(bool)

                for i in range(hidden.shape[0]):
                    valid = attn[i] & (~special[i])
                    arr = hidden[i][valid]
                    if arr.size == 0:
                        arr = hidden[i][attn[i]]
                    if arr.size == 0:
                        arr = np.zeros((1, self.embedding_dim), dtype=np.float32)
                    token_matrices.append(arr.astype(np.float32))
                    pooled_vectors.append(self._pool(arr, pooling=pooling))

        return token_matrices, np.vstack(pooled_vectors).astype(np.float32)

    def encode_ontology_token_bundle(
        self,
        ontology_loader,
        use_synonyms: bool = False,
        batch_size: int = 32,
        pooling: str = "mean",
        max_tokens: int = 64,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        iris: List[str] = []
        texts: List[str] = []
        for iri, info in ontology_loader.classes.items():
            iris.append(iri)
            if use_synonyms and info["synonyms"]:
                texts.append(" | ".join([info["label"]] + info["synonyms"]))
            else:
                texts.append(info["label"])

        print(f"Encodage token+vector de {len(texts)} classes...")
        token_matrices, pooled_vectors = self.encode_token_and_vector(
            texts=texts,
            batch_size=batch_size,
            show_progress=True,
            pooling=pooling,
            max_tokens=max_tokens,
        )
        vectors_df = pd.DataFrame(
            pooled_vectors,
            index=iris,
            columns=[f"dim_{i}" for i in range(pooled_vectors.shape[1])],
        )
        return vectors_df, {iri: mat for iri, mat in zip(iris, token_matrices)}

    def get_pair_embeddings(
        self,
        pairs_df: pd.DataFrame,
        src_embeddings_df: pd.DataFrame,
        tgt_embeddings_df: pd.DataFrame,
        src_col: str = "src_iri",
        tgt_col: str = "tgt_iri",
    ) -> Tuple[np.ndarray, np.ndarray]:
        dim = len(src_embeddings_df.columns)
        src_embeddings = []
        tgt_embeddings = []
        for src_iri, tgt_iri in zip(pairs_df[src_col].tolist(), pairs_df[tgt_col].tolist()):
            src_embeddings.append(
                src_embeddings_df.loc[src_iri].values if src_iri in src_embeddings_df.index else np.zeros(dim)
            )
            tgt_embeddings.append(
                tgt_embeddings_df.loc[tgt_iri].values if tgt_iri in tgt_embeddings_df.index else np.zeros(dim)
            )
        return np.asarray(src_embeddings), np.asarray(tgt_embeddings)

    def get_pair_token_embeddings(
        self,
        pairs_df: pd.DataFrame,
        src_tokens_map: Dict[str, np.ndarray],
        tgt_tokens_map: Dict[str, np.ndarray],
        src_col: str = "src_iri",
        tgt_col: str = "tgt_iri",
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        empty = np.empty((0, self.embedding_dim), dtype=np.float32)
        src_tokens = [src_tokens_map.get(x, empty) for x in pairs_df[src_col].tolist()]
        tgt_tokens = [tgt_tokens_map.get(x, empty) for x in pairs_df[tgt_col].tolist()]
        return src_tokens, tgt_tokens


def _clean_model_name(model_name: str) -> str:
    if model_name == "sapbert":
        return SAPBERT_MODEL
    return model_name


def _resolve_features(requested: Sequence[str], available: Sequence[str], include_bricks: bool) -> List[str]:
    aliases = {
        "overlap_sym": "tda_overlap_sym",
    }
    available_set = set(available)
    selected: List[str] = []
    missing: List[str] = []

    for raw in requested:
        name = aliases.get(raw, raw)
        if name in available_set and name not in selected:
            selected.append(name)
        else:
            missing.append(raw)

    if include_bricks:
        for name in BRICK_FEATURES:
            if name in available_set and name not in selected:
                selected.append(name)

    if missing:
        print("  Features ignorées car absentes:", ", ".join(missing))
    if not selected:
        raise ValueError("Aucune feature sélectionnée après résolution des noms.")
    return selected


def _apply_candidate_filter(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df.copy()
    if mode == "consensus":
        return df[df["brick_consensus"] == 1].copy()
    if mode == "embed_bidir":
        return df[(df["brick_embedding_src"] == 1) & (df["brick_embedding_tgt"] == 1)].copy()
    if mode == "overlap_tfidf":
        return df[(df["brick_overlap"] == 1) & (df["brick_tfidf"] == 1)].copy()
    if mode == "embed_bidir_or_consensus":
        return df[
            ((df["brick_embedding_src"] == 1) & (df["brick_embedding_tgt"] == 1))
            | (df["brick_consensus"] == 1)
        ].copy()
    raise ValueError(f"Unknown candidate filter: {mode}")


def _label(onto: OntologyLoader, label_map: Dict[str, str], iri: str) -> str:
    return label_map.get(iri) or onto.get_label(iri) or ""


def _pair_row(
    src_onto: OntologyLoader,
    tgt_onto: OntologyLoader,
    src_label_map: Dict[str, str],
    tgt_label_map: Dict[str, str],
    s: str,
    t: str,
    label: int,
    origins: Iterable[str],
) -> dict:
    origin_set = set(origins)
    return {
        "src_iri": s,
        "tgt_iri": t,
        "src_label": _label(src_onto, src_label_map, s),
        "tgt_label": _label(tgt_onto, tgt_label_map, t),
        "label": int(label),
        "origin": "+".join(sorted(origin_set)),
        "brick_embedding_src": int("embedding_src" in origin_set),
        "brick_embedding_tgt": int("embedding_tgt" in origin_set),
        "brick_overlap": int("overlap" in origin_set),
        "brick_tfidf": int("tfidf" in origin_set),
    }


def _dedupe_with_origins(rows: List[dict]) -> pd.DataFrame:
    by_pair: Dict[Tuple[str, str], dict] = {}
    origins: Dict[Tuple[str, str], Set[str]] = {}
    for row in rows:
        key = (row["src_iri"], row["tgt_iri"])
        row_origins = set(str(row.get("origin", "")).split("+")) - {""}
        if key not in by_pair:
            by_pair[key] = dict(row)
            origins[key] = set(row_origins)
        else:
            origins[key].update(row_origins)
            if int(row.get("label", 0)) > int(by_pair[key].get("label", 0)):
                by_pair[key]["label"] = int(row["label"])

    out: List[dict] = []
    for key, row in by_pair.items():
        org = origins[key]
        row["origin"] = "+".join(sorted(org))
        row["brick_embedding_src"] = int("embedding_src" in org)
        row["brick_embedding_tgt"] = int("embedding_tgt" in org)
        row["brick_overlap"] = int("overlap" in org)
        row["brick_tfidf"] = int("tfidf" in org)
        out.append(row)
    return pd.DataFrame(out)


@dataclass
class EmbeddingIndex:
    iris: List[str]
    matrix: np.ndarray
    norm_matrix: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "EmbeddingIndex":
        iris = df.index.astype(str).tolist()
        matrix = df.to_numpy(dtype=np.float32, copy=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm_matrix = matrix / np.maximum(norms, 1e-12)
        return cls(iris=iris, matrix=matrix, norm_matrix=norm_matrix)

    def topk_l2(self, query_vec: np.ndarray, k: int, exclude: Optional[Set[str]] = None) -> List[Tuple[str, float, int]]:
        if k <= 0:
            return []
        exclude = exclude or set()
        diff = self.matrix - query_vec.astype(np.float32, copy=False)
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        order = np.argsort(dist)
        out: List[Tuple[str, float, int]] = []
        rank = 0
        for idx in order:
            iri = self.iris[int(idx)]
            if iri in exclude:
                continue
            rank += 1
            score = 1.0 / (1.0 + float(dist[int(idx)]))
            out.append((iri, score, rank))
            if len(out) >= k:
                break
        return out


def _embedding_lookup(df: pd.DataFrame, iri: str) -> Optional[np.ndarray]:
    if iri not in df.index:
        return None
    return df.loc[iri].to_numpy(dtype=np.float32, copy=False)


def _embedding_candidates(
    query_iri: str,
    query_df: pd.DataFrame,
    cand_index: EmbeddingIndex,
    k: int,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    q = _embedding_lookup(query_df, query_iri)
    if q is None:
        return []
    return [iri for iri, _, _ in cand_index.topk_l2(q, k=k, exclude=exclude)]


def _build_rank_maps(
    pairs: pd.DataFrame,
    src_emb_df: pd.DataFrame,
    tgt_emb_df: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
    if pairs.empty:
        return {}, {}, {}

    src_vecs = np.vstack([
        _embedding_lookup(src_emb_df, s) if _embedding_lookup(src_emb_df, s) is not None else np.zeros(src_emb_df.shape[1], dtype=np.float32)
        for s in pairs["src_iri"]
    ])
    tgt_vecs = np.vstack([
        _embedding_lookup(tgt_emb_df, t) if _embedding_lookup(tgt_emb_df, t) is not None else np.zeros(tgt_emb_df.shape[1], dtype=np.float32)
        for t in pairs["tgt_iri"]
    ])
    dist = np.sqrt(np.sum((src_vecs - tgt_vecs) ** 2, axis=1))
    sim = 1.0 / (1.0 + dist)

    work = pairs[["src_iri", "tgt_iri"]].copy()
    work["bio_l2_sim"] = sim
    work["rank_src"] = work.groupby("src_iri")["bio_l2_sim"].rank(method="first", ascending=False).astype(int)
    work["rank_tgt"] = work.groupby("tgt_iri")["bio_l2_sim"].rank(method="first", ascending=False).astype(int)

    sim_map = {
        (r.src_iri, r.tgt_iri): float(r.bio_l2_sim)
        for r in work.itertuples(index=False)
    }
    src_rank_map = {
        (r.src_iri, r.tgt_iri): int(r.rank_src)
        for r in work.itertuples(index=False)
    }
    tgt_rank_map = {
        (r.src_iri, r.tgt_iri): int(r.rank_tgt)
        for r in work.itertuples(index=False)
    }
    return sim_map, src_rank_map, tgt_rank_map


def _add_brick_features(pairs: pd.DataFrame, src_emb_df: pd.DataFrame, tgt_emb_df: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    sim_map, src_rank_map, tgt_rank_map = _build_rank_maps(out, src_emb_df, tgt_emb_df)
    keys = list(zip(out["src_iri"], out["tgt_iri"]))
    out["bio_l2_sim"] = [sim_map.get(k, 0.0) for k in keys]
    out["bio_l2_rank_src"] = [src_rank_map.get(k, 999999) for k in keys]
    out["bio_l2_rank_tgt"] = [tgt_rank_map.get(k, 999999) for k in keys]
    src_norm = out["src_label"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    tgt_norm = out["tgt_label"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    out["brick_lex_exact_norm"] = (src_norm == tgt_norm).astype(int)
    # syn_jaro_winkler/syn_jaccard_tokens sont calculés plus tard; ici on pose le consensus structurel.
    out["brick_consensus"] = (
        out[["brick_embedding_src", "brick_embedding_tgt", "brick_overlap", "brick_tfidf"]].sum(axis=1) >= 2
    ).astype(int)
    out["brick_lex_high"] = 0
    return out


def _candidate_union_for_query(
    query_iri: str,
    query_label: str,
    query_tokens: Set[str],
    cand_token_map: Dict[str, Set[str]],
    cand_label_map: Dict[str, str],
    inv_index: Dict[str, Set[str]],
    tfidf_iris: List[str],
    tfidf_vec,
    tfidf_mat,
    emb_cands: List[str],
    k_overlap: int,
    k_tfidf: int,
    k_final: int,
    min_common_tokens: int,
    exclude: Optional[Set[str]],
    heuristic_filter: bool,
    cand_token_freq: Dict[str, int],
    heuristic_max_keep: int,
    heuristic_min_score: float,
    heuristic_min_keep: int,
) -> List[Tuple[str, Set[str]]]:
    overlap = topk_token_overlap_candidates(
        query_tokens=query_tokens,
        query_label=query_label,
        cand_token_map=cand_token_map,
        cand_label_map=cand_label_map,
        inv_index=inv_index,
        k=k_overlap,
        min_common_tokens=min_common_tokens,
        exclude=exclude or set(),
    )
    tfidf = topk_tfidf_candidates(
        query_label=query_label,
        cand_iris=tfidf_iris,
        vectorizer=tfidf_vec,
        cand_matrix=tfidf_mat,
        k=k_tfidf,
        exclude=exclude or set(),
    )
    merged = merge_candidates(emb_cands, overlap, tfidf, limit=k_final)
    if heuristic_filter:
        merged = heuristic_filter_candidates(
            query_tokens=query_tokens,
            query_label=query_label,
            candidates=merged,
            cand_token_map=cand_token_map,
            cand_label_map=cand_label_map,
            cand_token_freq=cand_token_freq,
            max_keep=heuristic_max_keep,
            min_score=heuristic_min_score,
            min_keep=heuristic_min_keep,
        )

    out: List[Tuple[str, Set[str]]] = []
    overlap_set = set(overlap)
    tfidf_set = set(tfidf)
    emb_set = set(emb_cands)
    for cand in merged:
        origins: Set[str] = set()
        if cand in emb_set:
            origins.add("embedding_src")
        if cand in overlap_set:
            origins.add("overlap")
        if cand in tfidf_set:
            origins.add("tfidf")
        out.append((cand, origins))
    return out


def run(args: argparse.Namespace) -> Dict[str, float]:
    pair_dir = BIOML_DIR / args.pair
    output_dir = OUTPUTS_DIR / args.pair / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"BioGITOM + briques + MetaSpace: {args.pair}")
    print("=" * 72)

    print("\n[1/7] Chargement OWL + mappings...")
    t0 = time.time()
    src_file, tgt_file = infer_src_tgt_files(pair_dir, args.pair)
    src_onto = OntologyLoader(src_file).load()
    tgt_onto = OntologyLoader(tgt_file).load()
    train_df, test_df, _ = load_bioml_task(pair_dir)
    train_gold = set(zip(train_df["SrcEntity"], train_df["TgtEntity"]))
    test_gold = set(zip(test_df["SrcEntity"], test_df["TgtEntity"]))
    print(f"  Source: {src_file.name} ({len(src_onto)} classes)")
    print(f"  Target: {tgt_file.name} ({len(tgt_onto)} classes)")
    print(f"  Train mappings: {len(train_df)}")
    print(f"  Test mappings: {len(test_df)}")
    print(f"  Temps: {time.time() - t0:.1f}s")

    print("\n[2/7] Index lexical/tags...")
    t0 = time.time()
    if args.token_source == "all_literals":
        src_label_map, src_tok_map, src_inv = build_all_literal_token_maps(src_onto)
        tgt_label_map, tgt_tok_map, tgt_inv = build_all_literal_token_maps(tgt_onto)
    else:
        src_label_map, src_tok_map, src_inv = build_label_token_maps(src_onto)
        tgt_label_map, tgt_tok_map, tgt_inv = build_label_token_maps(tgt_onto)

    src_token_freq = build_token_freq(src_inv)
    tgt_token_freq = build_token_freq(tgt_inv)
    tfidf_tgt_iris, tfidf_tgt_vec, tfidf_tgt_mat = build_tfidf_index(
        cand_label_map=tgt_label_map,
        extra_corpus_labels=list(src_label_map.values()),
    )
    tfidf_src_iris, tfidf_src_vec, tfidf_src_mat = build_tfidf_index(
        cand_label_map=src_label_map,
        extra_corpus_labels=list(tgt_label_map.values()),
    )
    print(f"  Token source: {args.token_source}")
    print(f"  Vocab source/target: {len(src_inv)} / {len(tgt_inv)}")
    print(f"  Temps: {time.time() - t0:.1f}s")

    print("\n[3/7] Embeddings SapBERT/BioGITOM...")
    t0 = time.time()
    model_name = _clean_model_name(args.model)
    if args.model == "sapbert" or model_name == SAPBERT_MODEL:
        encoder = DirectTransformerLabelEncoder(model_name=args.model, device=args.device)
    else:
        encoder = LabelEncoder(model_name=model_name, device=args.device)
    cache = EmbeddingCache(output_dir / "embeddings_cache")
    src_emb_df, src_tok_emb_map = cache.get_or_compute_token_bundle(
        src_onto,
        f"{args.pair}_src",
        encoder,
        use_synonyms=args.use_synonyms,
        pooling=args.pooling,
        max_tokens=args.max_tokens,
    )
    tgt_emb_df, tgt_tok_emb_map = cache.get_or_compute_token_bundle(
        tgt_onto,
        f"{args.pair}_tgt",
        encoder,
        use_synonyms=args.use_synonyms,
        pooling=args.pooling,
        max_tokens=args.max_tokens,
    )
    src_index = EmbeddingIndex.from_df(src_emb_df)
    tgt_index = EmbeddingIndex.from_df(tgt_emb_df)
    print(f"  Modèle: {model_name}")
    print(f"  Dim: {src_emb_df.shape[1]}")
    print(f"  Temps: {time.time() - t0:.1f}s")

    print("\n[4/7] Génération train candidates...")
    t0 = time.time()
    train_rows: List[dict] = []
    for row in tqdm(train_df.itertuples(index=False), total=len(train_df), desc="Train candidates"):
        s = row.SrcEntity
        t = row.TgtEntity
        train_rows.append(
            _pair_row(src_onto, tgt_onto, src_label_map, tgt_label_map, s, t, 1, {"positive"})
        )

        emb_tgts = _embedding_candidates(s, src_emb_df, tgt_index, args.k_embedding, exclude={t})
        cands_t = _candidate_union_for_query(
            query_iri=s,
            query_label=_label(src_onto, src_label_map, s),
            query_tokens=src_tok_map.get(s, set()),
            cand_token_map=tgt_tok_map,
            cand_label_map=tgt_label_map,
            inv_index=tgt_inv,
            tfidf_iris=tfidf_tgt_iris,
            tfidf_vec=tfidf_tgt_vec,
            tfidf_mat=tfidf_tgt_mat,
            emb_cands=emb_tgts,
            k_overlap=args.k_overlap,
            k_tfidf=args.k_tfidf,
            k_final=args.k_final,
            min_common_tokens=args.min_common_tokens,
            exclude={t},
            heuristic_filter=args.heuristic_filter,
            cand_token_freq=tgt_token_freq,
            heuristic_max_keep=args.heuristic_max_keep,
            heuristic_min_score=args.heuristic_min_score,
            heuristic_min_keep=args.heuristic_min_keep,
        )
        for nt, origins in cands_t:
            if (s, nt) not in train_gold:
                train_rows.append(_pair_row(src_onto, tgt_onto, src_label_map, tgt_label_map, s, nt, 0, origins))

        if args.bidirectional_embedding:
            emb_srcs = _embedding_candidates(t, tgt_emb_df, src_index, args.k_embedding, exclude={s})
            for ns in emb_srcs:
                if (ns, t) not in train_gold:
                    train_rows.append(
                        _pair_row(src_onto, tgt_onto, src_label_map, tgt_label_map, ns, t, 0, {"embedding_tgt"})
                    )

    train_pairs = _dedupe_with_origins(train_rows)
    train_pairs = _add_brick_features(train_pairs, src_emb_df, tgt_emb_df)
    print(f"  Train pairs: {len(train_pairs)} pos={int(train_pairs['label'].sum())} neg={int((train_pairs['label'] == 0).sum())}")
    print(f"  Temps: {time.time() - t0:.1f}s")

    print("\n[5/7] Génération test candidates...")
    t0 = time.time()
    test_rows: List[dict] = []
    test_sources = sorted(set(test_df["SrcEntity"]))
    test_targets = sorted(set(test_df["TgtEntity"]))

    for s in tqdm(test_sources, desc="Test src->tgt"):
        emb_tgts = _embedding_candidates(s, src_emb_df, tgt_index, args.k_embedding, exclude=set())
        cands = _candidate_union_for_query(
            query_iri=s,
            query_label=_label(src_onto, src_label_map, s),
            query_tokens=src_tok_map.get(s, set()),
            cand_token_map=tgt_tok_map,
            cand_label_map=tgt_label_map,
            inv_index=tgt_inv,
            tfidf_iris=tfidf_tgt_iris,
            tfidf_vec=tfidf_tgt_vec,
            tfidf_mat=tfidf_tgt_mat,
            emb_cands=emb_tgts,
            k_overlap=args.k_overlap,
            k_tfidf=args.k_tfidf,
            k_final=args.k_final,
            min_common_tokens=args.min_common_tokens,
            exclude=set(),
            heuristic_filter=args.heuristic_filter,
            cand_token_freq=tgt_token_freq,
            heuristic_max_keep=args.heuristic_max_keep,
            heuristic_min_score=args.heuristic_min_score,
            heuristic_min_keep=args.heuristic_min_keep,
        )
        for t, origins in cands:
            key = (s, t)
            test_rows.append(
                _pair_row(src_onto, tgt_onto, src_label_map, tgt_label_map, s, t, int(key in test_gold), origins)
            )

    if args.bidirectional_embedding:
        for t in tqdm(test_targets, desc="Test tgt->src"):
            emb_srcs = _embedding_candidates(t, tgt_emb_df, src_index, args.k_embedding, exclude=set())
            for s in emb_srcs:
                key = (s, t)
                test_rows.append(
                    _pair_row(src_onto, tgt_onto, src_label_map, tgt_label_map, s, t, int(key in test_gold), {"embedding_tgt"})
                )

    test_pairs = _dedupe_with_origins(test_rows)
    test_pairs = _add_brick_features(test_pairs, src_emb_df, tgt_emb_df)
    if args.max_test_pairs > 0 and len(test_pairs) > args.max_test_pairs:
        test_pairs = test_pairs.sample(n=args.max_test_pairs, random_state=42).reset_index(drop=True)

    candidate_set = set(zip(test_pairs["src_iri"], test_pairs["tgt_iri"]))
    coverage = len(candidate_set & test_gold)
    print(f"  Test sources/targets: {len(test_sources)} / {len(test_targets)}")
    print(f"  Test candidate pairs: {len(test_pairs)}")
    print(f"  Gold coverage ceiling: {coverage}/{len(test_gold)} = {coverage / max(len(test_gold), 1):.4f}")
    print(f"  Temps: {time.time() - t0:.1f}s")

    print("\n[6/7] Meta-features...")
    t0 = time.time()
    train_src_emb, train_tgt_emb = encoder.get_pair_embeddings(train_pairs, src_emb_df, tgt_emb_df)
    test_src_emb, test_tgt_emb = encoder.get_pair_embeddings(test_pairs, src_emb_df, tgt_emb_df)
    train_src_tok, train_tgt_tok = encoder.get_pair_token_embeddings(train_pairs, src_tok_emb_map, tgt_tok_emb_map)
    test_src_tok, test_tgt_tok = encoder.get_pair_token_embeddings(test_pairs, src_tok_emb_map, tgt_tok_emb_map)

    pipeline = FeaturePipeline(
        use_syntax=True,
        use_classical=True,
        use_spectral=False,
        use_topological=args.use_topological,
        use_nlp=False,
    )
    train_feat = pipeline.compute_features_batch(
        train_pairs,
        src_embeddings=train_src_emb,
        tgt_embeddings=train_tgt_emb,
        src_token_embeddings=train_src_tok,
        tgt_token_embeddings=train_tgt_tok,
        tda_cache_dir=output_dir / "tda_cache",
        show_progress=True,
    )
    test_feat = pipeline.compute_features_batch(
        test_pairs,
        src_embeddings=test_src_emb,
        tgt_embeddings=test_tgt_emb,
        src_token_embeddings=test_src_tok,
        tgt_token_embeddings=test_tgt_tok,
        tda_cache_dir=output_dir / "tda_cache",
        show_progress=True,
    )

    for col in BRICK_FEATURES:
        train_feat[col] = train_pairs[col].to_numpy()
        test_feat[col] = test_pairs[col].to_numpy()
    train_feat["brick_lex_high"] = (
        (train_feat.get("syn_jaro_winkler", 0) >= args.lex_high_threshold)
        | (train_feat.get("syn_jaccard_tokens", 0) >= args.lex_high_threshold)
        | (train_pairs["brick_lex_exact_norm"].to_numpy() == 1)
    ).astype(int)
    test_feat["brick_lex_high"] = (
        (test_feat.get("syn_jaro_winkler", 0) >= args.lex_high_threshold)
        | (test_feat.get("syn_jaccard_tokens", 0) >= args.lex_high_threshold)
        | (test_pairs["brick_lex_exact_norm"].to_numpy() == 1)
    ).astype(int)

    requested_features = [x.strip() for x in args.features.split(",") if x.strip()]
    feature_cols = _resolve_features(requested_features, train_feat.columns.tolist(), args.include_brick_features)
    print(f"  Features utilisées ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"  Temps: {time.time() - t0:.1f}s")

    print("\n[7/7] Train + greedy 1-to-1...")
    t0 = time.time()
    X_train, y_train, feature_cols = prepare_training_data(train_feat, train_pairs, feature_cols=feature_cols)
    X_test = test_feat[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).to_numpy()

    threshold = args.threshold
    if args.auto_threshold_train:
        cv_trainer = MetaMatchTrainer(model_type=args.model_family, threshold=0.5)
        cv = cv_trainer.cross_validate(X_train, y_train, n_folds=args.cv_folds, feature_names=feature_cols, verbose=False)
        threshold, best_f1 = find_optimal_threshold(y_train, cv["y_proba_oof"], metric="f1")
        print(f"  Threshold OOF train: {threshold:.4f} (OOF F1={best_f1:.4f})")

    trainer = MetaMatchTrainer(model_type=args.model_family, threshold=threshold)
    trainer.train(X_train, y_train, feature_names=feature_cols, verbose=False)
    model_score = trainer.predict_proba(X_test)

    pred_df = test_pairs.copy()
    pred_df["model_score"] = model_score
    pred_df["bio_score"] = test_feat["bio_l2_sim"].to_numpy()
    if args.score_mode == "model":
        pred_df["score"] = pred_df["model_score"]
    elif args.score_mode == "bio_l2":
        pred_df["score"] = pred_df["bio_score"]
    elif args.score_mode == "max":
        pred_df["score"] = np.maximum(pred_df["model_score"], pred_df["bio_score"])
    elif args.score_mode == "blend":
        pred_df["score"] = (args.model_weight * pred_df["model_score"]) + ((1.0 - args.model_weight) * pred_df["bio_score"])
    else:
        raise ValueError(f"score_mode inconnu: {args.score_mode}")

    filtered_pred_df = _apply_candidate_filter(pred_df, args.candidate_filter)
    print(f"  Filtre candidats final: {args.candidate_filter} ({len(filtered_pred_df)}/{len(pred_df)} gardés)")

    match_df = build_matches_with_postfilter(
        pred_df=filtered_pred_df,
        threshold=threshold,
        mode=args.post_filter_mode,
        rank_src_max=args.rank_src_max,
        rank_tgt_max=args.rank_tgt_max,
    )
    pred_set = set(zip(match_df["SrcEntity"], match_df["TgtEntity"]))
    metrics = compute_alignment_metrics(pred_set, test_gold)

    train_pairs.to_csv(output_dir / "train_pairs_biogitom_briques.csv", index=False)
    test_pairs.to_csv(output_dir / "test_candidates_biogitom_briques.csv", index=False)
    train_feat.to_csv(output_dir / "train_features_biogitom_briques.csv", index=False)
    test_feat.to_csv(output_dir / "test_features_biogitom_briques.csv", index=False)
    pred_df.to_csv(output_dir / "test_predictions_biogitom_briques.csv", index=False)
    match_df.to_csv(output_dir / "matches.tsv", sep="\t", index=False)

    result = {
        **metrics,
        "pair": args.pair,
        "model": model_name,
        "model_family": args.model_family,
        "token_source": args.token_source,
        "candidate_gold_coverage": coverage,
        "candidate_gold_total": len(test_gold),
        "candidate_gold_coverage_ratio": coverage / max(len(test_gold), 1),
        "candidate_size": len(test_pairs),
        "candidate_filter": args.candidate_filter,
        "candidate_size_after_filter": len(filtered_pred_df),
        "candidate_gold_after_filter": int(filtered_pred_df["label"].sum()) if "label" in filtered_pred_df.columns else None,
        "train_pairs": len(train_pairs),
        "threshold": float(threshold),
        "post_filter_mode": args.post_filter_mode,
        "score_mode": args.score_mode,
        "model_weight": args.model_weight,
        "k_embedding": args.k_embedding,
        "k_overlap": args.k_overlap,
        "k_tfidf": args.k_tfidf,
        "k_final": args.k_final,
        "min_common_tokens": args.min_common_tokens,
        "features": ",".join(feature_cols),
    }
    pd.DataFrame([result]).to_csv(output_dir / "results_biogitom_briques.csv", index=False)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | result, f, indent=2, ensure_ascii=False)

    print(f"  Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} Pred={metrics['n_predicted']}")
    print(f"  Sorties: {output_dir}")
    print(f"  Temps: {time.time() - t0:.1f}s")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BioGITOM-like candidate selection + briques + MetaSpace for ontology matching")
    parser.add_argument("--pair", default="omim-ordo")
    parser.add_argument("--model", default="sapbert", help="'sapbert' ou chemin/nom HuggingFace")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-family", default="xgboost", choices=["xgboost", "extra_trees", "random_forest", "stacking"])
    parser.add_argument("--token-source", default="all_literals", choices=["label", "all_literals"])
    parser.add_argument("--min-common-tokens", type=int, default=5)
    parser.add_argument("--k-overlap", type=int, default=10)
    parser.add_argument("--k-tfidf", type=int, default=5)
    parser.add_argument("--k-final", type=int, default=15)
    parser.add_argument("--k-embedding", type=int, default=5)
    parser.add_argument("--bidirectional-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heuristic-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--heuristic-max-keep", type=int, default=15)
    parser.add_argument("--heuristic-min-score", type=float, default=0.0)
    parser.add_argument("--heuristic-min-keep", type=int, default=3)
    parser.add_argument("--use-synonyms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pooling", default="mean", choices=["mean", "max"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--use-topological", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--include-brick-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lex-high-threshold", type=float, default=0.90)
    parser.add_argument("--score-mode", default="bio_l2", choices=["model", "bio_l2", "max", "blend"])
    parser.add_argument("--model-weight", type=float, default=0.70)
    parser.add_argument(
        "--candidate-filter",
        default="embed_bidir",
        choices=["all", "consensus", "embed_bidir", "overlap_tfidf", "embed_bidir_or_consensus"],
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--auto-threshold-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--post-filter-mode", default="greedy_1to1", choices=["none", "top1_src", "mutual_best", "greedy_1to1"])
    parser.add_argument("--rank-src-max", type=int, default=0)
    parser.add_argument("--rank-tgt-max", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--output-subdir", default="biogitom_briques_sapbert")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
