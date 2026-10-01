#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline ontology matching sans métafeatures:
SapBERT embeddings + top-k L2 bidirectionnel + greedy 1-to-1.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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

from src.config import BIOML_DIR, ONTOLOGY_PAIRS, OUTPUTS_DIR
from src.data.ontology_loader import OntologyLoader
from src.model.evaluate import compute_alignment_metrics
from src.scripts.run_token_overlap_pipeline import build_matches_with_postfilter, infer_src_tgt_files

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def _clean_model_name(model_name: str) -> str:
    return SAPBERT_MODEL if model_name == "sapbert" else model_name


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


class SapBertVectorEncoder:
    def __init__(self, model_name: str = "sapbert", device: str = "cpu"):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Installe transformers et torch pour utiliser SapBERT.") from exc

        self.torch = torch
        self.model_name = model_name
        self.model_path = _clean_model_name(model_name)
        self.device = device
        print(f"Chargement modèle: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)
        print(f"  Dimension: {self.embedding_dim}")

    def encode_texts(self, texts: Sequence[str], batch_size: int, max_length: int) -> np.ndarray:
        vectors: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Embedding batches")
        with self.torch.no_grad():
            for start in iterator:
                batch = [str(x or "") for x in texts[start : start + batch_size]]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(vectors).astype(np.float32)

    def encode_ontology(
        self,
        onto: OntologyLoader,
        cache_path: Path,
        use_synonyms: bool,
        batch_size: int,
        max_length: int,
    ) -> pd.DataFrame:
        if cache_path.exists():
            print(f"Chargement cache: {cache_path}")
            return pd.read_pickle(cache_path)

        iris: List[str] = []
        texts: List[str] = []
        for iri, info in onto.classes.items():
            iris.append(iri)
            label = info.get("label") or onto.get_label(iri) or ""
            if use_synonyms and info.get("synonyms"):
                label = " | ".join([label] + list(info["synonyms"]))
            texts.append(label)

        print(f"Encodage de {len(texts)} classes...")
        vectors = self.encode_texts(texts, batch_size=batch_size, max_length=max_length)
        df = pd.DataFrame(vectors, index=iris, columns=[f"dim_{i}" for i in range(vectors.shape[1])])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        print(f"Cache sauvegardé: {cache_path}")
        return df


class L2Index:
    def __init__(self, embeddings_df: pd.DataFrame):
        self.iris = embeddings_df.index.astype(str).tolist()
        self.matrix = embeddings_df.to_numpy(dtype=np.float32, copy=True)
        self.sq_norm = np.sum(self.matrix * self.matrix, axis=1)

    def topk(self, query_vectors: np.ndarray, query_iris: Sequence[str], k: int, batch_size: int) -> Dict[str, List[Tuple[str, float, int]]]:
        out: Dict[str, List[Tuple[str, float, int]]] = {}
        for start in tqdm(range(0, len(query_iris), batch_size), desc="L2 top-k"):
            q = query_vectors[start : start + batch_size].astype(np.float32, copy=False)
            q_norm = np.sum(q * q, axis=1, keepdims=True)
            dist_sq = np.maximum(q_norm + self.sq_norm[None, :] - 2.0 * q @ self.matrix.T, 0.0)
            take = min(k, dist_sq.shape[1])
            part = np.argpartition(dist_sq, take - 1, axis=1)[:, :take]
            row_dist = np.take_along_axis(dist_sq, part, axis=1)
            order = np.argsort(row_dist, axis=1)
            sorted_idx = np.take_along_axis(part, order, axis=1)
            sorted_dist = np.sqrt(np.take_along_axis(row_dist, order, axis=1))
            for local_i, qiri in enumerate(query_iris[start : start + batch_size]):
                hits: List[Tuple[str, float, int]] = []
                for rank, idx in enumerate(sorted_idx[local_i], start=1):
                    score = 1.0 / (1.0 + float(sorted_dist[local_i, rank - 1]))
                    hits.append((self.iris[int(idx)], score, rank))
                out[qiri] = hits
        return out


def _vectors_for(df: pd.DataFrame, iris: Sequence[str]) -> np.ndarray:
    dim = len(df.columns)
    rows = []
    for iri in iris:
        if iri in df.index:
            rows.append(df.loc[iri].to_numpy(dtype=np.float32, copy=False))
        else:
            rows.append(np.zeros(dim, dtype=np.float32))
    return np.vstack(rows).astype(np.float32)


def _load_test_mappings(pair_dir: Path, task_type: str) -> pd.DataFrame:
    test_path = pair_dir / f"refs_{task_type}" / "test.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    return pd.read_csv(test_path, sep="\t")


def _discover_pairs() -> List[str]:
    configured = list(ONTOLOGY_PAIRS.keys())
    present = [p for p in configured if (BIOML_DIR / p).exists()]
    extra = sorted([p.name for p in BIOML_DIR.iterdir() if p.is_dir() and p.name not in present])
    return present + extra


def run_pair(pair_name: str, args: argparse.Namespace) -> dict:
    pair_dir = BIOML_DIR / pair_name
    output_dir = OUTPUTS_DIR / pair_name / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"SapBERT bidirectionnel sans métafeatures: {pair_name}")
    print("=" * 72)
    t_all = time.time()

    src_file, tgt_file = infer_src_tgt_files(pair_dir, pair_name)
    src_onto = OntologyLoader(src_file).load()
    tgt_onto = OntologyLoader(tgt_file).load()
    test_df = _load_test_mappings(pair_dir, args.task_type)
    ref_set = set(zip(test_df["SrcEntity"], test_df["TgtEntity"]))
    test_sources = sorted(set(test_df["SrcEntity"]))
    test_targets = sorted(set(test_df["TgtEntity"]))
    print(f"Source classes: {len(src_onto)} | Target classes: {len(tgt_onto)}")
    print(f"Test mappings: {len(test_df)} | Test src/tgt: {len(test_sources)}/{len(test_targets)}")

    encoder = SapBertVectorEncoder(model_name=args.model, device=args.device)
    safe_model = _safe_model_name(_clean_model_name(args.model))
    cache_dir = output_dir / "embeddings_cache"
    src_emb = encoder.encode_ontology(
        src_onto,
        cache_dir / f"{pair_name}_src_{safe_model}_vec.pkl",
        use_synonyms=args.use_synonyms,
        batch_size=args.embedding_batch_size,
        max_length=args.max_length,
    )
    tgt_emb = encoder.encode_ontology(
        tgt_onto,
        cache_dir / f"{pair_name}_tgt_{safe_model}_vec.pkl",
        use_synonyms=args.use_synonyms,
        batch_size=args.embedding_batch_size,
        max_length=args.max_length,
    )

    print("Recherche top-k source -> target...")
    src_query_vecs = _vectors_for(src_emb, test_sources)
    src_to_tgt = L2Index(tgt_emb).topk(src_query_vecs, test_sources, k=args.k, batch_size=args.search_batch_size)

    print("Recherche top-k target -> source...")
    tgt_query_vecs = _vectors_for(tgt_emb, test_targets)
    tgt_to_src = L2Index(src_emb).topk(tgt_query_vecs, test_targets, k=args.k, batch_size=args.search_batch_size)
    reverse_pairs = {(s, t): (score, rank) for t, hits in tgt_to_src.items() for s, score, rank in hits}

    rows: List[dict] = []
    for s, hits in src_to_tgt.items():
        src_label = src_onto.get_label(s)
        for t, score_src, rank_src in hits:
            reverse = reverse_pairs.get((s, t))
            if args.candidate_filter == "embed_bidir" and reverse is None:
                continue
            score_tgt, rank_tgt = reverse if reverse is not None else (0.0, 0)
            score = min(score_src, score_tgt) if reverse is not None else score_src
            rows.append(
                {
                    "src_iri": s,
                    "tgt_iri": t,
                    "src_label": src_label,
                    "tgt_label": tgt_onto.get_label(t),
                    "label": int((s, t) in ref_set),
                    "bio_l2_sim": score,
                    "bio_l2_sim_src": score_src,
                    "bio_l2_sim_tgt": score_tgt,
                    "bio_l2_rank_src": rank_src,
                    "bio_l2_rank_tgt": rank_tgt,
                    "brick_embedding_src": 1,
                    "brick_embedding_tgt": int(reverse is not None),
                    "score": score,
                }
            )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        matches = pd.DataFrame(columns=["SrcEntity", "TgtEntity", "Score"])
    else:
        matches = build_matches_with_postfilter(
            candidates,
            threshold=args.threshold,
            mode=args.post_filter_mode,
            rank_src_max=args.rank_src_max,
            rank_tgt_max=args.rank_tgt_max,
        )

    pred_set = set(zip(matches["SrcEntity"], matches["TgtEntity"]))
    metrics = compute_alignment_metrics(pred_set, ref_set)
    result = {
        **metrics,
        "pair": pair_name,
        "model": _clean_model_name(args.model),
        "task_type": args.task_type,
        "k": args.k,
        "candidate_filter": args.candidate_filter,
        "candidate_size": len(candidates),
        "candidate_gold_coverage": int(candidates["label"].sum()) if not candidates.empty else 0,
        "candidate_gold_total": len(ref_set),
        "candidate_gold_coverage_ratio": (int(candidates["label"].sum()) / max(len(ref_set), 1)) if not candidates.empty else 0.0,
        "threshold": args.threshold,
        "post_filter_mode": args.post_filter_mode,
        "elapsed_seconds": time.time() - t_all,
    }

    candidates.to_csv(output_dir / "test_candidates_sapbert_bidir.csv", index=False)
    matches.to_csv(output_dir / "matches.tsv", sep="\t", index=False)
    pd.DataFrame([result]).to_csv(output_dir / "results_sapbert_bidir.csv", index=False)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | result, f, indent=2, ensure_ascii=False)

    print(
        f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} Pred={metrics['n_predicted']}")
    print(f"Candidates={len(candidates)} | Sorties={output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SapBERT/L2 bidirectionnel sans calcul des métafeatures")
    parser.add_argument("--pair", default="omim-ordo", help="Nom de paire, liste séparée par virgules, ou 'all'")
    parser.add_argument("--task-type", default="equiv")
    parser.add_argument("--model", default="sapbert")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-filter", default="embed_bidir", choices=["embed_bidir", "all"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--post-filter-mode", default="greedy_1to1", choices=["none", "top1_src", "mutual_best", "greedy_1to1"])
    parser.add_argument("--rank-src-max", type=int, default=0)
    parser.add_argument("--rank-tgt-max", type=int, default=0)
    parser.add_argument("--use-synonyms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--search-batch-size", type=int, default=256)
    parser.add_argument("--output-subdir", default="sapbert_bidir_baseline")
    args = parser.parse_args()

    if args.pair == "all":
        pairs = _discover_pairs()
    else:
        pairs = [p.strip() for p in args.pair.split(",") if p.strip()]
    if not pairs:
        raise ValueError("Aucune paire à exécuter.")

    results = []
    for pair_name in pairs:
        if not (BIOML_DIR / pair_name).exists():
            print(f"SKIP {pair_name}: dossier absent dans {BIOML_DIR}")
            continue
        results.append(run_pair(pair_name, args))

    if len(results) > 1:
        out_path = OUTPUTS_DIR / f"sapbert_bidir_summary_{args.task_type}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\nRésumé multi-paires: {out_path}")


if __name__ == "__main__":
    main()
