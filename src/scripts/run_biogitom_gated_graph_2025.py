#!/usr/bin/env python3
"""Leakage-free BioGITOM-style graph encoder on a saved 2025 candidate space.

The public BioGITOM implementation builds one Sentence-SapBERT vector from a
concept label and its exact synonyms, propagates these vectors over an
undirected ``subClassOf`` graph with a GIN/Transformer layer, and learns a
dimension-wise gate between semantic and structural vectors.  This script
implements the same representation strategy while keeping MetaMatch's strict
protocol:

* model selection uses ``train.tsv`` candidates only;
* the saved test candidate space is scored only after selection is frozen;
* ``test.tsv`` is opened once, after blind alignments have been written;
* the hierarchy is stored as an edge list, never as a dense N x N matrix.

This is an experimental comparison branch.  It deliberately reuses a saved
candidate space so that the graph encoder, rather than candidate retrieval, is
the experimental variable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.data.ontology_loader import OntologyLoader
from src.scripts.enrich_saved_metaspace_sentence_sapbert import (
    _extra_synonym_map,
    _normalize_rows,
    _unique,
)
from src.scripts.run_sapbert_bidir_baseline import SapBertVectorEncoder
from src.scripts.graph_embedding_workers import encode_missing
from src.scripts.run_token_overlap_pipeline import infer_src_tgt_files


Pair = Tuple[str, str]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_candidate_dir(pair: str, requested: str) -> Path:
    root = OUTPUTS_DIR / pair
    if requested != "auto":
        path = root / requested
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    patterns = (
        "union_label_synonyms_metaspace_*",
        "frozen_protocol_sentence_sapbert_synonyms",
        "frozen_protocol_final",
        "best_graph_metaspace",
        "adaptive_full_tda_metaspace",
        "adaptive_base_metaspace",
    )
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_dir())
        if matches:
            break
    valid = [
        path
        for path in matches
        if (path / "train_oof_predictions.csv").exists()
        and (path / "test_candidates_pre_gold.csv").exists()
    ]
    if not valid:
        raise FileNotFoundError(
            f"No saved candidate MetaSpace found for {pair} below {root}"
        )
    chosen = max(valid, key=lambda path: path.stat().st_mtime)
    print(f"MetaSpace candidats auto: {chosen}")
    return chosen


def _encode_all_entities(
    ontology: OntologyLoader,
    cache_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    max_tokens: int,
    max_synonyms: int,
    checkpoint_size: int,
    side: str,
    workers: int = 1,
    threads: int = 1,
) -> pd.DataFrame:
    iris = sorted(str(iri) for iri in ontology.classes)
    cached = pd.read_pickle(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cached.empty:
        cached.index = cached.index.astype(str)
    missing = sorted(set(iris) - set(cached.index))
    if missing:
        extra = _extra_synonym_map(ontology, set(missing), max_synonyms)
        texts = []
        for iri in missing:
            labels = _unique(
                list(ontology.get_all_labels(iri)) + extra.get(iri, []),
                1 + max_synonyms,
            )
            # This comma-joined representation follows BioGITOM's public CFE
            # implementation. Parents/children are represented by graph edges,
            # not inserted into the transformer text.
            texts.append(", ".join(labels) if labels else ontology.get_label(iri))
        cached = encode_missing(cached, missing, texts, cache_path, model_name, device,
                                batch_size, max_tokens, checkpoint_size, workers, threads, side)
    result = cached.reindex(iris)
    if result.isna().any().any():
        raise RuntimeError(f"Missing {side} semantic embeddings after encoding")
    return result.astype(np.float32)


def _edge_index(ontology: OntologyLoader, iris: Sequence[str]) -> Tensor:
    position = {iri: idx for idx, iri in enumerate(iris)}
    edges = set()
    for child in iris:
        child_idx = position[child]
        for parent in ontology.get_parents(child):
            parent_idx = position.get(str(parent))
            if parent_idx is None or parent_idx == child_idx:
                continue
            edges.add((child_idx, parent_idx))
            edges.add((parent_idx, child_idx))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    array = np.asarray(sorted(edges), dtype=np.int64).T
    return torch.from_numpy(array)


def _iri_digest(iris: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for iri in iris:
        digest.update(str(iri).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_precomputed_edges(path: Path, iris: Sequence[str], side: str) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
        node_count = int(np.asarray(payload["node_count"]).item())
        expected_digest = str(np.asarray(payload["iri_sha256"]).item())
    observed_digest = _iri_digest(iris)
    if node_count != len(iris) or expected_digest != observed_digest:
        raise ValueError(
            f"{side} edge-cache contract mismatch: nodes={node_count}/{len(iris)}, "
            f"digest={expected_digest}/{observed_digest}"
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Invalid {side} edge_index shape: {edge_index.shape}")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= len(iris)
    ):
        raise ValueError(f"{side} edge cache contains an invalid node index")
    print(
        f"Cache arêtes {side}: {len(iris)} nœuds/{edge_index.shape[1]} arcs <- {path}",
        flush=True,
    )
    return torch.from_numpy(edge_index)


def _load_complete_semantic_cache(cache_path: Path, side: str) -> pd.DataFrame:
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    frame = pd.read_pickle(cache_path)
    if frame.empty:
        raise ValueError(f"Empty {side} semantic cache: {cache_path}")
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate {side} semantic-cache IRIs: {cache_path}")
    frame = frame.sort_index()
    if frame.isna().any().any():
        raise ValueError(f"NaN in {side} semantic cache: {cache_path}")
    print(
        f"Cache sémantique {side}: {len(frame)} entités x {frame.shape[1]} dimensions",
        flush=True,
    )
    return frame.astype(np.float32)


class GraphIsomorphismTransformer(nn.Module):
    """One-head GIN/Transformer message-passing layer.

    Attention is computed in edge chunks to keep SNOMED/FMA memory bounded.
    ``edge_index[0]`` is the sender and ``edge_index[1]`` the receiver.
    """

    def __init__(self, dim: int, attention_dim: int, edge_chunk_size: int):
        super().__init__()
        self.query = nn.Linear(dim, attention_dim, bias=False)
        self.key = nn.Linear(dim, attention_dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.PReLU(dim), nn.Linear(dim, dim), nn.PReLU(dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_chunk_size = edge_chunk_size
        self.scale = math.sqrt(attention_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return self.mlp((1.0 + self.eps) * x)
        sender, receiver = edge_index
        q, k, v = self.query(x), self.key(x), self.value(x)
        node_max = torch.full(
            (x.shape[0],), -torch.inf, device=x.device, dtype=x.dtype
        )
        for start in range(0, sender.numel(), self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, sender.numel())
            src, dst = sender[start:stop], receiver[start:stop]
            score = (q[dst] * k[src]).sum(dim=1) / self.scale
            # The group maximum is only a numerical-stability offset; softmax
            # is invariant to it. Detaching avoids an in-place autograd version
            # conflict when the maximum is accumulated over several chunks.
            node_max.scatter_reduce_(
                0, dst, score.detach(), reduce="amax", include_self=True
            )
        denominator = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        aggregate = torch.zeros_like(x)
        for start in range(0, sender.numel(), self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, sender.numel())
            src, dst = sender[start:stop], receiver[start:stop]
            score = (q[dst] * k[src]).sum(dim=1) / self.scale
            weight = torch.exp(score - node_max[dst])
            denominator = denominator.index_add(0, dst, weight)
            aggregate = aggregate.index_add(0, dst, weight.unsqueeze(1) * v[src])
        aggregate = aggregate / denominator.clamp_min(1e-12).unsqueeze(1)
        return self.mlp(aggregate + (1.0 + self.eps) * x)


class SharedGIT(nn.Module):
    def __init__(self, dim: int, attention_dim: int, edge_chunk_size: int):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(dim, dim), nn.PReLU(dim))
        self.git = GraphIsomorphismTransformer(dim, attention_dim, edge_chunk_size)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.git(self.input(x), edge_index)


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.source_gate = nn.Linear(dim, dim)
        self.target_gate = nn.Linear(dim, dim)

    @staticmethod
    def _mix(structural: Tensor, semantic: Tensor, layer: nn.Linear) -> Tensor:
        gate = torch.sigmoid(layer(structural))
        return gate * structural + (1.0 - gate) * semantic

    def forward(
        self, src_struct: Tensor, src_sem: Tensor, tgt_struct: Tensor, tgt_sem: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return (
            self._mix(src_struct, src_sem, self.source_gate),
            self._mix(tgt_struct, tgt_sem, self.target_gate),
        )


def _contrastive_loss(left: Tensor, right: Tensor, label: Tensor, margin: float) -> Tensor:
    distance = F.pairwise_distance(left, right)
    return (
        label * distance.square()
        + (1.0 - label) * F.relu(margin - distance).square()
    ).mean()


def _random_training_pairs(
    positives: Sequence[Pair],
    src_position: Dict[str, int],
    tgt_position: Dict[str, int],
    negatives_per_positive: int,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    positive_set = set(positives)
    src_rows: List[int] = []
    tgt_rows: List[int] = []
    labels: List[float] = []
    tgt_count = len(tgt_position)
    for src, tgt in positives:
        if src not in src_position or tgt not in tgt_position:
            continue
        src_rows.append(src_position[src])
        tgt_rows.append(tgt_position[tgt])
        labels.append(1.0)
        used = set()
        while len(used) < min(negatives_per_positive, tgt_count - 1):
            candidate = int(rng.integers(0, tgt_count))
            if candidate in used:
                continue
            tgt_iri = next_iris_tgt[candidate]
            if (src, tgt_iri) in positive_set:
                continue
            used.add(candidate)
            src_rows.append(src_position[src])
            tgt_rows.append(candidate)
            labels.append(0.0)
    return (
        torch.tensor(src_rows, dtype=torch.long),
        torch.tensor(tgt_rows, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


# Set by run() before negative sampling.  Keeping the ordered list separate
# avoids rebuilding an inverse target dictionary inside the hot loop.
next_iris_tgt: List[str] = []


@dataclass
class FitResult:
    git: SharedGIT
    gate: GatedFusion
    best_epoch: int
    validation_loss: float


def _fit_models(
    src_sem: Tensor,
    tgt_sem: Tensor,
    src_edges: Tensor,
    tgt_edges: Tensor,
    train_triplets: Tuple[Tensor, Tensor, Tensor],
    valid_triplets: Tuple[Tensor, Tensor, Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> FitResult:
    dim = src_sem.shape[1]
    git = SharedGIT(dim, args.attention_dim, args.edge_chunk_size).to(device)
    gate = GatedFusion(dim).to(device)
    src_sem, tgt_sem = src_sem.to(device), tgt_sem.to(device)
    src_edges, tgt_edges = src_edges.to(device), tgt_edges.to(device)
    train = tuple(value.to(device) for value in train_triplets)
    valid = tuple(value.to(device) for value in valid_triplets)
    optimizer = torch.optim.Adam(
        list(git.parameters()) + list(gate.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        git.train()
        gate.train()
        optimizer.zero_grad()
        src_struct = git(src_sem, src_edges)
        tgt_struct = git(tgt_sem, tgt_edges)
        src_final, tgt_final = gate(src_struct, src_sem, tgt_struct, tgt_sem)
        loss = _contrastive_loss(
            src_final[train[0]], tgt_final[train[1]], train[2], args.margin
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(git.parameters()) + list(gate.parameters()), args.gradient_clip
        )
        optimizer.step()
        git.eval()
        gate.eval()
        with torch.no_grad():
            src_struct = git(src_sem, src_edges)
            tgt_struct = git(tgt_sem, tgt_edges)
            src_final, tgt_final = gate(src_struct, src_sem, tgt_struct, tgt_sem)
            valid_loss = float(
                _contrastive_loss(
                    src_final[valid[0]], tgt_final[valid[1]], valid[2], args.margin
                ).cpu()
            )
        if valid_loss < best_loss - args.min_delta:
            best_loss, best_epoch, patience = valid_loss, epoch, 0
            best_state = {
                "git": {k: v.detach().cpu().clone() for k, v in git.state_dict().items()},
                "gate": {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
            }
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"BioGITOM epoch {epoch}/{args.epochs}: "
                f"train_loss={float(loss.detach().cpu()):.6f} "
                f"valid_loss={valid_loss:.6f}"
            )
        if patience >= args.early_stopping_patience:
            print(f"Early stopping epoch {epoch}; best={best_epoch}")
            break
    if best_state is None:
        raise RuntimeError("BioGITOM model selection failed")
    git.load_state_dict(best_state["git"])
    gate.load_state_dict(best_state["gate"])
    return FitResult(git, gate, best_epoch, best_loss)


def _final_embeddings(
    fit: FitResult,
    src_sem: Tensor,
    tgt_sem: Tensor,
    src_edges: Tensor,
    tgt_edges: Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    fit.git.eval()
    fit.gate.eval()
    with torch.no_grad():
        ss, ts = src_sem.to(device), tgt_sem.to(device)
        structural_src = fit.git(ss, src_edges.to(device))
        structural_tgt = fit.git(ts, tgt_edges.to(device))
        final_src, final_tgt = fit.gate(structural_src, ss, structural_tgt, ts)
        final_src = F.normalize(final_src, dim=1).cpu().numpy().astype(np.float32)
        final_tgt = F.normalize(final_tgt, dim=1).cpu().numpy().astype(np.float32)
    return final_src, final_tgt


def _score_pairs(
    pairs: pd.DataFrame,
    src_vectors: np.ndarray,
    tgt_vectors: np.ndarray,
    src_position: Dict[str, int],
    tgt_position: Dict[str, int],
) -> pd.DataFrame:
    src_idx = pairs["src_iri"].astype(str).map(src_position)
    tgt_idx = pairs["tgt_iri"].astype(str).map(tgt_position)
    valid = src_idx.notna() & tgt_idx.notna()
    out = pairs.loc[valid, ["src_iri", "tgt_iri"]].reset_index(drop=True)
    left = src_vectors[src_idx[valid].astype(int).to_numpy()]
    right = tgt_vectors[tgt_idx[valid].astype(int).to_numpy()]
    cosine = np.sum(left * right, axis=1)
    out["score"] = cosine
    out["rank_src_tgt"] = out.groupby("src_iri")["score"].rank(
        method="min", ascending=False
    )
    out["rank_tgt_src"] = out.groupby("tgt_iri")["score"].rank(
        method="min", ascending=False
    )
    return out


def _topk_indices(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
    device: torch.device,
    query_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k without materialising the full similarity matrix."""
    if not len(query) or not len(corpus) or k <= 0:
        return (
            np.empty((len(query), 0), dtype=np.int64),
            np.empty((len(query), 0), dtype=np.float32),
        )
    k = min(k, len(corpus))
    corpus_tensor = torch.from_numpy(corpus).to(device)
    all_indices: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(query), query_chunk_size):
            stop = min(start + query_chunk_size, len(query))
            query_tensor = torch.from_numpy(query[start:stop]).to(device)
            values, indices = torch.topk(query_tensor @ corpus_tensor.T, k=k, dim=1)
            all_indices.append(indices.cpu().numpy().astype(np.int64, copy=False))
            all_scores.append(values.cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(all_indices), np.vstack(all_scores)


def _expanded_retrieval_pairs(
    existing: pd.DataFrame,
    src_query_iris: Sequence[str],
    tgt_query_iris: Sequence[str],
    src_vectors: np.ndarray,
    tgt_vectors: np.ndarray,
    src_position: Dict[str, int],
    tgt_position: Dict[str, int],
    topk: int,
    device: torch.device,
    query_chunk_size: int,
) -> pd.DataFrame:
    """Union saved candidates with forward and reverse embedding retrieval."""
    rows: Dict[Pair, set[str]] = {}
    for row in existing[["src_iri", "tgt_iri"]].astype(str).itertuples(index=False):
        rows.setdefault((row.src_iri, row.tgt_iri), set()).add("saved")

    src_queries = [iri for iri in dict.fromkeys(map(str, src_query_iris)) if iri in src_position]
    if src_queries:
        query = src_vectors[[src_position[iri] for iri in src_queries]]
        indices, _ = _topk_indices(
            query, tgt_vectors, topk, device, query_chunk_size
        )
        tgt_iris_by_position = [None] * len(tgt_position)
        for iri, position in tgt_position.items():
            tgt_iris_by_position[position] = iri
        for src_iri, neighbours in zip(src_queries, indices):
            for target_idx in neighbours:
                rows.setdefault(
                    (src_iri, str(tgt_iris_by_position[int(target_idx)])), set()
                ).add("ann_src")

    tgt_queries = [iri for iri in dict.fromkeys(map(str, tgt_query_iris)) if iri in tgt_position]
    # Reverse retrieval must stay inside the requested source population
    # (validation sources or blind-test sources inferred from saved candidates).
    if tgt_queries and src_queries:
        query = tgt_vectors[[tgt_position[iri] for iri in tgt_queries]]
        source_corpus = src_vectors[[src_position[iri] for iri in src_queries]]
        indices, _ = _topk_indices(
            query, source_corpus, topk, device, query_chunk_size
        )
        for tgt_iri, neighbours in zip(tgt_queries, indices):
            for source_idx in neighbours:
                rows.setdefault(
                    (src_queries[int(source_idx)], tgt_iri), set()
                ).add("ann_tgt")

    result = pd.DataFrame(
        (
            {"src_iri": src, "tgt_iri": tgt, "candidate_origin": "+".join(sorted(origin))}
            for (src, tgt), origin in rows.items()
        )
    )
    return result.sort_values(["src_iri", "tgt_iri"]).reset_index(drop=True)


def _select_matches(scored: pd.DataFrame, threshold: float, mode: str) -> pd.DataFrame:
    frame = scored.loc[scored.score >= threshold].sort_values(
        ["score", "src_iri", "tgt_iri"], ascending=[False, True, True]
    )
    if mode == "mutual_best":
        frame = frame.loc[(frame.rank_src_tgt == 1) & (frame.rank_tgt_src == 1)]
    elif mode == "greedy_1to1":
        rows = []
        seen_src, seen_tgt = set(), set()
        for row in frame.itertuples(index=False):
            if row.src_iri in seen_src or row.tgt_iri in seen_tgt:
                continue
            seen_src.add(row.src_iri)
            seen_tgt.add(row.tgt_iri)
            rows.append(row)
        frame = pd.DataFrame(rows, columns=frame.columns) if rows else frame.iloc[:0]
    return frame.reset_index(drop=True)


def _metrics(predicted: Iterable[Pair], gold: Iterable[Pair]) -> Dict[str, float]:
    predicted, gold = set(predicted), set(gold)
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": len(predicted) - tp,
        "fn": len(gold) - tp,
    }


def _tune_selection(scored: pd.DataFrame, gold: set[Pair]) -> Dict[str, object]:
    best = None
    thresholds = np.unique(
        np.concatenate([np.linspace(-0.1, 0.95, 106), scored.score.quantile(
            np.linspace(0.5, 0.995, 50)
        ).to_numpy()])
    )
    for mode in ("none", "greedy_1to1", "mutual_best"):
        for threshold in thresholds:
            matches = _select_matches(scored, float(threshold), mode)
            metrics = _metrics(zip(matches.src_iri, matches.tgt_iri), gold)
            row = {"threshold": float(threshold), "post_filter_mode": mode, **metrics}
            if best is None or (row["f1"], row["recall"], row["precision"]) > (
                best["f1"], best["recall"], best["precision"]
            ):
                best = row
    assert best is not None
    return best


def run(args: argparse.Namespace) -> Dict[str, object]:
    global next_iris_tgt
    started = time.time()
    _seed_everything(args.seed)
    candidate_dir = _resolve_candidate_dir(args.pair, args.input_subdir)
    output_dir = OUTPUTS_DIR / args.pair / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = BIOML_DIR / args.pair
    cache_dir = OUTPUTS_DIR / args.pair / args.cache_subdir
    tag = f"label_synonyms_syn{args.max_synonyms}_tok{args.max_tokens}"
    src_cache = cache_dir / f"src_biogitom_{tag}.pkl"
    tgt_cache = cache_dir / f"tgt_biogitom_{tag}.pkl"
    use_precomputed_edges = bool(args.src_edge_cache or args.tgt_edge_cache)
    if use_precomputed_edges:
        if not args.src_edge_cache or not args.tgt_edge_cache:
            raise ValueError("Both --src-edge-cache and --tgt-edge-cache are required")
        # Fast rescue path: ontology-wide semantic caches and edge caches are
        # complete, so Graph06 must not reopen or re-index the OWL graphs.
        src_onto = tgt_onto = None
        src_df = _load_complete_semantic_cache(src_cache, "source")
        tgt_df = _load_complete_semantic_cache(tgt_cache, "cible")
    else:
        src_file, tgt_file = infer_src_tgt_files(pair_dir, args.pair)
        src_onto = OntologyLoader(src_file).load()
        tgt_onto = OntologyLoader(tgt_file).load()
        src_df = _encode_all_entities(
            src_onto, src_cache, args.model,
            args.device, args.batch_size, args.max_tokens, args.max_synonyms,
            args.checkpoint_size, "source", args.embedding_workers, args.embedding_threads,
        )
        tgt_df = _encode_all_entities(
            tgt_onto, tgt_cache, args.model,
            args.device, args.batch_size, args.max_tokens, args.max_synonyms,
            args.checkpoint_size, "cible", args.embedding_workers, args.embedding_threads,
        )
    if args.prepare_only:
        payload = {
            "pair": args.pair,
            "source_entities": len(src_df),
            "target_entities": len(tgt_df),
            "cache_subdir": args.cache_subdir,
            "representation": "Sentence-SapBERT(label+synonyms)",
        }
        (output_dir / "semantic_cache_ready.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(payload)
        return payload
    src_iris, tgt_iris = src_df.index.tolist(), tgt_df.index.tolist()
    next_iris_tgt = tgt_iris
    src_pos = {iri: idx for idx, iri in enumerate(src_iris)}
    tgt_pos = {iri: idx for idx, iri in enumerate(tgt_iris)}
    src_sem = torch.from_numpy(src_df.to_numpy(dtype=np.float32))
    tgt_sem = torch.from_numpy(tgt_df.to_numpy(dtype=np.float32))
    if use_precomputed_edges:
        src_edges = _load_precomputed_edges(args.src_edge_cache, src_iris, "source")
        tgt_edges = _load_precomputed_edges(args.tgt_edge_cache, tgt_iris, "cible")
    else:
        assert src_onto is not None and tgt_onto is not None
        src_edges = _edge_index(src_onto, src_iris)
        tgt_edges = _edge_index(tgt_onto, tgt_iris)
    print(
        f"Graphes BioGITOM: source={len(src_iris)} nœuds/{src_edges.shape[1]} arcs; "
        f"cible={len(tgt_iris)} nœuds/{tgt_edges.shape[1]} arcs"
    )

    train_gold_df = pd.read_csv(pair_dir / "refs_equiv" / "train.tsv", sep="\t")
    positives = list(zip(
        train_gold_df.SrcEntity.astype(str), train_gold_df.TgtEntity.astype(str)
    ))
    # Hold out complete source groups.  A correspondence-level split can leave
    # another gold target for the same source in training and then count it as
    # a false positive against an incomplete validation subset.
    rng = np.random.default_rng(args.seed)
    positive_by_source: Dict[str, List[Pair]] = {}
    for pair in positives:
        positive_by_source.setdefault(pair[0], []).append(pair)
    source_order = list(positive_by_source)
    rng.shuffle(source_order)
    requested_valid = max(1, int(round(len(positives) * args.validation_fraction)))
    valid_sources_grouped: set[str] = set()
    valid_size = 0
    for source in source_order:
        if valid_size >= requested_valid:
            break
        valid_sources_grouped.add(source)
        valid_size += len(positive_by_source[source])
    valid_positive = [pair for pair in positives if pair[0] in valid_sources_grouped]
    train_positive = [pair for pair in positives if pair[0] not in valid_sources_grouped]
    train_triplets = _random_training_pairs(
        train_positive, src_pos, tgt_pos, args.negatives_per_positive, args.seed
    )
    valid_triplets = _random_training_pairs(
        valid_positive, src_pos, tgt_pos, args.negatives_per_positive,
        args.seed + 1,
    )
    device = torch.device(args.device)
    fit = _fit_models(
        src_sem, tgt_sem, src_edges, tgt_edges, train_triplets, valid_triplets,
        args, device,
    )
    src_vectors, tgt_vectors = _final_embeddings(
        fit, src_sem, tgt_sem, src_edges, tgt_edges, device
    )

    train_candidates = pd.read_csv(candidate_dir / "train_oof_predictions.csv")
    valid_sources = {src for src, _ in valid_positive}
    valid_candidates = train_candidates.loc[
        train_candidates.src_iri.astype(str).isin(valid_sources)
    ].copy()
    if args.retrieval_topk > 0:
        # Mirror blind test retrieval: the validation ANN target universe must
        # come from the already generated candidate space, never from the gold
        # validation correspondences.  The former gold-target restriction made
        # validation artificially easy and inflated V5 validation F1.
        valid_target_universe = sorted(
            valid_candidates.tgt_iri.astype(str).unique()
        )
        valid_candidates = _expanded_retrieval_pairs(
            valid_candidates,
            src_query_iris=sorted(valid_sources),
            tgt_query_iris=valid_target_universe,
            src_vectors=src_vectors,
            tgt_vectors=tgt_vectors,
            src_position=src_pos,
            tgt_position=tgt_pos,
            topk=args.retrieval_topk,
            device=device,
            query_chunk_size=args.retrieval_query_chunk_size,
        )
    valid_scores = _score_pairs(valid_candidates, src_vectors, tgt_vectors, src_pos, tgt_pos)
    valid_scores.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(valid_positive, columns=["SrcEntity", "TgtEntity"]).to_csv(
        output_dir / "validation_gold.tsv", sep="\t", index=False
    )
    selected = _tune_selection(valid_scores, set(valid_positive))
    print(f"Sélection train-only BioGITOM: {selected}")

    # The hold-out above freezes the epoch count and decision protocol.  Fit a
    # fresh final encoder on every train.tsv positive before blind inference.
    # Using the full triplet set as the monitoring set here does not select a
    # new hyperparameter; it only retains the numerically best state within the
    # already frozen number of epochs.
    all_triplets = _random_training_pairs(
        positives, src_pos, tgt_pos, args.negatives_per_positive, args.seed + 2
    )
    final_args = copy.copy(args)
    final_args.epochs = max(1, fit.best_epoch)
    final_args.early_stopping_patience = final_args.epochs + 1
    print(
        f"Réentraînement final sur tout train.tsv: {len(positives)} positifs, "
        f"{final_args.epochs} epochs"
    )
    final_fit = _fit_models(
        src_sem, tgt_sem, src_edges, tgt_edges, all_triplets, all_triplets,
        final_args, device,
    )
    src_vectors, tgt_vectors = _final_embeddings(
        final_fit, src_sem, tgt_sem, src_edges, tgt_edges, device
    )

    # Blind test prediction. The representation has never used test.tsv.
    test_candidates = pd.read_csv(candidate_dir / "test_candidates_pre_gold.csv")
    if args.retrieval_topk > 0:
        # Test entity membership is inferred from the already saved blind
        # candidate space.  test.tsv is still not opened here.
        test_candidates = _expanded_retrieval_pairs(
            test_candidates,
            src_query_iris=sorted(test_candidates.src_iri.astype(str).unique()),
            tgt_query_iris=sorted(test_candidates.tgt_iri.astype(str).unique()),
            src_vectors=src_vectors,
            tgt_vectors=tgt_vectors,
            src_position=src_pos,
            tgt_position=tgt_pos,
            topk=args.retrieval_topk,
            device=device,
            query_chunk_size=args.retrieval_query_chunk_size,
        )
    test_scores = _score_pairs(test_candidates, src_vectors, tgt_vectors, src_pos, tgt_pos)
    matches = _select_matches(
        test_scores, float(selected["threshold"]), str(selected["post_filter_mode"])
    )
    alignment = matches.rename(
        columns={"src_iri": "SrcEntity", "tgt_iri": "TgtEntity", "score": "Score"}
    )[["SrcEntity", "TgtEntity", "Score"]]
    alignment.to_csv(output_dir / "matches_pre_gold.tsv", sep="\t", index=False)
    test_scores.to_csv(output_dir / "test_predictions_pre_gold.csv", index=False)
    torch.save(
        {"git": final_fit.git.state_dict(), "gate": final_fit.gate.state_dict()},
        output_dir / "biogitom_gated_graph.pt",
    )
    train_only = {
        "pair": args.pair,
        "candidate_dir": str(candidate_dir),
        "selected_train_only": selected,
        "best_epoch": fit.best_epoch,
        "final_fit_epoch": final_fit.best_epoch,
        "validation_loss": fit.validation_loss,
        "negative_per_positive": args.negatives_per_positive,
        "retrieval_topk_bidirectional": args.retrieval_topk,
        "representation": "Sentence-SapBERT(label+synonyms)+undirected-subClassOf-GIT+gate",
        "blind_alignment_path": str(output_dir / "matches_pre_gold.tsv"),
        "test_tsv_access": "never before blind alignment save",
    }
    (output_dir / "train_only_results.json").write_text(
        json.dumps(train_only, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.train_only:
        print("TRAIN-ONLY terminé; test.tsv n'a jamais été chargé")
        return train_only

    # Evaluation is intentionally last and is never used for selection.
    test_gold_df = pd.read_csv(pair_dir / "refs_equiv" / "test.tsv", sep="\t")
    test_gold = set(zip(
        test_gold_df.SrcEntity.astype(str), test_gold_df.TgtEntity.astype(str)
    ))
    final_metrics = _metrics(zip(alignment.SrcEntity, alignment.TgtEntity), test_gold)
    result = {
        **train_only,
        **final_metrics,
        "alignment_count": len(alignment),
        "elapsed_seconds": time.time() - started,
        "test_tsv_access": "once after blind alignment save",
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"FINAL BioGITOM-style {args.pair}: P={final_metrics['precision']:.4f} "
        f"R={final_metrics['recall']:.4f} F1={final_metrics['f1']:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--input-subdir", default="auto")
    parser.add_argument("--output-subdir", default="biogitom_gated_graph_2025")
    parser.add_argument("--cache-subdir", default="biogitom_gated_graph_cache")
    parser.add_argument("--model", default="sapbert")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-size", type=int, default=1024)
    parser.add_argument("--embedding-workers", type=int, default=1)
    parser.add_argument("--embedding-threads", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-synonyms", type=int, default=999)
    parser.add_argument("--negatives-per-positive", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--edge-chunk-size", type=int, default=100000)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retrieval-topk",
        type=int,
        default=0,
        help="Add exact forward/reverse top-k candidates in the learned space.",
    )
    parser.add_argument("--retrieval-query-chunk-size", type=int, default=256)
    parser.add_argument("--src-edge-cache", type=Path)
    parser.add_argument("--tgt-edge-cache", type=Path)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0,1)")
    run(args)


if __name__ == "__main__":
    main()
