#!/usr/bin/env python3
"""Chunked label+synonyms embeddings, semantic gate, and exact 79 features."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.classical import CLASSICAL_FEATURES
from src.features.pipeline import FeaturePipeline
from src.features.spectral import SPECTRAL_FEATURES, compute_spectral_features
from src.features.syntax import SYNTAX_FEATURES
from src.features.topological import TOPOLOGICAL_FEATURES, compute_topological_features
from src.scripts.run_biogitom_briques_pipeline import DirectTransformerLabelEncoder
from src.scripts.embedding_storage import atomic_pickle, merge_embedding_shards, reuse_legacy_shards


SPECTRAL_PAIR_FEATURES = [
    f"{prefix}_{name.replace('spc_', '')}"
    for prefix in ("spc_src", "spc_tgt", "spc_combined")
    for name in SPECTRAL_FEATURES
]
FEATURES79 = list(SYNTAX_FEATURES) + list(CLASSICAL_FEATURES) + SPECTRAL_PAIR_FEATURES + list(TOPOLOGICAL_FEATURES)


def _bounds(size: int, index: int, total: int) -> tuple[int, int]:
    return (size * index) // total, (size * (index + 1)) // total


def prepare_entities(args: argparse.Namespace) -> None:
    shared, embed = Path(args.shared_dir), Path(args.embedding_dir)
    embed.mkdir(parents=True, exist_ok=True)
    src: set[str] = set()
    tgt: set[str] = set()
    for candidate_dir in args.candidate_dirs:
        root = Path(candidate_dir)
        for split in ("train", "unseen"):
            frame = pd.read_pickle(root / f"{split}_lexical_candidates.pkl")
            src.update(frame.src_iri.astype(str))
            tgt.update(frame.tgt_iri.astype(str))
    for side, values in (("src", src), ("tgt", tgt)):
        texts = pd.read_pickle(shared / f"{side}_texts.pkl")
        selected = texts[texts.iri.astype(str).isin(values)].copy()
        if len(selected) != len(values):
            missing = values - set(selected.iri.astype(str))
            raise RuntimeError(f"Textes {side} manquants: {list(missing)[:5]}")
        selected = selected.sort_values("iri").reset_index(drop=True)
        atomic_pickle(selected, embed / f"{side}_entities.pkl")
        reuse_legacy_shards(embed, side, selected.iri.astype(str).tolist(), args.num_shards)
        print(f"Entités {side} à encoder une seule fois: {len(selected)}")
    (embed / "embedding_manifest.json").write_text(json.dumps({
        "representation": "label_synonyms", "max_synonyms": args.max_synonyms,
        "max_tokens": args.max_tokens, "candidate_dirs": args.candidate_dirs,
        "hidden_test_tsv_access": "never",
    }, indent=2), encoding="utf-8")


def encode_shard(args: argparse.Namespace) -> None:
    embed = Path(args.embedding_dir)
    frame = pd.read_pickle(embed / f"{args.side}_entities.pkl")
    start, end = _bounds(len(frame), args.shard_index, args.num_shards)
    part = frame.iloc[start:end].reset_index(drop=True)
    encoder = DirectTransformerLabelEncoder(model_name="sapbert", device=args.device)
    clouds, pooled = encoder.encode_token_and_vector(
        part.text.astype(str).tolist(), batch_size=args.batch_size,
        show_progress=True, pooling="mean", max_tokens=args.max_tokens,
    )
    out = embed / "shards"
    out.mkdir(parents=True, exist_ok=True)
    atomic_pickle({
        "iris": part.iri.astype(str).tolist(), "clouds": clouds,
        "pooled": np.asarray(pooled, dtype=np.float32),
    }, out / f"{args.side}_{args.shard_index:04d}_of_{args.num_shards:04d}.pkl")
    print(f"Embedding {args.side} {args.shard_index + 1}/{args.num_shards}: {len(part)}")


def merge_embeddings(args: argparse.Namespace) -> None:
    merge_embedding_shards(args.embedding_dir, args.num_shards, args.max_tokens)


def _embedding_lookup(embed: Path, side: str):
    iris = json.loads((embed / f"{side}_iris.json").read_text())
    return {iri: i for i, iri in enumerate(iris)}, np.load(embed / f"{side}_pooled.npy", mmap_mode="r")


def semantic_shard(args: argparse.Namespace) -> None:
    work, embed = Path(args.work_dir), Path(args.embedding_dir)
    src_pos, src_vec = _embedding_lookup(embed, "src")
    tgt_pos, tgt_vec = _embedding_lookup(embed, "tgt")
    out = work / "semantic_shards"; out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "unseen"):
        frame = pd.read_pickle(work / f"{split}_lexical_candidates.pkl")
        start, end = _bounds(len(frame), args.shard_index, args.num_shards)
        part = frame.iloc[start:end].copy().reset_index(drop=True)
        if len(part):
            left = np.vstack([src_vec[src_pos[x]] for x in part.src_iri.astype(str)]).astype(np.float32)
            right = np.vstack([tgt_vec[tgt_pos[x]] for x in part.tgt_iri.astype(str)]).astype(np.float32)
            left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
            right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
            part["semantic_cosine"] = np.sum(left * right, axis=1)
        else:
            part["semantic_cosine"] = pd.Series(dtype=float)
        part.to_pickle(out / f"{split}_{args.shard_index:04d}_of_{args.num_shards:04d}.pkl")


def merge_semantic(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    gold_df = pd.read_csv(work / "train_gold.tsv", sep="\t")
    gold = set(zip(gold_df.SrcEntity.astype(str), gold_df.TgtEntity.astype(str)))
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "unseen"):
        paths = [work / "semantic_shards" / f"{split}_{i:04d}_of_{args.num_shards:04d}.pkl" for i in range(args.num_shards)]
        missing = [str(x) for x in paths if not x.is_file()]
        if missing:
            raise FileNotFoundError(f"Shards sémantiques manquants: {missing[:5]}")
        frame = pd.concat((pd.read_pickle(x) for x in paths), ignore_index=True)
        # Gold-only injected rows are training examples, never retrieval
        # evidence.  Excluding them from rank groups prevents gold leakage.
        retrieved_mask = frame.retrieved.astype(bool)
        frame["rank_src_tgt"] = np.inf
        frame["rank_tgt_src"] = np.inf
        retrieved_frame = frame.loc[retrieved_mask]
        frame.loc[retrieved_mask, "rank_src_tgt"] = retrieved_frame.semantic_cosine.groupby(
            retrieved_frame.src_iri
        ).rank(method="min", ascending=False)
        frame.loc[retrieved_mask, "rank_tgt_src"] = retrieved_frame.semantic_cosine.groupby(
            retrieved_frame.tgt_iri
        ).rank(method="min", ascending=False)
        frames[split] = frame
    rows = []
    retrieved = frames["train"][frames["train"].retrieved.astype(bool)]
    configs = [("none", 0)] + list(product(("and", "or"), args.k_semantic_grid))
    for rule, k in configs:
        if rule == "none":
            kept = retrieved
        elif rule == "and":
            kept = retrieved[(retrieved.rank_src_tgt <= k) & (retrieved.rank_tgt_src <= k)]
        else:
            kept = retrieved[(retrieved.rank_src_tgt <= k) | (retrieved.rank_tgt_src <= k)]
        pairs = set(zip(kept.src_iri, kept.tgt_iri)); covered = len(pairs & gold)
        rows.append({"filter_rule": rule, "k_semantic": k, "candidate_count": len(kept),
                     "gold_covered": covered, "gold_total": len(gold),
                     "candidate_recall": covered / len(gold) if gold else 0.0})
    grid = pd.DataFrame(rows)
    eligible = grid[grid.candidate_recall >= args.min_candidate_recall]
    if eligible.empty:
        eligible = grid[grid.candidate_recall == grid.candidate_recall.max()]
    selected = eligible.sort_values(["candidate_count", "candidate_recall"], ascending=[True, False]).iloc[0]
    rule, k = str(selected.filter_rule), int(selected.k_semantic)
    for split, frame in frames.items():
        base = frame[frame.retrieved.astype(bool)]
        if rule == "and":
            base = base[(base.rank_src_tgt <= k) & (base.rank_tgt_src <= k)]
        elif rule == "or":
            base = base[(base.rank_src_tgt <= k) | (base.rank_tgt_src <= k)]
        if split == "train":
            positive_rows = frame[frame.label.astype(int) == 1]
            base = pd.concat([base, positive_rows], ignore_index=True)
        base = base.drop_duplicates(["src_iri", "tgt_iri"]).sort_values(["src_iri", "tgt_iri"]).reset_index(drop=True)
        base.to_pickle(work / f"{split}_final_candidates.pkl")
        print(f"{args.branch} {split} après filtre sémantique: {len(base)}")
    grid.to_csv(work / "semantic_grid_train_only.csv", index=False)
    (work / "selected_semantic_protocol.json").write_text(json.dumps({
        "selected": selected.to_dict(), "representation": "label_synonyms",
        "gold_injected_train_only": True, "test_tsv_access": "never",
    }, indent=2), encoding="utf-8")


def _cloud(embed: Path, side: str):
    iris = json.loads((embed / f"{side}_iris.json").read_text())
    return (
        {iri: i for i, iri in enumerate(iris)},
        np.load(embed / f"{side}_clouds.npy", mmap_mode="r"),
        np.load(embed / f"{side}_lengths.npy", mmap_mode="r"),
        np.load(embed / f"{side}_pooled.npy", mmap_mode="r"),
    )


def feature_shard(args: argparse.Namespace) -> None:
    work, embed = Path(args.work_dir), Path(args.embedding_dir)
    src_pos, src_clouds, src_lengths, src_pool = _cloud(embed, "src")
    tgt_pos, tgt_clouds, tgt_lengths, tgt_pool = _cloud(embed, "tgt")
    pipeline = FeaturePipeline(use_syntax=True, use_classical=True, use_spectral=False, use_topological=False, use_nlp=False)
    out = work / "feature_shards"; out.mkdir(parents=True, exist_ok=True)
    # A private cache per shard avoids concurrent writes to the same pickle.
    cache_dir = work / "tda_entity_cache" / f"shard_{args.shard_index:04d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    spectral_bases = [x.replace("spc_", "") for x in SPECTRAL_FEATURES]
    for split in ("train", "unseen"):
        pairs_all = pd.read_pickle(work / f"{split}_final_candidates.pkl")
        start, end = _bounds(len(pairs_all), args.shard_index, args.num_shards)
        pairs = pairs_all.iloc[start:end].copy().reset_index(drop=True)
        if len(pairs):
            left = np.vstack([src_pool[src_pos[x]] for x in pairs.src_iri.astype(str)]).astype(np.float32)
            right = np.vstack([tgt_pool[tgt_pos[x]] for x in pairs.tgt_iri.astype(str)]).astype(np.float32)
        else:
            left = right = np.empty((0, 768), np.float32)
        if len(pairs):
            base = pipeline.compute_features_batch(
                pairs, left, right, show_progress=False
            ).reset_index(drop=True)
        else:
            base = pd.DataFrame(columns=list(SYNTAX_FEATURES) + list(CLASSICAL_FEATURES))
        tda_rows, spectral_rows = [], []
        diagram_cache: dict = {}; entity_cache: dict = {}
        src_spc: dict[str, dict] = {}; tgt_spc: dict[str, dict] = {}
        for row in pairs.itertuples(index=False):
            si, ti = src_pos[str(row.src_iri)], tgt_pos[str(row.tgt_iri)]
            sc = np.asarray(src_clouds[si, : int(src_lengths[si])], dtype=np.float32)
            tc = np.asarray(tgt_clouds[ti, : int(tgt_lengths[ti])], dtype=np.float32)
            tda_rows.append(compute_topological_features(
                sc, tc, diagram_cache=diagram_cache, entity_cache=entity_cache,
                disk_cache_dir=str(cache_dir), src_key=str(row.src_iri), tgt_key=str(row.tgt_iri),
            ))
            if str(row.src_iri) not in src_spc:
                src_spc[str(row.src_iri)] = compute_spectral_features(sc, "spc_src") if sc.size else {f"spc_src_{x}": 0.0 for x in spectral_bases}
            if str(row.tgt_iri) not in tgt_spc:
                tgt_spc[str(row.tgt_iri)] = compute_spectral_features(tc, "spc_tgt") if tc.size else {f"spc_tgt_{x}": 0.0 for x in spectral_bases}
            combined = compute_spectral_features(np.vstack([sc, tc]), "spc_combined") if sc.size and tc.size else {f"spc_combined_{x}": 0.0 for x in spectral_bases}
            spectral_rows.append({**src_spc[str(row.src_iri)], **tgt_spc[str(row.tgt_iri)], **combined})
        spectral = pd.DataFrame(spectral_rows, columns=SPECTRAL_PAIR_FEATURES).replace([np.inf, -np.inf], 0).fillna(0)
        topology = pd.DataFrame(tda_rows, columns=TOPOLOGICAL_FEATURES).replace([np.inf, -np.inf], 0).fillna(0)
        features = pd.concat([base, spectral, topology], axis=1)
        if list(features.columns) != FEATURES79 or len(features.columns) != 79:
            raise RuntimeError(f"MetaSpace invalide: {len(features.columns)} features; attendu 79")
        pd.to_pickle({"pairs": pairs, "features": features}, out / f"{split}_{args.shard_index:04d}_of_{args.num_shards:04d}.pkl")
        print(f"{args.branch} MetaSpace {split} shard {args.shard_index + 1}/{args.num_shards}: {len(pairs)}")


def merge_features(args: argparse.Namespace) -> None:
    work, output = Path(args.work_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "unseen"):
        paths = [work / "feature_shards" / f"{split}_{i:04d}_of_{args.num_shards:04d}.pkl" for i in range(args.num_shards)]
        missing = [str(x) for x in paths if not x.is_file()]
        if missing:
            raise FileNotFoundError(f"Shards features manquants: {missing[:5]}")
        payloads = [pd.read_pickle(x) for x in paths]
        pairs = pd.concat([x["pairs"] for x in payloads], ignore_index=True)
        features = pd.concat([x["features"] for x in payloads], ignore_index=True)
        if len(pairs) != len(features) or len(features.columns) != 79 or pairs[["src_iri", "tgt_iri"]].duplicated().any():
            raise RuntimeError(f"Fusion MetaSpace {split} invalide")
        pair_name = "train_oof_predictions.csv" if split == "train" else "test_candidates_pre_gold.csv"
        pairs.to_csv(output / pair_name, index=False)
        features.to_csv(output / f"{'train' if split == 'train' else 'test'}_features.csv", index=False)
        counts[split] = len(pairs)
    metadata = {
        "pair": args.pair.upper().replace("_", "-"), "branch": args.branch,
        "representation": "label_synonyms", "features": FEATURES79,
        "feature_count": 79, "feature_breakdown": {"syntax": 22, "classical": 8, "spectral": 15, "tda": 34},
        "train_candidate_count": counts["train"], "test_candidate_count": counts["unseen"],
        "candidate_protocol": json.loads((work / "selected_lexical_protocol.json").read_text()),
        "semantic_protocol": json.loads((work / "selected_semantic_protocol.json").read_text()),
        "negative_regeneration": "all non-gold candidates rebuilt for this branch",
        "strict_checks": {"feature_count_79": True, "duplicate_pairs": False},
        "test_tsv_access": "never",
    }
    (output / "final_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"MetaSpace 79 fusionné {args.branch}: train={counts['train']} unseen={counts['unseen']}")


def _csv_ints(raw: str) -> list[int]:
    return sorted({int(x) for x in raw.split(",") if x.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[
        "prepare-entities", "encode-shard", "merge-embeddings", "semantic-shard",
        "merge-semantic", "feature-shard", "merge-features",
    ])
    parser.add_argument("--pair", required=True)
    parser.add_argument("--shared-dir")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--candidate-dirs", nargs="*", default=[])
    parser.add_argument("--work-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--branch", default="holdout")
    parser.add_argument("--side", choices=["src", "tgt"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-synonyms", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--k-semantic-grid", type=_csv_ints, default=_csv_ints("3,5,10,15,20"))
    parser.add_argument("--min-candidate-recall", type=float, default=0.90)
    args = parser.parse_args()
    if args.mode == "encode-shard" and not args.side:
        parser.error("--side requis")
    if args.mode not in ("prepare-entities", "encode-shard", "merge-embeddings") and not args.work_dir:
        parser.error("--work-dir requis")
    {
        "prepare-entities": prepare_entities,
        "encode-shard": encode_shard,
        "merge-embeddings": merge_embeddings,
        "semantic-shard": semantic_shard,
        "merge-semantic": merge_semantic,
        "feature-shard": feature_shard,
        "merge-features": merge_features,
    }[args.mode](args)
    print("GUARD: texte label+synonyms; exactement 79 features; aucun test.tsv caché")


if __name__ == "__main__":
    main()
