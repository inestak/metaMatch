#!/usr/bin/env python3
"""Sharded, leakage-free candidate generation for the canonical 79-feature run.

The ontology-dependent sparse matrices are prepared once.  Holdout and refit
splits then run independent bidirectional retrieval arrays and independently
rebuild their negatives.  Every textual channel uses the same canonical
``label + synonyms`` representation.  No hidden test reference is read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.data.ontology_loader import OntologyLoader
from src.scripts.enrich_saved_metaspace_sentence_sapbert import _concept_texts
from src.scripts.run_token_overlap_pipeline import infer_src_tgt_files


def _pair(value: str) -> str:
    return value.upper().replace("_", "-")


def _bounds(size: int, index: int, total: int) -> tuple[int, int]:
    return (size * index) // total, (size * (index + 1)) // total


def _read_gold(data_dir: Path, pair: str) -> pd.DataFrame:
    path = data_dir / pair / "refs_equiv" / "train.tsv"
    frame = pd.read_csv(path, sep="\t").drop_duplicates(["SrcEntity", "TgtEntity"])
    frame["SrcEntity"] = frame.SrcEntity.astype(str)
    frame["TgtEntity"] = frame.TgtEntity.astype(str)
    return frame


def _save_matrix(root: Path, name: str, matrix: sparse.spmatrix) -> None:
    sparse.save_npz(root / f"{name}.npz", matrix, compressed=False)
    sparse.save_npz(root / f"{name}_t.npz", matrix.T.tocsc(), compressed=False)


def prepare_shared(args: argparse.Namespace) -> None:
    started = time.time()
    root = Path(args.shared_dir)
    root.mkdir(parents=True, exist_ok=True)
    pair = _pair(args.pair)
    pair_dir = Path(args.ontology_data_dir) / pair
    src_file, tgt_file = infer_src_tgt_files(pair_dir, pair)
    src_onto = OntologyLoader(src_file).load()
    tgt_onto = OntologyLoader(tgt_file).load()
    src_iris = sorted(map(str, src_onto.classes))
    tgt_iris = sorted(map(str, tgt_onto.classes))
    src_texts, _, _ = _concept_texts(src_onto, src_iris, args.max_synonyms, 0, 0, 0)
    tgt_texts, _, _ = _concept_texts(tgt_onto, tgt_iris, args.max_synonyms, 0, 0, 0)
    src_text = dict(zip(src_iris, src_texts))
    tgt_text = dict(zip(tgt_iris, tgt_texts))
    pd.DataFrame({"iri": src_iris, "text": src_texts}).to_pickle(root / "src_texts.pkl")
    pd.DataFrame({"iri": tgt_iris, "text": tgt_texts}).to_pickle(root / "tgt_texts.pkl")

    corpus = src_texts + tgt_texts
    overlap = CountVectorizer(
        lowercase=True, binary=True, dtype=np.int32,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    overlap_matrix = overlap.fit_transform(corpus).tocsr()
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
        min_df=2, max_features=args.max_features, dtype=np.float32,
    )
    tfidf_matrix = char.fit_transform(corpus).tocsr()

    def neighbourhood(onto: OntologyLoader, iris: list[str], texts: dict[str, str]) -> list[str]:
        docs: list[str] = []
        for iri in iris:
            neighbours = sorted(
                {str(x) for x in onto.get_parents(iri)}
                | {str(x) for x in onto.get_children(iri)}
            )[: args.max_neighbors]
            docs.append(" | ".join(texts.get(x, "") for x in neighbours if texts.get(x, "")))
        return docs

    neighbourhood_corpus = neighbourhood(src_onto, src_iris, src_text) + neighbourhood(
        tgt_onto, tgt_iris, tgt_text
    )
    neigh = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), lowercase=True,
        min_df=2, max_features=args.max_features, sublinear_tf=True,
        dtype=np.float32,
    )
    if any(x.strip() for x in neighbourhood_corpus):
        neigh_matrix = neigh.fit_transform(neighbourhood_corpus).tocsr()
    else:
        neigh_matrix = sparse.csr_matrix((len(corpus), 1), dtype=np.float32)

    cut = len(src_iris)
    for name, matrix in (
        ("src_overlap", overlap_matrix[:cut]), ("tgt_overlap", overlap_matrix[cut:]),
        ("src_tfidf", tfidf_matrix[:cut]), ("tgt_tfidf", tfidf_matrix[cut:]),
        ("src_neigh", neigh_matrix[:cut]), ("tgt_neigh", neigh_matrix[cut:]),
    ):
        _save_matrix(root, name, matrix)
    (root / "src_iris.json").write_text(json.dumps(src_iris), encoding="utf-8")
    (root / "tgt_iris.json").write_text(json.dumps(tgt_iris), encoding="utf-8")
    (root / "shared_manifest.json").write_text(
        json.dumps({
            "pair": pair, "representation": "label_synonyms",
            "max_synonyms": args.max_synonyms, "source_entities": len(src_iris),
            "target_entities": len(tgt_iris), "elapsed_seconds": time.time() - started,
            "hidden_test_tsv_access": "never",
        }, indent=2), encoding="utf-8",
    )
    print(f"Matrices partagées label+synonyms: {len(src_iris)} x {len(tgt_iris)}")


def prepare_split(args: argparse.Namespace) -> None:
    shared, work = Path(args.shared_dir), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    pair = _pair(args.pair)
    train = _read_gold(Path(args.data_dir), pair)
    src_all = json.loads((shared / "src_iris.json").read_text())
    tgt_all = json.loads((shared / "tgt_iris.json").read_text())
    seen_src, seen_tgt = set(train.SrcEntity), set(train.TgtEntity)
    lists = {
        "train_src": sorted(seen_src), "train_tgt": sorted(seen_tgt),
        "unseen_src": sorted(set(src_all) - seen_src),
        "unseen_tgt": sorted(set(tgt_all) - seen_tgt),
    }
    for name, values in lists.items():
        (work / f"{name}.json").write_text(json.dumps(values), encoding="utf-8")
    train.to_csv(work / "train_gold.tsv", sep="\t", index=False)
    print(
        f"Split {args.branch}: gold={len(train)}, unseen="
        f"{len(lists['unseen_src'])}/{len(lists['unseen_tgt'])}"
    )


def _top_rows(
    query: sparse.csr_matrix,
    candidate_t: sparse.csc_matrix,
    query_iris: list[str], candidate_iris: list[str], k: int,
    batch_size: int,
) -> list[list[tuple[str, float, int]]]:
    output: list[list[tuple[str, float, int]]] = []
    take = min(k, candidate_t.shape[1])
    for start in range(0, query.shape[0], batch_size):
        block = (query[start:start + batch_size] @ candidate_t).tocsr()
        for local in range(block.shape[0]):
            lo, hi = block.indptr[local], block.indptr[local + 1]
            indices, values = block.indices[lo:hi], block.data[lo:hi]
            if len(values) > take:
                chosen = np.argpartition(values, -take)[-take:]
                indices, values = indices[chosen], values[chosen]
            order = np.lexsort((indices, -values))
            output.append([
                (candidate_iris[int(idx)], float(values[pos]), rank + 1)
                for rank, pos in enumerate(order) for idx in [indices[pos]]
            ])
    return output


def rank_shard(args: argparse.Namespace) -> None:
    shared, work = Path(args.shared_dir), Path(args.work_dir)
    src_iris = json.loads((shared / "src_iris.json").read_text())
    tgt_iris = json.loads((shared / "tgt_iris.json").read_text())
    src_pos = {iri: i for i, iri in enumerate(src_iris)}
    tgt_pos = {iri: i for i, iri in enumerate(tgt_iris)}
    side = args.side
    destination = work / "retrieval_shards"
    destination.mkdir(parents=True, exist_ok=True)

    if side == "src":
        qpos, opposite_pos = src_pos, tgt_pos
        source_prefix, target_prefix = "src", "tgt"
    else:
        qpos, opposite_pos = tgt_pos, src_pos
        source_prefix, target_prefix = "tgt", "src"
    channel_specs = (
        ("overlap", args.max_k_overlap),
        ("tfidf", args.max_k_tfidf),
        ("neigh", args.max_k_neighborhood),
    )
    # Load every large sparse matrix only once per worker, then reuse it for
    # both the training and unseen query slices.
    matrices = {
        channel: (
            sparse.load_npz(shared / f"{source_prefix}_{channel}.npz").tocsr(),
            sparse.load_npz(shared / f"{target_prefix}_{channel}_t.npz").tocsc(),
        )
        for channel, k in channel_specs if k > 0
    }
    for split in ("train", "unseen"):
        queries_all = json.loads((work / f"{split}_{side}.json").read_text())
        # The opposite candidate universe must follow the same entity split as
        # the queries.  Using all ontology entities here leaks public training
        # endpoints into the holdout/hidden pool and also corrupts the top-k
        # ranks after those endpoints are removed downstream.
        opposite_side = "tgt" if side == "src" else "src"
        candidates = json.loads(
            (work / f"{split}_{opposite_side}.json").read_text()
        )
        candidate_indices = [opposite_pos[iri] for iri in candidates]
        start, end = _bounds(len(queries_all), args.shard_index, args.num_shards)
        queries = queries_all[start:end]
        rows: dict[tuple[str, str], dict] = {}
        for channel, k in channel_specs:
            if k <= 0 or not queries:
                continue
            matrix, candidate_t_all = matrices[channel]
            candidate_t = candidate_t_all[:, candidate_indices]
            query_matrix = matrix[[qpos[x] for x in queries]]
            ranked = _top_rows(query_matrix, candidate_t, queries, candidates, k, args.batch_size)
            for query, values in zip(queries, ranked):
                for candidate, score, rank in values:
                    key = (query, candidate) if side == "src" else (candidate, query)
                    row = rows.setdefault(key, {
                        "src_iri": key[0], "tgt_iri": key[1],
                        "overlap_score": 0.0, "overlap_rank": np.inf,
                        "tfidf_rank": np.inf, "neigh_rank": np.inf,
                    })
                    if channel == "overlap":
                        row["overlap_score"] = max(row["overlap_score"], score)
                    row[f"{channel}_rank"] = min(row[f"{channel}_rank"], rank)
        frame = pd.DataFrame(rows.values(), columns=[
            "src_iri", "tgt_iri", "overlap_score", "overlap_rank",
            "tfidf_rank", "neigh_rank",
        ])
        path = destination / f"{split}_{side}_{args.shard_index:04d}_of_{args.num_shards:04d}.pkl"
        frame.to_pickle(path)
        print(f"{split} {side} shard {args.shard_index + 1}/{args.num_shards}: {len(frame)}")


def _candidate_mask(frame: pd.DataFrame, m: int, ko: int, kt: int, kn: int) -> pd.Series:
    return (
        ((frame.overlap_score >= m) & (frame.overlap_rank <= ko))
        | (frame.tfidf_rank <= kt)
        | ((kn > 0) & (frame.neigh_rank <= kn))
    )


def merge_candidates(args: argparse.Namespace) -> None:
    shared, work = Path(args.shared_dir), Path(args.work_dir)
    src_text = pd.read_pickle(shared / "src_texts.pkl").set_index("iri").text.astype(str)
    tgt_text = pd.read_pickle(shared / "tgt_texts.pkl").set_index("iri").text.astype(str)
    merged: dict[str, pd.DataFrame] = {}
    for split in ("train", "unseen"):
        paths = [
            work / "retrieval_shards" / f"{split}_{side}_{i:04d}_of_{args.num_shards:04d}.pkl"
            for side in ("src", "tgt") for i in range(args.num_shards)
        ]
        missing = [str(x) for x in paths if not x.is_file()]
        if missing:
            raise FileNotFoundError(f"Shards candidats manquants: {missing[:5]}")
        frame = pd.concat((pd.read_pickle(x) for x in paths), ignore_index=True)
        frame = frame.groupby(["src_iri", "tgt_iri"], as_index=False).agg({
            "overlap_score": "max", "overlap_rank": "min",
            "tfidf_rank": "min", "neigh_rank": "min",
        })
        merged[split] = frame

    gold_df = pd.read_csv(work / "train_gold.tsv", sep="\t")
    gold = set(zip(gold_df.SrcEntity.astype(str), gold_df.TgtEntity.astype(str)))
    rows = []
    for m, ko, kt, kn in product(
        args.min_common_grid, args.k_overlap_grid,
        args.k_tfidf_grid, args.k_neighborhood_grid,
    ):
        selected = merged["train"].loc[_candidate_mask(merged["train"], m, ko, kt, kn)]
        pairs = set(zip(selected.src_iri, selected.tgt_iri))
        covered = len(pairs & gold)
        rows.append({
            "min_common_tokens": m, "k_overlap": ko, "k_tfidf": kt,
            "k_neighborhood_tfidf": kn, "candidate_count": len(pairs),
            "gold_covered": covered, "gold_total": len(gold),
            "candidate_recall": covered / len(gold) if gold else 0.0,
        })
    grid = pd.DataFrame(rows)
    eligible = grid[grid.candidate_recall >= args.min_candidate_recall]
    if eligible.empty:
        best_recall = grid.candidate_recall.max()
        eligible = grid[grid.candidate_recall == best_recall]
        print(f"WARNING recall lexical cible non atteint; meilleur={best_recall:.4f}")
    selected = eligible.sort_values(
        ["candidate_count", "candidate_recall"], ascending=[True, False]
    ).iloc[0]
    key = tuple(int(selected[x]) for x in (
        "min_common_tokens", "k_overlap", "k_tfidf", "k_neighborhood_tfidf"
    ))
    grid.to_csv(work / "lexical_grid_train_only.csv", index=False)
    for split in ("train", "unseen"):
        frame = merged[split].loc[_candidate_mask(merged[split], *key)].copy()
        frame["retrieved"] = True
        if split == "train":
            present = set(zip(frame.src_iri, frame.tgt_iri))
            injected = pd.DataFrame(
                [{"src_iri": s, "tgt_iri": t, "overlap_score": 0.0,
                  "overlap_rank": np.inf, "tfidf_rank": np.inf, "neigh_rank": np.inf,
                  "retrieved": False}
                 for s, t in gold - present]
            )
            frame = pd.concat([frame, injected], ignore_index=True)
        frame = frame.drop_duplicates(["src_iri", "tgt_iri"])
        frame["label"] = [int((s, t) in gold) for s, t in zip(frame.src_iri, frame.tgt_iri)]
        frame["src_label"] = frame.src_iri.map(src_text).fillna("")
        frame["tgt_label"] = frame.tgt_iri.map(tgt_text).fillna("")
        frame.to_pickle(work / f"{split}_lexical_candidates.pkl")
        print(f"{args.branch} {split}: {len(frame)} candidats")
    (work / "selected_lexical_protocol.json").write_text(
        json.dumps({"selected": selected.to_dict(), "representation": "label_synonyms",
                    "gold_injected_train_only": True, "test_tsv_access": "never"}, indent=2),
        encoding="utf-8",
    )


def _csv_ints(raw: str) -> list[int]:
    return sorted({int(x) for x in raw.split(",") if x.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[
        "prepare-shared", "prepare-split", "rank-shard", "merge-candidates"
    ])
    parser.add_argument("--pair", required=True)
    parser.add_argument("--ontology-data-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--branch", default="holdout")
    parser.add_argument("--side", choices=["src", "tgt"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-synonyms", type=int, default=4)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=250000)
    parser.add_argument("--max-k-overlap", type=int, default=20)
    parser.add_argument("--max-k-tfidf", type=int, default=10)
    parser.add_argument("--max-k-neighborhood", type=int, default=3)
    parser.add_argument("--min-common-grid", type=_csv_ints, default=_csv_ints("1,2,3"))
    parser.add_argument("--k-overlap-grid", type=_csv_ints, default=_csv_ints("5,10,20"))
    parser.add_argument("--k-tfidf-grid", type=_csv_ints, default=_csv_ints("3,5,10"))
    parser.add_argument("--k-neighborhood-grid", type=_csv_ints, default=_csv_ints("0,3"))
    parser.add_argument("--min-candidate-recall", type=float, default=0.90)
    args = parser.parse_args()
    if args.mode != "prepare-shared" and not args.work_dir:
        parser.error("--work-dir requis")
    if args.mode == "rank-shard" and not args.side:
        parser.error("--side requis")
    {
        "prepare-shared": prepare_shared,
        "prepare-split": prepare_split,
        "rank-shard": rank_shard,
        "merge-candidates": merge_candidates,
    }[args.mode](args)
    print("GUARD: aucun test.tsv caché n'a été lu")


if __name__ == "__main__":
    main()
