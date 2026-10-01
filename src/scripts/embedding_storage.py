"""Bounded-memory embedding storage; no model or ontology imports."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _close_memmap(value):
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def embedding_side_ready(embed, side, expected_iris=None):
    """Validate one published embedding side without loading dense arrays."""
    embed = Path(embed)
    paths = [embed / f"{side}_{key}.npy" for key in ("pooled", "clouds", "lengths")]
    iris_path = embed / f"{side}_iris.json"
    if not iris_path.is_file() or not all(path.is_file() for path in paths):
        return False
    iris = json.loads(iris_path.read_text())
    if expected_iris is not None and iris != list(expected_iris):
        return False
    arrays = []
    try:
        arrays = [np.load(path, mmap_mode="r") for path in paths]
        if any(len(value) != len(iris) for value in arrays):
            return False
        return arrays[0].ndim == 2 and arrays[1].ndim == 3 and arrays[2].ndim == 1 \
            and arrays[0].shape[1] == arrays[1].shape[2]
    finally:
        for value in arrays:
            _close_memmap(value)


def validate_and_prune_embeddings(embed):
    """Validate completed caches while keeping dense matrices memory-mapped."""
    embed = Path(embed)
    for side in ("src", "tgt"):
        if not embedding_side_ready(embed, side):
            raise ValueError(f"Invalid embedding cache: {embed}/{side}")


def reuse_embedding_side(destination, side, expected_iris, search_root):
    """Reuse an identical ontology representation from another pair cache."""
    destination = Path(destination)
    if embedding_side_ready(destination, side, expected_iris):
        return True
    search_root = Path(search_root)
    own = destination.resolve()
    for iris_path in sorted(search_root.glob("*/*/embeddings*/[st][rg][ct]_iris.json")):
        source_dir = iris_path.parent
        try:
            if source_dir.resolve() == own:
                continue
            source_side = iris_path.name.split("_", 1)[0]
            if not embedding_side_ready(source_dir, source_side, expected_iris):
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for suffix in ("iris.json", "pooled.npy", "clouds.npy", "lengths.npy"):
            source = source_dir / f"{source_side}_{suffix}"
            target = destination / f"{side}_{suffix}"
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                os.link(source, target)
            except OSError:
                target.symlink_to(source.resolve())
        print(
            f"[reuse] {side} embeddings from {source_dir.parent.parent.name}/"
            f"{source_side}; {len(expected_iris)} entities",
            flush=True,
        )
        return True
    return False


def atomic_pickle(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        pd.to_pickle(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def reuse_legacy_shards(embed, side, iris, total):
    """Split completed coarse shards when boundaries align; keep originals."""
    root = Path(embed) / "shards"
    for path in sorted(root.glob(f"{side}_*_of_*.pkl")):
        _, index, _, old_total = path.stem.split("_")
        index, old_total = int(index), int(old_total)
        if old_total >= total or total % old_total:
            continue
        first, last = index * total // old_total, (index + 1) * total // old_total
        targets = [root / f"{side}_{i:04d}_of_{total:04d}.pkl" for i in range(first, last)]
        if all(p.exists() for p in targets):
            continue
        payload = pd.read_pickle(path)
        start, end = len(iris) * index // old_total, len(iris) * (index + 1) // old_total
        if payload["iris"] != iris[start:end]:
            raise ValueError(f"Legacy shard does not match entities: {path}")
        for i, target in zip(range(first, last), targets):
            if target.exists():
                continue
            a, b = len(iris) * i // total - start, len(iris) * (i + 1) // total - start
            atomic_pickle({k: payload[k][a:b] for k in ("iris", "clouds", "pooled")}, target)
        del payload
        print(f"[reuse] {path.name} réparti en {last - first} petits shards", flush=True)


def merge_embedding_shards(embed, total, max_tokens):
    """Read only one shard at a time; publish readiness after both sides."""
    embed = Path(embed)
    marker = embed / "merged_embeddings.ready.json"
    marker.unlink(missing_ok=True)
    for side in ("src", "tgt"):
        iris = pd.read_pickle(embed / f"{side}_entities.pkl").iri.astype(str).tolist()
        if embedding_side_ready(embed, side, iris):
            print(f"[reuse] MERGE EMBEDDINGS {side}: already complete", flush=True)
            continue
        maps = {}
        shapes = {"pooled": ((len(iris), 768), np.float32),
                  "clouds": ((len(iris), max_tokens, 768), np.float16),
                  "lengths": ((len(iris),), np.int16)}
        try:
            for name, (shape, dtype) in shapes.items():
                maps[name] = np.lib.format.open_memmap(
                    embed / f"{side}_{name}.partial.npy", mode="w+", dtype=dtype, shape=shape)
            for index in range(total):
                path = embed / "shards" / f"{side}_{index:04d}_of_{total:04d}.pkl"
                payload = pd.read_pickle(path)
                a, b = len(iris) * index // total, len(iris) * (index + 1) // total
                if payload["iris"] != iris[a:b] or len(payload["clouds"]) != b - a:
                    raise ValueError(f"Invalid shard/order: {path}")
                maps["pooled"][a:b] = payload["pooled"]
                for offset, cloud in enumerate(payload["clouds"]):
                    matrix = np.asarray(cloud)
                    take = min(max_tokens, len(matrix))
                    maps["clouds"][a + offset] = 0
                    maps["clouds"][a + offset, :take] = matrix[:take]
                    maps["lengths"][a + offset] = take
                del payload
                for mm in maps.values():
                    mm.flush()
                print(f"MERGE EMBEDDINGS {side}: {index + 1}/{total}", flush=True)
        finally:
            for mm in maps.values():
                mm._mmap.close()
        for name in shapes:
            os.replace(embed / f"{side}_{name}.partial.npy", embed / f"{side}_{name}.npy")
        tmp = embed / f"{side}_iris.json.tmp"
        tmp.write_text(json.dumps(iris), encoding="utf-8")
        os.replace(tmp, embed / f"{side}_iris.json")
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps({"num_shards": total, "max_tokens": max_tokens}))
    os.replace(tmp, marker)
