"""Spawn-based Graph06 encoding with independent resumable checkpoints."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.embedding_storage import atomic_pickle

_encoder = None


def _init_encoder(model, device, threads):
    global _encoder
    import torch
    from src.scripts.run_sapbert_bidir_baseline import SapBertVectorEncoder
    torch.set_num_threads(threads)
    _encoder = SapBertVectorEncoder(model_name=model, device=device)


def _encode_part(job):
    path, iris, texts, batch_size, max_tokens = job
    values = np.asarray(_encoder.encode_texts(
        texts, batch_size=batch_size, max_length=max_tokens), dtype=np.float32)
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    if values.shape[0] != len(iris) or not np.isfinite(values).all():
        raise ValueError("Invalid Graph06 embeddings")
    atomic_pickle(pd.DataFrame(values, index=iris), path)
    return path


def encode_missing(cached, missing, texts, cache_path, model, device,
                   batch_size, max_tokens, checkpoint_size, workers, threads, side):
    if checkpoint_size < 1 or workers < 1 or threads < 1:
        raise ValueError("workers, threads and checkpoint size must be positive")
    root = Path(str(cache_path) + ".parts")
    root.mkdir(parents=True, exist_ok=True)
    paths, jobs = [], []
    for a in range(0, len(missing), checkpoint_size):
        iris, part_texts = missing[a:a + checkpoint_size], texts[a:a + checkpoint_size]
        fingerprint = hashlib.sha256(json.dumps(
            [model, max_tokens, iris, part_texts], ensure_ascii=False).encode()).hexdigest()
        path = root / f"{fingerprint}.pkl"
        paths.append(path)
        if path.exists():
            part = pd.read_pickle(path)
            if list(part.index) != iris or not np.isfinite(part.to_numpy()).all():
                raise ValueError(f"Invalid Graph06 checkpoint: {path}")
        else:
            jobs.append((path, iris, part_texts, batch_size, max_tokens))
    completed = len(paths) - len(jobs)
    print(f"GRAPH EMBEDDINGS {side}: {completed}/{len(paths)} checkpoints réutilisés; "
          f"workers={workers}, threads/worker={threads}", flush=True)
    if jobs:
        # Pool's context manager terminates only its own workers on interruption.
        with mp.get_context("spawn").Pool(
                min(workers, len(jobs)), initializer=_init_encoder,
                initargs=(model, device, threads)) as pool:
            for path in pool.imap_unordered(_encode_part, jobs, chunksize=1):
                completed += 1
                print(f"GRAPH EMBEDDINGS {side}: {completed}/{len(paths)} checkpoints terminés",
                      flush=True)
    # Only sentence vectors (not token clouds) are assembled here.
    result = pd.concat([cached] + [pd.read_pickle(p) for p in paths])
    result = result.loc[~result.index.duplicated(keep="last")]
    atomic_pickle(result, cache_path)
    return result
