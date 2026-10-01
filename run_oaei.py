#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK_ROOT = Path(os.environ.get("BIOML_WORK_ROOT", ROOT)).expanduser().resolve()
DATA = Path(os.environ.get("BIOML_DATA_DIR", ROOT / "bioml_2026")).expanduser().resolve()
PUBLIC70 = WORK_ROOT / "bioml_2026_public70"
OUT_H = WORK_ROOT / "outputs_bioml2026_parallel79_holdout"
OUT_R = WORK_ROOT / "outputs_bioml2026_parallel79_refit_public70"
OUT_U = WORK_ROOT / "outputs_bioml2026_union403"
CACHE = WORK_ROOT / "cache"
ART = WORK_ROOT / "artifacts"
EXPERT_ROOT = WORK_ROOT / "TEN_EXPERT_HPO_V3"
FUSION_ROOT = WORK_ROOT / "FINAL_META_XGB"
SUBMISSION_ROOT = ROOT / "submission"

PAIRS = ("NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT")
EXPERTS = [
    "Global403", "Lexical", "Syntaxique", "Semantic", "ROUGE",
    "Retrieval", "StructuralHierarchy", "StructuralNeighborhood",
    "Spectral", "TDA",
]
MAX_PARALLEL = max(1, min(128, int(os.environ.get("BIOML_LOCAL_WORKERS", str(os.cpu_count() or 1)))))
# On a local workstation, 128 shards spend a disproportionate amount of time
# starting Python and repeatedly loading the same ontology/model.  Four shards
# per worker keeps all CPUs busy while substantially reducing that overhead.
N_SHARDS = max(1, min(128, int(os.environ.get("BIOML_LOCAL_SHARDS", str(MAX_PARALLEL * 4)))))
EMBED_WORKERS = max(1, min(MAX_PARALLEL, int(os.environ.get("BIOML_EMBED_WORKERS", str(min(2, MAX_PARALLEL))))))
EMBED_SHARDS = max(1, min(256, int(os.environ.get("BIOML_EMBED_SHARDS", "32"))))
GRAPH_EMBED_WORKERS = max(1, min(MAX_PARALLEL, int(os.environ.get("BIOML_GRAPH_EMBED_WORKERS", str(EMBED_WORKERS)))))
HEAVY_WORKERS = max(1, min(MAX_PARALLEL, int(os.environ.get("BIOML_HEAVY_WORKERS", str(min(2, MAX_PARALLEL))))))
MODEL_WORKERS = max(1, min(MAX_PARALLEL, int(os.environ.get("BIOML_MODEL_WORKERS", str(MAX_PARALLEL)))))
GRAPH_THREADS = max(1, int(os.environ.get("BIOML_GRAPH_THREADS", str(MODEL_WORKERS))))
GRAPH_EPOCHS = max(1, int(os.environ.get("BIOML_GRAPH_EPOCHS", "200")))
GRAPH_PATIENCE = max(1, int(os.environ.get("BIOML_GRAPH_PATIENCE", "20")))
GRAPH_NEGATIVES = max(1, int(os.environ.get("BIOML_GRAPH_NEGATIVES", "100")))
GRAPH_TOPK = max(0, int(os.environ.get("BIOML_GRAPH_RETRIEVAL_TOPK", "20")))
CLASSES_ONLY = os.environ.get("BIOML_CLASSES_ONLY", "0").strip().lower() in {"1", "true", "yes"}
EMBED_BATCH_SIZE = max(1, int(os.environ.get("BIOML_EMBED_BATCH_SIZE", "16")))
FORCE_REBUILD = os.environ.get("BIOML_FORCE_REBUILD", "0").strip().lower() in {"1", "true", "yes"}
PREFLIGHT_ONLY = os.environ.get("BIOML_PREFLIGHT_ONLY", "0").strip().lower() in {"1", "true", "yes"}
SKIP_LEGACY_ENSEMBLE = os.environ.get("BIOML_SKIP_LEGACY_ENSEMBLE", "0").strip().lower() in {"1", "true", "yes"}


def normalize_pair(x: str) -> str:
    p = x.strip().upper().replace("_", "-")
    if p not in PAIRS:
        raise SystemExit(f"Paire invalide: {x}. Valeurs: {', '.join(PAIRS)}")
    return p


def run(cmd, env=None):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    e = os.environ.copy()
    e.update({
        "PYTHONPATH": str(ROOT),
        "BIOML_PROJECT_ROOT": str(ROOT),
        "BIOML_DATA_DIR": str(DATA),
        "BIOML_ARTIFACTS_DIR": str(ART),
        "HF_HOME": str(CACHE / "huggingface"),
        "TORCH_HOME": str(CACHE / "torch"),
        "XDG_CACHE_HOME": str(CACHE),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BIOML_XGB_N_JOBS": "1",
    })
    if env:
        e.update({str(k): str(v) for k, v in env.items()})
    subprocess.run([str(x) for x in cmd], cwd=ROOT, env=e, check=True)


def py_module(module, *args, env=None):
    run([sys.executable, "-m", module, *map(str, args)], env=env)


def _module_call(module, *args, env=None):
    return ([sys.executable, "-m", module, *map(str, args)], env or {})

def _script_call(script, *args, env=None):
    return ([sys.executable, str(script), *map(str, args)], env or {})

def _base_env(extra=None):
    e = os.environ.copy()
    e.update({
        "PYTHONPATH": str(ROOT),
        "BIOML_PROJECT_ROOT": str(ROOT),
        "BIOML_DATA_DIR": str(DATA),
        "BIOML_ARTIFACTS_DIR": str(ART),
        "HF_HOME": str(CACHE / "huggingface"),
        "TORCH_HOME": str(CACHE / "torch"),
        "XDG_CACHE_HOME": str(CACHE),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "BIOML_XGB_N_JOBS": "1",
    })
    if extra:
        e.update({str(k): str(v) for k, v in extra.items()})
    return e

_active_children = set()
_children_lock = threading.Lock()
_stop_children = threading.Event()


def _run_one(call):
    cmd, extra = call
    with _children_lock:
        if _stop_children.is_set():
            raise RuntimeError("Parallel batch cancelled")
        proc = subprocess.Popen([str(x) for x in cmd], cwd=ROOT, env=_base_env(extra))
        _active_children.add(proc)
    try:
        code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, cmd)
    finally:
        with _children_lock:
            _active_children.discard(proc)
    return " ".join(map(str, cmd))

def _ready(*paths: Path) -> bool:
    return not FORCE_REBUILD and bool(paths) and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths)


def _reuse(label: str, *paths: Path) -> bool:
    ok = _ready(*paths)
    if ok:
        print(f"[reuse] {label}", flush=True)
    return ok


def _thread_env(base: dict, workers: int) -> dict:
    result = dict(base)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result[name] = str(max(1, workers))
    return result


def _graph_env(base: dict) -> dict:
    result = dict(base)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result[name] = str(GRAPH_THREADS)
    return result


def parallel_calls(calls, max_workers=None, label="LOCAL PARALLEL"):
    if not calls:
        return
    workers = min(max_workers or MAX_PARALLEL, len(calls))
    print(f"\n>>> {label}: {len(calls)} tasks, max_workers={workers}", flush=True)
    _stop_children.clear()
    ex = ThreadPoolExecutor(max_workers=workers)
    futs = []
    try:
        futs=[ex.submit(_run_one,c) for c in calls]
        for k,f in enumerate(as_completed(futs),1):
            f.result()
            if k % 10 == 0 or k == len(futs):
                print(f"    completed {k}/{len(futs)}", flush=True)
    except BaseException:
        with _children_lock:
            _stop_children.set()
            children = list(_active_children)
            for proc in children:
                if proc.poll() is None:
                    proc.terminate()
        for future in futs:
            future.cancel()
        for proc in children:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def ensure_inputs(pair: str):
    d = DATA / pair
    refs = d / "refs_equiv"
    if not d.exists():
        raise FileNotFoundError(f"Dossier absent: {d}")
    for f in (refs / "train.tsv", refs / "valid.tsv"):
        if not f.exists():
            raise FileNotFoundError(f)
    owls = list(d.glob("*.owl")) + list(d.glob("*.rdf")) + list(d.glob("*.xml"))
    if len(owls) < 2:
        raise RuntimeError(f"Deux ontologies attendues dans {d}; trouvé: {owls}")


def mode_tag(entity_mode: str) -> str:
    return "classes" if entity_mode == "classes" else "classes_properties"


def phase_parallel79(pair: str, entity_mode: str):
    tag = mode_tag(entity_mode)
    run_id = f"parallel79_label_synonyms_eligible_v2_2026_{tag}"
    shared = CACHE / "parallel79_label_synonyms_2026" / pair / tag / "shared"
    old_embed = CACHE / "parallel79_label_synonyms_2026" / pair / tag / "embeddings"
    new_embed = CACHE / "parallel79_label_synonyms_2026" / pair / tag / "embeddings_eligible_v2"
    embed = old_embed if all((old_embed / x).exists() for x in [
        "src_iris.json", "tgt_iris.json", "src_pooled.npy", "tgt_pooled.npy",
        "src_clouds.npy", "tgt_clouds.npy", "src_lengths.npy", "tgt_lengths.npy"
    ]) else new_embed

    hwork = OUT_H / pair / f"{run_id}_holdout_work"
    rwork = OUT_R / pair / f"{run_id}_refit_work"
    hmeta_name = f"{run_id}_holdout_metaspace"
    rmeta_name = f"{run_id}_refit_metaspace"
    hmeta = OUT_H / pair / hmeta_name
    rmeta = OUT_R / pair / rmeta_name

    for p in [shared, embed, hwork, rwork, CACHE / "huggingface", CACHE / "torch"]:
        p.mkdir(parents=True, exist_ok=True)

    env = {"METAMATCH_ENTITY_MODE": entity_mode}

    if not (shared / "shared_manifest.json").exists():
        py_module("src.scripts.parallel79_candidates_2026",
                  "--mode", "prepare-shared", "--pair", pair,
                  "--ontology-data-dir", DATA, "--shared-dir", shared,
                  "--max-synonyms", 4, "--max-neighbors", 20, env=env)

    py_module("src.scripts.prepare_bioml2026_public70",
              "--source-root", DATA, "--output-root", PUBLIC70, "--pair", pair, env=env)

    required_embed = ["src_pooled.npy", "tgt_pooled.npy", "src_clouds.npy", "tgt_clouds.npy",
                      "src_lengths.npy", "tgt_lengths.npy", "src_iris.json", "tgt_iris.json",
                      "merged_embeddings.ready.json"]
    embeddings_ready = all((embed / x).exists() for x in required_embed)
    for branch, data_root, work, output_meta in [
        ("holdout", DATA, hwork, hmeta), ("refit", PUBLIC70, rwork, rmeta)
    ]:
        if embeddings_ready and _reuse(
            f"Parallel79 {pair}/{tag}/{branch}",
            output_meta / "final_results.json",
            output_meta / "train_oof_predictions.csv",
            output_meta / "test_candidates_pre_gold.csv",
            output_meta / "train_features.csv",
            output_meta / "test_features.csv",
        ):
            continue
        py_module("src.scripts.parallel79_candidates_2026",
                  "--mode", "prepare-split", "--pair", pair,
                  "--data-dir", data_root, "--shared-dir", shared,
                  "--work-dir", work, "--branch", branch, env=env)
        lexical_ready = _reuse(
            f"candidats lexicaux {pair}/{tag}/{branch}",
            work / "selected_lexical_protocol.json",
            work / "train_lexical_candidates.pkl",
            work / "unseen_lexical_candidates.pkl",
        )
        if not lexical_ready:
            rank_calls = []
            for side in ("src", "tgt"):
                for i in range(N_SHARDS):
                    shard_root = work / "retrieval_shards"
                    if _ready(
                        shard_root / f"train_{side}_{i:04d}_of_{N_SHARDS:04d}.pkl",
                        shard_root / f"unseen_{side}_{i:04d}_of_{N_SHARDS:04d}.pkl",
                    ):
                        continue
                    rank_calls.append(_module_call(
                        "src.scripts.parallel79_candidates_2026",
                        "--mode", "rank-shard", "--pair", pair,
                        "--shared-dir", shared, "--work-dir", work,
                        "--branch", branch, "--side", side,
                        "--num-shards", N_SHARDS, "--shard-index", i,
                        "--batch-size", 128, env=env,
                    ))
            parallel_calls(rank_calls, label="RETRIEVAL")
            py_module("src.scripts.parallel79_candidates_2026",
                      "--mode", "merge-candidates", "--pair", pair,
                      "--shared-dir", shared, "--work-dir", work,
                      "--branch", branch, "--num-shards", N_SHARDS,
                      "--min-candidate-recall", 0.90, env=env)

    if not all((embed / x).exists() for x in required_embed):
        py_module("src.scripts.parallel79_features_2026",
                  "--mode", "prepare-entities", "--pair", pair,
                  "--shared-dir", shared, "--embedding-dir", embed,
                  "--candidate-dirs", hwork, rwork,
                  "--max-synonyms", 4, "--max-tokens", 96, "--num-shards", EMBED_SHARDS, env=env)
        from src.scripts.embedding_storage import embedding_side_ready, reuse_embedding_side
        entity_iris = {
            side: pd.read_pickle(embed / f"{side}_entities.pkl").iri.astype(str).tolist()
            for side in ("src", "tgt")
        }
        embedding_search_root = CACHE / "parallel79_label_synonyms_2026"
        for side in ("src", "tgt"):
            reuse_embedding_side(embed, side, entity_iris[side], embedding_search_root)
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        batch = 64 if device == "cuda" else EMBED_BATCH_SIZE
        embed_calls = []
        for side in ("src", "tgt"):
            if embedding_side_ready(embed, side, entity_iris[side]):
                continue
            for i in range(EMBED_SHARDS):
                output = embed / "shards" / f"{side}_{i:04d}_of_{EMBED_SHARDS:04d}.pkl"
                if _ready(output):
                    continue
                embed_calls.append(_module_call(
                    "src.scripts.parallel79_features_2026",
                    "--mode", "encode-shard", "--pair", pair,
                    "--embedding-dir", embed, "--side", side,
                    "--device", device, "--batch-size", batch,
                    "--max-tokens", 96, "--num-shards", EMBED_SHARDS,
                    "--shard-index", i,
                    env=_thread_env(env, max(1, MAX_PARALLEL // EMBED_WORKERS)),
                ))
        parallel_calls(embed_calls, max_workers=EMBED_WORKERS, label="EMBEDDINGS")
        py_module("src.scripts.parallel79_features_2026",
                  "--mode", "merge-embeddings", "--pair", pair,
                  "--embedding-dir", embed, "--num-shards", EMBED_SHARDS,
                  "--max-tokens", 96, env=env)

    for branch, work, outroot, metaname in [
        ("holdout", hwork, OUT_H, hmeta_name),
        ("refit", rwork, OUT_R, rmeta_name),
    ]:
        output_meta = outroot / pair / metaname
        if _reuse(
            f"MetaSpace79 {pair}/{tag}/{branch}",
            output_meta / "final_results.json",
            output_meta / "train_oof_predictions.csv",
            output_meta / "test_candidates_pre_gold.csv",
            output_meta / "train_features.csv",
            output_meta / "test_features.csv",
        ):
            continue
        semantic_ready = _reuse(
            f"filtre sémantique {pair}/{tag}/{branch}",
            work / "selected_semantic_protocol.json",
            work / "train_final_candidates.pkl",
            work / "unseen_final_candidates.pkl",
        )
        if not semantic_ready:
            semantic_calls = []
            for i in range(N_SHARDS):
                shard_root = work / "semantic_shards"
                if _ready(
                    shard_root / f"train_{i:04d}_of_{N_SHARDS:04d}.pkl",
                    shard_root / f"unseen_{i:04d}_of_{N_SHARDS:04d}.pkl",
                ):
                    continue
                semantic_calls.append(_module_call(
                    "src.scripts.parallel79_features_2026",
                    "--mode", "semantic-shard", "--pair", pair,
                    "--embedding-dir", embed, "--work-dir", work,
                    "--branch", branch, "--num-shards", N_SHARDS,
                    "--shard-index", i, env=env,
                ))
            parallel_calls(semantic_calls, label="SEMANTIC")
            py_module("src.scripts.parallel79_features_2026",
                      "--mode", "merge-semantic", "--pair", pair,
                      "--embedding-dir", embed, "--work-dir", work,
                      "--branch", branch, "--num-shards", N_SHARDS,
                      "--min-candidate-recall", 0.90, env=env)
        feature_calls = []
        for i in range(N_SHARDS):
            shard_root = work / "feature_shards"
            if _ready(
                shard_root / f"train_{i:04d}_of_{N_SHARDS:04d}.pkl",
                shard_root / f"unseen_{i:04d}_of_{N_SHARDS:04d}.pkl",
            ):
                continue
            feature_calls.append(_module_call(
                "src.scripts.parallel79_features_2026",
                "--mode", "feature-shard", "--pair", pair,
                "--embedding-dir", embed, "--work-dir", work,
                "--branch", branch, "--num-shards", N_SHARDS,
                "--shard-index", i, env=env,
            ))
        parallel_calls(feature_calls, max_workers=HEAVY_WORKERS, label="FEATURES 79/TDA")
        py_module("src.scripts.parallel79_features_2026",
                  "--mode", "merge-features", "--pair", pair,
                  "--embedding-dir", embed, "--work-dir", work,
                  "--output-dir", outroot / pair / metaname,
                  "--branch", branch, "--num-shards", N_SHARDS, env=env)

    return {
        "run_id": run_id, "shared": shared, "embed": embed,
        "hmeta": hmeta, "rmeta": rmeta, "hwork": hwork, "rwork": rwork,
    }


def phase_union403(pair: str, entity_mode: str, p79: dict):
    tag = mode_tag(entity_mode)
    root = OUT_U / pair / tag
    graph_cache = CACHE / "union403_xgb_2026" / pair / tag / "graph_semantic"
    hgraph_name = f"union403_graph06_holdout_{tag}"
    rgraph_name = f"union403_graph06_refit_{tag}"
    hunion = root / "holdout" / "union"; runion = root / "refit" / "union"
    hwork = root / "holdout" / "feature_work"; rwork = root / "refit" / "feature_work"
    h79 = root / "holdout" / "meta79"; r79 = root / "refit" / "meta79"
    h90 = root / "holdout" / "meta90"; r90 = root / "refit" / "meta90"
    hdeep = root / "holdout" / "deep"; rdeep = root / "refit" / "deep"
    select = root / "public_valid_selection"; final = root / "final_refit"
    for p in [root, graph_cache, hunion, runion, hwork, rwork, h79, r79, h90, r90, hdeep, rdeep, select, final]:
        p.mkdir(parents=True, exist_ok=True)
    env_h = {"METAMATCH_ENTITY_MODE": entity_mode, "METAMATCH_BIOML_DIR": DATA, "METAMATCH_OUTPUTS_DIR": OUT_H}
    env_r = {"METAMATCH_ENTITY_MODE": entity_mode, "METAMATCH_BIOML_DIR": PUBLIC70, "METAMATCH_OUTPUTS_DIR": OUT_R}

    # In the standard all-pairs order, SNOMED and NCIT have already been
    # encoded by the two preceding pairs. Reuse their Graph06 semantic caches
    # after verifying that the exact ordered IRI universe is identical.
    if pair == "SNOMED-NCIT" and tag == "classes":
        graph_cache.mkdir(parents=True, exist_ok=True)
        reuse_specs = [
            ("src", "SNOMED-FMA", "src"),
            ("tgt", "NCIT-DOID", "src"),
        ]
        current_iris = {
            side: json.loads((p79["embed"] / f"{side}_iris.json").read_text())
            for side in ("src", "tgt")
        }
        graph_tag = "label_synonyms_syn999_tok128"
        for destination_side, source_pair, source_side in reuse_specs:
            destination = graph_cache / f"{destination_side}_biogitom_{graph_tag}.pkl"
            if destination.exists():
                continue
            source_embed = None
            for candidate_name in ("embeddings_eligible_v2", "embeddings"):
                candidate = CACHE / "parallel79_label_synonyms_2026" / source_pair / "classes" / candidate_name
                if (candidate / f"{source_side}_iris.json").is_file():
                    source_embed = candidate
                    break
            source_graph = (
                CACHE / "union403_xgb_2026" / source_pair / "classes" / "graph_semantic" /
                f"{source_side}_biogitom_{graph_tag}.pkl"
            )
            if source_embed is None or not source_graph.is_file():
                continue
            source_iris = json.loads((source_embed / f"{source_side}_iris.json").read_text())
            if source_iris != current_iris[destination_side]:
                continue
            try:
                os.link(source_graph, destination)
            except OSError:
                destination.symlink_to(source_graph.resolve())
            print(
                f"[reuse] Graph06 {destination_side} semantic cache from "
                f"{source_pair}/{source_side}",
                flush=True,
            )

    # Graph cache + train-only fits.
    graph_prepare = OUT_H / pair / "union403_graph_cache_ready" / "semantic_cache_ready.json"
    if not _reuse(f"cache Graph06 {pair}/{tag}", graph_prepare):
        py_module("src.scripts.run_biogitom_gated_graph_2025",
                  "--pair", pair, "--input-subdir", p79["hmeta"].name,
                  "--output-subdir", "union403_graph_cache_ready",
                  "--cache-subdir", graph_cache, "--device", "cpu", "--batch-size", 8,
                  "--checkpoint-size", 256, "--max-tokens", 128, "--max-synonyms", 999,
                  "--embedding-workers", GRAPH_EMBED_WORKERS,
                  "--embedding-threads", max(1, MAX_PARALLEL // GRAPH_EMBED_WORKERS),
                  "--prepare-only", env=_thread_env(env_h, max(1, MAX_PARALLEL // GRAPH_EMBED_WORKERS)))
    for branch, input_meta, out_name, env in [
        ("holdout", p79["hmeta"].name, hgraph_name, env_h),
        ("refit", p79["rmeta"].name, rgraph_name, env_r),
    ]:
        graph_out = (OUT_H if branch == "holdout" else OUT_R) / pair / out_name
        if _reuse(
            f"Graph06 {pair}/{tag}/{branch}",
            graph_out / "train_only_results.json",
            graph_out / "validation_predictions.csv",
            graph_out / "test_predictions_pre_gold.csv",
        ):
            continue
        py_module("src.scripts.run_biogitom_gated_graph_2025",
                  "--pair", pair, "--input-subdir", input_meta,
                  "--output-subdir", out_name, "--cache-subdir", graph_cache,
                  "--device", "cpu", "--batch-size", 8, "--checkpoint-size", 256,
                  "--max-tokens", 128, "--max-synonyms", 999,
                  "--negatives-per-positive", GRAPH_NEGATIVES, "--attention-dim", 128,
                  "--retrieval-topk", GRAPH_TOPK, "--epochs", GRAPH_EPOCHS,
                  "--early-stopping-patience", GRAPH_PATIENCE, "--seed", 42, "--train-only",
                  env=_graph_env(env))

    hgraph = OUT_H / pair / hgraph_name
    rgraph = OUT_R / pair / rgraph_name
    for branch, meta, graph, out in [
        ("holdout", p79["hmeta"], hgraph, hunion),
        ("refit", p79["rmeta"], rgraph, runion),
    ]:
        if _reuse(
            f"union {pair}/{tag}/{branch}",
            out / "union_manifest.json",
            out / "train_oof_predictions.csv",
            out / "test_candidates_pre_gold.csv",
        ):
            continue
        py_module("src.scripts.union403_2026", "build-union",
                  "--meta-dir", meta, "--graph-dir", graph, "--output-dir", out)

    py_module("src.scripts.union403_2026", "prepare-work",
              "--union-dir", hunion, "--shared-dir", p79["shared"],
              "--train-gold", DATA / pair / "refs_equiv" / "train.tsv", "--work-dir", hwork)
    py_module("src.scripts.union403_2026", "prepare-work",
              "--union-dir", runion, "--shared-dir", p79["shared"],
              "--train-gold", PUBLIC70 / pair / "refs_equiv" / "train.tsv", "--work-dir", rwork)

    for branch, work, out79 in [("holdout", hwork, h79), ("refit", rwork, r79)]:
        if _reuse(
            f"Union MetaSpace79 {pair}/{tag}/{branch}",
            out79 / "final_results.json",
            out79 / "train_oof_predictions.csv",
            out79 / "test_candidates_pre_gold.csv",
            out79 / "train_features.csv",
            out79 / "test_features.csv",
        ):
            continue
        feature_calls = []
        for i in range(N_SHARDS):
            shard_root = work / "feature_shards"
            if _ready(
                shard_root / f"train_{i:04d}_of_{N_SHARDS:04d}.pkl",
                shard_root / f"unseen_{i:04d}_of_{N_SHARDS:04d}.pkl",
            ):
                continue
            feature_calls.append(_module_call(
                "src.scripts.parallel79_features_2026",
                "--mode", "feature-shard", "--pair", pair,
                "--embedding-dir", p79["embed"], "--work-dir", work,
                "--branch", branch, "--num-shards", N_SHARDS, "--shard-index", i,
            ))
        parallel_calls(feature_calls, max_workers=HEAVY_WORKERS, label="UNION FEATURES 79/TDA")
        py_module("src.scripts.parallel79_features_2026",
                  "--mode", "merge-features", "--pair", pair,
                  "--embedding-dir", p79["embed"], "--work-dir", work,
                  "--output-dir", out79, "--branch", branch, "--num-shards", N_SHARDS)

    def deep_branch(data_root, union, gold, outdeep):
        shards = outdeep / "shards"
        calls = []
        for split, fname in [("train", "train_oof_predictions.csv"), ("unseen", "test_candidates_pre_gold.csv")]:
            merged_output = outdeep / f"{split}_deep.csv"
            if _reuse(f"deep fusionné {pair}/{tag}/{split}", merged_output):
                continue
            for i in range(N_SHARDS):
                sdir = shards / f"{split}_{i:04d}"
                sdir.mkdir(parents=True, exist_ok=True)
                if _ready(sdir / "candidate_deep_features.csv"):
                    continue
                calls.append(_script_call(ROOT / "deep_shard_worker.py",
                    "--pair", pair, "--pairs", union / fname,
                    "--num-shards", N_SHARDS, "--shard-index", i,
                    "--output-dir", sdir, "--train-gold", gold,
                    "--data-root", data_root))
        parallel_calls(calls, max_workers=HEAVY_WORKERS, label="DEEP/STRUCTURAL")
        for split in ("train", "unseen"):
            output = outdeep / f"{split}_deep.csv"
            if _ready(output):
                continue
            py_module("src.scripts.union403_2026", "merge-deep",
                      "--shard-dir", shards, "--split", split,
                      "--num-shards", N_SHARDS, "--output", output)

    deep_branch(DATA, hunion, DATA / pair / "refs_equiv" / "train.tsv", hdeep)
    deep_branch(PUBLIC70, runion, PUBLIC70 / pair / "refs_equiv" / "train.tsv", rdeep)

    for branch, m79, union, out90 in [
        ("holdout", h79, hunion, h90), ("refit", r79, runion, r90),
    ]:
        if _reuse(
            f"MetaSpace90 {pair}/{tag}/{branch}",
            out90 / "manifest.json", out90 / "train_features.csv",
            out90 / "test_features.csv", out90 / "train_oof_predictions.csv",
            out90 / "test_candidates_pre_gold.csv",
        ):
            continue
        py_module("src.scripts.union403_2026", "add-rouge",
                  "--meta79-dir", m79, "--union-dir", union,
                  "--shared-dir", p79["shared"], "--output-dir", out90)

    workers = MODEL_WORKERS
    if not _reuse(f"sélection publique {pair}/{tag}", select / "selection_results.json"):
        py_module("src.scripts.union403_2026", "train-valid",
                  "--train-pairs", hunion / "train_oof_predictions.csv",
                  "--unseen-pairs", hunion / "test_candidates_pre_gold.csv",
                  "--train-features", h90 / "train_features.csv",
                  "--unseen-features", h90 / "test_features.csv",
                  "--train-deep", hdeep / "train_deep.csv",
                  "--unseen-deep", hdeep / "unseen_deep.csv",
                  "--train-gold", DATA / pair / "refs_equiv" / "train.tsv",
                  "--valid-gold", DATA / pair / "refs_equiv" / "valid.tsv",
                  "--output-dir", select, "--workers", workers)
    if not _reuse(
        f"refit Union403 {pair}/{tag}",
        final / "final_results.json", final / "submission.tsv",
        final / "hidden_predictions_pre_gold.csv",
    ):
        py_module("src.scripts.union403_2026", "refit",
                  "--train-pairs", runion / "train_oof_predictions.csv",
                  "--unseen-pairs", runion / "test_candidates_pre_gold.csv",
                  "--train-features", r90 / "train_features.csv",
                  "--unseen-features", r90 / "test_features.csv",
                  "--train-deep", rdeep / "train_deep.csv",
                  "--unseen-deep", rdeep / "unseen_deep.csv",
                  "--train-gold", PUBLIC70 / pair / "refs_equiv" / "train.tsv",
                  "--selection-json", select / "selection_results.json",
                  "--output-dir", final, "--workers", workers)

    return dict(root=root, hunion=hunion, runion=runion, h90=h90, r90=r90,
                hdeep=hdeep, rdeep=rdeep, select=select, final=final)


def public_f1(core: dict) -> float:
    return float(json.loads((core["select"] / "selection_results.json").read_text())["selected"]["f1"])


def export_403_zip(pair: str, core: dict):
    from src.scripts.union403_2026 import _matrix

    hp, hx = _matrix(core["hunion"] / "train_oof_predictions.csv",
                     core["h90"] / "train_features.csv", core["hdeep"] / "train_deep.csv")
    hvp, hvx = _matrix(core["hunion"] / "test_candidates_pre_gold.csv",
                       core["h90"] / "test_features.csv", core["hdeep"] / "unseen_deep.csv")
    rp, rx = _matrix(core["runion"] / "test_candidates_pre_gold.csv",
                     core["r90"] / "test_features.csv", core["rdeep"] / "unseen_deep.csv")

    pair_columns = ["src_iri", "tgt_iri"]
    public = pd.concat([
        pd.concat([hp[pair_columns].reset_index(drop=True), hx.reset_index(drop=True)], axis=1),
        pd.concat([hvp[pair_columns].reset_index(drop=True), hvx.reset_index(drop=True)], axis=1),
    ], ignore_index=True).drop_duplicates(["src_iri", "tgt_iri"], keep="first")
    hidden = pd.concat([rp[pair_columns].reset_index(drop=True), rx.reset_index(drop=True)], axis=1)

    if len([c for c in public.columns if c not in {"src_iri", "tgt_iri"}]) != 403:
        raise RuntimeError("Le MetaSpace public n'a pas exactement 403 features")
    if len([c for c in hidden.columns if c not in {"src_iri", "tgt_iri"}]) != 403:
        raise RuntimeError("Le MetaSpace hidden n'a pas exactement 403 features")

    ART.mkdir(parents=True, exist_ok=True)
    slug = pair.replace("-", "_")
    zpath = ART / f"{slug}_Union403_for_Ines.zip"
    if _reuse(f"export Union403 {pair}", zpath):
        return zpath
    tmp = ART / f"_{slug}_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    pub_csv = tmp / "train_valid_metaspace_403.csv"
    hid_csv = tmp / "test_hidden_metaspace_403.csv"
    public.to_csv(pub_csv, index=False)
    hidden.to_csv(hid_csv, index=False)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pub_csv, arcname=pub_csv.name)
        z.write(hid_csv, arcname=hid_csv.name)
    shutil.rmtree(tmp)
    print(f"MetaSpace 403 exporté: {zpath}")
    return zpath


def train_10_experts_and_fuse(pair: str):
    EXPR = EXPERT_ROOT
    EXPR.mkdir(parents=True, exist_ok=True)
    calls = []
    for expert in EXPERTS:
        expert_dir = EXPR / pair / "experts" / expert
        if _reuse(
            f"expert {pair}/{expert}",
            expert_dir / "FINAL_REPORT.json",
            expert_dir / "OOF_FROZEN_SCORES.csv",
            expert_dir / "HIDDEN_BASE_PREDICTIONS.tsv",
            expert_dir / "HIDDEN_V3_PREDICTIONS.tsv",
        ):
            continue
        calls.append(_script_call(ROOT / "ten_expert_hpo_v3_worker.py",
                     "--pair", pair, "--expert", expert, "--output-root", EXPR))
    parallel_calls(calls, label="10 EXPERTS XGBOOST/V3")

    FUSION_ROOT.mkdir(parents=True, exist_ok=True)
    fusion_pair = FUSION_ROOT / pair
    if not _reuse(
        f"fusion finale {pair}",
        fusion_pair / "META_XGB_REPORT.json",
        fusion_pair / f"{pair}.tsv",
    ):
        run([sys.executable, ROOT / "metafusion_xgb_10meta_v3.py",
             "--existing-root", EXPR, "--output-root", FUSION_ROOT, "--pair", pair],
            env={"BIOML_XGB_N_JOBS": str(MODEL_WORKERS)})

    candidates = [
        FUSION_ROOT / pair / f"{pair}.tsv",
        FUSION_ROOT / pair / "submission.tsv",
    ]
    src = next((x for x in candidates if x.exists()), None)
    if src is None:
        raise FileNotFoundError(f"TSV final introuvable dans {FUSION_ROOT / pair}")
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    dst = SUBMISSION_ROOT / f"{pair}.tsv"
    submission = pd.read_csv(src, sep="\t", dtype=str)
    required = ["SrcEntity", "TgtEntity", "Relation"]
    if any(column not in submission.columns for column in required):
        raise RuntimeError(f"Colonnes de soumission invalides dans {src}: {list(submission.columns)}")
    submission[required].to_csv(dst, sep="\t", index=False)
    matches_name = f"matches_{pair.replace('-', '_')}.tsv"
    matches_dst = SUBMISSION_ROOT / matches_name
    submission[required].to_csv(matches_dst, sep="\t", index=False)
    print("\n" + "=" * 80)
    print(f"SOUMISSION FINALE: {dst}")
    print(f"MATCHES OAEI: {matches_dst}")
    print("=" * 80)
    return dst


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {Path(__file__).name} PAIR")
    pair = normalize_pair(sys.argv[1])
    ensure_inputs(pair)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"Configuration locale: workers={MAX_PARALLEL}, shards={N_SHARDS}, "
        f"embedding_workers={EMBED_WORKERS}, embedding_shards={EMBED_SHARDS}, "
        f"heavy_workers={HEAVY_WORKERS}, model_workers={MODEL_WORKERS}, "
        f"graph_threads={GRAPH_THREADS}, graph_epochs={GRAPH_EPOCHS}, "
        f"resume={'off' if FORCE_REBUILD else 'on'}, "
        f"work_root={WORK_ROOT}",
        flush=True,
    )
    free_gib = shutil.disk_usage(WORK_ROOT).free / (1024 ** 3)
    if free_gib < 40:
        print(
            f"WARNING: seulement {free_gib:.1f} Gio libres dans {WORK_ROOT}. "
            "Définir BIOML_WORK_ROOT vers un disque ayant davantage d'espace.",
            flush=True,
        )
    if PREFLIGHT_ONLY:
        print("PREFLIGHT OK: entrées valides; aucun calcul lancé.", flush=True)
        return
    ART.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    modes = ["classes"] if pair != "SNOMED-NCIT" or CLASSES_ONLY else ["classes", "classes_and_properties"]
    built = []
    for mode in modes:
        print(f"\n### {pair} — mode={mode} ###")
        p79 = phase_parallel79(pair, mode)
        core = phase_union403(pair, mode, p79)
        built.append((public_f1(core), mode, core))

    best_f1, best_mode, best_core = max(built, key=lambda x: x[0])
    print(f"\nMode retenu sur validation publique: {best_mode} (F1={best_f1:.6f})")
    export_403_zip(pair, best_core)
    if SKIP_LEGACY_ENSEMBLE:
        print("[pipeline] Legacy ensemble skipped; unified XGBoost ensemble runs next.", flush=True)
    else:
        train_10_experts_and_fuse(pair)


if __name__ == "__main__":
    main()
