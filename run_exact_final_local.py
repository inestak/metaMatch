#!/usr/bin/env python3
"""Complete memory-aware local reproduction of the historical final protocol."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PAIRS = ("NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT")
EXPERTS = (
    "Global403", "Lexical", "Syntaxique", "Semantic", "ROUGE",
    "Retrieval", "Structural", "Spectral", "TDA",
)
ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("BIOML_WORK_ROOT", ROOT / "bioml2026_work")).expanduser().resolve()
DATA = Path(os.environ.get("BIOML_DATA_DIR", ROOT / "bioml_2026")).expanduser().resolve()
ARTIFACTS = Path(os.environ.get("BIOML_ARTIFACTS_DIR", WORK / "artifacts")).expanduser().resolve()
OUTPUT_BASE = WORK / "outputs_bioml2026_union403"
EXPERT_ROOT = OUTPUT_BASE / "NINE_EXPERT_HPO_V3_CONSENSUS"
TRIAL_ROOT = OUTPUT_BASE / "NINE_EXPERT_HPO_PARALLEL_TRIALS"
ENTITY_ROOT = OUTPUT_BASE / "FULL_ENTITY_ALIGNABILITY_FILTER"
LOGS = ROOT / "logs_exact_local"
PYTHON = os.environ.get("BIOML_PYTHON", sys.executable)


def memory_gib() -> float:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        try:
            value = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(value.strip()) / 2**30
        except Exception:
            return 16.0


def cpu_slots() -> int:
    count = os.cpu_count() or 1
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        count = max(1, count // 2)
    return count


RAM_GIB = memory_gib()
CPU_SLOTS = cpu_slots()
AUTO_TASKS = max(1, min(128, CPU_SLOTS, int(max(1.0, RAM_GIB - 8.0) // 4.0)))
TASK_WORKERS = max(1, min(128, int(os.environ.get("BIOML_LOCAL_MAX_TASKS", AUTO_TASKS))))
AUTO_PIPELINES = max(1, min(3, int(max(1.0, RAM_GIB - 8.0) // 28.0)))
PIPELINE_WORKERS = max(1, min(3, int(os.environ.get("BIOML_LOCAL_PAIR_WORKERS", AUTO_PIPELINES))))


def environment() -> dict[str, str]:
    env = os.environ.copy()
    per_pair_workers = max(1, TASK_WORKERS // PIPELINE_WORKERS)
    env.update({
        "PYTHONPATH": str(ROOT), "BIOML_PROJECT_ROOT": str(ROOT),
        "BIOML_DATA_DIR": str(DATA), "BIOML_WORK_ROOT": str(WORK),
        "BIOML_ARTIFACTS_DIR": str(ARTIFACTS),
        "BIOML_NINE_EXPERT_ROOT": str(EXPERT_ROOT),
        "BIOML_CONSENSUS_ROOT": str(ENTITY_ROOT), "BIOML_XGB_N_JOBS": "1",
        # The exact nine-expert protocol below supersedes run_oaei.py's older
        # ten-expert fusion stage.  Avoid training that obsolete stage twice.
        "BIOML_SKIP_LEGACY_ENSEMBLE": "1",
        "BIOML_LOCAL_WORKERS": env.get("BIOML_LOCAL_WORKERS", str(per_pair_workers)),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
    })
    return env


def complete(*paths: Path) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def run_logged(name: str, command: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    print(f"START {name}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment(), stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-50:])
        raise RuntimeError(f"{name} failed with exit code {result.returncode}\n{tail}")
    print(f"COMPLETE {name}", flush=True)


def run_parallel(stage: str, tasks: list[tuple[str, list[str]]], workers: int) -> None:
    if not tasks:
        print(f"REUSE {stage}: all outputs already exist", flush=True)
        return
    print(f"{stage}: tasks={len(tasks)} concurrency={workers} CPU/task=1", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_logged, name, command) for name, command in tasks]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    if not DATA.is_dir():
        raise FileNotFoundError(f"Input directory not found: {DATA}")
    WORK.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"Local resources: CPU slots={CPU_SLOTS}, RAM={RAM_GIB:.1f} GiB, "
          f"general concurrency={TASK_WORKERS}, pair concurrency={PIPELINE_WORKERS}", flush=True)

    if os.environ.get("BIOML_SKIP_PIPELINE", "0").lower() not in {"1", "true", "yes"}:
        run_parallel("ONTOLOGY/METASPACE PIPELINES",
                     [(f"pipeline_{p}", ["bash", "run_local.sh", p]) for p in PAIRS],
                     PIPELINE_WORKERS)

    artifact_tasks = []
    for pair in PAIRS:
        archive = ARTIFACTS / f"{pair.replace('-', '_')}_Union403_for_Ines.zip"
        if complete(archive):
            continue
        artifact_tasks.append((
            f"artifact_{pair}",
            [PYTHON, "build_union403_artifact.py", "--pair", pair,
             "--work-root", str(WORK), "--artifact-dir", str(ARTIFACTS),
             "--pipeline-project", str(ROOT)],
        ))
    run_parallel("UNION403 ARTIFACTS", artifact_tasks, min(PIPELINE_WORKERS, 3))

    run_parallel("INDEPENDENT XGBOOST HPO", [
        (f"hpo_trial_{i:03d}", [PYTHON, "parallel_hpo_trial.py", "--task-index", str(i),
         "--output-root", str(TRIAL_ROOT)]) for i in range(810)
    ], TASK_WORKERS)

    tasks = []
    for pair in PAIRS:
        for expert in EXPERTS:
            out = EXPERT_ROOT / pair / "experts" / expert
            if complete(out / "FINAL_REPORT.json", out / "OOF_FROZEN_SCORES.csv",
                        out / "HIDDEN_BASE_PREDICTIONS.tsv", out / "HIDDEN_V3_PREDICTIONS.tsv"):
                continue
            tasks.append((f"expert_{pair}_{expert}", [PYTHON, "parallel_hpo_finalize.py",
                "--pair", pair, "--expert", expert, "--output-root", str(EXPERT_ROOT),
                "--trial-root", str(TRIAL_ROOT)]))
    run_parallel("EXPERT OOF/REFIT", tasks, TASK_WORKERS)

    run_parallel("NINE-EXPERT BASE CONSENSUS", [
        (f"base_consensus_{p}", [PYTHON, "nine_expert_hpo_v3_consensus.py",
         "--pair", p, "--output-root", str(EXPERT_ROOT)]) for p in PAIRS
    ], min(3, TASK_WORKERS))

    tasks = []
    for pair in PAIRS:
        for expert in EXPERTS:
            out = ENTITY_ROOT / pair / "experts" / expert
            if complete(out / "FINAL_REPORT.json", out / "HIDDEN_FILTERED_SAFE.tsv",
                        out / "HIDDEN_FILTERED_BALANCED.tsv", out / "HIDDEN_FILTERED_AGGRESSIVE.tsv"):
                continue
            tasks.append((f"entity_{pair}_{expert}", [PYTHON, "full_entity_alignability_worker.py",
                "--pair", pair, "--expert", expert, "--output-root", str(ENTITY_ROOT)]))
    run_parallel("ENTITY ALIGNABILITY", tasks, TASK_WORKERS)

    run_parallel("FILTERED CONSENSUS", [
        (f"filtered_consensus_{p}", [PYTHON, "full_entity_alignability_consensus.py",
         "--pair", p, "--input-root", str(ENTITY_ROOT)]) for p in PAIRS
    ], min(3, TASK_WORKERS))
    run_logged("final_submission", [PYTHON, "build_final_oaei_submission.py"])
    archive = ENTITY_ROOT / "FINAL_OAEI_2026" / "BIOML2026_FINAL_SUBMISSION.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    print(f"FINAL ARCHIVE: {archive}", flush=True)


if __name__ == "__main__":
    main()
