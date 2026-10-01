#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -ne 1 ]; then echo "Usage: bash run_local.sh PAIR"; exit 2; fi
PAIR="$1"
# This release trains its exact nine-expert protocol in
# run_exact_final_local.py after the canonical Union403 artifact is ready.
export BIOML_SKIP_LEGACY_ENSEMBLE="${BIOML_SKIP_LEGACY_ENSEMBLE:-1}"

# BIOML_FAST=1 is an explicit deadline-oriented profile. The complete research
# protocol remains the default when BIOML_FAST is unset.
if [ "${BIOML_FAST:-0}" = "1" ]; then
  export BIOML_GRAPH_EPOCHS="${BIOML_GRAPH_EPOCHS:-30}"
  export BIOML_GRAPH_PATIENCE="${BIOML_GRAPH_PATIENCE:-8}"
  export BIOML_GRAPH_NEGATIVES="${BIOML_GRAPH_NEGATIVES:-30}"
  export BIOML_GRAPH_RETRIEVAL_TOPK="${BIOML_GRAPH_RETRIEVAL_TOPK:-10}"
  export BIOML_CLASSES_ONLY="${BIOML_CLASSES_ONLY:-1}"
  export BIOML_EMBED_BATCH_SIZE="${BIOML_EMBED_BATCH_SIZE:-32}"
fi

# Prefer an explicitly selected Python. Otherwise use the current python3 when
# it has the scientific stack, then fall back to the parent Pipenv environment.
if [ -n "${BIOML_PYTHON:-}" ]; then
  PYTHON_CMD=("$BIOML_PYTHON")
elif python3 -c 'import numpy,pandas,sklearn,xgboost,torch,rdflib,owlready2' >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v pipenv >/dev/null 2>&1 && pipenv run python -c 'import numpy,pandas,sklearn,xgboost,torch,rdflib,owlready2' >/dev/null 2>&1; then
  PYTHON_CMD=(pipenv run python)
else
  echo "ERROR: incomplete Python environment." >&2
  echo "Install requirements.txt or set BIOML_PYTHON=/path/to/python." >&2
  exit 2
fi
if [ -z "${BIOML_LOCAL_WORKERS:-}" ]; then
  # On Intel Macs, os.cpu_count() includes Hyper-Threading.  CPU-bound Python
  # workers are faster and use much less RAM when based on physical cores.
  BIOML_LOCAL_WORKERS="$(python3 -c 'import os,platform; n=os.cpu_count() or 1; print(max(1, n//2) if platform.system()=="Darwin" and platform.machine()=="x86_64" else min(128,n))')"
fi
if [ -z "${BIOML_LOCAL_SHARDS:-}" ]; then
  if [ "${BIOML_FAST:-0}" = "1" ]; then
    BIOML_LOCAL_SHARDS=$((BIOML_LOCAL_WORKERS * 8))
  else
    BIOML_LOCAL_SHARDS=$((BIOML_LOCAL_WORKERS * 4))
  fi
  if [ "$BIOML_LOCAL_SHARDS" -gt 128 ]; then BIOML_LOCAL_SHARDS=128; fi
fi
if [ -z "${BIOML_EMBED_WORKERS:-}" ]; then
  BIOML_EMBED_WORKERS="$(python3 -c 'import os,subprocess,sys
try:
    ram=os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
except Exception:
    try:
        ram=int(subprocess.check_output(["sysctl","-n","hw.memsize"], stderr=subprocess.DEVNULL)) / 2**30
    except Exception:
        ram=16
print(max(1,min(3,int(sys.argv[1]),int((ram-10)//6))))' "$BIOML_LOCAL_WORKERS")"
fi
if [ -z "${BIOML_EMBED_SHARDS:-}" ]; then BIOML_EMBED_SHARDS=32; fi
export BIOML_GRAPH_EMBED_WORKERS="${BIOML_GRAPH_EMBED_WORKERS:-$BIOML_EMBED_WORKERS}"
if [ -z "${BIOML_HEAVY_WORKERS:-}" ]; then
  BIOML_HEAVY_WORKERS="$BIOML_LOCAL_WORKERS"
  if [ "$BIOML_HEAVY_WORKERS" -gt 2 ]; then BIOML_HEAVY_WORKERS=2; fi
fi
if [ -z "${BIOML_MODEL_WORKERS:-}" ]; then BIOML_MODEL_WORKERS="$BIOML_LOCAL_WORKERS"; fi
export BIOML_GRAPH_THREADS="${BIOML_GRAPH_THREADS:-$BIOML_MODEL_WORKERS}"
export BIOML_LOCAL_WORKERS BIOML_LOCAL_SHARDS BIOML_EMBED_WORKERS BIOML_EMBED_SHARDS BIOML_HEAVY_WORKERS BIOML_MODEL_WORKERS
export BIOML_XGB_N_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "PAIR=$PAIR | workers=$BIOML_LOCAL_WORKERS | shards=$BIOML_LOCAL_SHARDS | embedding=$BIOML_EMBED_WORKERS/$BIOML_EMBED_SHARDS | heavy=$BIOML_HEAVY_WORKERS | model=$BIOML_MODEL_WORKERS"
echo "Graph06 embedding workers=$BIOML_GRAPH_EMBED_WORKERS | progressive memory-mapped merge"
if [ "${BIOML_FAST:-0}" = "1" ]; then
  echo "FAST PROFILE: graph_epochs=$BIOML_GRAPH_EPOCHS | negatives=$BIOML_GRAPH_NEGATIVES | retrieval_topk=$BIOML_GRAPH_RETRIEVAL_TOPK | classes_only=$BIOML_CLASSES_ONLY"
fi
"${PYTHON_CMD[@]}" run_oaei.py "$PAIR"
