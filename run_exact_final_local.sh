#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BIOML_DATA_DIR="${BIOML_DATA_DIR:-$PWD/bioml_2026}"
export BIOML_WORK_ROOT="${BIOML_WORK_ROOT:-$PWD/bioml2026_work}"
export BIOML_ARTIFACTS_DIR="${BIOML_ARTIFACTS_DIR:-$BIOML_WORK_ROOT/artifacts}"
export BIOML_SKIP_LEGACY_ENSEMBLE=1
export BIOML_XGB_N_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ -n "${BIOML_PYTHON:-}" ]; then
  exec "$BIOML_PYTHON" -u run_exact_final_local.py
elif command -v pipenv >/dev/null 2>&1 && pipenv run python -c 'import numpy,pandas,sklearn,xgboost,torch,rdflib,owlready2' >/dev/null 2>&1; then
  exec pipenv run python -u run_exact_final_local.py
else
  exec python3 -u run_exact_final_local.py
fi
