# BioML 2026 exact historical final protocol — local A to Z

This package runs the complete local workflow for NCIT-DOID, SNOMED-FMA, and
SNOMED-NCIT: candidates, embeddings, MetaSpaces, Graph06/Union403, nine
independently tuned XGBoost experts, source/target entity-alignability filters,
nine-expert consensus, greedy one-to-one matching, and final packaging.
The obsolete ten-expert fusion embedded in the base pipeline is explicitly
skipped, because the exact nine-expert protocol is trained immediately after
the canonical 403-feature artifacts are produced.

The frozen final rules are SAFE vote >=7/9 for NCIT-DOID, SAFE vote >=8/9 for
SNOMED-FMA, and SAFE vote >=7/9 for SNOMED-NCIT. The reference run produced
2,082, 4,575, and 10,003 predictions respectively.

Every subprocess uses one CPU. The 810 HPO trial slots are independent. Local
concurrency is automatically limited by CPU count and physical memory. Override
it carefully with `BIOML_LOCAL_MAX_TASKS`; the hard ceiling is 128. Concurrent
ontology-pair pipelines are controlled by `BIOML_LOCAL_PAIR_WORKERS` and never
exceed three.

## Installation

Python 3.10 is recommended. The orchestration and XGBoost smoke tests were run
with Python 3.10.19, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2,
XGBoost 3.2.0, PyTorch 2.2.2, SciPy 1.15.3, RDFLib 7.6.0, and Owlready2 0.50.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run

```bash
export BIOML_DATA_DIR=/absolute/path/to/bioml_2026
export BIOML_WORK_ROOT=/absolute/path/to/bioml2026_work
bash run_exact_final_local.sh
```

On the author's Mac, the existing input path is:

```bash
export BIOML_DATA_DIR="/Users/nahawandkired/Documents/onto_matching/metamatch_code_nour/BIOML2026_OAEI_LOCAL_PARALLEL/bioml_2026"
export BIOML_WORK_ROOT="/Users/nahawandkired/Documents/onto_matching/bioml2026_work"
bash run_exact_final_local.sh
```

Example for at most six one-CPU model tasks and one large ontology pipeline:

```bash
export BIOML_LOCAL_MAX_TASKS=6
export BIOML_LOCAL_PAIR_WORKERS=1
bash run_exact_final_local.sh
```

The workflow is resumable. Each HPO trial writes only one compact JSON report;
large OOF and hidden prediction tables are written once per expert.

Final output:

```text
$BIOML_WORK_ROOT/outputs_bioml2026_union403/FULL_ENTITY_ALIGNABILITY_FILTER/FINAL_OAEI_2026/BIOML2026_FINAL_SUBMISSION.zip
```

Hidden reference alignments are never read. The historical final vote levels
for NCIT-DOID and SNOMED-FMA were selected using a 2025 pseudo-test proxy; this
must be disclosed when documenting the protocol.
