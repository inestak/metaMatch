# MetaMatch

MetaMatch is an ontology matching pipeline developed for the **Bio-ML 2026** benchmark. It performs ontology alignment for the following biomedical ontology pairs:

* **NCIT–DOID**
* **SNOMED–FMA**
* **SNOMED–NCIT**

The pipeline combines ontology-aware candidate generation, semantic embeddings, MetaSpaces, graph-based and lexical features, XGBoost-based matching models, consensus-based prediction, entity alignability filtering, and one-to-one matching.

The complete workflow can be executed locally from candidate generation to the final OAEI alignment package.

---

## Pipeline Overview

The MetaMatch pipeline follows the workflow below:

```text
Input ontologies
      │
      ▼
Candidate generation
      │
      ▼
Semantic embeddings
      │
      ▼
MetaSpaces
      │
      ▼
Graph06 / Union403 features
      │
      ▼
XGBoost experts
      │
      ▼
Entity alignability filtering
      │
      ▼
Expert consensus
      │
      ▼
Greedy one-to-one matching
      │
      ▼
OAEI alignment files
      │
      ▼
Final submission ZIP
```

The pipeline is designed to be **resumable**. Intermediate results are preserved by the workflow, allowing an interrupted execution to continue without recomputing completed stages.

---

## Requirements

Python **3.10** is recommended.

The pipeline has been tested with:

```text
Python       3.10.19
NumPy        1.26.4
pandas       2.3.3
scikit-learn 1.7.2
XGBoost      3.2.0
PyTorch      2.2.2
SciPy        1.15.3
RDFLib       7.6.0
Owlready2    0.50
```

Install the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

## Input Data

The pipeline requires the Bio-ML 2026 ontology data.

Set the input data directory:

```bash
export BIOML_DATA_DIR="/path/to/bioml_2026"
```

---

## Running MetaMatch

Once the environment and input data path are configured, launch the complete pipeline with:

```bash
bash run_exact_final_local.sh
```

This command runs the complete local workflow, including:

* candidate generation;
* semantic embeddings;
* MetaSpaces;
* feature generation;
* model training and prediction;
* entity alignability filtering;
* expert consensus;
* one-to-one matching;
* final OAEI packaging.

### Recommended local configuration

For a machine where several model tasks can run concurrently while keeping ontology-pair processing limited to one worker:

```bash
export BIOML_LOCAL_MAX_TASKS=6
export BIOML_LOCAL_PAIR_WORKERS=1

bash run_exact_final_local.sh
```

The two variables control local execution parallelism:

```text
BIOML_LOCAL_MAX_TASKS
    Maximum number of concurrent model tasks.

BIOML_LOCAL_PAIR_WORKERS
    Number of workers used for ontology-pair processing.
```

No additional command is required to launch the individual pipeline stages.

---

## Complete Execution

A complete execution from a clean environment can be started with:

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

export BIOML_DATA_DIR="/path/to/bioml_2026"

export BIOML_LOCAL_MAX_TASKS=6
export BIOML_LOCAL_PAIR_WORKERS=1

bash run_exact_final_local.sh
```

---

## Pipeline Stages

### 1. Candidate Generation

The pipeline first generates candidate correspondences between entities from the source and target ontologies.

The candidate-generation stage reduces the search space before the more expensive semantic and structural computations.

### 2. Semantic Embeddings

Ontology entities are represented using semantic embeddings.

These representations provide semantic similarity information used during candidate scoring and subsequent feature construction.

### 3. MetaSpaces

MetaSpaces are generated from the ontology entities and their associated information.

They provide the basis for the MetaMatch feature representation used by the downstream matching models.

### 4. Graph06 / Union403 Features

The pipeline constructs the feature space used by the matching models.

The canonical representation contains **403 features**, combining MetaMatch features with additional lexical, retrieval/provenance, and ontology structural information.

These features are generated automatically by the pipeline.

### 5. XGBoost Matching Models

The generated feature representation is used by the XGBoost matching models.

Hyperparameter optimization is performed for the corresponding experts.

Each HPO trial produces a compact JSON report.

Large OOF and prediction tables are generated once per expert rather than being repeatedly written for every HPO trial.

### 6. Entity Alignability Filtering

Source and target entities are filtered according to their alignability before final correspondence selection.

This removes candidates that do not satisfy the required entity-level constraints.

### 7. Expert Consensus

The predictions produced by the matching experts are combined using the consensus procedure implemented in the pipeline.

### 8. One-to-One Matching

The final candidate correspondences are processed using greedy one-to-one matching.

This produces the final set of correspondences satisfying the required one-to-one constraints.

### 9. Final Packaging

The resulting alignments are converted into the final OAEI submission package.

---

## Output

After a successful execution, the pipeline generates the final OAEI submission package in its configured output directory:

```text
outputs_bioml2026_union403/FULL_ENTITY_ALIGNABILITY_FILTER/FINAL_OAEI_2026/BIOML2026_FINAL_SUBMISSION.zip
```

The generated package contains the final alignment files for the supported Bio-ML 2026 ontology matching tracks.

---

## Resuming an Execution

MetaMatch is designed to be resumable.

Intermediate artifacts are preserved during execution.

If an execution is interrupted, running:

```bash
bash run_exact_final_local.sh
```

again allows the pipeline to reuse completed artifacts rather than recomputing the entire workflow from the beginning.

---

## Benchmark Tracks

The pipeline supports the following Bio-ML 2026 ontology matching tasks:

```text
NCIT–DOID
SNOMED–FMA
SNOMED–NCIT
```

The same orchestration script is used to execute the complete workflow across the supported ontology pairs.

---

## Repository Entry Point

The main entry point for the local pipeline is:

```text
run_exact_final_local.sh
```

The dependency specification is:

```text
requirements.txt
```

The recommended execution sequence is:

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

export BIOML_DATA_DIR="/path/to/bioml_2026"

export BIOML_LOCAL_MAX_TASKS=6
export BIOML_LOCAL_PAIR_WORKERS=1

bash run_exact_final_local.sh
```

The final submission package is generated at:

```text
outputs_bioml2026_union403/FULL_ENTITY_ALIGNABILITY_FILTER/FINAL_OAEI_2026/BIOML2026_FINAL_SUBMISSION.zip
```

---

## Reproducibility

To reproduce the MetaMatch experiments:

1. Use the specified Python environment.
2. Install the dependencies from `requirements.txt`.
3. Provide the Bio-ML 2026 input data.
4. Set `BIOML_DATA_DIR` to the location of `bioml_2026`.
5. Configure the local execution parallelism if required.
6. Run `run_exact_final_local.sh`.

The complete workflow, from candidate generation to final OAEI packaging, is executed by the orchestration script.
