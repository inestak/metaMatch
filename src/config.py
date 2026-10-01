#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration globale (chemins, paires d'ontologies, hyperparamètres).
"""

import os
from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).parent.parent
SPEC_DIR = ROOT_DIR / "spec"
BIOML_DIR = Path(os.environ.get("METAMATCH_BIOML_DIR", ROOT_DIR / "bioml")).resolve()
OUTPUTS_DIR = Path(os.environ.get("METAMATCH_OUTPUTS_DIR", ROOT_DIR / "outputs")).resolve()

# URLs
ZENODO_BIOML_2024 = "https://zenodo.org/records/13119437"

# Paires d'ontologies Bio-ML
ONTOLOGY_PAIRS = {
    "omim-ordo": {
        "source": "omim",
        "target": "ordo",
        "domain": "disease",
    },
    "ncit-doid": {
        "source": "ncit",
        "target": "doid",
        "domain": "disease",
    },
    "snomed-fma": {
        "source": "snomed",
        "target": "fma",
        "domain": "body",
    },
    "snomed-ncit-pharm": {
        "source": "snomed",
        "target": "ncit",
        "domain": "pharm",
    },
    "snomed-ncit-neoplas": {
        "source": "snomed",
        "target": "ncit",
        "domain": "neoplas",
    },
}

# Modèles d'embeddings (ordre de priorité)
EMBEDDING_MODELS = [
    "sentence-transformers/all-roberta-large-v1",  # RoBERTa (prioritaire)
    # "mistralai/Mistral-7B-v0.1",                 # Mistral (optionnel)
    # "KaLM",                                      # KaLM (optionnel)
]

# Hyperparamètres
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

CV_FOLDS = 10  # Cross-validation folds

# Features
FEATURE_CATEGORIES = [
    "syntactic",      # 22 features
    "classical",      # 7 distances
    "spectral",       # 5 x 4 matrices
    "topological",    # TDA features
    "nlp_metrics",    # BLEU, ROUGE
]
