#!/usr/bin/env python3
from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Parser des fichiers TSV Bio-ML (train.tsv, test.tsv, test.cands.tsv).
"""

import ast
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


def load_mappings(tsv_path: str | Path) -> pd.DataFrame:
    """
    Charge un fichier de mappings (train.tsv, test.tsv, full.tsv).

    Args:
        tsv_path: Chemin vers le fichier TSV

    Returns:
        DataFrame avec colonnes: SrcEntity, TgtEntity, Score
    """
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def load_candidates(cands_path: str | Path) -> pd.DataFrame:
    """
    Charge un fichier de candidats (test.cands.tsv).

    Args:
        cands_path: Chemin vers le fichier de candidats

    Returns:
        DataFrame avec colonnes: SrcEntity, TgtEntity, TgtCandidates (liste)
    """
    df = pd.read_csv(cands_path, sep="\t")

    # Parser la colonne TgtCandidates (tuple Python en string)
    def parse_candidates(s):
        if pd.isna(s):
            return []
        try:
            # Convertir le tuple string en liste
            parsed = ast.literal_eval(s)
            if isinstance(parsed, tuple):
                return list(parsed)
            return parsed
        except (ValueError, SyntaxError):
            return []

    df["TgtCandidates"] = df["TgtCandidates"].apply(parse_candidates)
    df["n_candidates"] = df["TgtCandidates"].apply(len)

    return df


def load_bioml_task(
    task_dir: str | Path,
    task_type: str = "equiv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge tous les fichiers d'une tâche Bio-ML.

    Args:
        task_dir: Répertoire de la tâche (ex: data/bioml/omim-ordo)
        task_type: "equiv" pour équivalence, "subs" pour subsumption

    Returns:
        Tuple (train_df, test_df, candidates_df)
    """
    task_dir = Path(task_dir)
    refs_dir = task_dir / f"refs_{task_type}"

    train_df = load_mappings(refs_dir / "train.tsv")
    test_df = load_mappings(refs_dir / "test.tsv")
    cands_df = load_candidates(refs_dir / "test.cands.tsv")

    return train_df, test_df, cands_df


def get_all_entities(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[set, set]:
    """
    Extrait tous les IRIs source et cible des mappings.

    Returns:
        Tuple (source_iris, target_iris)
    """
    source_iris = set(train_df["SrcEntity"]) | set(test_df["SrcEntity"])
    target_iris = set(train_df["TgtEntity"]) | set(test_df["TgtEntity"])
    return source_iris, target_iris


# -----------------------------------------------------------------------------
# CLI pour tester
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tsv_parser.py <task_dir>")
        print("Example: python tsv_parser.py data/bioml/omim-ordo")
        sys.exit(1)

    task_dir = sys.argv[1]
    train_df, test_df, cands_df = load_bioml_task(task_dir)

    print(f"=== {task_dir} ===\n")

    print("Train set:")
    print(f"  {len(train_df)} mappings")
    print(train_df.head(3).to_string())

    print(f"\nTest set:")
    print(f"  {len(test_df)} mappings")
    print(test_df.head(3).to_string())

    print(f"\nCandidates:")
    print(f"  {len(cands_df)} queries")
    print(f"  Moyenne candidats par query: {cands_df['n_candidates'].mean():.1f}")
    print(f"  Min/Max candidats: {cands_df['n_candidates'].min()} / {cands_df['n_candidates'].max()}")

    src_iris, tgt_iris = get_all_entities(train_df, test_df)
    print(f"\nEntités uniques:")
    print(f"  Sources: {len(src_iris)}")
    print(f"  Targets: {len(tgt_iris)}")
