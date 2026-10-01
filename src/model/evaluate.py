#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métriques d'évaluation OAEI.

Global matching: Precision, Recall, F1
Local ranking: MRR, Hits@1, Hits@5, Hits@10
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


# =============================================================================
# Global Matching Metrics
# =============================================================================

def compute_global_matching_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calcule les métriques de matching global.

    Args:
        y_true: Labels vrais (0/1)
        y_pred: Prédictions (0/1)

    Returns:
        Dict avec precision, recall, f1
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_alignment_metrics(
    predicted_mappings: Set[Tuple[str, str]],
    reference_mappings: Set[Tuple[str, str]],
) -> Dict[str, float]:
    """
    Calcule les métriques à partir d'ensembles de mappings.

    Args:
        predicted_mappings: Set de tuples (src_iri, tgt_iri) prédits
        reference_mappings: Set de tuples (src_iri, tgt_iri) de référence

    Returns:
        Dict avec precision, recall, f1
    """
    tp = len(predicted_mappings & reference_mappings)
    fp = len(predicted_mappings - reference_mappings)
    fn = len(reference_mappings - predicted_mappings)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_predicted": len(predicted_mappings),
        "n_reference": len(reference_mappings),
    }


# =============================================================================
# Local Ranking Metrics
# =============================================================================

def compute_mrr(
    rankings: List[int],
) -> float:
    """
    Calcule le Mean Reciprocal Rank.

    Args:
        rankings: Liste des rangs du correct (1-indexé), 0 si pas trouvé

    Returns:
        MRR score
    """
    reciprocal_ranks = []
    for rank in rankings:
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_hits_at_k(
    rankings: List[int],
    k: int,
) -> float:
    """
    Calcule Hits@K.

    Args:
        rankings: Liste des rangs du correct (1-indexé), 0 si pas trouvé
        k: Seuil K

    Returns:
        Hits@K score (proportion de rangs <= k)
    """
    hits = sum(1 for rank in rankings if 0 < rank <= k)
    return hits / len(rankings) if rankings else 0.0


def compute_ranking_metrics(
    rankings: List[int],
) -> Dict[str, float]:
    """
    Calcule toutes les métriques de ranking.

    Args:
        rankings: Liste des rangs du correct (1-indexé)

    Returns:
        Dict avec MRR, Hits@1, Hits@5, Hits@10
    """
    return {
        "mrr": compute_mrr(rankings),
        "hits_at_1": compute_hits_at_k(rankings, 1),
        "hits_at_5": compute_hits_at_k(rankings, 5),
        "hits_at_10": compute_hits_at_k(rankings, 10),
        "n_queries": len(rankings),
        "n_found": sum(1 for r in rankings if r > 0),
    }


def compute_rankings_from_predictions(
    pairs_df: pd.DataFrame,
    scores: np.ndarray,
    src_col: str = "src_iri",
    tgt_col: str = "tgt_iri",
    label_col: str = "label",
) -> List[int]:
    """
    Calcule les rangs à partir des scores de prédiction.

    Pour chaque query (src), classe les candidats (tgt) par score
    décroissant et trouve le rang du candidat correct.

    Args:
        pairs_df: DataFrame avec colonnes src_iri, tgt_iri, label
        scores: Scores de match (plus élevé = plus probable)
        src_col: Nom de la colonne source
        tgt_col: Nom de la colonne cible
        label_col: Nom de la colonne label

    Returns:
        Liste des rangs pour chaque query ayant un positif
    """
    df = pairs_df.copy()
    df["_score"] = scores

    rankings = []

    # Grouper par source
    for src, group in df.groupby(src_col):
        # Trier par score décroissant
        sorted_group = group.sort_values("_score", ascending=False).reset_index(drop=True)

        # Trouver le rang du positif
        positive_indices = sorted_group[sorted_group[label_col] == 1].index.tolist()

        if positive_indices:
            # Rang 1-indexé
            rank = positive_indices[0] + 1
            rankings.append(rank)

    return rankings


# =============================================================================
# Évaluation complète OAEI
# =============================================================================

class OAEIEvaluator:
    """Évaluateur complet pour le track Bio-ML OAEI."""

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def evaluate_global_matching(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name: str = "default",
    ) -> Dict[str, float]:
        """
        Évalue le matching global.

        Args:
            y_true: Labels vrais
            y_pred: Prédictions
            name: Nom pour stocker les résultats

        Returns:
            Dict de métriques
        """
        metrics = compute_global_matching_metrics(y_true, y_pred)
        self.results[f"{name}_global"] = metrics
        return metrics

    def evaluate_local_ranking(
        self,
        pairs_df: pd.DataFrame,
        scores: np.ndarray,
        name: str = "default",
    ) -> Dict[str, float]:
        """
        Évalue le ranking local.

        Args:
            pairs_df: DataFrame avec colonnes src_iri, tgt_iri, label
            scores: Scores de match
            name: Nom pour stocker les résultats

        Returns:
            Dict de métriques
        """
        rankings = compute_rankings_from_predictions(pairs_df, scores)
        metrics = compute_ranking_metrics(rankings)
        self.results[f"{name}_ranking"] = metrics
        return metrics

    def evaluate_full(
        self,
        pairs_df: pd.DataFrame,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        name: str = "default",
    ) -> Dict[str, Dict]:
        """
        Évaluation complète (global + ranking).

        Args:
            pairs_df: DataFrame avec labels
            y_proba: Probabilités prédites
            threshold: Seuil pour le matching global
            name: Nom de l'expérience

        Returns:
            Dict avec métriques global et ranking
        """
        y_true = pairs_df["label"].values
        y_pred = (y_proba >= threshold).astype(int)

        global_metrics = self.evaluate_global_matching(y_true, y_pred, name)
        ranking_metrics = self.evaluate_local_ranking(pairs_df, y_proba, name)

        return {
            "global": global_metrics,
            "ranking": ranking_metrics,
        }

    def summary(self) -> pd.DataFrame:
        """Retourne un résumé des résultats."""
        rows = []
        for name, metrics in self.results.items():
            row = {"experiment": name}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def print_report(self, name: str = "default") -> None:
        """Affiche un rapport formaté."""
        global_key = f"{name}_global"
        ranking_key = f"{name}_ranking"

        print(f"\n{'='*50}")
        print(f"OAEI Evaluation Report: {name}")
        print(f"{'='*50}")

        if global_key in self.results:
            g = self.results[global_key]
            print(f"\n--- Global Matching ---")
            print(f"Precision: {g['precision']:.4f}")
            print(f"Recall:    {g['recall']:.4f}")
            print(f"F1:        {g['f1']:.4f}")
            print(f"TP={g['tp']}, FP={g['fp']}, FN={g['fn']}")

        if ranking_key in self.results:
            r = self.results[ranking_key]
            print(f"\n--- Local Ranking ---")
            print(f"MRR:      {r['mrr']:.4f}")
            print(f"Hits@1:   {r['hits_at_1']:.4f}")
            print(f"Hits@5:   {r['hits_at_5']:.4f}")
            print(f"Hits@10:  {r['hits_at_10']:.4f}")
            print(f"Queries: {r['n_queries']}, Found: {r['n_found']}")


# =============================================================================
# Utilitaires
# =============================================================================

def extract_mappings_from_predictions(
    pairs_df: pd.DataFrame,
    y_pred: np.ndarray,
    src_col: str = "src_iri",
    tgt_col: str = "tgt_iri",
) -> Set[Tuple[str, str]]:
    """
    Extrait l'ensemble des mappings prédits.

    Args:
        pairs_df: DataFrame des paires
        y_pred: Prédictions binaires
        src_col: Colonne source
        tgt_col: Colonne cible

    Returns:
        Set de tuples (src_iri, tgt_iri)
    """
    positive_mask = y_pred == 1
    mappings = set()

    for idx in pairs_df[positive_mask].index:
        src = pairs_df.loc[idx, src_col]
        tgt = pairs_df.loc[idx, tgt_col]
        mappings.add((src, tgt))

    return mappings


def select_best_candidates(
    pairs_df: pd.DataFrame,
    scores: np.ndarray,
    src_col: str = "src_iri",
    tgt_col: str = "tgt_iri",
    threshold: Optional[float] = None,
) -> Set[Tuple[str, str]]:
    """
    Sélectionne le meilleur candidat pour chaque source.

    Stratégie: pour chaque source, garde le candidat avec le score
    le plus élevé (au-dessus du seuil si spécifié).

    Args:
        pairs_df: DataFrame des paires
        scores: Scores de match
        src_col: Colonne source
        tgt_col: Colonne cible
        threshold: Seuil minimum (None = pas de seuil)

    Returns:
        Set de tuples (src_iri, tgt_iri)
    """
    df = pairs_df.copy()
    df["_score"] = scores

    mappings = set()

    for src, group in df.groupby(src_col):
        best_idx = group["_score"].idxmax()
        best_score = group.loc[best_idx, "_score"]

        if threshold is None or best_score >= threshold:
            tgt = group.loc[best_idx, tgt_col]
            mappings.add((src, tgt))

    return mappings


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test Global Matching
    print("=== Test Global Matching ===")
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0])

    metrics = compute_global_matching_metrics(y_true, y_pred)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

    # Test Local Ranking
    print("\n=== Test Local Ranking ===")

    # Simuler des données de ranking
    # Pour 3 queries, avec 10 candidats chacun
    data = []
    for q in range(3):
        for c in range(10):
            label = 1 if c == 0 else 0  # Premier candidat est le correct
            data.append({
                "src_iri": f"src_{q}",
                "tgt_iri": f"tgt_{q}_{c}",
                "label": label,
            })

    pairs_df = pd.DataFrame(data)

    # Scores: le correct a un score élevé mais pas toujours le max
    scores = np.random.rand(len(pairs_df))
    # Booster le score du correct pour query 0 et 1
    scores[0] = 0.95  # Query 0: rank 1
    scores[10] = 0.80  # Query 1: pourrait être rank > 1
    scores[11] = 0.90  # Concurrent pour query 1

    rankings = compute_rankings_from_predictions(pairs_df, scores)
    print(f"Rankings: {rankings}")

    rank_metrics = compute_ranking_metrics(rankings)
    for k, v in rank_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Test OAEIEvaluator
    print("\n=== Test OAEIEvaluator ===")
    evaluator = OAEIEvaluator()

    # Données plus réalistes
    n_queries = 100
    candidates_per_query = 101

    data = []
    for q in range(n_queries):
        correct_pos = np.random.randint(0, candidates_per_query)
        for c in range(candidates_per_query):
            data.append({
                "src_iri": f"src_{q}",
                "tgt_iri": f"tgt_{q}_{c}",
                "label": 1 if c == correct_pos else 0,
            })

    pairs_df = pd.DataFrame(data)

    # Scores simulés (le correct a tendance à avoir un score plus élevé)
    scores = np.random.rand(len(pairs_df))
    positive_mask = pairs_df["label"] == 1
    scores[positive_mask] += np.random.rand(positive_mask.sum()) * 0.5

    results = evaluator.evaluate_full(pairs_df, scores, threshold=0.5, name="test")
    evaluator.print_report("test")
