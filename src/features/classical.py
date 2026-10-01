#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distances classiques entre embeddings (8 features).

Cosine, euclidean, pearson, etc.
Refactoré depuis compute_features.py de Nour.
"""

from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist


# =============================================================================
# Distances individuelles
# =============================================================================

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance euclidienne."""
    return float(np.linalg.norm(v1 - v2))


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance cosinus (1 - similarité)."""
    return float(cdist([v1], [v2], metric="cosine")[0][0])


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similarité cosinus."""
    return 1.0 - cosine_distance(v1, v2)


def pearson_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Corrélation de Pearson."""
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    corr, _ = pearsonr(v1, v2)
    return float(corr) if not np.isnan(corr) else 0.0


def spearman_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Corrélation de Spearman."""
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    corr, _ = spearmanr(v1, v2)
    return float(corr) if not np.isnan(corr) else 0.0


def minkowski_distance(v1: np.ndarray, v2: np.ndarray, p: int = 3) -> float:
    """Distance de Minkowski (p=3 par défaut)."""
    return float(cdist([v1], [v2], metric="minkowski", p=p)[0][0])


def canberra_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance de Canberra."""
    return float(cdist([v1], [v2], metric="canberra")[0][0])


def chebyshev_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance de Chebyshev (max des différences absolues)."""
    return float(cdist([v1], [v2], metric="chebyshev")[0][0])


# =============================================================================
# Liste des features
# =============================================================================

CLASSICAL_FEATURES = [
    "cls_euclidean",
    "cls_cosine_dist",
    "cls_cosine_sim",
    "cls_pearson",
    "cls_spearman",
    "cls_minkowski",
    "cls_canberra",
    "cls_chebyshev",
]


def compute_classical_features(v1: np.ndarray, v2: np.ndarray) -> Dict[str, float]:
    """
    Calcule toutes les distances/similarités classiques entre deux vecteurs.

    Args:
        v1: Premier vecteur (embedding)
        v2: Second vecteur (embedding)

    Returns:
        Dictionnaire {feature_name: value}
    """
    return {
        "cls_euclidean": euclidean_distance(v1, v2),
        "cls_cosine_dist": cosine_distance(v1, v2),
        "cls_cosine_sim": cosine_similarity(v1, v2),
        "cls_pearson": pearson_correlation(v1, v2),
        "cls_spearman": spearman_correlation(v1, v2),
        "cls_minkowski": minkowski_distance(v1, v2),
        "cls_canberra": canberra_distance(v1, v2),
        "cls_chebyshev": chebyshev_distance(v1, v2),
    }


def compute_classical_features_batch(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    show_progress: bool = True
) -> List[Dict[str, float]]:
    """
    Calcule les features classiques pour un batch de paires.

    Args:
        embeddings1: Array (n_pairs, embedding_dim)
        embeddings2: Array (n_pairs, embedding_dim)
        show_progress: Afficher la progression

    Returns:
        Liste de dictionnaires de features
    """
    from tqdm import tqdm

    n = len(embeddings1)
    iterator = tqdm(range(n), desc="Classical features") if show_progress else range(n)

    results = []
    for i in iterator:
        results.append(compute_classical_features(embeddings1[i], embeddings2[i]))

    return results


def compute_classical_features_vectorized(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    chunk_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """
    Version vectorisée (plus rapide) pour de grands batches.

    Args:
        embeddings1: Array (n_pairs, embedding_dim)
        embeddings2: Array (n_pairs, embedding_dim)

    Returns:
        Dictionnaire {feature_name: array de valeurs}
    """
    left = np.asarray(embeddings1, dtype=np.float32)
    right = np.asarray(embeddings2, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"Embedding matrices must have the same 2-D shape; "
            f"got {left.shape} and {right.shape}"
        )
    n = len(left)
    outputs = {
        name: np.zeros(n, dtype=np.float64)
        for name in CLASSICAL_FEATURES
    }

    # Chunking is deliberate.  A 50k x 768 temporary array can exceed one GB
    # once argsort creates int64 ranks.  It has also triggered native SciPy /
    # Accelerate crashes on Apple Silicon.  All pair rows are independent.
    for start in range(0, n, max(1, chunk_size)):
        stop = min(start + max(1, chunk_size), n)
        a = left[start:stop]
        b = right[start:stop]
        diff = a - b
        absolute = np.abs(diff)

        euclidean = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64))
        dot = np.sum(a * b, axis=1, dtype=np.float64)
        norm1 = np.sqrt(np.sum(a * a, axis=1, dtype=np.float64))
        norm2 = np.sqrt(np.sum(b * b, axis=1, dtype=np.float64))
        cosine_sim = np.divide(
            dot,
            norm1 * norm2,
            out=np.zeros_like(dot),
            where=(norm1 * norm2) > 1e-12,
        )

        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        pearson_num = np.sum(ac * bc, axis=1, dtype=np.float64)
        pearson_den = np.sqrt(
            np.sum(ac * ac, axis=1, dtype=np.float64)
            * np.sum(bc * bc, axis=1, dtype=np.float64)
        )
        pearson = np.divide(
            pearson_num,
            pearson_den,
            out=np.zeros_like(pearson_num),
            where=pearson_den > 1e-12,
        )

        # Transformer coordinates are continuous, so exact ties are extremely
        # rare.  Double argsort yields the standard rank correlation without
        # thousands of calls into scipy.stats (the source of the macOS crash).
        rank_a = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float32)
        rank_b = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float32)
        rank_a -= rank_a.mean(axis=1, keepdims=True)
        rank_b -= rank_b.mean(axis=1, keepdims=True)
        spearman_num = np.sum(rank_a * rank_b, axis=1, dtype=np.float64)
        spearman_den = np.sqrt(
            np.sum(rank_a * rank_a, axis=1, dtype=np.float64)
            * np.sum(rank_b * rank_b, axis=1, dtype=np.float64)
        )
        spearman = np.divide(
            spearman_num,
            spearman_den,
            out=np.zeros_like(spearman_num),
            where=spearman_den > 1e-12,
        )
        nonconstant = (np.std(a, axis=1) > 0) & (np.std(b, axis=1) > 0)
        spearman = np.where(nonconstant, spearman, 0.0)

        minkowski = np.power(
            np.sum(absolute * absolute * absolute, axis=1, dtype=np.float64),
            1.0 / 3.0,
        )
        canberra = np.sum(
            np.divide(
                absolute,
                np.abs(a) + np.abs(b),
                out=np.zeros_like(absolute),
                where=(np.abs(a) + np.abs(b)) > 0,
            ),
            axis=1,
            dtype=np.float64,
        )

        outputs["cls_euclidean"][start:stop] = euclidean
        outputs["cls_cosine_dist"][start:stop] = 1.0 - cosine_sim
        outputs["cls_cosine_sim"][start:stop] = cosine_sim
        outputs["cls_pearson"][start:stop] = pearson
        outputs["cls_spearman"][start:stop] = spearman
        outputs["cls_minkowski"][start:stop] = minkowski
        outputs["cls_canberra"][start:stop] = canberra
        outputs["cls_chebyshev"][start:stop] = np.max(absolute, axis=1)

    return outputs


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    # Test avec des vecteurs aléatoires
    np.random.seed(42)

    v1 = np.random.randn(384)
    v2 = np.random.randn(384)
    v3 = v1 + np.random.randn(384) * 0.1  # Proche de v1

    print("v1 vs v2 (aléatoires):")
    for k, v in compute_classical_features(v1, v2).items():
        print(f"  {k}: {v:.4f}")

    print("\nv1 vs v3 (proches):")
    for k, v in compute_classical_features(v1, v3).items():
        print(f"  {k}: {v:.4f}")

    print("\nv1 vs v1 (identiques):")
    for k, v in compute_classical_features(v1, v1).items():
        print(f"  {k}: {v:.4f}")
