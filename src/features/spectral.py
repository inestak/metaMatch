#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features spectrales via SVD (15 features).

alpha_req, ne_sum, rank_me, stable_rank, self_cluster
pour src, tgt et combined.
Refactoré depuis compute_features.py de Nour.
"""

from typing import Dict

import numpy as np
from scipy.linalg import svd


def alpha_req(matrix: np.ndarray) -> float:
    """
    Alpha de la loi de puissance des valeurs singulières.

    Mesure la décroissance des valeurs singulières.
    """
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 0.0

    try:
        _, S, _ = svd(matrix, full_matrices=False)
        S = S[S > 1e-10]  # Filtrer les valeurs quasi-nulles

        if len(S) < 2:
            return 0.0

        log_indices = np.log(np.arange(1, len(S) + 1))
        log_singular_values = np.log(S)
        slope, _ = np.polyfit(log_indices, log_singular_values, 1)
        return -slope
    except Exception:
        return 0.0


def ne_sum(matrix: np.ndarray) -> float:
    """
    NESum: Somme des valeurs propres normalisées de la matrice de covariance.

    Mesure la dimensionnalité effective.
    """
    if matrix.size == 0 or matrix.shape[0] < 2:
        return 0.0

    try:
        C = np.cov(matrix, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(C)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Tri décroissant

        lambda_0 = eigenvalues[0] if eigenvalues[0] > 1e-10 else 1e-10
        return float(np.sum(eigenvalues / lambda_0))
    except Exception:
        return 0.0


def rank_me(matrix: np.ndarray) -> float:
    """
    RankMe: Rang effectif basé sur l'entropie des valeurs singulières.

    Plus le rang est élevé, plus l'information est distribuée.
    """
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 0.0

    try:
        _, S, _ = svd(matrix, full_matrices=False)
        S = S[S > 1e-10]

        if len(S) == 0:
            return 0.0

        S_normalized = S / np.sum(S)
        entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-12))
        return float(np.exp(entropy))
    except Exception:
        return 0.0


def stable_rank(matrix: np.ndarray) -> float:
    """
    Rang stable: Ratio de la norme de Frobenius au carré sur la norme spectrale au carré.

    Mesure robuste du rang.
    """
    if matrix.size == 0:
        return 0.0

    try:
        frobenius_norm = np.linalg.norm(matrix, 'fro')
        spectral_norm = np.linalg.norm(matrix, 2)

        if spectral_norm < 1e-10:
            return 0.0

        return float((frobenius_norm ** 2) / (spectral_norm ** 2))
    except Exception:
        return 0.0


def self_cluster(matrix: np.ndarray) -> float:
    """
    Self-clustering coefficient.

    Mesure la tendance des points à former des clusters.
    """
    if matrix.size == 0 or matrix.shape[0] < 2:
        return 0.0

    try:
        n, d = matrix.shape
        if d < 2 or n < 2:
            return 0.0

        dot_product = matrix @ matrix.T
        frobenius_norm = np.linalg.norm(dot_product, 'fro')

        denominator = (d - 1) * (n - 1) * n
        if denominator == 0:
            return 0.0

        return float((d * frobenius_norm - n * (d + n - 1)) / denominator)
    except Exception:
        return 0.0


# =============================================================================
# Liste des features
# =============================================================================

SPECTRAL_FEATURES = [
    "spc_alpha_req",
    "spc_ne_sum",
    "spc_rank_me",
    "spc_stable_rank",
    "spc_self_cluster",
]


def compute_spectral_features(matrix: np.ndarray, prefix: str = "spc") -> Dict[str, float]:
    """
    Calcule toutes les features spectrales pour une matrice.

    Args:
        matrix: Matrice (n_samples, n_features)
        prefix: Préfixe pour les noms de features

    Returns:
        Dictionnaire {feature_name: value}
    """
    return {
        f"{prefix}_alpha_req": alpha_req(matrix),
        f"{prefix}_ne_sum": ne_sum(matrix),
        f"{prefix}_rank_me": rank_me(matrix),
        f"{prefix}_stable_rank": stable_rank(matrix),
        f"{prefix}_self_cluster": self_cluster(matrix),
    }


def compute_spectral_features_pair(
    token_matrix1: np.ndarray,
    token_matrix2: np.ndarray,
) -> Dict[str, float]:
    """
    Calcule les features spectrales pour une paire de matrices token-level.

    Returns:
        Dictionnaire de features
    """
    if (
        token_matrix1 is None
        or token_matrix2 is None
        or token_matrix1.size == 0
        or token_matrix2.size == 0
    ):
        base_names = [f.replace("spc_", "") for f in SPECTRAL_FEATURES]
        zeros = {}
        for prefix in ("spc_src", "spc_tgt", "spc_combined"):
            for b in base_names:
                zeros[f"{prefix}_{b}"] = 0.0
        return zeros

    cloud1 = token_matrix1
    cloud2 = token_matrix2

    features = {}

    # Features pour chaque cloud
    for k, v in compute_spectral_features(cloud1, "spc_src").items():
        features[k] = v
    for k, v in compute_spectral_features(cloud2, "spc_tgt").items():
        features[k] = v

    # Features pour le cloud combiné
    combined = np.vstack([cloud1, cloud2])
    for k, v in compute_spectral_features(combined, "spc_combined").items():
        features[k] = v

    return features


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Matrice aléatoire
    matrix = np.random.randn(100, 50)

    print("Features spectrales pour matrice aléatoire (100x50):")
    for k, v in compute_spectral_features(matrix).items():
        print(f"  {k}: {v:.4f}")

    # Matrice de rang faible
    low_rank = np.random.randn(100, 5) @ np.random.randn(5, 50)

    print("\nFeatures spectrales pour matrice rang faible:")
    for k, v in compute_spectral_features(low_rank).items():
        print(f"  {k}: {v:.4f}")

    # Test pair
    emb1 = np.random.randn(32, 384)
    emb2 = np.random.randn(28, 384)

    print("\nFeatures spectrales pour paire d'embeddings:")
    for k, v in compute_spectral_features_pair(emb1, emb2).items():
        print(f"  {k}: {v:.4f}")
