#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features TDA (étendues) - OPTIONNEL (lent sur mon pc).

Persistent homology, bottleneck, wasserstein.
Désactivé par défaut (lent + dépendances complexes).
Refactoré depuis compute_features.py de Nour.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import hashlib
import os
import pickle
import tempfile

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Imports optionnels
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    from persim.persistent_entropy import persistent_entropy
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

try:
    from gudhi.bottleneck import bottleneck_distance
    HAS_GUDHI_BOTTLENECK = True
except ImportError:
    HAS_GUDHI_BOTTLENECK = False

try:
    from gudhi.wasserstein import wasserstein_distance
    HAS_GUDHI_WASSERSTEIN = True
except ImportError:
    HAS_GUDHI_WASSERSTEIN = False


def check_tda_available() -> bool:
    """Vérifie si les bibliothèques TDA sont disponibles."""
    # TDA core features need ripser + persim.
    # GUDHI-based distances are optional and guarded per metric.
    return HAS_RIPSER and HAS_PERSIM


def compute_persistence_diagram(
    point_cloud: np.ndarray,
    metric: str = "euclidean",
    maxdim: int = 2
) -> Optional[list]:
    """
    Calcule le diagramme de persistance d'un nuage de points.

    Args:
        point_cloud: Matrice (n_points, n_dims)
        metric: Métrique de distance
        maxdim: Dimension homologique maximale

    Returns:
        Liste des diagrammes par dimension, ou None si erreur
    """
    if not HAS_RIPSER:
        return None

    try:
        result = ripser(point_cloud, metric=metric, maxdim=maxdim)
        return result["dgms"]
    except Exception:
        return None


def _get_or_compute_diagram(
    point_cloud: np.ndarray,
    metric: str,
    maxdim: int,
    diagram_cache: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
):
    """
    Retourne un diagramme depuis le cache ou le calcule.
    """
    if diagram_cache is not None and cache_key is not None and cache_key in diagram_cache:
        return diagram_cache[cache_key]

    dgms = compute_persistence_diagram(point_cloud, metric=metric, maxdim=maxdim)

    if diagram_cache is not None and cache_key is not None and dgms is not None:
        diagram_cache[cache_key] = dgms

    return dgms


def _extract_homology_groups(dgms):
    """Extrait H0/H1/H2 depuis la sortie ripser."""
    h0 = dgms[0] if len(dgms) > 0 else np.array([])
    h1 = dgms[1] if len(dgms) > 1 else np.array([])
    h2 = dgms[2] if len(dgms) > 2 else np.array([])
    return h0, h1, h2


def _build_unary_tda_stats(dgms) -> Dict[str, float]:
    """Stats topologiques unaires (indépendantes des paires)."""
    h0, h1, h2 = _extract_homology_groups(dgms)
    return {
        "h0_count": h0_count(h0),
        "h1_count": h1_count(h1),
        "h2_count": h2_count(h2),
        "h0_max_life": max_lifetime(h0),
        "h1_max_life": max_lifetime(h1),
        "h2_max_life": max_lifetime(h2),
        "h0_entropy": persistence_entropy(h0),
        "h1_entropy": persistence_entropy(h1),
        "h2_entropy": persistence_entropy(h2),
    }


def _entity_cache_file(disk_cache_dir: str, entity_key: str, metric: str, maxdim: int) -> Path:
    key = f"{entity_key}|{metric}|{maxdim}|tda_entity_v1"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return Path(disk_cache_dir) / f"{digest}.pkl"


def _load_entity_payload(
    disk_cache_dir: Optional[str],
    entity_key: Optional[str],
    metric: str,
    maxdim: int,
) -> Optional[Dict[str, Any]]:
    if not disk_cache_dir or not entity_key:
        return None
    path = _entity_cache_file(disk_cache_dir, entity_key, metric, maxdim)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if (
            payload.get("version") != 1
            or payload.get("entity_key") != entity_key
            or payload.get("metric") != metric
            or payload.get("maxdim") != maxdim
        ):
            return None
        return payload
    except Exception:
        return None


def _save_entity_payload(
    disk_cache_dir: Optional[str],
    payload: Dict[str, Any],
):
    if not disk_cache_dir:
        return
    entity_key = payload.get("entity_key")
    metric = payload.get("metric")
    maxdim = payload.get("maxdim")
    if not entity_key or metric is None or maxdim is None:
        return
    path = _entity_cache_file(disk_cache_dir, entity_key, metric, maxdim)
    temp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        return
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _get_or_compute_entity_payload(
    point_cloud: np.ndarray,
    metric: str,
    maxdim: int,
    diagram_cache: Optional[Dict[str, Any]] = None,
    entity_cache: Optional[Dict[str, Any]] = None,
    entity_key: Optional[str] = None,
    disk_cache_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Cache multi-niveaux pour une entité:
    1) cache RAM payload
    2) cache disque payload
    3) calcul puis sauvegarde
    """
    if entity_key and entity_cache is not None and entity_key in entity_cache:
        return entity_cache[entity_key]

    payload = _load_entity_payload(
        disk_cache_dir=disk_cache_dir,
        entity_key=entity_key,
        metric=metric,
        maxdim=maxdim,
    )
    if payload is not None:
        if entity_cache is not None and entity_key is not None:
            entity_cache[entity_key] = payload
        if diagram_cache is not None and entity_key is not None and payload.get("dgms") is not None:
            diagram_cache[entity_key] = payload["dgms"]
        return payload

    dgms = _get_or_compute_diagram(
        point_cloud,
        metric=metric,
        maxdim=maxdim,
        diagram_cache=diagram_cache,
        cache_key=entity_key,
    )
    if dgms is None:
        return None

    payload = {
        "version": 1,
        "entity_key": entity_key,
        "metric": metric,
        "maxdim": maxdim,
        "features": _build_unary_tda_stats(dgms),
        "dgms": dgms,
    }

    if entity_cache is not None and entity_key is not None:
        entity_cache[entity_key] = payload
    _save_entity_payload(disk_cache_dir, payload)
    return payload


def h0_count(dgm: np.ndarray) -> int:
    """Nombre de composantes connexes (H0)."""
    return len(dgm) if dgm is not None else 0


def h1_count(dgm: np.ndarray) -> int:
    """Nombre de cycles (H1)."""
    return len(dgm) if dgm is not None else 0


def h2_count(dgm: np.ndarray) -> int:
    """Nombre de cavités (H2)."""
    return len(dgm) if dgm is not None else 0


def max_lifetime(dgm: np.ndarray, exclude_inf: bool = True) -> float:
    """Durée de vie maximale d'une feature."""
    if dgm is None or len(dgm) == 0:
        return 0.0

    lifetimes = []
    for birth, death in dgm:
        if exclude_inf and death == np.inf:
            continue
        lifetimes.append(death - birth)

    return max(lifetimes) if lifetimes else 0.0


def total_persistence(dgm: np.ndarray, exclude_inf: bool = True) -> float:
    """Somme des durées de vie."""
    if dgm is None or len(dgm) == 0:
        return 0.0

    total = 0.0
    for birth, death in dgm:
        if exclude_inf and death == np.inf:
            continue
        total += death - birth

    return total


def persistence_entropy(dgm: np.ndarray) -> float:
    """Entropie de la persistance."""
    if not HAS_PERSIM or dgm is None or len(dgm) == 0:
        return 0.0

    try:
        entropy = persistent_entropy(dgm)
        return float(entropy[0]) if len(entropy) > 0 else 0.0
    except Exception:
        return 0.0


def compute_bottleneck(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Distance bottleneck entre deux diagrammes."""
    if not HAS_GUDHI_BOTTLENECK or dgm1 is None or dgm2 is None:
        return 0.0

    try:
        # Filtrer les points à l'infini
        dgm1_clean = dgm1[dgm1[:, 1] != np.inf] if len(dgm1) > 0 else dgm1
        dgm2_clean = dgm2[dgm2[:, 1] != np.inf] if len(dgm2) > 0 else dgm2

        if len(dgm1_clean) == 0 or len(dgm2_clean) == 0:
            return 0.0

        return float(bottleneck_distance(dgm1_clean, dgm2_clean))
    except Exception:
        return 0.0


def compute_wasserstein(dgm1: np.ndarray, dgm2: np.ndarray, p: int = 2) -> float:
    """Distance de Wasserstein entre deux diagrammes."""
    if not HAS_GUDHI_WASSERSTEIN or dgm1 is None or dgm2 is None:
        return 0.0

    try:
        dgm1_clean = dgm1[dgm1[:, 1] != np.inf] if len(dgm1) > 0 else dgm1
        dgm2_clean = dgm2[dgm2[:, 1] != np.inf] if len(dgm2) > 0 else dgm2

        if len(dgm1_clean) == 0 or len(dgm2_clean) == 0:
            return 0.0

        return float(wasserstein_distance(dgm1_clean, dgm2_clean, order=p))
    except Exception:
        return 0.0


def overlap_percentage_3d(cloud_a: np.ndarray, cloud_b: np.ndarray) -> tuple[float, float, float]:
    """
    Estime le recouvrement entre deux nuages de points via voisinage local.

    Returns:
        (overlap_sym, overlap_a_to_b, overlap_b_to_a)
    """
    if cloud_a.size == 0 or cloud_b.size == 0:
        return 0.0, 0.0, 0.0
    if cloud_a.shape[0] < 2 or cloud_b.shape[0] < 2:
        return 0.0, 0.0, 0.0

    try:
        nn_a = NearestNeighbors(n_neighbors=2).fit(cloud_a)
        da, _ = nn_a.kneighbors(cloud_a)
        eps_a = da[:, 1]

        nn_b = NearestNeighbors(n_neighbors=2).fit(cloud_b)
        db, _ = nn_b.kneighbors(cloud_b)
        eps_b = db[:, 1]

        nn_b2 = NearestNeighbors(n_neighbors=1).fit(cloud_b)
        dist_a2b, _ = nn_b2.kneighbors(cloud_a)
        overlap_a = float(np.mean(dist_a2b[:, 0] <= eps_a))

        nn_a2 = NearestNeighbors(n_neighbors=1).fit(cloud_a)
        dist_b2a, _ = nn_a2.kneighbors(cloud_b)
        overlap_b = float(np.mean(dist_b2a[:, 0] <= eps_b))

        overlap_sym = 0.5 * (overlap_a + overlap_b)
        return overlap_sym, overlap_a, overlap_b
    except Exception:
        return 0.0, 0.0, 0.0


def overlap_percentage_cosine(
    cloud_a: np.ndarray, cloud_b: np.ndarray
) -> tuple[float, float, float]:
    """Adaptive local overlap using cosine distance (1 - cosine similarity).

    The internal radius of every point is its cosine distance to the nearest
    *other* point of the same cloud.  The directional coverage is the fraction
    of points whose nearest point in the opposite cloud lies inside that
    radius.  Returns ``(symmetric, a_to_b, b_to_a)``.
    """
    cloud_a = np.asarray(cloud_a, dtype=np.float32)
    cloud_b = np.asarray(cloud_b, dtype=np.float32)
    if cloud_a.ndim != 2 or cloud_b.ndim != 2 or len(cloud_a) < 2 or len(cloud_b) < 2:
        return 0.0, 0.0, 0.0

    try:
        def cosine_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            left_norm = np.linalg.norm(left, axis=1, keepdims=True)
            right_norm = np.linalg.norm(right, axis=1, keepdims=True)
            left_unit = np.divide(left, left_norm, out=np.zeros_like(left), where=left_norm > 0)
            right_unit = np.divide(right, right_norm, out=np.zeros_like(right), where=right_norm > 0)
            return np.clip(1.0 - left_unit @ right_unit.T, 0.0, 2.0)

        distances_aa = cosine_distances(cloud_a, cloud_a)
        distances_bb = cosine_distances(cloud_b, cloud_b)
        np.fill_diagonal(distances_aa, np.inf)
        np.fill_diagonal(distances_bb, np.inf)
        radius_a = distances_aa.min(axis=1)
        radius_b = distances_bb.min(axis=1)

        distances_ab = cosine_distances(cloud_a, cloud_b)
        overlap_a_to_b = float(np.mean(distances_ab.min(axis=1) <= radius_a))
        overlap_b_to_a = float(np.mean(distances_ab.min(axis=0) <= radius_b))
        overlap_symmetric = 0.5 * (overlap_a_to_b + overlap_b_to_a)
        return overlap_symmetric, overlap_a_to_b, overlap_b_to_a
    except Exception:
        return 0.0, 0.0, 0.0


# =============================================================================
# Liste des features
# =============================================================================

TOPOLOGICAL_FEATURES = [
    # Source
    "tda_h0_count_src",
    "tda_h1_count_src",
    "tda_h2_count_src",
    "tda_h0_max_life_src",
    "tda_h1_max_life_src",
    "tda_h2_max_life_src",
    "tda_h0_entropy_src",
    "tda_h1_entropy_src",
    "tda_h2_entropy_src",
    # Target
    "tda_h0_count_tgt",
    "tda_h1_count_tgt",
    "tda_h2_count_tgt",
    "tda_h0_max_life_tgt",
    "tda_h1_max_life_tgt",
    "tda_h2_max_life_tgt",
    "tda_h0_entropy_tgt",
    "tda_h1_entropy_tgt",
    "tda_h2_entropy_tgt",
    # Combined (stacked token-level clouds)
    "tda_h0_count_combined",
    "tda_h1_count_combined",
    "tda_h2_count_combined",
    "tda_h0_max_life_combined",
    "tda_h1_max_life_combined",
    "tda_h2_max_life_combined",
    "tda_h0_entropy_combined",
    "tda_h1_entropy_combined",
    "tda_h2_entropy_combined",
    # Distances inter-diagrammes
    "tda_h0_bottleneck",
    "tda_h0_wasserstein",
    "tda_h1_bottleneck",
    "tda_h1_wasserstein",
    "tda_overlap_sym",
    "tda_overlap_a2b",
    "tda_overlap_b2a",
]


def compute_topological_features(
    token_matrix1: np.ndarray,
    token_matrix2: np.ndarray,
    metric: str = "euclidean",
    diagram_cache: Optional[Dict[str, Any]] = None,
    entity_cache: Optional[Dict[str, Any]] = None,
    disk_cache_dir: Optional[str] = None,
    src_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calcule les features topologiques pour une paire de matrices token-level.

    Args:
        token_matrix1: Matrice token-level source (n_tokens, emb_dim)
        token_matrix2: Matrice token-level cible (n_tokens, emb_dim)
        metric: Métrique de distance
        diagram_cache: Cache mutable des diagrammes persistants
        entity_cache: Cache mutable des payloads TDA par entité
        disk_cache_dir: Répertoire de cache persistant sur disque
        src_key: Identifiant stable de l'entité source pour la mise en cache
        tgt_key: Identifiant stable de l'entité cible pour la mise en cache

    Returns:
        Dictionnaire de features (vide si TDA non disponible)
    """
    if not check_tda_available():
        return {k: 0.0 for k in TOPOLOGICAL_FEATURES}

    if (
        token_matrix1 is None
        or token_matrix2 is None
        or token_matrix1.size == 0
        or token_matrix2.size == 0
    ):
        return {k: 0.0 for k in TOPOLOGICAL_FEATURES}

    cloud1 = token_matrix1
    cloud2 = token_matrix2
    cloud_combined = np.vstack([cloud1, cloud2])
    overlap_sym, overlap_a2b, overlap_b2a = overlap_percentage_3d(cloud1, cloud2)

    # Calculer les diagrammes
    src_entity_key = f"src::{src_key}" if src_key else None
    tgt_entity_key = f"tgt::{tgt_key}" if tgt_key else None

    src_payload = _get_or_compute_entity_payload(
        cloud1,
        metric=metric,
        maxdim=2,
        diagram_cache=diagram_cache,
        entity_cache=entity_cache,
        entity_key=src_entity_key,
        disk_cache_dir=disk_cache_dir,
    )
    tgt_payload = _get_or_compute_entity_payload(
        cloud2,
        metric=metric,
        maxdim=2,
        diagram_cache=diagram_cache,
        entity_cache=entity_cache,
        entity_key=tgt_entity_key,
        disk_cache_dir=disk_cache_dir,
    )
    dgms_combined = compute_persistence_diagram(cloud_combined, metric, maxdim=2)

    if src_payload is None or tgt_payload is None or dgms_combined is None:
        return {k: 0.0 for k in TOPOLOGICAL_FEATURES}

    dgms1 = src_payload["dgms"]
    dgms2 = tgt_payload["dgms"]
    src_stats = src_payload["features"]
    tgt_stats = tgt_payload["features"]

    # Extraire H0/H1/H2
    h0_1, h1_1, h2_1 = _extract_homology_groups(dgms1)
    h0_2, h1_2, h2_2 = _extract_homology_groups(dgms2)

    h0_c, h1_c, h2_c = _extract_homology_groups(dgms_combined)

    return {
        # Source
        "tda_h0_count_src": src_stats["h0_count"],
        "tda_h1_count_src": src_stats["h1_count"],
        "tda_h2_count_src": src_stats["h2_count"],
        "tda_h0_max_life_src": src_stats["h0_max_life"],
        "tda_h1_max_life_src": src_stats["h1_max_life"],
        "tda_h2_max_life_src": src_stats["h2_max_life"],
        "tda_h0_entropy_src": src_stats["h0_entropy"],
        "tda_h1_entropy_src": src_stats["h1_entropy"],
        "tda_h2_entropy_src": src_stats["h2_entropy"],
        # Target
        "tda_h0_count_tgt": tgt_stats["h0_count"],
        "tda_h1_count_tgt": tgt_stats["h1_count"],
        "tda_h2_count_tgt": tgt_stats["h2_count"],
        "tda_h0_max_life_tgt": tgt_stats["h0_max_life"],
        "tda_h1_max_life_tgt": tgt_stats["h1_max_life"],
        "tda_h2_max_life_tgt": tgt_stats["h2_max_life"],
        "tda_h0_entropy_tgt": tgt_stats["h0_entropy"],
        "tda_h1_entropy_tgt": tgt_stats["h1_entropy"],
        "tda_h2_entropy_tgt": tgt_stats["h2_entropy"],
        # Combined (stacked clouds)
        "tda_h0_count_combined": h0_count(h0_c),
        "tda_h1_count_combined": h1_count(h1_c),
        "tda_h2_count_combined": h2_count(h2_c),
        "tda_h0_max_life_combined": max_lifetime(h0_c),
        "tda_h1_max_life_combined": max_lifetime(h1_c),
        "tda_h2_max_life_combined": max_lifetime(h2_c),
        "tda_h0_entropy_combined": persistence_entropy(h0_c),
        "tda_h1_entropy_combined": persistence_entropy(h1_c),
        "tda_h2_entropy_combined": persistence_entropy(h2_c),
        # Distances between source and target diagrams
        "tda_h0_bottleneck": compute_bottleneck(h0_1, h0_2),
        "tda_h0_wasserstein": compute_wasserstein(h0_1, h0_2),
        "tda_h1_bottleneck": compute_bottleneck(h1_1, h1_2),
        "tda_h1_wasserstein": compute_wasserstein(h1_1, h1_2),
        "tda_overlap_sym": overlap_sym,
        "tda_overlap_a2b": overlap_a2b,
        "tda_overlap_b2a": overlap_b2a,
    }


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    print(f"TDA disponible: {check_tda_available()}")
    print(f"  ripser: {HAS_RIPSER}")
    print(f"  persim: {HAS_PERSIM}")
    print(f"  gudhi bottleneck: {HAS_GUDHI_BOTTLENECK}")
    print(f"  gudhi wasserstein: {HAS_GUDHI_WASSERSTEIN}")

    if check_tda_available():
        np.random.seed(42)

        emb1 = np.random.randn(32, 384)
        emb2 = np.random.randn(32, 384)
        emb3 = emb1 + np.random.randn(32, 384) * 0.1

        print("\nFeatures TDA (emb1 vs emb2 - différents):")
        for k, v in compute_topological_features(emb1, emb2).items():
            print(f"  {k}: {v:.4f}")

        print("\nFeatures TDA (emb1 vs emb3 - proches):")
        for k, v in compute_topological_features(emb1, emb3).items():
            print(f"  {k}: {v:.4f}")
    else:
        print("\nInstaller ripser, persim, gudhi pour les features TDA:")
        print("  pip install ripser persim gudhi")
