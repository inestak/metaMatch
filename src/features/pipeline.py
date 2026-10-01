#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline pour calculer les méta-features.

55 features par défaut:
- syntax.py: 22 (Levenshtein, Jaro-Winkler, Jaccard, etc.)
- classical.py: 8 (cosine, euclidean, etc.)
- spectral.py: 15 (SVD)
- nlp_metrics.py: 10 (BLEU, ROUGE)

+12 avec TDA (optionnel, lent): topological.py
"""

# TODO: ajouter option pour sauvegarder les features intermédiaires
# TODO: paralléliser le calcul des features (multiprocessing)
# DONE: désactiver TDA par défaut (trop lent)

from typing import Dict, List, Optional, Set

try:
    from ..debug import debug, debug_var
except ImportError:
    def debug(msg, level=1): pass
    def debug_var(name, value, level=2): pass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .syntax import compute_syntax_features, SYNTAX_FEATURES
from .classical import compute_classical_features, compute_classical_features_vectorized, CLASSICAL_FEATURES
from .spectral import compute_spectral_features_pair, SPECTRAL_FEATURES
from .topological import compute_topological_features, TOPOLOGICAL_FEATURES, check_tda_available
from .nlp_metrics import compute_nlp_features, NLP_FEATURES


# =============================================================================
# Configuration des features
# =============================================================================

ALL_FEATURES = (
    SYNTAX_FEATURES +
    CLASSICAL_FEATURES +
    [f"spc_src_{f.replace('spc_', '')}" for f in SPECTRAL_FEATURES] +
    [f"spc_tgt_{f.replace('spc_', '')}" for f in SPECTRAL_FEATURES] +
    [f"spc_combined_{f.replace('spc_', '')}" for f in SPECTRAL_FEATURES] +
    TOPOLOGICAL_FEATURES +
    NLP_FEATURES
)


class FeaturePipeline:
    """Pipeline de calcul des méta-features."""

    def __init__(
        self,
        use_syntax: bool = True,
        use_classical: bool = True,
        use_spectral: bool = True,
        use_topological: bool = False,  # Désactivé par défaut (lent)
        use_nlp: bool = True,
    ):
        """
        Initialise le pipeline.

        Args:
            use_syntax: Calculer les features syntaxiques
            use_classical: Calculer les distances classiques
            use_spectral: Calculer les features spectrales
            use_topological: Calculer les features TDA (lent)
            use_nlp: Calculer BLEU/ROUGE
        """
        self.use_syntax = use_syntax
        self.use_classical = use_classical
        self.use_spectral = use_spectral
        self.use_topological = use_topological and check_tda_available()
        self.use_nlp = use_nlp

        debug(f"FeaturePipeline: syntax={use_syntax}, classical={use_classical}, spectral={use_spectral}, tda={self.use_topological}, nlp={use_nlp}")
        # if use_topological:
        #     print("WARNING: TDA activé, ça va être long...")  # TODO: virer ce print

        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> List[str]:
        """Retourne la liste des features actives."""
        if self._feature_names is None:
            names = []
            if self.use_syntax:
                names.extend(SYNTAX_FEATURES)
            if self.use_classical:
                names.extend(CLASSICAL_FEATURES)
            if self.use_spectral:
                # 3 variantes: src, tgt, combined
                # SPECTRAL_FEATURES = ["spc_alpha_req", "spc_ne_sum", ...]
                # On veut: spc_src_alpha_req, spc_src_ne_sum, ...
                base_names = [f.replace("spc_", "") for f in SPECTRAL_FEATURES]
                for prefix in ["spc_src", "spc_tgt", "spc_combined"]:
                    names.extend([f"{prefix}_{b}" for b in base_names])
            if self.use_topological:
                names.extend(TOPOLOGICAL_FEATURES)
            if self.use_nlp:
                names.extend(NLP_FEATURES)
            self._feature_names = names
        return self._feature_names

    def compute_features_single(
        self,
        src_label: str,
        tgt_label: str,
        src_embedding: Optional[np.ndarray] = None,
        tgt_embedding: Optional[np.ndarray] = None,
        src_token_matrix: Optional[np.ndarray] = None,
        tgt_token_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calcule toutes les features pour une paire.

        Args:
            src_label: Label source
            tgt_label: Label cible
            src_embedding: Embedding source (requis pour classical)
            tgt_embedding: Embedding cible (requis pour classical)

        Returns:
            Dictionnaire de features
        """
        features = {}

        if self.use_syntax:
            features.update(compute_syntax_features(src_label, tgt_label))

        if self.use_classical and src_embedding is not None and tgt_embedding is not None:
            features.update(compute_classical_features(src_embedding, tgt_embedding))

        if self.use_spectral:
            features.update(
                compute_spectral_features_pair(
                    src_token_matrix,
                    tgt_token_matrix,
                )
            )

        if self.use_topological:
            features.update(
                compute_topological_features(
                    src_token_matrix,
                    tgt_token_matrix,
                )
            )

        if self.use_nlp:
            features.update(compute_nlp_features(src_label, tgt_label))

        return features

    def compute_features_batch(
        self,
        pairs_df: pd.DataFrame,
        src_embeddings: Optional[np.ndarray] = None,
        tgt_embeddings: Optional[np.ndarray] = None,
        src_token_embeddings: Optional[List[np.ndarray]] = None,
        tgt_token_embeddings: Optional[List[np.ndarray]] = None,
        tda_cache_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Calcule les features pour un batch de paires.

        Args:
            pairs_df: DataFrame avec colonnes src_label, tgt_label
            src_embeddings: Array (n_pairs, embedding_dim)
            tgt_embeddings: Array (n_pairs, embedding_dim)
            show_progress: Afficher la progression

        Returns:
            DataFrame avec les features
        """
        n = len(pairs_df)
        has_embeddings = src_embeddings is not None and tgt_embeddings is not None

        # Initialiser les colonnes
        feature_data = {name: [] for name in self.feature_names}

        # Calcul vectorisé pour les features classiques si possible
        classical_vectorized = None
        if self.use_classical and has_embeddings:
            print("Calcul vectorisé des features classiques...")
            classical_vectorized = compute_classical_features_vectorized(
                src_embeddings, tgt_embeddings
            )

        # Cache des diagrammes TDA par entité (évite de recalculer par paire)
        tda_diagram_cache = {} if self.use_topological else None
        # Cache des payloads TDA unaires (stats + diagrammes)
        tda_entity_cache = {} if self.use_topological else None
        tda_cache_dir_str = None
        if self.use_topological and tda_cache_dir is not None:
            tda_cache_dir = Path(tda_cache_dir)
            tda_cache_dir.mkdir(parents=True, exist_ok=True)
            tda_cache_dir_str = str(tda_cache_dir)

        # Boucle sur les paires
        if show_progress:
            iterator = tqdm(range(n), desc="Computing features", leave=False)
        else:
            iterator = range(n)

        for i in iterator:
            row = pairs_df.iloc[i]
            src_label = row["src_label"]
            tgt_label = row["tgt_label"]

            src_emb = src_embeddings[i] if has_embeddings else None
            tgt_emb = tgt_embeddings[i] if has_embeddings else None
            src_tok = src_token_embeddings[i] if src_token_embeddings is not None else None
            tgt_tok = tgt_token_embeddings[i] if tgt_token_embeddings is not None else None

            # Syntaxiques
            if self.use_syntax:
                syn_feats = compute_syntax_features(src_label, tgt_label)
                for k in SYNTAX_FEATURES:
                    feature_data[k].append(syn_feats[k])

            # Classiques (utiliser le vectorisé)
            if self.use_classical and has_embeddings:
                for k in CLASSICAL_FEATURES:
                    feature_data[k].append(classical_vectorized[k][i])

            # Spectrales
            if self.use_spectral:
                spec_feats = compute_spectral_features_pair(
                    src_tok,
                    tgt_tok,
                )
                for k in spec_feats:
                    if k in feature_data:
                        feature_data[k].append(spec_feats[k])

            # Topologiques (lent!)
            if self.use_topological:
                src_key = row["src_iri"] if "src_iri" in row.index else None
                tgt_key = row["tgt_iri"] if "tgt_iri" in row.index else None
                topo_feats = compute_topological_features(
                    src_tok,
                    tgt_tok,
                    diagram_cache=tda_diagram_cache,
                    entity_cache=tda_entity_cache,
                    disk_cache_dir=tda_cache_dir_str,
                    src_key=src_key,
                    tgt_key=tgt_key,
                )
                for k in TOPOLOGICAL_FEATURES:
                    feature_data[k].append(topo_feats[k])

            # NLP
            if self.use_nlp:
                nlp_feats = compute_nlp_features(src_label, tgt_label)
                for k in NLP_FEATURES:
                    feature_data[k].append(nlp_feats[k])

        if show_progress and hasattr(iterator, "close"):
            iterator.close()

        return pd.DataFrame(feature_data)

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Retourne les features groupées par catégorie."""
        groups = {}
        if self.use_syntax:
            groups["syntax"] = SYNTAX_FEATURES
        if self.use_classical:
            groups["classical"] = CLASSICAL_FEATURES
        if self.use_spectral:
            groups["spectral"] = [f for f in self.feature_names if f.startswith("spc_")]
        if self.use_topological:
            groups["topological"] = TOPOLOGICAL_FEATURES
        if self.use_nlp:
            groups["nlp"] = NLP_FEATURES
        return groups


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Test simple
    pipeline = FeaturePipeline(
        use_syntax=True,
        use_classical=True,
        use_spectral=True,
        use_topological=False,
        use_nlp=True,
    )

    print(f"Features actives: {len(pipeline.feature_names)}")
    print("\nGroupes:")
    for group, feats in pipeline.get_feature_groups().items():
        print(f"  {group}: {len(feats)} features")

    # Test avec des données
    np.random.seed(42)

    src_label = "diabetes mellitus type 2"
    tgt_label = "type 2 diabetes"
    src_emb = np.random.randn(384)
    tgt_emb = np.random.randn(384)

    print(f"\nTest paire: '{src_label}' vs '{tgt_label}'")
    features = pipeline.compute_features_single(src_label, tgt_label, src_emb, tgt_emb)

    print(f"\n{len(features)} features calculées:")
    for k, v in list(features.items())[:10]:
        print(f"  {k}: {v:.4f}")
    print("  ...")
