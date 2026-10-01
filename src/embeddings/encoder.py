#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encodage des labels en embeddings (sentence-transformers).

Modèles: roberta, minilm, mpnet
Inclut un cache pour éviter de recalculer (comment j'ai mal codé la première fois...).
"""

# TODO: tester avec RoBERTa (meilleur mais plus lent)
# TODO: option pour concaténer les synonymes
# DONE: cache des embeddings pour éviter recalcul

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# Modèles disponibles
MODELS = {
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


class LabelEncoder:
    """Encode les labels textuels en vecteurs denses."""

    def __init__(
        self,
        model_name: str = "roberta",
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialise l'encodeur.

        Args:
            model_name: Nom du modèle ("roberta", "minilm", "mpnet") ou chemin HuggingFace
            device: "cpu" ou "cuda"
            cache_dir: Répertoire pour le cache des embeddings
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers requis: pip install sentence-transformers")

        # Résoudre le nom du modèle
        if model_name in MODELS:
            self.model_path = MODELS[model_name]
        else:
            self.model_path = model_name

        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        print(f"Chargement du modèle {self.model_path}...")
        # Prefer safetensors first to avoid torch.load restrictions on old torch versions.
        try:
            self.model = SentenceTransformer(
                self.model_path,
                device=device,
                model_kwargs={"use_safetensors": True},
            )
        except Exception as e:
            print(f"  Fallback chargement standard (sans safetensors forcé): {e}")
            self.model = SentenceTransformer(self.model_path, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode une liste de textes.

        Args:
            texts: Liste de textes à encoder
            batch_size: Taille des batches
            show_progress: Afficher la progression

        Returns:
            Array numpy (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    @staticmethod
    def _pool_token_embeddings(token_matrix: np.ndarray, pooling: str = "mean") -> np.ndarray:
        """
        Pooling d'une matrice token-level en vecteur sentence-level.

        Args:
            token_matrix: Array (n_tokens, emb_dim)
            pooling: "mean" (défaut) ou "max"

        Returns:
            Array (emb_dim,)
        """
        if token_matrix.size == 0:
            return np.array([], dtype=np.float32)

        if pooling == "max":
            return token_matrix.max(axis=0).astype(np.float32)
        # défaut: mean pooling
        return token_matrix.mean(axis=0).astype(np.float32)

    def encode_token_and_vector(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        pooling: str = "mean",
        max_tokens: int = 64,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Encode les textes en:
        1) matrice token embeddings (variable-length)
        2) vecteur poolé (mean/max)

        Args:
            texts: Textes à encoder
            batch_size: Batch size
            show_progress: Barre de progression ST
            pooling: "mean" ou "max"
            max_tokens: Troncature du nombre de tokens (0/None = pas de troncature)

        Returns:
            (token_matrices, pooled_vectors)
        """
        token_tensors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=False,
            output_value="token_embeddings",
        )

        token_matrices: List[np.ndarray] = []
        pooled_vectors: List[np.ndarray] = []

        for tok in token_tensors:
            arr = tok.detach().cpu().numpy().astype(np.float32)
            if max_tokens and max_tokens > 0 and arr.shape[0] > max_tokens:
                arr = arr[:max_tokens]

            token_matrices.append(arr)
            pooled_vectors.append(self._pool_token_embeddings(arr, pooling=pooling))

        return token_matrices, np.vstack(pooled_vectors)

    def encode_ontology(
        self,
        ontology_loader,
        use_synonyms: bool = False,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Encode toutes les classes d'une ontologie.

        Args:
            ontology_loader: OntologyLoader chargé
            use_synonyms: Concaténer les synonymes au label principal
            batch_size: Taille des batches

        Returns:
            DataFrame avec index=IRI, colonnes=dimensions embedding
        """
        iris = []
        texts = []

        for iri, info in ontology_loader.classes.items():
            iris.append(iri)

            if use_synonyms and info["synonyms"]:
                # Concaténer label + synonymes
                all_labels = [info["label"]] + info["synonyms"]
                text = " | ".join(all_labels)
            else:
                text = info["label"]

            texts.append(text)

        print(f"Encodage de {len(texts)} classes...")
        embeddings = self.encode(texts, batch_size=batch_size)

        # Créer le DataFrame
        df = pd.DataFrame(
            embeddings,
            index=iris,
            columns=[f"dim_{i}" for i in range(self.embedding_dim)],
        )
        return df

    def encode_ontology_token_bundle(
        self,
        ontology_loader,
        use_synonyms: bool = False,
        batch_size: int = 32,
        pooling: str = "mean",
        max_tokens: int = 64,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Encode une ontologie en bundle token+vector.

        Returns:
            - vectors_df: DataFrame index=IRI, colonnes dim_*
            - tokens_map: dict IRI -> matrice (n_tokens, emb_dim)
        """
        iris: List[str] = []
        texts: List[str] = []

        for iri, info in ontology_loader.classes.items():
            iris.append(iri)
            if use_synonyms and info["synonyms"]:
                text = " | ".join([info["label"]] + info["synonyms"])
            else:
                text = info["label"]
            texts.append(text)

        print(f"Encodage token+vector de {len(texts)} classes...")
        token_matrices, pooled_vectors = self.encode_token_and_vector(
            texts=texts,
            batch_size=batch_size,
            show_progress=True,
            pooling=pooling,
            max_tokens=max_tokens,
        )

        vectors_df = pd.DataFrame(
            pooled_vectors,
            index=iris,
            columns=[f"dim_{i}" for i in range(pooled_vectors.shape[1])],
        )
        tokens_map = {iri: mat for iri, mat in zip(iris, token_matrices)}
        return vectors_df, tokens_map

    def encode_pairs(
        self,
        pairs_df: pd.DataFrame,
        batch_size: int = 32,
    ) -> tuple:
        """
        Encode les labels des paires.

        Args:
            pairs_df: DataFrame avec colonnes src_label, tgt_label

        Returns:
            Tuple (src_embeddings, tgt_embeddings) - arrays numpy
        """
        # Encoder les labels uniques pour éviter les doublons
        unique_src = pairs_df["src_label"].unique().tolist()
        unique_tgt = pairs_df["tgt_label"].unique().tolist()

        print(f"Encodage {len(unique_src)} labels source...")
        src_emb_dict = dict(zip(
            unique_src,
            self.encode(unique_src, batch_size=batch_size)
        ))

        print(f"Encodage {len(unique_tgt)} labels cible...")
        tgt_emb_dict = dict(zip(
            unique_tgt,
            self.encode(unique_tgt, batch_size=batch_size)
        ))

        # Mapper sur les paires
        src_embeddings = np.array([src_emb_dict[l] for l in pairs_df["src_label"]])
        tgt_embeddings = np.array([tgt_emb_dict[l] for l in pairs_df["tgt_label"]])

        return src_embeddings, tgt_embeddings

    def get_pair_embeddings(
        self,
        pairs_df: pd.DataFrame,
        src_embeddings_df: pd.DataFrame,
        tgt_embeddings_df: pd.DataFrame,
        src_col: str = "src_iri",
        tgt_col: str = "tgt_iri",
    ) -> tuple:
        """
        Extrait les embeddings pour les paires depuis des DataFrames pré-calculés.

        Args:
            pairs_df: DataFrame avec colonnes src_iri, tgt_iri
            src_embeddings_df: DataFrame des embeddings source (index=IRI)
            tgt_embeddings_df: DataFrame des embeddings cible (index=IRI)
            src_col: Nom de la colonne source
            tgt_col: Nom de la colonne cible

        Returns:
            Tuple (src_embeddings, tgt_embeddings) - arrays numpy
        """
        # Extraire les embeddings pour chaque paire
        src_iris = pairs_df[src_col].tolist()
        tgt_iris = pairs_df[tgt_col].tolist()

        # Vérifier que les IRIs existent
        missing_src = set(src_iris) - set(src_embeddings_df.index)
        missing_tgt = set(tgt_iris) - set(tgt_embeddings_df.index)

        if missing_src:
            print(f"Warning: {len(missing_src)} source IRIs not in embeddings")
        if missing_tgt:
            print(f"Warning: {len(missing_tgt)} target IRIs not in embeddings")

        # Extraire les embeddings
        src_embeddings = []
        tgt_embeddings = []

        for src_iri, tgt_iri in zip(src_iris, tgt_iris):
            if src_iri in src_embeddings_df.index:
                src_embeddings.append(src_embeddings_df.loc[src_iri].values)
            else:
                # Vecteur nul si manquant
                src_embeddings.append(np.zeros(len(src_embeddings_df.columns)))

            if tgt_iri in tgt_embeddings_df.index:
                tgt_embeddings.append(tgt_embeddings_df.loc[tgt_iri].values)
            else:
                tgt_embeddings.append(np.zeros(len(tgt_embeddings_df.columns)))

        return np.array(src_embeddings), np.array(tgt_embeddings)

    def get_pair_token_embeddings(
        self,
        pairs_df: pd.DataFrame,
        src_tokens_map: Dict[str, np.ndarray],
        tgt_tokens_map: Dict[str, np.ndarray],
        src_col: str = "src_iri",
        tgt_col: str = "tgt_iri",
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extrait les matrices token-level pour chaque paire.
        """
        src_tokens: List[np.ndarray] = []
        tgt_tokens: List[np.ndarray] = []

        for src_iri, tgt_iri in zip(pairs_df[src_col].tolist(), pairs_df[tgt_col].tolist()):
            src_tokens.append(src_tokens_map.get(src_iri, np.empty((0, self.embedding_dim), dtype=np.float32)))
            tgt_tokens.append(tgt_tokens_map.get(tgt_iri, np.empty((0, self.embedding_dim), dtype=np.float32)))

        return src_tokens, tgt_tokens

    def save_embeddings(self, df: pd.DataFrame, path: Path):
        """Sauvegarde les embeddings."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
        print(f"Embeddings sauvegardés: {path}")

    def load_embeddings(self, path: Path) -> pd.DataFrame:
        """Charge les embeddings."""
        return pd.read_pickle(path)


class EmbeddingCache:
    """Cache pour les embeddings d'ontologies."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, ontology_name: str, model_name: str, use_synonyms: bool) -> Path:
        suffix = "_syn" if use_synonyms else ""
        # Sanitize model name (replace / with _)
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{ontology_name}_{safe_model_name}{suffix}.pkl"

    def _get_token_bundle_paths(
        self,
        ontology_name: str,
        model_name: str,
        pooling: str,
        max_tokens: int,
        use_synonyms: bool,
    ) -> Tuple[Path, Path]:
        suffix = "_syn" if use_synonyms else ""
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        base = f"{ontology_name}_{safe_model_name}_{pooling}_tok{max_tokens}{suffix}"
        return (
            self.cache_dir / f"{base}_vec.pkl",
            self.cache_dir / f"{base}_tok.pkl",
        )

    def exists(self, ontology_name: str, model_name: str, use_synonyms: bool = False) -> bool:
        return self._get_path(ontology_name, model_name, use_synonyms).exists()

    def load(self, ontology_name: str, model_name: str, use_synonyms: bool = False) -> pd.DataFrame:
        path = self._get_path(ontology_name, model_name, use_synonyms)
        print(f"Chargement cache: {path}")
        return pd.read_pickle(path)

    def save(self, df: pd.DataFrame, ontology_name: str, model_name: str, use_synonyms: bool = False):
        path = self._get_path(ontology_name, model_name, use_synonyms)
        df.to_pickle(path)
        print(f"Cache sauvegardé: {path}")

    def get_or_compute(
        self,
        ontology_loader,
        ontology_name: str,
        encoder: LabelEncoder,
        use_synonyms: bool = False,
    ) -> pd.DataFrame:
        """Charge depuis le cache ou calcule les embeddings."""
        if self.exists(ontology_name, encoder.model_name, use_synonyms):
            return self.load(ontology_name, encoder.model_name, use_synonyms)

        df = encoder.encode_ontology(ontology_loader, use_synonyms=use_synonyms)
        self.save(df, ontology_name, encoder.model_name, use_synonyms)
        return df

    def get_or_compute_token_bundle(
        self,
        ontology_loader,
        ontology_name: str,
        encoder: LabelEncoder,
        use_synonyms: bool = False,
        pooling: str = "mean",
        max_tokens: int = 64,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Charge ou calcule le bundle (vectors_df + tokens_map).
        """
        vec_path, tok_path = self._get_token_bundle_paths(
            ontology_name=ontology_name,
            model_name=encoder.model_name,
            pooling=pooling,
            max_tokens=max_tokens,
            use_synonyms=use_synonyms,
        )

        if vec_path.exists() and tok_path.exists():
            print(f"Chargement cache vectors: {vec_path}")
            print(f"Chargement cache tokens: {tok_path}")
            vectors_df = pd.read_pickle(vec_path)
            tokens_map = pd.read_pickle(tok_path)
            return vectors_df, tokens_map

        vectors_df, tokens_map = encoder.encode_ontology_token_bundle(
            ontology_loader=ontology_loader,
            use_synonyms=use_synonyms,
            pooling=pooling,
            max_tokens=max_tokens,
        )
        vectors_df.to_pickle(vec_path)
        pd.to_pickle(tokens_map, tok_path)
        print(f"Cache sauvegardé vectors: {vec_path}")
        print(f"Cache sauvegardé tokens: {tok_path}")
        return vectors_df, tokens_map


# -----------------------------------------------------------------------------
# CLI pour tester
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.ontology_loader import OntologyLoader

    if len(sys.argv) < 2:
        print("Usage: python encoder.py <owl_file> [model_name]")
        print("Example: python encoder.py data/bioml/omim-ordo/omim.owl roberta")
        sys.exit(1)

    owl_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "minilm"  # MiniLM par défaut (plus rapide)

    # Charger l'ontologie
    onto = OntologyLoader(owl_path).load()

    # Encoder
    encoder = LabelEncoder(model_name=model_name)
    embeddings_df = encoder.encode_ontology(onto, use_synonyms=False)

    print(f"\n=== Embeddings ===")
    print(f"Shape: {embeddings_df.shape}")
    print(f"\nExemples (5 premières classes, 5 premières dimensions):")
    print(embeddings_df.iloc[:5, :5].to_string())

    # Statistiques
    print(f"\nStatistiques:")
    print(f"  Min: {embeddings_df.values.min():.4f}")
    print(f"  Max: {embeddings_df.values.max():.4f}")
    print(f"  Mean: {embeddings_df.values.mean():.4f}")
    print(f"  Std: {embeddings_df.values.std():.4f}")
