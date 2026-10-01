#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement XGBoost.

- Cross-validation k-fold stratifiée
- Recherche du seuil optimal (F1)
- Sauvegarde modèle + feature importance
"""

# TODO: grid search pour les hyperparamètres
# DONE: early stopping pour éviter overfitting

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib


# =============================================================================
# Configuration par défaut
# =============================================================================

DEFAULT_XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    # Keep thread usage conservative on macOS to avoid runtime instability.
    "n_jobs": 1,
}

DEFAULT_EXTRATREES_PARAMS = {
    "n_estimators": 900,
    "random_state": 42,
    "n_jobs": 1,
    "class_weight": "balanced_subsample",
}

DEFAULT_RANDOMFOREST_PARAMS = {
    "n_estimators": 700,
    "random_state": 42,
    "n_jobs": 1,
    "class_weight": "balanced_subsample",
}

DEFAULT_STACKING_META_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
}


# =============================================================================
# Classe principale
# =============================================================================

class MetaMatchTrainer:
    """Entraîneur XGBoost pour MetaMatch."""

    def __init__(
        self,
        model_type: str = "xgboost",
        xgb_params: Optional[Dict[str, Any]] = None,
        extratrees_params: Optional[Dict[str, Any]] = None,
        randomforest_params: Optional[Dict[str, Any]] = None,
        stacking_meta_params: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialise le trainer.

        Args:
            model_type: "xgboost" | "extra_trees" | "random_forest" | "stacking"
            xgb_params: Paramètres XGBoost (utilise les défauts si None)
            threshold: Seuil de classification
        """
        self.model_type = model_type
        self.xgb_params = {**DEFAULT_XGBOOST_PARAMS, **(xgb_params or {})}
        self.extratrees_params = {**DEFAULT_EXTRATREES_PARAMS, **(extratrees_params or {})}
        self.randomforest_params = {**DEFAULT_RANDOMFOREST_PARAMS, **(randomforest_params or {})}
        self.stacking_meta_params = {**DEFAULT_STACKING_META_PARAMS, **(stacking_meta_params or {})}
        self.threshold = threshold
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def _build_model(self):
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(**self.xgb_params)
        if self.model_type == "extra_trees":
            return ExtraTreesClassifier(**self.extratrees_params)
        if self.model_type == "random_forest":
            return RandomForestClassifier(**self.randomforest_params)
        if self.model_type == "stacking":
            base_estimators = [
                ("xgb", xgb.XGBClassifier(**self.xgb_params)),
                ("et", ExtraTreesClassifier(**self.extratrees_params)),
                ("rf", RandomForestClassifier(**self.randomforest_params)),
            ]
            meta = LogisticRegression(**self.stacking_meta_params)
            return StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta,
                stack_method="predict_proba",
                passthrough=False,
                n_jobs=1,
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = True,
    ) -> "MetaMatchTrainer":
        """
        Entraîne le modèle XGBoost.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: Noms des features
            eval_set: Tuple (X_val, y_val) pour early stopping
            early_stopping_rounds: Patience pour early stopping
            verbose: Afficher la progression

        Returns:
            Self
        """
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        self.model = self._build_model()

        fit_params = {}
        if self.model_type == "xgboost" and eval_set is not None:
            fit_params["eval_set"] = [(eval_set[0], eval_set[1])]
            if early_stopping_rounds:
                fit_params["early_stopping_rounds"] = early_stopping_rounds
        if self.model_type == "xgboost":
            fit_params["verbose"] = verbose

        self.model.fit(X, y, **fit_params)

        # Feature importance
        self._compute_feature_importance()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prédit les probabilités de match."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Prédit les labels binaires.

        Args:
            X: Features
            threshold: Seuil (utilise self.threshold si None)

        Returns:
            Array de prédictions (0 ou 1)
        """
        threshold = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur un jeu de données.

        Args:
            X: Features
            y: Labels vrais
            threshold: Seuil de classification

        Returns:
            Dict de métriques
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X, threshold)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            "n_samples": len(y),
            "n_positives": int(y.sum()),
            "n_predicted_positives": int(y_pred.sum()),
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 10,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Cross-validation k-fold stratifiée.

        Args:
            X: Features
            y: Labels
            n_folds: Nombre de folds
            feature_names: Noms des features
            verbose: Afficher la progression

        Returns:
            Dict avec métriques par fold et moyennes
        """
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_metrics = []
        y_proba_all = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            if verbose:
                print(f"Fold {fold_idx + 1}/{n_folds}...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Entraîner sur ce fold
            fold_model = self._build_model()
            if self.model_type == "xgboost":
                fold_model.fit(X_train, y_train, verbose=False)
            else:
                fold_model.fit(X_train, y_train)

            # Prédire
            y_proba_fold = fold_model.predict_proba(X_val)[:, 1]
            y_proba_all[val_idx] = y_proba_fold
            y_pred_fold = (y_proba_fold >= self.threshold).astype(int)

            # Métriques du fold
            metrics = {
                "fold": fold_idx + 1,
                "accuracy": accuracy_score(y_val, y_pred_fold),
                "precision": precision_score(y_val, y_pred_fold, zero_division=0),
                "recall": recall_score(y_val, y_pred_fold, zero_division=0),
                "f1": f1_score(y_val, y_pred_fold, zero_division=0),
                "roc_auc": roc_auc_score(y_val, y_proba_fold) if len(np.unique(y_val)) > 1 else 0.0,
                "n_val": len(y_val),
                "n_pos_val": int(y_val.sum()),
            }
            fold_metrics.append(metrics)

            if verbose:
                print(f"  F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

        # Métriques globales OOF (out-of-fold)
        y_pred_oof = (y_proba_all >= self.threshold).astype(int)
        global_metrics = {
            "accuracy": accuracy_score(y, y_pred_oof),
            "precision": precision_score(y, y_pred_oof, zero_division=0),
            "recall": recall_score(y, y_pred_oof, zero_division=0),
            "f1": f1_score(y, y_pred_oof, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba_all) if len(np.unique(y)) > 1 else 0.0,
        }

        # Moyennes par fold
        avg_metrics = {
            k: np.mean([m[k] for m in fold_metrics])
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        }
        std_metrics = {
            f"{k}_std": np.std([m[k] for m in fold_metrics])
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        }

        if verbose:
            print(f"\n=== CV Results ({n_folds}-fold) ===")
            print(f"OOF F1: {global_metrics['f1']:.4f}")
            print(f"OOF AUC: {global_metrics['roc_auc']:.4f}")
            print(f"Avg F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")

        return {
            "fold_metrics": fold_metrics,
            "oof_metrics": global_metrics,
            "avg_metrics": avg_metrics,
            "std_metrics": std_metrics,
            "y_proba_oof": y_proba_all,
        }

    def _compute_feature_importance(self) -> None:
        """Calcule l'importance des features."""
        if self.model is None or self.feature_names is None:
            return

        if not hasattr(self.model, "feature_importances_"):
            self.feature_importance = None
            return

        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Retourne les N features les plus importantes."""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.feature_importance.head(n)

    def save(self, path: str) -> None:
        """Sauvegarde le modèle et les métadonnées."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modèle
        if self.model_type == "xgboost":
            model_path = path.with_suffix(".xgb")
            self.model.save_model(str(model_path))
        else:
            model_path = path.with_suffix(".joblib")
            joblib.dump(self.model, model_path)

        # Sauvegarder les métadonnées
        meta = {
            "model_type": self.model_type,
            "xgb_params": self.xgb_params,
            "extratrees_params": self.extratrees_params,
            "randomforest_params": self.randomforest_params,
            "stacking_meta_params": self.stacking_meta_params,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Sauvegarder l'importance des features
        if self.feature_importance is not None:
            importance_path = path.with_suffix(".importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)

    @classmethod
    def load(cls, path: str) -> "MetaMatchTrainer":
        """Charge un modèle sauvegardé."""
        path = Path(path)

        # Charger les métadonnées
        meta_path = path.with_suffix(".json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        trainer = cls(
            model_type=meta.get("model_type", "xgboost"),
            xgb_params=meta["xgb_params"],
            extratrees_params=meta.get("extratrees_params"),
            randomforest_params=meta.get("randomforest_params"),
            stacking_meta_params=meta.get("stacking_meta_params"),
            threshold=meta["threshold"],
        )
        trainer.feature_names = meta["feature_names"]

        # Charger le modèle
        if trainer.model_type == "xgboost":
            model_path = path.with_suffix(".xgb")
            trainer.model = xgb.XGBClassifier()
            trainer.model.load_model(str(model_path))
        else:
            model_path = path.with_suffix(".joblib")
            trainer.model = joblib.load(model_path)

        # Charger l'importance si disponible
        importance_path = path.with_suffix(".importance.csv")
        if importance_path.exists():
            trainer.feature_importance = pd.read_csv(importance_path)

        return trainer


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def prepare_training_data(
    features_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prépare les données pour l'entraînement.

    Args:
        features_df: DataFrame des features
        pairs_df: DataFrame des paires avec colonne 'label'
        feature_cols: Colonnes à utiliser (toutes si None)

    Returns:
        X, y, feature_names
    """
    if feature_cols is None:
        feature_cols = [c for c in features_df.columns if c not in ["src_iri", "tgt_iri", "label"]]

    X = features_df[feature_cols].values
    y = pairs_df["label"].values

    # Gérer les NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Trouve le seuil optimal pour une métrique donnée.

    Args:
        y_true: Labels vrais
        y_proba: Probabilités prédites
        metric: Métrique à optimiser ("f1", "precision", "recall")
        thresholds: Seuils à tester

    Returns:
        (optimal_threshold, best_score)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Test avec données synthétiques
    np.random.seed(42)

    n_samples = 1000
    n_features = 55

    # Générer des features
    X = np.random.randn(n_samples, n_features)

    # Labels: dépendant de quelques features
    logits = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.5
    y = (logits > 0).astype(int)

    print(f"Données: {X.shape}, {y.sum()} positifs / {len(y)} total")

    # Entraîner
    trainer = MetaMatchTrainer()

    print("\n=== Entraînement ===")
    trainer.train(X, y, verbose=False)

    train_metrics = trainer.evaluate(X, y)
    print(f"Train F1: {train_metrics['f1']:.4f}")
    print(f"Train AUC: {train_metrics['roc_auc']:.4f}")

    # Cross-validation
    print("\n=== Cross-validation 5-fold ===")
    cv_results = trainer.cross_validate(X, y, n_folds=5, verbose=True)

    # Top features
    print("\n=== Top 10 Features ===")
    print(trainer.get_top_features(10))

    # Test sauvegarde/chargement
    print("\n=== Test Save/Load ===")
    trainer.save("/tmp/test_model")
    loaded = MetaMatchTrainer.load("/tmp/test_model")
    loaded_metrics = loaded.evaluate(X, y)
    print(f"Loaded model F1: {loaded_metrics['f1']:.4f}")
