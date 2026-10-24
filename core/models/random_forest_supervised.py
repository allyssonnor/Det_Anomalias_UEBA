# core/models/random_forest_supervised.py
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings

from core.models.base_model import BaseModel


class RandomForestSupervisedModel(BaseModel):
    """
    Random Forest supervisionado para detecção de anomalias.
    Treinado com rótulos explícitos (0 = Normal, 1 = Anomalia).
    Retorna a probabilidade da classe 1 (anomalia).
    """

    def __init__(self, config):
        super().__init__(config)

        model_cfg = config.get("model", {}).get("random_forest_supervised", {})

        self.n_estimators = model_cfg.get("n_estimators", 150)
        self.max_depth = model_cfg.get("max_depth", 8)
        self.min_samples_split = model_cfg.get("min_samples_split", 10)
        self.min_samples_leaf = model_cfg.get("min_samples_leaf", 5)
        self.class_weight = model_cfg.get("class_weight", "balanced_subsample")
        self.random_state = config.get("training", {}).get("seed", 42)
        self.n_jobs = model_cfg.get("n_jobs", -1)

        self.model = None

    def _prepare_data(self, X):
        """Aceita dados 2D ou 3D. Se 3D, achata (mas usamos sequence_length=1, então é 2D)."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 3:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)
        return X_arr

    def fit(self, X, y=None):
        X_processed = self._prepare_data(X)

        if y is None:
            raise ValueError("O modelo supervisionado exige rótulos 'y'.")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_processed, y)

    def score(self, X):
        X_processed = self._prepare_data(X)

        if self.model is None:
            raise ValueError("O modelo precisa ser treinado antes de calcular scores.")

        # Retorna a probabilidade da classe 1 (anomalia)
        probs = self.model.predict_proba(X_processed)
        if probs.shape[1] == 1:
            return np.zeros(len(X_processed))
        
        scores = probs[:, 1]
        
        # 🔍 Diagnóstico (opcional, remove depois)
        # print(f"🔍 Score range: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        return self._ensure_higher_is_anomaly(scores, invert=False)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
