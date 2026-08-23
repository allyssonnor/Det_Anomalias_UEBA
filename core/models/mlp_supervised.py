# Arquivo: core/models/mlp_supervised.py
import os
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings

from core.models.base_model import BaseModel

class MLPSupervisedModel(BaseModel):
    """
    Especialista Supervisionado (IDS Clássico).
    Treinado com rótulos explícitos (0 = Normal, 1 = Ataque Alvo).
    Retorna a probabilidade da janela pertencer à classe de ataque.
    """

    def __init__(self, config):
        super().__init__(config)
        
        model_cfg = config.get("model", {}).get("mlp_supervised", config.get("model", {}).get("mlp_single", {}))
        
        hidden_layer_sizes = model_cfg.get("hidden_layer_sizes", (32,))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
            
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = model_cfg.get("activation", "relu")
        self.solver = model_cfg.get("solver", "adam")
        self.alpha = model_cfg.get("alpha", 0.0001)
        self.batch_size = model_cfg.get("batch_size", "auto")
        self.learning_rate_init = model_cfg.get("learning_rate_init", 0.001)
        self.max_iter = model_cfg.get("max_iter", 200)
        self.early_stopping = model_cfg.get("early_stopping", True)
        self.validation_fraction = model_cfg.get("validation_fraction", 0.1)
        self.random_state = config.get("training", {}).get("seed", 42)

        self.model = None

    def _prepare_data(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 3:
            X_last_event = X_arr[:, -1, :]
            X_mean_context = np.mean(X_arr, axis=1)
            return np.hstack([X_last_event, X_mean_context])
        return X_arr

    def fit(self, X, y=None):
        X_processed = self._prepare_data(X)
        
        if y is None or len(np.unique(y)) < 2:
            raise ValueError("O modelo supervisionado exige rótulos 'y' com pelo menos duas classes (0 e 1).")

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_processed, y)

    def score(self, X):
        X_processed = self._prepare_data(X)

        if self.model is None:
            raise ValueError("O modelo supervisionado precisa ser treinado antes de calcular scores.")

        probs = self.model.predict_proba(X_processed)
        
        if probs.shape[1] == 1:
            return np.zeros(len(X_processed))
            
        scores = probs[:, 1]
        return self._ensure_higher_is_anomaly(scores, invert=False)

    def save(self, path):
        """Exporta o modelo matematicamente treinado para o disco."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        """Carrega o modelo do disco para inferência futura."""
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
