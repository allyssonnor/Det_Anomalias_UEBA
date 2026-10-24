# core/models/mlp_single.py
# CORRIGIDO: Representação da janela agora usa 6 estatísticas (last, mean, std, min, max, sum)
# Baseado nos resultados do diagnóstico: Teste 3 (janela estatística) atingiu PR-AUC 0.96.
# Isso substitui a antiga representação (last + mean) que perdia sinal.
import os
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
import warnings

from core.models.base_model import BaseModel


class MLPSingleModel(BaseModel):
    """
    MLP Single (Otimizado para CPU usando Scikit-Learn).
    Versão corrigida: expande a janela temporal para 6 estatísticas por feature:
      - último evento
      - média
      - desvio padrão
      - mínimo
      - máximo
      - soma
    Entrada: sequence_length (20) × 16 features → 96 features de entrada.
    """

    def __init__(self, config):
        super().__init__(config)

        model_cfg = config.get("model", {}).get("mlp_single", {})

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

    # =====================================================
    # PREPARAÇÃO DIMENSIONAL (CORRIGIDA)
    # =====================================================
    def _prepare_data(self, X):
        """
        Converte a matriz 3D do TemporalBuilder (janelas) em uma representação
        2D enriquecida com 6 estatísticas por feature.
        Antes: último evento + média (32 features)
        Agora:  último + média + std + min + max + soma (96 features)
        """
        X_arr = np.asarray(X, dtype=np.float32)

        if X_arr.ndim == 3:
            # (n_samples, sequence_length, n_features)
            X_last = X_arr[:, -1, :]                  # último evento
            X_mean = np.mean(X_arr, axis=1)           # média
            X_std = np.std(X_arr, axis=1)             # desvio padrão
            X_min = np.min(X_arr, axis=1)             # mínimo
            X_max = np.max(X_arr, axis=1)             # máximo
            X_sum = np.sum(X_arr, axis=1)             # soma

            return np.hstack([X_last, X_mean, X_std, X_min, X_max, X_sum])

        return X_arr

    # =====================================================
    # TREINAMENTO
    # =====================================================
    #def fit(self, X): aqui seria para nunca aceitar treinar com dados de anomalia
    def fit(self, X, y=None, **kwargs): #aqui deixei passar
        X_processed = self._prepare_data(X)

        self.model = MLPRegressor(
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
            self.model.fit(X_processed, X_processed)

    # =====================================================
    # CÁLCULO DE SCORE DE ANOMALIA
    # =====================================================
    def score(self, X):
        X_processed = self._prepare_data(X)

        if self.model is None:
            raise ValueError("O modelo MLP Single precisa ser treinado antes de calcular scores.")

        recon = self.model.predict(X_processed)
        errors = (X_processed - recon) ** 2
        scores = np.mean(errors, axis=1)

        return self._ensure_higher_is_anomaly(scores, invert=False)
   # =====================================================
    # SALVAR E CARREGAR MODELO
    # =====================================================
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
