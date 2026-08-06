#NOVO MODELO DO MLP_SINGLE
import numpy as np
from sklearn.neural_network import MLPRegressor
import warnings

from core.models.base_model import BaseModel


class MLPSingleModel(BaseModel):
    """
    MLP Single (Otimizado para CPU usando Scikit-Learn).
    
    Recebe os dados 3D do TemporalBuilder e transforma em uma representação 2D,
    combinando o último evento do usuário com a média histórica do seu contexto temporal.
    A rede tenta reconstruir essa representação compacta. O erro quadrático médio (MSE)
    indica a taxa de anômala.
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
        self.alpha = model_cfg.get("alpha", 0.0001)  # Penalização L2 para regularização
        self.batch_size = model_cfg.get("batch_size", "auto")
        self.learning_rate_init = model_cfg.get("learning_rate_init", 0.001)
        self.max_iter = model_cfg.get("max_iter", 200)
        self.early_stopping = model_cfg.get("early_stopping", True)
        self.validation_fraction = model_cfg.get("validation_fraction", 0.1)
        self.random_state = config.get("training", {}).get("seed", 42)

        self.model = None

    # =====================================================
    # PREPARAÇÃO DIMENSIONAL (COMPATIBILIDADE 3D -> 2D)
    # =====================================================
    
    def _prepare_data(self, X):
        """
        Converte a matriz 3D do TemporalBuilder para 2D.
        Funde o estado comportamental corrente do usuário com seu comportamento médio recente.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 3:
            X_last_event = X_arr[:, -1, :]
            X_mean_context = np.mean(X_arr, axis=1)
            return np.hstack([X_last_event, X_mean_context])
        return X_arr

    # =====================================================
    # TREINAMENTO
    # =====================================================

    def fit(self, X):
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

        # Treina o MLP para reconstruir sua própria entrada consolidada (Autoencoder leve)
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

        # Reconstrução executada pela rede neural
        recon = self.model.predict(X_processed)

        # Erro quadrático médio por amostra
        errors = (X_processed - recon) ** 2
        scores = np.mean(errors, axis=1)

        # Contrato: maior erro = mais anômalo
        return self._ensure_higher_is_anomaly(scores, invert=False)