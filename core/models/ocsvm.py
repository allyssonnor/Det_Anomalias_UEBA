#NOVO MODELO DE OCSVM
import numpy as np
from sklearn.svm import OneClassSVM

from core.models.base_model import BaseModel


class OCSVMModel(BaseModel):
    """
    Wrapper para o algoritmo One-Class SVM.
    Adaptado para aceitar a estrutura de tensores 3D com achatamento em 2D.
    """

    def __init__(self, config):
        super().__init__(config)

        model_cfg = config.get("model", {}).get("ocsvm", {})

        self.kernel = model_cfg.get("kernel", "rbf")
        self.nu = model_cfg.get("nu", 0.05)
        self.gamma = model_cfg.get("gamma", "scale")
        self.degree = model_cfg.get("degree", 3)
        self.coef0 = model_cfg.get("coef0", 0.0)
        self.tol = model_cfg.get("tol", 1e-3)
        self.cache_size = model_cfg.get("cache_size", 200)
        self.max_iter = model_cfg.get("max_iter", -1)

        self.model = None

    # =====================================================
    # TREINAMENTO
    # =====================================================

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Adaptação para o pipeline temporal 3D do projeto
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            tol=self.tol,
            cache_size=self.cache_size,
            max_iter=self.max_iter
        )

        self.model.fit(X)

    # =====================================================
    # CÁLCULO DE SCORE DE ANOMALIA
    # =====================================================

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Adaptação para o pipeline temporal 3D do projeto
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        if self.model is None:
            raise ValueError("O modelo OCSVM precisa ser treinado antes de computar os scores.")

        # decision_function original retorna positivo para normal e negativo para anômalo
        scores = self.model.decision_function(X)

        # Inverte os valores para seguir o contrato global (scores maiores indicam anomalias)
        scores = self._ensure_higher_is_anomaly(scores, invert=True)

        return scores
