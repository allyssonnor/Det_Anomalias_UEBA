# core/models/lof_model.py
#NOVO MODELO DO LOF
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from core.models.base_model import BaseModel


class LOFModel(BaseModel):
    """
    Wrapper do Local Outlier Factor (LOF) adaptado para o ecossistema do projeto.
    Dá suporte a inferência em dados novos (test set) através de novelty=True
    e suporta o pipeline temporal 3D realizando achatamento dimensional.
    """

    def __init__(self, config):
        super().__init__(config)

        model_cfg = config.get("model", {}).get("lof", {})

        self.n_neighbors = model_cfg.get("n_neighbors", 20)
        self.algorithm = model_cfg.get("algorithm", "auto")
        self.leaf_size = model_cfg.get("leaf_size", 30)
        self.metric = model_cfg.get("metric", "minkowski")
        self.p = model_cfg.get("p", 2)
        self.n_jobs = model_cfg.get("n_jobs", -1)
        
        # Valor estático de contaminação para evitar avisos de deprecabilidade
        self.contamination = model_cfg.get("contamination", 0.1)

        self.model = None

    # =====================================================
    # TREINAMENTO
    # =====================================================

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Suporte para o tensor 3D de janelas temporais
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
            contamination=self.contamination,
            novelty=True  # Habilita a predição/escoragem de dados fora da amostra original de treino
        )

        self.model.fit(X)

    # =====================================================
    # CÁLCULO DE SCORE DE ANOMALIA
    # =====================================================

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Suporte para o tensor 3D de janelas temporais
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        if self.model is None:
            raise ValueError("O modelo LOF deve ser treinado antes do cálculo de scores.")

        # O decision_function do Scikit-Learn retorna valores negativos onde quanto menor, mais anômalo
        scores = self.model.decision_function(X)

        # Inverte o sinal seguindo o contrato BaseModel (maior valor = mais anômalo)
        scores = self._ensure_higher_is_anomaly(scores, invert=True)

        return scores