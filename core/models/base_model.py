
##### ✅✅✅✅✅ core/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Classe base para todos os modelos de detecção de anomalia.

    CONTRATO OBRIGATÓRIO:
    - fit(X)
    - score(X) → retorna anomaly score (quanto maior, mais anômalo)
    """

    def __init__(self, config):
        self.config = config

    # =====================================================
    # TRAIN
    # =====================================================

    @abstractmethod
    def fit(self, X: np.ndarray):
        """
        Treina o modelo.

        Parâmetros:
            X (np.ndarray): dados de treino (apenas normais idealmente)
        """
        pass

    # =====================================================
    # SCORE
    # =====================================================

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna anomaly scores.

        REGRA GLOBAL (CRÍTICA):
        - scores maiores ⇒ MAIS ANOMALIA

        Isso padroniza todos os modelos e permite:
        - threshold único
        - comparação justa
        - cálculo consistente de PR-AUC

        Retorna:
            np.ndarray shape (n_samples,)
        """
        pass

    # =====================================================
    # PREDICT (PADRONIZAÇÃO)
    # =====================================================

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Garante que qualquer chamada legado usando .predict() 
        retorne exatamente o mesmo que .score(), mantendo a 
        consistência da interface em todo o pipeline.
        """
        return self.score(X)

    # =====================================================
    # OPTIONAL: FIT + SCORE (para modelos que precisam)
    # =====================================================

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        """
        Alguns modelos podem querer treinar e já retornar score.
        (opcional, não obrigatório)
        """
        self.fit(X)
        return self.score(X)

    # =====================================================
    # UTIL: GARANTIR FORMATO DE SAÍDA
    # =====================================================

    def _ensure_1d(self, scores):
        """
        Garante que scores sejam vetor 1D
        """
        scores = np.asarray(scores)

        if scores.ndim > 1:
            scores = scores.reshape(-1)

        return scores

    # =====================================================
    # UTIL: NORMALIZAR DIREÇÃO DO SCORE
    # =====================================================

    def _ensure_higher_is_anomaly(self, scores, invert=False):
        """
        Alguns modelos (ex: sklearn) retornam:
        - valores maiores = mais normal

        Este método padroniza:
        - valores maiores = mais anômalo

        Parâmetros:
            invert (bool): se True, inverte o sinal
        """
        scores = self._ensure_1d(scores)

        if invert:
            scores = -scores

        return scores
