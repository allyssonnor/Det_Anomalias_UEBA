
from sklearn.ensemble import IsolationForest
import numpy as np
from core.models.base_model import BaseModel

class IsolationForestModel(BaseModel):
    """
    Isolation Forest wrapper otimizado para detecção de anomalias.
    
    Tem a capacidade de processar janelas temporais (3D) do workflow atual.
    """
    def __init__(self, config):
        super().__init__(config)
        model_cfg = config.get("model", {}).get("isolation_forest", {})
        
        # Parâmetros originais para controle total
        self.n_estimators = model_cfg.get("n_estimators", 200)
        self.max_samples = model_cfg.get("max_samples", "auto")
        self.contamination = model_cfg.get("contamination", "auto")
        self.max_features = model_cfg.get("max_features", 1.0)
        self.bootstrap = model_cfg.get("bootstrap", False)
        self.n_jobs = model_cfg.get("n_jobs", -1)
        self.random_state = config.get("training", {}).get("seed", 42)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

    def fit(self, X):
        # Conversão defensiva para numpy
        X = np.asarray(X)
        
        # INTELIGÊNCIA DE PIPELINE: Isolation Forest espera (n_samples, n_features).
        # Se os dados forem 3D (janelas), faz o flattening automático.
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
            
        self.model.fit(X)

    def score(self, X):
        X = np.asarray(X)
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
            
        # sklearn: decision_function retorna valores onde quanto menor, mais anômalo.
        # Nosso contrato exige: scores maiores = MAIS ANOMALIA.
        scores = self.model.decision_function(X)
        
        # Padronização via BaseModel (invert=True para garantir a direção correta)
        return self._ensure_higher_is_anomaly(scores, invert=True)

