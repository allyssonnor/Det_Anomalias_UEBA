
import os
import numpy as np
from sklearn.preprocessing import RobustScaler
import importlib
from core.models.base_model import BaseModel

class EnsembleModel(BaseModel):
    """
    Detetor de Anomalias Híbrido (Esquadrão).
    
    Esta versão opera na estrutura de pastas plana e utiliza o RobustScaler
    para garantir que nenhum modelo domine a votação devido a outliers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # IMPORT DINÂMICO PARA EVITAR RECURSÃO CIRCULAR
        # Como ModelFactory importa Ensemble, Ensemble deve importar Factory dentro do método.
        try:
            module = importlib.import_module("core.models.model_factory")
            ModelFactory = getattr(module, "ModelFactory")
        except (ImportError, AttributeError) as e:
            raise ImportError(f"❌ Erro crítico ao recrutar o Esquadrão: Não foi possível carregar a ModelFactory. {e}")

        ens_cfg = config.get("model", {}).get("ensemble", {})
        model_names = ens_cfg.get("models", [])
        self.weights = ens_cfg.get("weights", [])

        if not model_names:
            raise ValueError("❌ O modelo 'ensemble' requer uma lista de 'models' no config.yaml.")

        # Garantia de pesos equilibrados, caso não definidos
        if not self.weights or len(self.weights) != len(model_names):
            self.weights = [1.0 / len(model_names)] * len(model_names)

        self.models = []
        self.scalers = [] 
        
        print("\n🤝 [ENSEMBLE] Recrutando Esquadrão (Estrutura Plana)...")
        for name in model_names:
            # Cria uma cópia da configuração com alteração apenas do tipo para o sub-modelo
            sub_config = config.copy()
            sub_config["model"] = config["model"].copy()
            sub_config["model"]["type"] = name
            
            try:
                m = ModelFactory.get_model(sub_config)
                self.models.append(m)
                # RobustScaler: Essencial para manter a democracia entre modelos heterogêneos
                self.scalers.append(RobustScaler())
            except Exception as e:
                print(f"⚠️ [ENSEMBLE] Falha ao recrutar sub-modelo '{name}': {e}")

        if not self.models:
            raise RuntimeError("❌ O Esquadrão não conseguiu recrutar nenhum modelo válido.")

    def fit(self, X):
        """
        Treina todos os membros do esquadrão sequencialmente.
        """
        X_arr = np.asarray(X)
        for i, model in enumerate(self.models):
            print(f"🏋️ [ENSEMBLE] Treinando sub-modelo {i+1}/{len(self.models)}...")
            model.fit(X_arr)
            
            # Calibra o scaler do esquadrão com os scores do treino (baseline de normalidade)
            raw_scores = model.score(X_arr)
            self.scalers[i].fit(raw_scores.reshape(-1, 1))

    def score(self, X):
        """
        Combina os votos de todos os modelos usando média ponderada normalizada.
        """
        X_arr = np.asarray(X)
        final_scores = np.zeros(len(X_arr))
        
        for i, model in enumerate(self.models):
            raw_scores = model.score(X_arr)
            
            # Normalização: impede que um erro grande de um modelo
            # anule a opinião dos outros membros do esquadrão.
            scaled_scores = self.scalers[i].transform(raw_scores.reshape(-1, 1)).flatten()
            
            # Aplicação do peso (votação ponderada)
            final_scores += scaled_scores * self.weights[i]
            
        # Garante que a saída siga o contrato: Maior = Mais Anômalo
        return self._ensure_higher_is_anomaly(final_scores)

    def save(self, path):
        """
        O Ensemble delega o salvamento para os seus membros (se implementado).
        """
        for i, model in enumerate(self.models):
            if hasattr(model, "save"):
                sub_path = os.path.join(path, f"sub_model_{i}")
                model.save(sub_path)

    def load(self, path):
        """
        O Ensemble delega o carregamento para os seus membros (se implementado).
        """
        for i, model in enumerate(self.models):
            if hasattr(model, "load"):
                sub_path = os.path.join(path, f"sub_model_{i}")
                model.load(sub_path)

