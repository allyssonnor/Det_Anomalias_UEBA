# core/models/model_factory.py
import re
import importlib
import inspect
import pkgutil
import logging
import core.models
from core.models.base_model import BaseModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}
_DISCOVERED = False

def _to_snake(name: str) -> str:
    """
    Converte CamelCase para snake_case.
    Ex: MLPAutoencoder -> mlp_autoencoder
        IsolationForestModel -> isolation_forest
        LOFModel -> lof
    """
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s).lower()
    return s.replace('_model', '')  # remove o sufixo '_model' se existir

def _discover_models():
    global _DISCOVERED
    if _DISCOVERED:
        return
        
    logger.info("🔍 [ModelFactory] Percorrendo core.models para descoberta automática...")
    
    package_path = getattr(core.models, "__path__", [])
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name in {"base_model", "model_factory"}:
            continue
        
        full_module_name = f"core.models.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModel) and obj is not BaseModel:
                    key = _to_snake(name)
                    if key not in MODEL_REGISTRY:
                        MODEL_REGISTRY[key] = obj
                        logger.debug(f"  ✅ {key} -> {name}")
                    else:
                        logger.warning(f"  ⚠️ Chave '{key}' já registrada. Ignorando {name}.")
        except Exception as e:
            logger.error(f"  ❌ Erro ao carregar {full_module_name}: {e}")
            
    _DISCOVERED = True

# Executa a descoberta no momento da importação
_discover_models()


class ModelFactory:
    @classmethod
    def get_model(cls, config: dict) -> BaseModel:
        model_cfg = config.get("model", {})
        model_name = model_cfg.get("type")
        
        if not model_name or model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"❌ Modelo '{model_name}' não encontrado.\n"
                f"   Modelos disponíveis: {list(MODEL_REGISTRY.keys())}"
            )
        
        model_class = MODEL_REGISTRY[model_name]
        logger.info(f"🤖 Instanciando: {model_name} ({model_class.__name__})")
        model = model_class(config)
        
        # Validação do contrato obrigatório (fit e score)
        # Idealmente isso deveria estar na __init_subclass__ da BaseModel para evitar checagem em runtime
        required = ("fit", "score")
        missing = [m for m in required if not callable(getattr(model, m, None))]
        if missing:
            raise NotImplementedError(
                f"🚨 O modelo '{model_name}' não implementa os métodos obrigatórios: {missing}"
            )
        return model

    @classmethod
    def register(cls, name: str, model_class):
        """Permite registro manual se necessário (útil para testes ou plugins externos)."""
        MODEL_REGISTRY[name] = model_class


def create(config):
    return ModelFactory.get_model(config)
