#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferência em lote com dois modos bem separados:

--mode official     (default)
    Usa os thresholds salvos (user_thresholds.json) do treino.
    Produz 1 linha por modelo por dataset. Esta é a métrica oficial.

--mode sensitivity
    Varre percentis de threshold sobre o próprio dataset alvo.
    É uma ANÁLISE DE SENSIBILIDADE, NÃO uma avaliação confirmatória.
    Produz N linhas por modelo (uma por percentil). Rotuladas como
    evaluation_mode='sensitivity'.

Falha alto em qualquer erro de carregamento de modelo.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import glob
import gc
import pickle
import yaml
import json
import numpy as np
import pandas as pd
import argparse
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
for _ in range(4):
    project_root = os.path.abspath(os.path.join(project_root, ".."))
    if os.path.isdir(os.path.join(project_root, "core")):
        break
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix

from core.data.dataset_adapter import DatasetAdapter
from core.temporal_builder import TemporalBuilder
from core.models.model_factory import create as create_model
from core.trainer import Trainer


def clear_memory():
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass


def _load_model_from_dir(base_model, model_path):
    """Falha alto em erro — nunca retorna base_model sem pesos."""
    candidates = [
        os.path.join(model_path, "model.pkl"),
        os.path.join(model_path, "model.pt"),
        os.path.join(model_path, "model.keras"),
    ]
    target_file = next((c for c in candidates if os.path.exists(c)), None)

    if target_file is None:
        if hasattr(base_model, "load"):
            base_model.load(model_path)
            return base_model
        raise RuntimeError(f"Nenhum peso encontrado em: {model_path}")

    if target_file.endswith('.pkl'):
        with open(target_file, "rb") as pf:
            loaded_obj = pickle.load(pf)
        obj_module = type(loaded_obj).__module__
        if obj_module.startswith(('sklearn', 'xgboost', 'lightgbm', 'catboost')):
            base_model.model = loaded_obj
            return base_model
        elif hasattr(loaded_obj, "score") or hasattr(loaded_obj, "predict"):
            return loaded_obj
        elif isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
            base_model.load_state_dict(loaded_obj)
            return base_model
        else:
            base_model.model = loaded_obj
            return base_model

    elif target_file.endswith('.keras'):
        try:
            base_model.load(model_path)
            return base_model
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar Keras em {target_file}: {e}")

    else:  # torch
        try:
            loaded_obj = torch.load(target_file, map_location=torch.device("cpu"), weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar torch model em {target_file}: {e}")
        if isinstance(loaded_obj, torch.nn.Module):
            return loaded_obj
        elif isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
            base_model.load_state_dict(loaded_obj)
            return base_model
        else:
            raise RuntimeError(f"Formato torch não reconhecido em {target_file}")

def _compute_metrics(y_test, y_pred, scores, eval_mode):
    if y_test is None or len(y_test) == 0:
        return {
            "evaluation_mode": eval_mode, "has_labels": 0,
            "pr_auc": float("nan"), "roc_auc": float("nan"),
            "f1_score": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        }

    if len(np.unique(y_test)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall_curve, precision_curve)
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except ValueError:
            roc_auc = float("nan")
    else:
        pr_auc = float("nan")
        roc_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() if len(y_test) > 0 else (0, 0, 0, 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else float("nan")

    return {
        "evaluation_mode": eval_mode, "has_labels": 1,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "f1_score": f1, "precision": precision, "recall": recall,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    
'''def _compute_metrics(y_test, y_pred, scores, eval_mode):
    if len(np.unique(y_test)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall_curve, precision_curve)
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except ValueError:
            roc_auc = float("nan")
    else:
        pr_auc = float("nan")
        roc_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() if len(y_test) > 0 else (0, 0, 0, 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "evaluation_mode": eval_mode,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
'''

def processar_modelo(model_dir, dataset_path, percentis_teste, mode):
    print(f"\n[{model_dir}] (mode={mode})")

    meta_path = os.path.join(model_dir, "trainer_meta.pkl")
    config_path = os.path.join(model_dir, "config_used.yaml")
    thresholds_path = os.path.join(model_dir, "user_thresholds.json")

    if not os.path.exists(config_path):
        print("  ⚠️ config_used.yaml ausente. Pulando.")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    scalers = meta['scalers']
    feature_columns = meta['feature_columns']

    if 'feature_processor' not in meta:
        print("  ❌ Meta sem 'feature_processor'. Pulando.")
        return []
    processor = meta['feature_processor']
    if not getattr(processor, "is_fitted", False):
        print("  ❌ FeatureProcessor não fitted. Pulando.")
        return []
    processor.strict = True

    # Configura para usar o dataset alvo
    config.setdefault("data", {})
    config["data"]["test_path"] = dataset_path
    config["data"]["train_path"] = ""
    config["data"]["val_path"] = ""
    config["data"]["dataset_path"] = ""
    config.setdefault("sampling", {})["mode"] = "full"

    # Modelo (falha alto em erro)
    base_model = create_model(config)
    model_path = os.path.join(model_dir, "saved_model")
    final_model = _load_model_from_dir(base_model, model_path)

    if hasattr(final_model, "eval"):
        final_model.eval()

    trainer = Trainer(config, final_model)
    trainer.scalers = scalers
    trainer.feature_columns = feature_columns

    # Dataset
    adapter = DatasetAdapter(config)
    _, _, test_raw = adapter.load()

    if test_raw is None or test_raw.empty:
        print("  ❌ Dataset vazio. Pulando.")
        return []

    test_scaled = processor.transform(test_raw)

    if 'temporal_builder' in meta:
        temporal = meta['temporal_builder']
    else:
        temporal = TemporalBuilder(config)

    test_df = temporal.transform(test_scaled)

    X_test, _, users_test, y_test = trainer._build_sequences(
        test_df, fit=False, return_labels=True, feature_columns=feature_columns
    )

    if X_test is None or len(X_test) == 0:
        print("  ⚠️ Sem janelas. Pulando.")
        return []

    scores = trainer._calculate_scores(X_test)
    #if y_test is None:
    #    y_test = np.zeros(len(scores))

    model_preset = config.get("model_presets", [{}])[0].get("name", "Unknown")
    split_strategy = config.get("sampling", {}).get("split_strategy", "Unknown")
    seed = config.get("training", {}).get("seed", "Unknown")
    dataset_tag = os.path.basename(dataset_path)

    base_info = {
        "source_dir": model_dir,
        "model_preset": model_preset,
        "split_strategy": split_strategy,
        "seed": seed,
        "dataset_tag": dataset_tag,
    }

    resultados = []

    if mode == "official":
        # Usa thresholds salvos (oficial, calibrado na validação)
        if not os.path.exists(thresholds_path):
            print("  ❌ user_thresholds.json ausente. Pulando (modo official exige thresholds salvos).")
            return []
        with open(thresholds_path, "r") as f:
            user_thresholds = json.load(f)

        thresh_mode = config.get("training", {}).get("threshold_mode", "per_user")
        global_fb = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)
        thresholds_aplicados = np.array([
            user_thresholds.get(str(u), global_fb)
            if thresh_mode == "per_user" and str(u) != "__GLOBAL_FALLBACK__"
            else global_fb
            for u in users_test
        ])
        y_pred = (scores > thresholds_aplicados).astype(int)

        metrics = _compute_metrics(y_test, y_pred, scores, "official")
        resultados.append({**base_info, "percentile": None, "threshold_value": None, **metrics})
        print(f"  ✅ official: n_alerts={int(np.sum(y_pred))}")

    elif mode == "sensitivity":
        # Varredura de percentis (exploratória)
        for p in percentis_teste:
            threshold = np.percentile(scores, p)
            y_pred = (scores > threshold).astype(int)
            metrics = _compute_metrics(y_test, y_pred, scores, "sensitivity")
            resultados.append({**base_info, "percentile": p, "threshold_value": threshold, **metrics})
        print(f"  ✅ sensitivity: {len(percentis_teste)} percentis avaliados.")

    else:
        raise ValueError(f"mode inválido: {mode}")

    return resultados


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="batch_inference_results.csv")
    parser.add_argument("--mode", type=str, default="official",
                        choices=["official", "sensitivity"],
                        help="official: thresholds salvos (métricas confirmatórias). "
                             "sensitivity: varredura de percentis (exploratória).")
    args = parser.parse_args()

    print("="*70)
    print(f"🚀 INFERÊNCIA EM LOTE | mode={args.mode.upper()}")
    print("="*70)
    print(f"📁 Raiz: {args.root}")
    print(f"🎯 Dataset: {args.dataset}")

    todos_resultados = []
    pastas_processadas = set()
    dataset_tag = os.path.basename(args.dataset)

    if os.path.exists(args.output):
        try:
            df_existente = pd.read_csv(args.output)
            if 'source_dir' in df_existente.columns and 'dataset_tag' in df_existente.columns:
                # Resume: só pula modelos JÁ processados NESTE dataset NESTE modo
                existing = df_existente[
                    (df_existente['dataset_tag'] == dataset_tag)
                    & (df_existente['evaluation_mode'] == args.mode)
                ]
                pastas_processadas = set(existing['source_dir'].unique())
                todos_resultados = df_existente.to_dict('records')
                print(f"🔄 Retomando: {len(pastas_processadas)} modelos já feitos neste dataset/modo.")
        except Exception as e:
            print(f"⚠️ Erro na retomada: {e}")

    pattern = os.path.join(glob.escape(args.root), "**", "trainer_meta.pkl")
    meta_files = glob.glob(pattern, recursive=True)

    if not meta_files:
        print(f"❌ Nenhum modelo em {args.root}.")
        return

    diretorios_modelos = sorted(set(os.path.dirname(f) for f in meta_files))
    print(f"🔍 {len(diretorios_modelos)} modelos encontrados.")

    percentis_teste = list(range(99, 89, -1))

    for i, model_dir in enumerate(diretorios_modelos, 1):
        print(f"\nProgresso: {i}/{len(diretorios_modelos)}")
        if model_dir in pastas_processadas:
            print("  ⏭️ Já processado (mesmo dataset e modo).")
            continue

        try:
            resultados = processar_modelo(model_dir, args.dataset, percentis_teste, args.mode)
            if resultados:
                todos_resultados.extend(resultados)
                pd.DataFrame(todos_resultados).to_csv(args.output, index=False)
                print(f"  💾 CSV atualizado ({len(todos_resultados)} linhas).")
        except Exception as e:
            print(f"  ❌ Erro crítico em {model_dir}: {e}")
            traceback.print_exc()
            print("  ⏭️ Continuando com o próximo modelo.")

        clear_memory()

    if todos_resultados:
        df_final = pd.DataFrame(todos_resultados)
        df_final.to_csv(args.output, index=False)
        print(f"\n✅ CONCLUÍDO — {len(df_final)} linhas em {args.output}")
        print(f"   Filtre por evaluation_mode='official' para métricas confirmatórias.")
    else:
        print("\n⚠️ Nada foi gerado.")


if __name__ == "__main__":
    main()