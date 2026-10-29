#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inferência em Lote (Batch Inference) com Varredura de Percentis e Resume.
Varre uma pasta raiz recursivamente, carrega cada modelo treinado, 
aponta para um dataset inédito e avalia o desempenho variando o fixed_percentile.
Possui salvamento incremental e retomada de onde parou.

Patches:
  PATCH 1 — zera data.dataset_path para forçar o uso de test_path.
  PATCH 2 — filtra builders do FeatureProcessor para os que o dataset novo suporta.
"""

import os
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix, classification_report

from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor, FEATURE_REGISTRY
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


def processar_modelo(model_dir, dataset_path, percentis_teste):
    print(f"\n[{model_dir}]")

    meta_path = os.path.join(model_dir, "trainer_meta.pkl")
    config_path = os.path.join(model_dir, "config_used.yaml")

    if not os.path.exists(config_path):
        print("  ⚠️ config_used.yaml não encontrado. Pulando.")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        scalers = meta['scalers']
        feature_columns = meta['feature_columns']

    # 🔥 PATCH 1 — zera dataset_path para forçar o uso do test_path
    config["data"]["test_path"] = dataset_path
    config["data"]["train_path"] = ""
    config["data"]["val_path"] = ""
    config["data"]["dataset_path"] = ""              # ← ESSENCIAL
    config["sampling"]["mode"] = "full"

    # ---- Carregar modelo ----
    base_model = create_model(config)
    target_file = None
    for cand in [
        os.path.join(model_dir, "saved_model", "model.pkl"),
        os.path.join(model_dir, "saved_model", "model.pt"),
        os.path.join(model_dir, "saved_model", "model.keras"),
    ]:
        if os.path.exists(cand):
            target_file = cand
            break

    if not target_file and hasattr(base_model, "load"):
        try:
            base_model.load(os.path.join(model_dir, "saved_model"))
            final_model = base_model
        except Exception:
            print("  ❌ Falha ao carregar pesos do modelo. Pulando.")
            return []
    elif target_file:
        if target_file.endswith('.pkl'):
            with open(target_file, "rb") as pf:
                loaded_obj = pickle.load(pf)
            obj_module = type(loaded_obj).__module__
            if obj_module.startswith(('sklearn', 'xgboost', 'lightgbm', 'catboost')):
                base_model.model = loaded_obj
                final_model = base_model
            elif (hasattr(loaded_obj, "score") or hasattr(loaded_obj, "predict")) and not obj_module.startswith('sklearn'):
                final_model = loaded_obj
            else:
                if isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
                    base_model.load_state_dict(loaded_obj)
                    final_model = base_model
                else:
                    base_model.model = loaded_obj
                    final_model = base_model
        else:
            try:
                loaded_obj = torch.load(target_file, map_location=torch.device("cpu"), weights_only=False)
                if isinstance(loaded_obj, torch.nn.Module):
                    final_model = loaded_obj
                elif isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
                    base_model.load_state_dict(loaded_obj)
                    final_model = base_model
                else:
                    final_model = loaded_obj
            except Exception:
                final_model = base_model
    else:
        print("  ❌ Nenhum arquivo de peso encontrado. Pulando.")
        return []

    if hasattr(final_model, "eval"):
        final_model.eval()

    trainer = Trainer(config, final_model)
    trainer.scalers = scalers
    trainer.feature_columns = feature_columns

    # ---- Carregar dataset novo ----
    adapter = DatasetAdapter(config)
    _, _, test_raw = adapter.load()

    if test_raw is None or test_raw.empty:
        print("  ❌ Dataset extraído está vazio. Pulando.")
        return []

    # ---- Carregar processor ----
    if 'feature_processor' in meta:
        processor = meta['feature_processor']
        if not hasattr(processor, '_is_train_transformed'):
            processor._is_train_transformed = True
        if not hasattr(processor, 'cumulative_state'):
            processor.cumulative_state = {}
    else:
        processor = FeatureProcessor(config)

    # 🔥 PATCH 2 — filtra builders para os que o dataset novo suporta
    available = set(test_raw.columns)
    builders_validos = [
        spec["builder"]
        for name, spec in FEATURE_REGISTRY.items()
        if set(spec["required"]).issubset(available)
    ]
    orig_builders = list(getattr(processor, "active_builders", []))
    processor.active_builders = builders_validos
    processor.features_frozen = True   # impede _detect_features de re-adicionar

    if len(orig_builders) != len(builders_validos):
        removidos = set(orig_builders) - set(builders_validos)
        print(f"  ⚠️ Builders filtrados: {len(orig_builders)} → {len(builders_validos)} "
              f"(removidos: {sorted(removidos)})")

    test_scaled = processor.transform(test_raw)

    if 'temporal_builder' in meta:
        temporal = meta['temporal_builder']
    else:
        temporal = TemporalBuilder(config)

    test_df = temporal.transform(test_scaled)

    X_test, _, _, y_test = trainer._build_sequences(
        test_df, fit=False, return_labels=True, feature_columns=feature_columns
    )

    if X_test is None or len(X_test) == 0:
        print("  ⚠️ Dataset não possui eventos suficientes para formar janelas temporais.")
        return []

    # ---- Inferência ----
    scores = trainer._calculate_scores(X_test)
    if y_test is None:
        y_test = np.zeros(len(scores))

    resultados_modelo = []

    model_type = config.get("model", {}).get("type", "Unknown")
    model_names = {
        "random_forest_supervised": "RandomForest",
        "mlp_single": "MLP_Single",
    }
    model_preset = model_names.get(model_type, model_type)
    split_strategy = config.get("sampling", {}).get("split_strategy", "Unknown")
    seed = config.get("training", {}).get(
        "seed", config.get("seed", config.get("settings", {}).get("seeds", ["Unknown"])[0])
    )
    threshold_mode = config.get("training", {}).get("threshold_mode", "Unknown")
    prevalence = float(np.mean(np.asarray(y_test).astype(int))) if len(y_test) else 0.0
    inference_dataset = os.path.basename(dataset_path)

    for p in percentis_teste:
        threshold = np.percentile(scores, p)
        y_pred = (scores > threshold).astype(int)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall_curve, precision_curve) if len(np.unique(y_test)) > 1 else 0.0
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except ValueError:
            roc_auc = 0.5

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() if len(y_test) > 0 else (0, 0, 0, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        resultados_modelo.append({
            "source_dir": model_dir,
            "inference_dataset": inference_dataset,
            "model_preset": model_preset,
            "threshold_mode": threshold_mode,
            "split_strategy": split_strategy,
            "seed": seed,
            "percentile": p,
            "threshold_value": threshold,
            "pr_auc": pr_auc,
            "lift_pr": (pr_auc / prevalence) if prevalence > 0 else 0.0,
            "prevalence": prevalence,
            "n_windows": int(len(y_test)),
            "n_anomalies_true": int(np.sum(y_test)),
            "n_alerts": int(np.sum(y_pred)),
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    print(f"  ✅ {len(percentis_teste)} percentis avaliados com sucesso.")
    return resultados_modelo


def main():
    parser = argparse.ArgumentParser(description="Inferência em Lote com Varredura de Percentis e Resume")
    parser.add_argument("--root", type=str, default="results", help="Pasta raiz contendo os experimentos")
    parser.add_argument("--dataset", type=str, required=True, help="Caminho do dataset inédito")
    parser.add_argument("--output", type=str, default="batch_inference_results.csv", help="CSV de saída")
    args = parser.parse_args()

    print("="*70)
    print("🚀 INICIANDO INFERÊNCIA EM LOTE (BATCH INFERENCE)")
    print("="*70)
    print(f"📁 Raiz de busca: {args.root}")
    print(f"🎯 Dataset alvo : {args.dataset}")
    if not os.path.isfile(args.dataset) or os.path.getsize(args.dataset) == 0:
        raise FileNotFoundError(f"Dataset alvo não encontrado ou vazio: {args.dataset}")

    todos_resultados = []
    pastas_processadas = set()

    if os.path.exists(args.output):
        try:
            df_existente = pd.read_csv(args.output)
            if "inference_dataset" in df_existente.columns and not df_existente["inference_dataset"].eq(os.path.basename(args.dataset)).all():
                raise ValueError("CSV de retomada pertence a outro dataset; use outro arquivo de saída.")
            if 'source_dir' in df_existente.columns:
                pastas_processadas = set(df_existente['source_dir'].unique())
                todos_resultados = df_existente.to_dict('records')
                print(f"🔄 Retomando: {len(pastas_processadas)} modelos já processados.")
        except Exception as e:
            print(f"⚠️ Erro ao ler CSV de retomada: {e}")

    pattern = os.path.join(glob.escape(args.root), "**", "trainer_meta.pkl")
    ############### pattern = os.path.join(args.root, "**", "trainer_meta.pkl")
    meta_files = glob.glob(pattern, recursive=True)

    if not meta_files:
        print(f"❌ Nenhum modelo encontrado em {args.root}.")
        return

    diretorios_modelos = sorted(set(os.path.dirname(f) for f in meta_files))
    print(f"🔍 {len(diretorios_modelos)} modelos encontrados.")

    percentis_teste = list(range(99, 89, -1))

    for i, model_dir in enumerate(diretorios_modelos, 1):
        print(f"\nProgresso: {i}/{len(diretorios_modelos)}")
        if model_dir in pastas_processadas:
            print("  ⏭️ Já processado.")
            continue

        try:
            resultados = processar_modelo(model_dir, args.dataset, percentis_teste)
            if resultados:
                todos_resultados.extend(resultados)
                pd.DataFrame(todos_resultados).to_csv(args.output, index=False)
                print(f"  💾 CSV mesclado ({len(todos_resultados)} linhas).")
        except Exception as e:
            print(f"  ❌ Erro crítico: {e}")
            traceback.print_exc()

        clear_memory()

    if todos_resultados:
        df_final = pd.DataFrame(todos_resultados)
        df_final.to_csv(args.output, index=False)
        print(f"\n✅ CONCLUÍDO — {len(df_final)} linhas salvas em {args.output}")
    else:
        print("\n⚠️ Nada foi gerado.")


if __name__ == "__main__":
    main()
