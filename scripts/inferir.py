#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inferência Consolidado e Isolado: Carrega um modelo fechado, processa um dataset qualquer,
respeitando estritamente o schema de features do treinamento, calcula métricas ao vivo e gera o Dashboard.
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import json
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix
from numpy.lib.stride_tricks import sliding_window_view

from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor
from core.temporal_builder import TemporalBuilder
from core.models.model_factory import create as create_model

def rodar_inferencia(config_path, dataset_novo_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    output_dir = config.get("output_dir", "results/colab_cpu_runs")
    model_path = os.path.join(output_dir, "saved_model")
    meta_path = os.path.join(output_dir, "trainer_meta.pkl")
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")

    if not os.path.exists(meta_path):
        print(f"⚠️ '{meta_path}' não encontrado. Buscando por trainer_meta.pkl em subdiretórios de results/...")
        for root, dirs, files in os.walk("results"):
            if "trainer_meta.pkl" in files:
                output_dir = root
                model_path = os.path.join(output_dir, "saved_model")
                meta_path = os.path.join(output_dir, "trainer_meta.pkl")
                thresholds_path = os.path.join(output_dir, "user_thresholds.json")
                print(f"🔍 Artefatos encontrados automaticamente em: {output_dir}")
                break

    print(f"📂 Carregando artefatos do modelo de: {output_dir}")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        scalers = meta['scalers']
        feature_columns = meta['feature_columns']
        print(f"✅ Schema recuperado: {len(feature_columns)} features esperadas pelo modelo.")

    with open(thresholds_path, "r") as f:
        user_thresholds = json.load(f)
        global_fallback = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)

    model = create_model(config)
    
    search_dirs = [model_path, output_dir]
    target_file = None
    
    for d in search_dirs:
        if os.path.exists(d):
            if os.path.isfile(d):
                target_file = d
                break
            for root, _, files in os.walk(d):
                for file in files:
                    if file.endswith(('.pt', '.pth', '.bin', '.pkl')) and file != 'trainer_meta.pkl':
                        target_file = os.path.join(root, file)
                        break
                if target_file:
                    break
        if target_file:
            break

    if target_file and os.path.exists(target_file):
        print(f"🔗 Encontrado arquivo de modelo em: {target_file}")
        try:
            if target_file.endswith('.pkl'):
                with open(target_file, "rb") as pf:
                    loaded_obj = pickle.load(pf)
            else:
                loaded_obj = torch.load(target_file, map_location=torch.device("cpu"), weights_only=False)

            if isinstance(loaded_obj, torch.nn.Module):
                model = loaded_obj
            elif isinstance(loaded_obj, dict) and hasattr(model, "load_state_dict"):
                model.load_state_dict(loaded_obj)
            else:
                model = loaded_obj
            print("🤖 Modelo carregado com sucesso.")
        except Exception as e:
            try:
                loaded_obj = torch.load(target_file, map_location=torch.device("cpu"), weights_only=False)
                if isinstance(loaded_obj, torch.nn.Module):
                    model = loaded_obj
                elif isinstance(loaded_obj, dict) and hasattr(model, "load_state_dict"):
                    model.load_state_dict(loaded_obj)
                else:
                    model = loaded_obj
                print("🤖 Modelo carregado via torch.load com sucesso.")
            except Exception as e2:
                if hasattr(model, "load"):
                    model.load(model_path)
                else:
                    raise RuntimeError(f"Erro ao carregar pesos {target_file}. Pickle: {e} | Torch: {e2}")
    elif hasattr(model, "load"):
        model.load(model_path)
        print("🤖 Modelo carregado via método .load() nativo.")
    else:
        raise RuntimeError(f"Não foi possível localizar nenhum arquivo de pesos válido em '{model_path}' ou '{output_dir}'.")

    # CARREGAMENTO DO DATASET EXCLUSIVO DA INFERÊNCIA
    config_infer = config.copy()
    config_infer["data"]["test_path"] = dataset_novo_path
    
    adapter = DatasetAdapter(config_infer)
    _, _, test_raw = adapter.load()

    processor = FeatureProcessor(config_infer)
    test_scaled = processor.transform(test_raw)

    temporal = TemporalBuilder(config_infer)
    test_df = temporal.transform(test_scaled)

    sequence_length = config.get("model", {}).get("sequence_length", 20)
    target_col = config.get("data", {}).get("target_col", "Is_Anomaly")
    
    X_all, indices_all, users_all, y_true_all = [], [], [], []
    for user_id, group in test_df.groupby("UserID"):
        scaler = scalers.get(str(user_id), list(scalers.values())[0] if scalers else None)
        
        # ALINHAMENTO ESTRITO COM AS COLUNAS DO TREINAMENTO
        X_num = group.reindex(columns=feature_columns, fill_value=0)
        X_arr = X_num.values.astype(np.float32)
        if scaler:
            X_arr = scaler.transform(X_arr)
            
        if len(X_arr) < sequence_length: continue
        
        windows = sliding_window_view(X_arr, window_shape=sequence_length, axis=0)
        windows = np.swapaxes(windows, 1, 2)
        X_all.append(windows)
        
        idx = group.index.values[sequence_length - 1:]
        indices_all.append(idx)
        users_all.append(np.full(len(idx), str(user_id)))
        
        if target_col in group.columns:
            windows_labels = sliding_window_view(group[target_col].values, window_shape=sequence_length)
            y_true_all.append(np.max(windows_labels, axis=1))

    if not X_all:
        print("⚠️ O dataset novo não possui eventos suficientes para formar janelas temporais.")
        return

    X_test = np.concatenate(X_all)
    indices_test = np.concatenate(indices_all)
    users_test = np.concatenate(users_all)
    y_test = np.concatenate(y_true_all) if y_true_all else np.zeros(len(X_test))

    expected_features = getattr(model, "n_features_in_", None)
    if not isinstance(model, torch.nn.Module):
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        if expected_features is not None and X_test_flat.shape[1] != expected_features:
            if X_test_flat.shape[1] > expected_features:
                X_test_flat = X_test_flat[:, :expected_features]
            else:
                pad_width = expected_features - X_test_flat.shape[1]
                X_test_flat = np.pad(X_test_flat, ((0, 0), (0, pad_width)), mode='constant')
    else:
        X_test_flat = X_test

    print("🚀 Executando cálculo de anomalias no dataset externo...")
    if hasattr(model, "eval"):
        model.eval()
    
    with torch.no_grad():
        if isinstance(model, torch.nn.Module):
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            scores = model(X_tensor).detach().cpu().numpy()
            if scores.ndim > 1:
                scores = scores.squeeze(-1)
        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_flat)
            scores = probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs.squeeze()
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_flat)
        elif hasattr(model, "predict"):
            scores = model.predict(X_test_flat)
        else:
            raise RuntimeError("Modelo sem métodos de inferência suportados.")

    thresh_mode = config.get("training", {}).get("threshold_mode", "per_user")
    thresholds_aplicados = np.array([
        user_thresholds.get(u, global_fallback) if thresh_mode == "per_user" and u != "__GLOBAL_FALLBACK__" else global_fallback 
        for u in users_test
    ])
    
    y_pred = (scores > thresholds_aplicados).astype(int)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(recall_curve, precision_curve) if len(np.unique(y_test)) > 1 else 0.0
    try:
        roc_auc = roc_auc_score(y_test, scores)
    except ValueError:
        roc_auc = 0.5 

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() if len(y_test) > 0 else (0, 0, 0, 0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_1 = report_dict.get("1", report_dict.get("1.0", report_dict.get("True", {})))

    print("\n" + "="*60)
    print("📊 RESULTADOS EXCLUSIVOS DA INFERÊNCIA ATUAL")
    print("="*60)
    print(f" 🔹 Janelas Avaliadas            : {len(y_test)}")
    print(f" 🔹 Anomalias Reais (Gabarito)   : {int(np.sum(y_test))}")
    print(f" 🔹 Alertas Disparados           : {int(np.sum(y_pred))}")
    print("-" * 60)
    print(f" ✅ Verdadeiros Positivos (TP) : {int(tp)}")
    print(f" ⚠️  Falsos Positivos (FP)     : {int(fp)}")
    print(f" ❌ Falsos Negativos (FN)     : {int(fn)}")
    print(f" 🛡️  Verdadeiros Negativos (TN) : {int(tn)}")
    print("-" * 60)
    print(f" 🎯 Precision                  : {class_1.get('precision', 0.0):.4f}")
    print(f" 🎯 Recall                     : {class_1.get('recall', 0.0):.4f}")
    print(f" 🎯 F1-Score                   : {class_1.get('f1-score', 0.0):.4f}")
    print(f" 🎯 PR-AUC                     : {pr_auc:.4f}")
    print(f" 🎯 ROC-AUC                    : {roc_auc:.4f}")
    print("="*60 + "\n")

    os.makedirs("results", exist_ok=True)
    df_resultado = test_df.loc[indices_test].copy().reset_index(drop=True)
    df_resultado["anomaly_score"] = scores
    df_resultado["is_alert"] = y_pred
    df_resultado.to_csv("results/resultado_inferencia.csv", index=False)
    np.save("results/y_pred_novo.npy", scores)
    np.save("results/y_true_novo.npy", y_test)

    # Geração exclusiva do Dashboard Atual
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Avaliação de Inferência - Defesa de Tese (UEBA)', fontsize=18, fontweight='bold', y=0.96)

    ax1 = fig.add_subplot(221)
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, annot_kws={"size": 16, "weight": "bold"})
    ax1.set_title('1. Matriz de Confusão (Inferência)', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(['Normal', 'Alarme (Predito)'], fontsize=11)
    ax1.set_yticklabels(['Normal', 'Anomalia (Real)'], fontsize=11, rotation=90)

    ax2 = fig.add_subplot(222)
    metrics_names = ['PR-AUC', 'ROC-AUC']
    metrics_vals = [pr_auc, roc_auc]
    sns.barplot(x=metrics_names, y=metrics_vals, hue=metrics_names, palette=['#17becf', '#9467bd'], legend=False, ax=ax2)
    ax2.set_title('2. Capacidade Preditiva (AUCs)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score', fontsize=12)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

    ax3 = fig.add_subplot(212)
    base_metrics = ['Precision', 'Recall', 'F1-Score']
    base_vals = [class_1.get('precision', 0.0), class_1.get('recall', 0.0), class_1.get('f1-score', 0.0)]
    sns.barplot(x=base_metrics, y=base_vals, hue=base_metrics, palette=['#ff7f0e', '#2ca02c', '#1f77b4'], legend=False, ax=ax3)
    ax3.set_title('3. Desempenho Operacional (Precision, Recall, F1)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Score', fontsize=12)
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    dashboard_path = "results/dashboard_inferencia_banca.png"
    plt.savefig(dashboard_path, dpi=300)
    plt.close()

    print(f"💾 Relatório exportado: results/resultado_inferencia.csv")
    print(f"📊 Dashboard gerado com sucesso em: {dashboard_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML de configuração do modelo.")
    parser.add_argument("--data", type=str, required=True, help="Caminho do novo dataset a ser processado.")
    args = parser.parse_args()

    rodar_inferencia(args.config, args.data)
