#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inferência de Alta Fidelidade (Pipeline Oficial)
Garante zero vazamento (leakage) reconstruindo o estado exato do treinamento.
"""
import os
# Silencia logs e avisos verbosos do TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import json
import numpy as np
import yaml
import copy
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.evaluate_performance import run_external_evaluation

from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor
from core.temporal_builder import TemporalBuilder
from core.models.model_factory import create as create_model
from core.trainer import Trainer

# Remove warnings chatos do scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

def rodar_inferencia(config_path, dataset_novo_path):
    print("\n" + "="*70)
    print("🔍 INFERÊNCIA DE ALTA FIDELIDADE E ZERO VAZAMENTO")
    print("="*70)

    # 1. Localiza a Configuração Temporária para achar o diretório raiz
    with open(config_path, "r", encoding="utf-8") as f:
        temp_config = yaml.safe_load(f)

    output_dir = temp_config.get("output_dir", "results/banca_runs")
    meta_path = os.path.join(output_dir, "trainer_meta.pkl")

    if not os.path.exists(meta_path):
        meta_encontrada = False
        for root, dirs, files in os.walk("results"):
            if "trainer_meta.pkl" in files:
                output_dir = root
                meta_path = os.path.join(output_dir, "trainer_meta.pkl")
                meta_encontrada = True
                break
        if not meta_encontrada:
            raise FileNotFoundError(f"❌ O arquivo 'trainer_meta.pkl' não foi localizado. Rode o detectar.py completo primeiro.")

    print(f"📂 Diretório de artefatos recuperado: {output_dir}")

    # 2. Carrega a CONFIGURAÇÃO OFICIAL DO TREINO (Elimina divergência de sequence_length)
    config_used_path = os.path.join(output_dir, "config_used.yaml")
    if os.path.exists(config_used_path):
        with open(config_used_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print("✅ Configuração oficial do treino (config_used.yaml) carregada com sucesso.")
    else:
        print("⚠️ config_used.yaml não encontrado. Usando a configuração inicial.")
        config = temp_config

    config["output_dir"] = output_dir

    # 3. Carrega os Metadados e as Instâncias Treinadas do Pipeline
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        scalers = meta.get('scalers', {})
        feature_columns = meta.get('feature_columns', [])
        processor = meta.get('feature_processor', None)
        temporal = meta.get('temporal_builder', None)

    if processor is None or temporal is None:
        raise ValueError("❌ O FeatureProcessor ou TemporalBuilder não estão no trainer_meta.pkl.")

    print(f"✅ Pipeline de features recuperado: {len(feature_columns)} features estruturais esperadas.")

    # 4. Instancia o Modelo (Wrapper) e injeta os pesos
    if "model" not in config: config["model"] = {}
    if not config["model"].get("type"): config["model"]["type"] = "mlp_supervised"

    model = create_model(config)
    
    saved_model_dir = os.path.join(output_dir, "saved_model")
    weight_file = os.path.join(saved_model_dir, "model.pkl")

    if not os.path.exists(weight_file):
        for f in os.listdir(saved_model_dir):
            if f.endswith(('.pkl', '.pt', '.pth')):
                weight_file = os.path.join(saved_model_dir, f)
                break

    print(f"🔗 Carregando pesos do modelo de: {weight_file}")
    with open(weight_file, "rb") as pf:
        loaded_obj = pickle.load(pf)

    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj
    elif isinstance(loaded_obj, dict) and hasattr(model, "load_state_dict"):
        model.load_state_dict(loaded_obj)
    else:
        # Injeta o classificador scikit-learn dentro do Wrapper Oficial
        injected = False
        for attr in ['model', 'clf', 'estimator', 'classifier', '_model']:
            if hasattr(model, attr):
                setattr(model, attr, loaded_obj)
                injected = True
                break
        if not injected:
            model = loaded_obj

    if hasattr(model, "eval"):
        model.eval()

    # 5. Processamento Limpo do Dataset de Inferência (Zero Vazamento)
    config_infer = copy.deepcopy(config)
    config_infer["data"]["test_path"] = dataset_novo_path
    
    adapter = DatasetAdapter(config_infer)
    _, _, test_raw = adapter.load()

    print("🧠 Transformando dados brutos com o FeatureProcessor oficial...")
    test_scaled = processor.transform(test_raw)

    print("⏳ Construindo sessões temporais com o TemporalBuilder oficial...")
    test_df = temporal.transform(test_scaled)

    # 6. Instancia o Trainer com o Wrapper restaurado e extrai sequências
    trainer = Trainer(config, model)
    trainer.scalers = scalers
    trainer.feature_columns = feature_columns
    
    print("⚙️ Extraindo janelas exatas...")
    X_test, indices_test, users_test, y_test = trainer._build_sequences(test_df, fit=False, return_labels=True)

    if len(X_test) == 0:
        print("⚠️ O dataset não possui eventos suficientes para formar as janelas.")
        return

    # 7. Cálculo de Scores Oficial
    print("🚀 Executando inferência probabilística delegada ao motor oficial...")
    scores = trainer._calculate_scores(X_test)

    # 8. Aplicação Estrita dos Limiares (Global e Por Usuário)
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            thresholds_data = json.load(f)
        fixed_threshold = thresholds_data.get("__GLOBAL_FALLBACK__", 0.5)
        # Preenche os dicionários para o limiar UEBA (Comportamental)
        trainer.user_thresholds = thresholds_data
    else:
        fixed_threshold = config.get("training", {}).get("fixed_threshold", 0.5)
        trainer.user_thresholds = {}
        
    trainer.global_threshold = fixed_threshold

    thresholds_aplicados = np.array([trainer.user_thresholds.get(str(u), fixed_threshold) for u in users_test])
    y_pred = (scores > thresholds_aplicados).astype(int)

    # 9. Geração de Relatórios e Auditoria Oficial
    audit_dir = os.path.join(output_dir, "auditoria_manual")
    os.makedirs(audit_dir, exist_ok=True)

    np.save(os.path.join(audit_dir, "scores.npy"), scores)
    np.save(os.path.join(audit_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(audit_dir, "indices.npy"), indices_test)
    if y_test is not None:
        np.save(os.path.join(audit_dir, "y_true.npy"), y_test)

    eval_config = copy.deepcopy(config)
    eval_config["output_dir"] = audit_dir

    metrics = run_external_evaluation(
        eval_config,
        test_df=test_df,
        scores=scores,
        indices=indices_test,
        verbose=True
    )

    print("\n" + "="*60)
    print("📊 RESULTADOS DA INFERÊNCIA PURA (Zero VAZAMENTO)")
    print("="*60)
    print(f" 🔹 Janelas Avaliadas            : {len(scores)}")
    print(f" 🔹 Anomalias Reais (Gabarito)   : {int(np.sum(y_test)) if y_test is not None else 0}")
    print(f" 🔹 Alertas Disparados           : {int(np.sum(y_pred))}")
    print("-"*60)
    print(f" ✅ Verdadeiros Positivos (TP) : {metrics.get('tp', 0)}")
    print(f" ⚠️  Falsos Positivos (FP)     : {metrics.get('fp', 0)}")
    print(f" ❌ Falsos Negativos (FN)     : {metrics.get('fn', 0)}")
    print(f" 🛡️  Verdadeiros Negativos (TN) : {metrics.get('tn', 0)}")
    print("-"*60)
    print(f" 🎯 Precision                  : {metrics.get('precision', 0.0):.4f}")
    print(f" 🎯 Recall                     : {metrics.get('recall', 0.0):.4f}")
    print(f" 🎯 F1-Score                   : {metrics.get('f1_score', 0.0):.4f}")
    print(f" 🎯 PR-AUC                     : {metrics.get('pr_auc', 0.0):.4f}")
    print(f" 🎯 ROC-AUC                    : {metrics.get('roc_auc', 0.0):.4f}")
    print("="*60 + "\n")

    # =====================================================================
    # 10. Geração do Dashboard para a Banca
    # =====================================================================
    print("📊 Gerando Dashboard Visual...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Avaliação de Inferência - Defesa de Tese (UEBA)', fontsize=18, fontweight='bold', y=0.96)

    # Pegando as métricas do dicionário oficial
    tn = metrics.get('tn', 0)
    fp = metrics.get('fp', 0)
    fn = metrics.get('fn', 0)
    tp = metrics.get('tp', 0)
    pr_auc = metrics.get('pr_auc', 0.0)
    roc_auc = metrics.get('roc_auc', 0.0)
    precision = metrics.get('precision', 0.0)
    recall = metrics.get('recall', 0.0)
    f1 = metrics.get('f1_score', 0.0)

    # Matriz 1: Matriz de Confusão
    ax1 = fig.add_subplot(221)
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, annot_kws={"size": 16, "weight": "bold"})
    ax1.set_title('1. Matriz de Confusão (Inferência)', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(['Normal', 'Alarme (Predito)'], fontsize=11)
    ax1.set_yticklabels(['Normal', 'Anomalia (Real)'], fontsize=11, rotation=90)

    # Gráfico 2: AUCs
    ax2 = fig.add_subplot(222)
    metrics_names = ['PR-AUC', 'ROC-AUC']
    metrics_vals = [pr_auc, roc_auc]
    sns.barplot(x=metrics_names, y=metrics_vals, hue=metrics_names, palette=['#17becf', '#9467bd'], legend=False, ax=ax2)
    ax2.set_title('2. Capacidade Preditiva (AUCs)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score', fontsize=12)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

    # Gráfico 3: Desempenho
    ax3 = fig.add_subplot(212)
    base_metrics = ['Precision', 'Recall', 'F1-Score']
    base_vals = [precision, recall, f1]
    sns.barplot(x=base_metrics, y=base_vals, hue=base_metrics, palette=['#ff7f0e', '#2ca02c', '#1f77b4'], legend=False, ax=ax3)
    ax3.set_title('3. Desempenho Operacional (Precision, Recall, F1)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Score', fontsize=12)
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Salva na raiz da pasta results para a célula do colab encontrar
    os.makedirs("results", exist_ok=True)
    dashboard_path = "results/dashboard_inferencia_banca.png"
    plt.savefig(dashboard_path, dpi=300)
    plt.close()
    
    print(f"✅ Dashboard salvo com sucesso em: {dashboard_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML genérico para apontar root.")
    parser.add_argument("--data", type=str, required=True, help="Caminho do novo dataset a ser processado.")
    args = parser.parse_args()

    rodar_inferencia(args.config, args.data)
