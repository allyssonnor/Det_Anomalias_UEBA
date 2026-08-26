#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inferência Consolidado (Tese)
Carrega o modelo salvo, restaura os metadados e limiares, aplica o pipeline oficial 
e gera o dashboard de resultados sem vazamento de dados.
"""
import os
# Supressão de logs incômodos do TensorFlow/CUDA
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
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix

from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor
from core.temporal_builder import TemporalBuilder
from core.models.model_factory import create as create_model
from core.trainer import Trainer

def rodar_inferencia(config_path, dataset_novo_path):
    print("="*70)
    print("🔍 INFERÊNCIA DE ALTA FIDELIDADE E ZERO VAZAMENTO")
    print("="*70)
    
    if not os.path.exists(dataset_novo_path):
        raise FileNotFoundError(f"❌ Arquivo de dataset não encontrado: {dataset_novo_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    output_dir = config.get("output_dir", "results")
    model_path = os.path.join(output_dir, "saved_model")
    meta_path = os.path.join(output_dir, "trainer_meta.pkl")
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")

    # Fallback de busca caso o output_dir direto não tenha os arquivos
    if not os.path.exists(meta_path):
        meta_encontrada = False
        for root, dirs, files in os.walk(config.get("settings", {}).get("results_root", "results")):
            if "trainer_meta.pkl" in files:
                output_dir = root
                model_path = os.path.join(output_dir, "saved_model")
                meta_path = os.path.join(output_dir, "trainer_meta.pkl")
                thresholds_path = os.path.join(output_dir, "user_thresholds.json")
                meta_encontrada = True
                break
        
        if not meta_encontrada:
            raise FileNotFoundError("❌ O arquivo de metadados 'trainer_meta.pkl' não foi encontrado. O modelo concluiu o treino?")

    print(f"📂 Diretório de artefatos recuperado: {output_dir}")

    config_used_path = os.path.join(output_dir, "config_used.yaml")
    if os.path.exists(config_used_path):
        print(f"✅ Configuração oficial do treino (config_used.yaml) carregada com sucesso.")
        with open(config_used_path, "r") as f:
            config = yaml.safe_load(f)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        scalers = meta['scalers']
        feature_columns = meta['feature_columns']
        print(f"✅ Pipeline de features recuperado: {len(feature_columns)} features esperadas.")

    with open(thresholds_path, "r") as f:
        user_thresholds = json.load(f)

    base_model = create_model(config)
    target_file = None
    
    search_candidates = [os.path.join(model_path, "model.pkl"), os.path.join(model_path, "model.pt")]
    for candidate in search_candidates:
        if os.path.exists(candidate):
            target_file = candidate
            break

    if target_file:
        print(f"🔗 Carregando pesos do modelo de: {target_file}")
        if target_file.endswith('.pkl'):
            with open(target_file, "rb") as pf:
                loaded_obj = pickle.load(pf)
                
            # --- ANTI-MATRIOSKA ABSOLUTO ---
            # Se o objeto carregado já for o nosso Wrapper completo, usamos ele e descartamos o base_model!
            if hasattr(loaded_obj, "score") or hasattr(loaded_obj, "predict"):
                final_model = loaded_obj
                print(f"🤖 Wrapper nativo blindado recuperado: {type(final_model).__name__}")
            else:
                # Fallback para PyTorch (se for apenas state_dict)
                if isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
                    base_model.load_state_dict(loaded_obj)
                    final_model = base_model
                else:
                    final_model = loaded_obj
        else:
            loaded_obj = torch.load(target_file, map_location=torch.device("cpu"), weights_only=False)
            if isinstance(loaded_obj, torch.nn.Module):
                final_model = loaded_obj
            elif isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
                base_model.load_state_dict(loaded_obj)
                final_model = base_model
            else:
                final_model = loaded_obj
    elif hasattr(base_model, "load"):
        base_model.load(model_path)
        final_model = base_model
        print("🤖 Modelo carregado via método .load() nativo.")
    else:
        raise RuntimeError(f"❌ Não foi possível encontrar pesos válidos em: {model_path}")

    # Inicializa o Trainer oficial delegando tudo a ele
    trainer = Trainer(config, final_model)
    trainer.scalers = scalers
    trainer.feature_columns = feature_columns
    trainer.user_thresholds = user_thresholds
    trainer.global_threshold = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)

    print(f"🎯 Forçando leitura do novo dataset alvo: {dataset_novo_path}")
    config_infer = config.copy()
    config_infer["data"]["test_path"] = dataset_novo_path
    config_infer["data"]["train_path"] = "" 
    config_infer["data"]["val_path"] = ""
    
    adapter = DatasetAdapter(config_infer)
    _, _, test_raw = adapter.load()
    
    if test_raw is None or test_raw.empty:
        raise ValueError("❌ Os dados extraídos estão vazios.")

    # NOVO: Recupera o estado histórico do processador para garantir os mesmos cálculos do treino
    if 'feature_processor' in meta:
        processor = meta['feature_processor']
        print("✅ FeatureProcessor histórico recuperado dos metadados (Zero Leakage garantido).")
    else:
        processor = FeatureProcessor(config_infer)
        print("⚠️ Aviso: FeatureProcessor recriado do zero (não encontrado no meta).")

    test_scaled = processor.transform(test_raw)

    if 'temporal_builder' in meta:
        temporal = meta['temporal_builder']
    else:
        temporal = TemporalBuilder(config_infer)
        
    test_df = temporal.transform(test_scaled)

    print("⚙️ Extraindo janelas exatas...")
    X_test, indices_test, users_test, y_test = trainer._build_sequences(test_df, fit=False, return_labels=True, feature_columns=feature_columns)

    if X_test is None or len(X_test) == 0:
        print("⚠️ O dataset novo não possui eventos suficientes para formar janelas temporais.")
        return

    print("🚀 Executando inferência probabilística delegada ao motor oficial...")
    if hasattr(final_model, "eval"):
        final_model.eval()

    # Delegação pura: O Trainer calcula os scores usando o wrapper correto
    scores = trainer._calculate_scores(X_test)

    thresh_mode = config.get("training", {}).get("threshold_mode", "per_user")
    thresholds_aplicados = np.array([
        trainer.user_thresholds.get(str(u), trainer.global_threshold) if thresh_mode == "per_user" and str(u) != "__GLOBAL_FALLBACK__" else trainer.global_threshold 
        for u in users_test
    ])
    
    y_pred = (scores > thresholds_aplicados).astype(int)
    
    if y_test is not None:
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
        print("📊 RESULTADOS DA INFERÊNCIA PURA (Zero VAZAMENTO)")
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

    # Salva os resultados CSV / NPY
    os.makedirs("results", exist_ok=True)
    df_resultado = test_df.loc[indices_test].copy().reset_index(drop=True)
    df_resultado["anomaly_score"] = scores
    df_resultado["is_alert"] = y_pred
    df_resultado.to_csv("results/resultado_inferencia.csv", index=False)
    np.save("results/y_pred_novo.npy", scores)
    
    if y_test is not None:
        np.save("results/y_true_novo.npy", y_test)

    if y_test is not None:
        print("📊 Gerando Dashboard Visual...")
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('Avaliação de Inferência - Defesa de Tese (UEBA)', fontsize=18, fontweight='bold', y=0.96)

        ax1 = fig.add_subplot(221)
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, annot_kws={"size": 16, "weight": "bold"})
        ax1.set_title('1. Matriz de Confusão', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(['Normal', 'Alarme (Predito)'], fontsize=11)
        ax1.set_yticklabels(['Normal', 'Anomalia (Real)'], fontsize=11, rotation=90)

        ax2 = fig.add_subplot(222)
        metrics_names = ['PR-AUC', 'ROC-AUC']
        metrics_vals = [pr_auc, roc_auc]
        sns.barplot(x=metrics_names, y=metrics_vals, hue=metrics_names, palette=['#17becf', '#9467bd'], legend=False, ax=ax2)
        ax2.set_title('2. Capacidade Preditiva (AUCs)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

        ax3 = fig.add_subplot(212)
        base_metrics = ['Precision', 'Recall', 'F1-Score']
        base_vals = [class_1.get('precision', 0.0), class_1.get('recall', 0.0), class_1.get('f1-score', 0.0)]
        sns.barplot(x=base_metrics, y=base_vals, hue=base_metrics, palette=['#ff7f0e', '#2ca02c', '#1f77b4'], legend=False, ax=ax3)
        ax3.set_title('3. Desempenho Operacional', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        dashboard_path = "results/dashboard_inferencia_banca.png"
        plt.savefig(dashboard_path, dpi=300)
        plt.close()
        print(f"✅ Dashboard salvo com sucesso em: {dashboard_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML de configuração do modelo.")
    parser.add_argument("--data", type=str, required=True, help="Caminho do novo dataset a ser processado.")
    args = parser.parse_args()

    rodar_inferencia(args.config, args.data)
