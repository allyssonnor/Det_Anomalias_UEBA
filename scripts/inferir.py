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

def rodar_inferencia(
    config_path,
    dataset_novo_path=None,
    model_run_dir=None,
    cached_test_dir=None,
    inference_output_dir=None,
):
    print("="*70)
    print("🔍 INFERÊNCIA DE ALTA FIDELIDADE E ZERO VAZAMENTO")
    print("="*70)

    if cached_test_dir is None:
        if not dataset_novo_path:
            raise ValueError("Informe --data ou --cached_test_dir.")
        if not os.path.exists(dataset_novo_path):
            raise FileNotFoundError(f"❌ Arquivo de dataset não encontrado: {dataset_novo_path}")
    elif not os.path.isdir(cached_test_dir):
        raise FileNotFoundError(f"❌ Cache de teste não encontrado: {cached_test_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = model_run_dir or config.get("output_dir", "results")
    model_path = os.path.join(output_dir, "saved_model")
    meta_path = os.path.join(output_dir, "trainer_meta.pkl")
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")

    # Busca automática mantida para uso legado; na demo o diretório é explícito.
    if not os.path.exists(meta_path) and model_run_dir is None:
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

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"❌ Artefatos do modelo não encontrados em {output_dir}. "
            "Treine o modelo antes ou informe --model_run_dir correto."
        )
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
        if target_file.endswith(".pkl"):
            with open(target_file, "rb") as pf:
                loaded_obj = pickle.load(pf)

            # Model wrappers save their underlying estimator (for example,
            # sklearn MLPClassifier). Its .score(X) is an accuracy API that
            # requires y, not an anomaly score. Reload through the wrapper.
            if isinstance(loaded_obj, type(base_model)):
                final_model = loaded_obj
                print(f"🤖 Wrapper nativo recuperado: {type(final_model).__name__}")
            elif isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
                base_model.load_state_dict(loaded_obj)
                final_model = base_model
            elif hasattr(base_model, "load"):
                base_model.load(model_path)
                final_model = base_model
                print(f"🤖 Estimador carregado pelo wrapper: {type(base_model).__name__}")
            else:
                final_model = loaded_obj
        else:
            loaded_obj = torch.load(
                target_file, map_location=torch.device("cpu"), weights_only=False
            )
            if isinstance(loaded_obj, torch.nn.Module):
                final_model = loaded_obj
            elif isinstance(loaded_obj, dict) and hasattr(base_model, "load_state_dict"):
                base_model.load_state_dict(loaded_obj)
                final_model = base_model
            elif hasattr(base_model, "load"):
                base_model.load(model_path)
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

    test_df = None
    indices_test = None
    if cached_test_dir:
        print(f"⚡ Carregando janelas de teste pré-processadas: {cached_test_dir}")
        cache_files = {
            "X_test": "X_test.npy",
            "indices_test": "test_indices.npy",
            "users_test": "test_users.npy",
            "y_test": "y_test.npy",
        }
        missing = [
            name for name in cache_files.values()
            if not os.path.isfile(os.path.join(cached_test_dir, name))
        ]
        if missing:
            raise FileNotFoundError(
                "Arquivos ausentes no cache de teste: " + ", ".join(missing)
            )
        X_test = np.load(os.path.join(cached_test_dir, cache_files["X_test"]))
        indices_test = np.load(os.path.join(cached_test_dir, cache_files["indices_test"]))
        users_test = np.load(
            os.path.join(cached_test_dir, cache_files["users_test"]),
            allow_pickle=True,
        )
        y_test = np.load(os.path.join(cached_test_dir, cache_files["y_test"]))
    else:
        print(f"🎯 Processando dataset alvo: {dataset_novo_path}")
        config_infer = config.copy()
        config_infer.setdefault("data", {})
        # O arquivo passado é avaliado como conjunto externo completo; não reutiliza
        # acidentalmente um dataset_path definido na configuração do treinamento.
        config_infer["data"]["dataset_path"] = ""
        config_infer["data"]["test_path"] = dataset_novo_path
        config_infer["data"]["train_path"] = ""
        config_infer["data"]["val_path"] = ""

        adapter = DatasetAdapter(config_infer)
        _, _, test_raw = adapter.load()
        if test_raw is None or test_raw.empty:
            raise ValueError("❌ Os dados extraídos estão vazios.")

        if "feature_processor" in meta:
            processor = meta["feature_processor"]
            print("✅ FeatureProcessor histórico recuperado dos metadados.")
        else:
            processor = FeatureProcessor(config_infer)
            print("⚠️ Aviso: FeatureProcessor recriado do zero (não encontrado no meta).")

        test_scaled = processor.transform(test_raw)
        temporal = meta.get("temporal_builder") or TemporalBuilder(config_infer)
        test_df = temporal.transform(test_scaled)

        print("⚙️ Extraindo janelas temporais...")
        X_test, indices_test, users_test, y_test = trainer._build_sequences(
            test_df,
            fit=False,
            return_labels=True,
            feature_columns=feature_columns,
        )

    if X_test is None or len(X_test) == 0:
        raise ValueError("O conjunto de teste não contém janelas válidas.")

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

    # Salva cada inferência no diretório do respectivo artefato.
    inference_output_dir = inference_output_dir or os.path.join(output_dir, "inference_demo")
    os.makedirs(inference_output_dir, exist_ok=True)
    if test_df is not None:
        df_resultado = test_df.loc[indices_test].copy().reset_index(drop=True)
        df_resultado["anomaly_score"] = scores
        df_resultado["is_alert"] = y_pred
        df_resultado.to_csv(
            os.path.join(inference_output_dir, "resultado_inferencia.csv"),
            index=False,
        )
    np.save(os.path.join(inference_output_dir, "scores.npy"), scores)
    np.save(os.path.join(inference_output_dir, "y_pred.npy"), y_pred)

    if y_test is not None:
        np.save(os.path.join(inference_output_dir, "y_true.npy"), y_test)
        prevalence = float(np.mean(y_test)) if len(y_test) else 0.0
        report = {
            "seed": int(config.get("training", {}).get("seed", 42)),
            "dataset": (
                "Synthetic_Chronological_Demo"
                if str(config.get("sampling", {}).get("split_strategy", "")).casefold()
                in {"chronological", "chrono"}
                else "Synthetic_Random_Demo"
            ),
            "model_preset": (
                "RandomForest"
                if config.get("model", {}).get("type") == "random_forest_supervised"
                else "MLP_Single"
            ),
            "training.threshold_mode": thresh_mode,
            "model.sequence_length": int(config.get("model", {}).get("sequence_length", 20)),
            "n_test_sequences": int(len(y_test)),
            "n_positive": int(np.sum(y_test)),
            "n_alerts": int(np.sum(y_pred)),
            "precision": float(class_1.get("precision", 0.0)),
            "recall": float(class_1.get("recall", 0.0)),
            "f1_score": float(class_1.get("f1-score", 0.0)),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "prevalence": prevalence,
            "lift_pr": float(pr_auc / prevalence) if prevalence > 0 else 0.0,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        with open(
            os.path.join(inference_output_dir, "classification_report.json"),
            "w",
            encoding="utf-8",
        ) as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)
    else:
        raise ValueError("O teste cacheado não contém rótulos para calcular métricas.")

    if y_test is not None:
        print("📊 Gerando Dashboard Visual...")
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Avaliação de Inferência - {report['model_preset']} · "
            f"{report['dataset']} · limiar {thresh_mode}",
            fontsize=16,
            fontweight="bold",
            y=0.96,
        )

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
        dashboard_path = os.path.join(inference_output_dir, "dashboard_inferencia.png")
        plt.savefig(dashboard_path, dpi=300)
        plt.close()
        print(f"✅ Dashboard salvo com sucesso em: {dashboard_path}")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML de configuração do modelo.")
    parser.add_argument("--data", type=str, default=None, help="Dataset externo a ser processado.")
    parser.add_argument("--model_run_dir", type=str, default=None, help="Diretório de artefatos da execução treinada.")
    parser.add_argument("--cached_test_dir", type=str, default=None, help="Cache de janelas de teste produzido por detectar.py.")
    parser.add_argument("--output_dir", type=str, default=None, help="Diretório para salvar métricas, predições e imagem.")
    args = parser.parse_args()

    rodar_inferencia(
        args.config,
        args.data,
        model_run_dir=args.model_run_dir,
        cached_test_dir=args.cached_test_dir,
        inference_output_dir=args.output_dir,
    )
