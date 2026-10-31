#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferência consolidada com integridade estrita de artefatos.

- Falha alto se o meta não contiver feature_processor fitted.
- Falha alto se user_thresholds.json não tiver __GLOBAL_FALLBACK__.
- Falha alto em qualquer erro de carregamento de modelo (nunca usa
  base_model sem pesos como fallback).
- Caminho do projeto corrigido (2 níveis acima).
- Métricas supervisionadas = NaN quando não há rótulos (não inventa).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
for _ in range(4):
    project_root = os.path.abspath(os.path.join(project_root, ".."))
    if os.path.isdir(os.path.join(project_root, "core")):
        break
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import copy
import pickle
import json
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
)

from core.data.dataset_adapter import DatasetAdapter
from core.temporal_builder import TemporalBuilder
from core.models.model_factory import create as create_model
from core.trainer import Trainer


def _load_model_from_dir(base_model, model_path):
    """
    Carrega pesos do modelo. Falha ALTO em erro (sem fallback silencioso).
    Suporta .pkl (sklearn/wrapper), .keras e .pt (torch).
    """
    candidates = [
        os.path.join(model_path, "model.pkl"),
        os.path.join(model_path, "model.pt"),
        os.path.join(model_path, "model.keras"),
    ]
    target_file = next((c for c in candidates if os.path.exists(c)), None)

    if target_file is None:
        if hasattr(base_model, "load"):
            try:
                base_model.load(model_path)
                print(f"🤖 Modelo carregado via .load() nativo")
                return base_model
            except Exception as e:
                raise RuntimeError(f"Falha em base_model.load({model_path}): {e}")
        raise RuntimeError(f"Nenhum arquivo de pesos encontrado em: {model_path}")

    print(f"🔗 Carregando pesos de: {target_file}")

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
            raise RuntimeError(f"Falha ao carregar modelo Keras em {target_file}: {e}")

    else:  # .pt / .pth (torch)
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
            raise RuntimeError(f"Formato torch não reconhecido em {target_file}: {type(loaded_obj)}")


def rodar_inferencia(config_path, dataset_novo_path, model_dir_override=None):
    print("=" * 70)
    print("🔍 INFERÊNCIA DE ALTA FIDELIDADE E ZERO VAZAMENTO")
    print("=" * 70)

    if not os.path.exists(dataset_novo_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {dataset_novo_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if model_dir_override:
        output_dir = model_dir_override
    else:
        output_dir = config.get("output_dir", "results")

    model_path = os.path.join(output_dir, "saved_model")
    meta_path = os.path.join(output_dir, "trainer_meta.pkl")
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")

    if not os.path.exists(meta_path) and not model_dir_override:
        print("⚠️ meta não encontrado no caminho direto. Buscando execução mais recente...")
        pasta_raiz = config.get("settings", {}).get("results_root", "results")
        encontrados = []
        for root, dirs, files in os.walk(pasta_raiz):
            if "trainer_meta.pkl" in files:
                encontrados.append(root)
        if encontrados:
            encontrados.sort(reverse=True)
            output_dir = encontrados[0]
            model_path = os.path.join(output_dir, "saved_model")
            meta_path = os.path.join(output_dir, "trainer_meta.pkl")
            thresholds_path = os.path.join(output_dir, "user_thresholds.json")
            print(f"✅ Selecionando o mais novo: {output_dir}")
        else:
            raise FileNotFoundError(f"❌ 'trainer_meta.pkl' não encontrado em {pasta_raiz}.")

    print(f"📂 Diretório de artefatos: {output_dir}")

    config_used_path = os.path.join(output_dir, "config_used.yaml")
    if os.path.exists(config_used_path):
        with open(config_used_path, "r") as f:
            config = yaml.safe_load(f)
        print("✅ config_used.yaml carregado.")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    scalers = meta['scalers']
    feature_columns = meta['feature_columns']
    print(f"✅ Pipeline de features: {len(feature_columns)} features.")

    # ---------- Thresholds: exigir __GLOBAL_FALLBACK__ ----------
    with open(thresholds_path, "r") as f:
        user_thresholds = json.load(f)

    if "__GLOBAL_FALLBACK__" not in user_thresholds:
        raise RuntimeError(
            f"user_thresholds.json sem __GLOBAL_FALLBACK__ em {output_dir}. "
            "Artefato incompatível. Não inventando threshold."
        )

    # ---------- FeatureProcessor: obrigatório, estrito ----------
    if 'feature_processor' not in meta:
        raise RuntimeError(
            "❌ Meta não contém 'feature_processor'. Retreine com versão atual "
            "do pipeline (detectar.py injeta post-hoc; ou anexar via "
            "trainer.feature_processor antes de run())."
        )
    processor = meta['feature_processor']
    if not getattr(processor, "is_fitted", False):
        raise RuntimeError("❌ FeatureProcessor do meta não está fitted. Artefato corrompido.")
    processor.strict = True
    print("✅ FeatureProcessor recuperado (fitted, strict=True).")

    # ---------- Modelo: falha alto em erro ----------
    base_model = create_model(config)
    final_model = _load_model_from_dir(base_model, model_path)

    trainer = Trainer(config, final_model)
    trainer.scalers = scalers
    trainer.feature_columns = feature_columns
    trainer.user_thresholds = user_thresholds
    trainer.global_threshold = user_thresholds["__GLOBAL_FALLBACK__"]

    # ---------- Dataset ----------
    config_infer = copy.deepcopy(config)
    config_infer.setdefault("data", {})["test_path"] = dataset_novo_path
    config_infer["data"]["train_path"] = ""
    config_infer["data"]["val_path"] = ""
    config_infer["data"]["dataset_path"] = ""
    config_infer.setdefault("sampling", {})["mode"] = "full"

    adapter = DatasetAdapter(config_infer)
    _, _, test_raw = adapter.load()

    if test_raw is None or test_raw.empty:
        raise ValueError("❌ Dados extraídos vazios.")

    test_scaled = processor.transform(test_raw)

    if 'temporal_builder' in meta:
        temporal = meta['temporal_builder']
    else:
        temporal = TemporalBuilder(config_infer)

    test_df = temporal.transform(test_scaled)

    X_test, indices_test, users_test, y_test = trainer._build_sequences(
        test_df, fit=False, return_labels=True, feature_columns=feature_columns
    )

    if X_test is None or len(X_test) == 0:
        print("⚠️ Dataset sem janelas temporais suficientes.")
        return

    if hasattr(final_model, "eval"):
        final_model.eval()

    scores = trainer._calculate_scores(X_test)

    thresh_mode = config.get("training", {}).get("threshold_mode", "per_user")
    thresholds_aplicados = np.array([
        trainer.user_thresholds.get(str(u), trainer.global_threshold)
        if thresh_mode == "per_user" and str(u) != "__GLOBAL_FALLBACK__"
        else trainer.global_threshold
        for u in users_test
    ])

    y_pred = (scores > thresholds_aplicados).astype(int)

    # ---------- Métricas (NaN quando indefinidas) ----------
    has_labels = y_test is not None and len(y_test) > 0
    tn = fp = fn = tp = 0
    pr_auc = float("nan")
    roc_auc = float("nan")
    class_1 = {"precision": float("nan"), "recall": float("nan"), "f1-score": float("nan")}

    if has_labels and len(np.unique(y_test)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall_curve, precision_curve)
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except ValueError:
            roc_auc = float("nan")
    elif has_labels:
        print("⚠️ Apenas uma classe nos rótulos. PR-AUC e ROC-AUC indefinidos (NaN).")
    else:
        print("ℹ️ Dataset sem rótulos. Apenas scores e alertas serão emitidos.")

    if has_labels:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() if len(y_test) > 0 else (0, 0, 0, 0)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        class_1 = report_dict.get("1", report_dict.get("1.0", report_dict.get("True", {})))

        print("\n" + "=" * 60)
        print("📊 RESULTADOS (Zero VAZAMENTO)")
        print("=" * 60)
        print(f" 🔹 Janelas: {len(y_test)} | Anomalias: {int(np.sum(y_test))} | Alertas: {int(np.sum(y_pred))}")
        print(f" ✅ TP={int(tp)} ⚠️ FP={int(fp)} ❌ FN={int(fn)} 🛡️ TN={int(tn)}")
        print(f" 🎯 Precision={class_1.get('precision', 0.0):.4f} Recall={class_1.get('recall', 0.0):.4f} F1={class_1.get('f1-score', 0.0):.4f}")
        print(f" 🎯 PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f}")
        print("=" * 60 + "\n")

    # ---------- CSV de eventos ----------
    dataset_tag = os.path.basename(dataset_novo_path).replace(".jsonl", "")
    out_csv = os.path.abspath(os.path.join(output_dir, f"inferencia_{dataset_tag}.csv"))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df_resultado = test_df.loc[indices_test].copy().reset_index(drop=True)
    df_resultado["anomaly_score"] = scores
    df_resultado["is_alert"] = y_pred
    df_resultado.to_csv(out_csv, index=False)
    print(f"✅ CSV de eventos: {out_csv}")

    # ---------- CSV de métricas ----------
    metrics_csv = os.path.abspath(os.path.join(output_dir, f"metricas_{dataset_tag}.csv"))
    pd.DataFrame([{
        "model_dir": output_dir,
        "dataset": dataset_tag,
        "evaluation_mode": "official",
        "n_windows": int(len(y_test)) if has_labels else int(len(scores)),
        "n_anomalies_true": int(np.sum(y_test)) if has_labels else 0,
        "n_alerts": int(np.sum(y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "has_labels": int(has_labels),
        "precision": float(class_1.get("precision", float("nan"))),
        "recall": float(class_1.get("recall", float("nan"))),
        "f1_score": float(class_1.get("f1-score", float("nan"))),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
    }]).to_csv(metrics_csv, index=False)
    print(f"✅ CSV de métricas: {metrics_csv}")

    # ---------- Dashboard (só com rótulos) ----------
    if has_labels:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('Avaliação de Inferência (UEBA)', fontsize=18, fontweight='bold', y=0.96)

        ax1 = fig.add_subplot(221)
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, annot_kws={"size": 16, "weight": "bold"})
        ax1.set_title('1. Matriz de Confusão', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(['Normal', 'Alarme'], fontsize=11)
        ax1.set_yticklabels(['Normal', 'Anomalia'], fontsize=11, rotation=90)

        ax2 = fig.add_subplot(222)
        metric_vals = [
            0.0 if np.isnan(pr_auc) else pr_auc,
            0.0 if np.isnan(roc_auc) else roc_auc,
        ]
        sns.barplot(x=['PR-AUC', 'ROC-AUC'], y=metric_vals,
                    hue=['PR-AUC', 'ROC-AUC'], palette=['#17becf', '#9467bd'], legend=False, ax=ax2)
        ax2.set_title('2. AUCs', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

        ax3 = fig.add_subplot(212)
        base_vals = [
            0.0 if np.isnan(class_1.get('precision', float('nan'))) else class_1.get('precision', 0.0),
            0.0 if np.isnan(class_1.get('recall', float('nan'))) else class_1.get('recall', 0.0),
            0.0 if np.isnan(class_1.get('f1-score', float('nan'))) else class_1.get('f1-score', 0.0),
        ]
        sns.barplot(x=['Precision', 'Recall', 'F1'], y=base_vals,
                    hue=['Precision', 'Recall', 'F1'], palette=['#ff7f0e', '#2ca02c', '#1f77b4'], legend=False, ax=ax3)
        ax3.set_title('3. Desempenho Operacional', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        dashboard_path = os.path.join(output_dir, f"dashboard_{dataset_tag}.png")
        plt.savefig(dashboard_path, dpi=300)
        plt.close()
        print(f"✅ Dashboard: {dashboard_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()
    rodar_inferencia(args.config, args.data, args.model_dir)