import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def run_external_evaluation(config, test_df=None, scores=None, indices=None, verbose=False):
    output_dir = config.get("output_dir", "./output")
    target_col = config.get("data", {}).get("target_col", "Is_Anomaly")
    sequence_length = config.get("model", {}).get("sequence_length", 10)
    thresh_mode = config.get("training", {}).get("threshold_mode", "per_user")

    # 1. Usa arrays da memória
    if scores is None or indices is None:
        scores = np.load(os.path.join(output_dir, "test_scores.npy"))
        indices = np.load(os.path.join(output_dir, "test_indices.npy"))

    # 2. Alinhamento de Rótulos
    y_test_list = []
    for user_id, group in test_df.groupby("UserID"):
        if len(group) < sequence_length: continue
        windows_labels = sliding_window_view(group[target_col].values, window_shape=sequence_length)
        y_test_list.append(np.max(windows_labels, axis=1))

    y_test = np.concatenate(y_test_list) if y_test_list else np.array([])
    if len(y_test) != len(scores):
        min_len = min(len(y_test), len(scores))
        y_test, scores = y_test[:min_len], scores[:min_len]
    y_test = np.nan_to_num(y_test, nan=0).astype(int)

    # 3. Limiares
    user_thresholds = {}
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f: user_thresholds = json.load(f)

    global_fallback = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)
    user_mapping = test_df["UserID"].to_dict()
    test_users = [str(user_mapping.get(idx, "unknown")) for idx in indices]

    sample_thresholds = np.array([
        user_thresholds.get(u, global_fallback) if thresh_mode == "per_user" and u != "__GLOBAL_FALLBACK__" else global_fallback 
        for u in test_users
    ])
    y_pred = (scores > sample_thresholds).astype(int)

    # 4. Cálculo de Métricas
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(recall_curve, precision_curve) if len(np.unique(y_test)) > 1 else 0.0
    try:
        roc_auc = roc_auc_score(y_test, scores)
    except ValueError:
        roc_auc = 0.5 

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() if len(y_test) > 0 else (0, 0, 0, 0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_1 = report_dict.get("1", report_dict.get("1.0", report_dict.get("True", {})))

    metrics_dict = {
        "pr_auc": float(pr_auc), "roc_auc": float(roc_auc),
        "f1_score": float(class_1.get("f1-score", 0.0)),
        "precision": float(class_1.get("precision", 0.0)),
        "recall": float(class_1.get("recall", 0.0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # 5. Dashboard Elegante no Terminal (Somente se verbose=True)
    if verbose:
        total_events = len(test_df)
        anom_events = int(test_df[target_col].sum())
        prev_events = (anom_events / total_events * 100) if total_events > 0 else 0.0

        total_windows = len(y_test)
        anom_windows = int(np.sum(y_test))
        prev_windows = (anom_windows / total_windows * 100) if total_windows > 0 else 0.0

        print("\n" + "="*60)
        print("📊 RESUMO DA AUDITORIA DE DETECÇÃO (DOUBLE CHECK)")
        print("="*60)
        print(f" 🔹 Eventos Totais (Brutos)      : {total_events}")
        print(f" 🔹 Eventos Anômalos (Brutos)    : {anom_events} (Prevalência: {prev_events:.2f}%)")
        print("-" * 60)
        print(f" 🔹 Janelas Avaliadas (Seq={sequence_length})   : {total_windows}")
        print(f" 🔹 Janelas Anômalas (Gabarito)  : {anom_windows} (Prevalência: {prev_windows:.2f}%)")
        print(f" 🔹 Alertas Disparados           : {int(np.sum(y_pred))}")
        print("-" * 60)
        print(f" ✅ Verdadeiros Positivos (TP) : {int(tp)}")
        print(f" ⚠️  Falsos Positivos (FP)     : {int(fp)}")
        print(f" ❌ Falsos Negativos (FN)     : {int(fn)}")
        print(f" 🛡️  Verdadeiros Negativos (TN) : {int(tn)}")
        print("-" * 60)
        print(f" 🎯 Precision (Precisão)      : {metrics_dict['precision']:.4f}")
        print(f" 🎯 Recall (Sensibilidade)    : {metrics_dict['recall']:.4f}")
        print(f" 🎯 F1-Score                  : {metrics_dict['f1_score']:.4f}")
        print(f" 🎯 PR-AUC                    : {metrics_dict['pr_auc']:.4f}")
        print(f" 🎯 ROC-AUC                   : {metrics_dict['roc_auc']:.4f}")
        print("="*60 + "\n")
        
    return metrics_dict
