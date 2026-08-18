# scripts/evaluate_performance.py
# VERSÃO COM CÁLCULO DE ROC-AUC INTEGRADO

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def run_external_evaluation(config, test_df=None):
    output_dir = config.get("output_dir", "./output")
    target_col = config.get("data", {}).get("target_col", "Is_Anomaly")
    sequence_length = config.get("model", {}).get("sequence_length", 10)
    
    train_cfg = config.get("training", {})
    thresh_mode = train_cfg.get("threshold_mode", "per_user")
    fixed_threshold = train_cfg.get("fixed_threshold", None)
    fixed_percentile = train_cfg.get("fixed_percentile", None)

    scores_path = os.path.join(output_dir, "test_scores.npy")
    indices_path = os.path.join(output_dir, "test_indices.npy")
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")

    # Verificação de segurança de existência dos escores gerados pelo Trainer
    if not os.path.exists(scores_path) or not os.path.exists(indices_path):
        print("❌ Erro Crítico: Arquivos de scores/indices de teste não encontrados no diretório!")
        return

    scores = np.load(scores_path)
    indices = np.load(indices_path)

    # ---------------------------------------------------------
    # 1. ALINHAMENTO DE RÓTULOS (NP.MAX)
    # ---------------------------------------------------------
    y_test_list = []
    for user_id, group in test_df.groupby("UserID"):
        if len(group) < sequence_length:
            continue
        windows_labels = sliding_window_view(group[target_col].values, window_shape=sequence_length)
        max_labels = np.max(windows_labels, axis=1)
        y_test_list.append(max_labels)

    if len(y_test_list) > 0:
        y_test = np.concatenate(y_test_list)
    else:
        y_test = np.array([])

    # Salvaguarda de consistência dimensional para evitar travamento do script de maratona
    if len(y_test) != len(scores):
        print(f"⚠️  Aviso: Desalinhamento detectado na avaliação! len(y_test)={len(y_test)} vs len(scores)={len(scores)}")
        min_len = min(len(y_test), len(scores))
        y_test = y_test[:min_len]
        scores = scores[:min_len]

    # Força os rótulos reais a serem inteiros (evita que chaves de floats como "1.0" quebrem o JSON)
    try:
        y_test = np.nan_to_num(y_test, nan=0).astype(int)
    except Exception as e:
        print(f"⚠️  Não foi possível converter y_test para inteiro: {e}")

    # ---------------------------------------------------------
    # 2. CARREGAMENTO E CONFIGURAÇÃO DO LIMIAR (METADADOS DO YAML)
    # ---------------------------------------------------------
    user_thresholds = {}
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, "r") as f:
                user_thresholds = json.load(f)
        except Exception as e:
            print(f"⚠️  Erro ao carregar o arquivo de thresholds: {e}")

    global_fallback = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)

    print("\n--- Iniciando Avaliação UEBA Rigorosa (Is_Anomaly) ---")
    
    if fixed_threshold is not None:
        print(f"⚖️  [YAML] Aplicando Limiar Absoluto Fixo: {fixed_threshold}")
    elif fixed_percentile is not None:
        print(f"⚖️  [YAML] Aplicando Percentil Fixo: {fixed_percentile}%")
        if thresh_mode == "per_user":
            print("   -> Lógica individual baseada em perfis dinâmicos de percentil.")
        else:
            print("   -> Lógica baseada em percentil global unificado.")
    else:
        print("🧠 [AUTO] Aplicando Limiares Otimizados Dinamicamente (F2-Score)")
        if thresh_mode == "per_user":
            print("   -> Modelo UEBA Clássico: Perfis comportamentais personalizados ativos.")
        else:
            print("   -> Modelo Clássico: Limiar global ajustado dinamicamente.")

    user_mapping = test_df["UserID"].to_dict()
    test_users = [str(user_mapping.get(idx, "unknown")) for idx in indices]

    sample_thresholds = []
    own_threshold_users = set()
    unique_test_users = np.unique(test_users)

    for u in test_users:
        if thresh_mode == "per_user" and u in user_thresholds and u != "__GLOBAL_FALLBACK__":
            sample_thresholds.append(user_thresholds[u])
            own_threshold_users.add(u)
        else:
            sample_thresholds.append(global_fallback)

    sample_thresholds = np.array(sample_thresholds)
    y_pred = (scores > sample_thresholds).astype(int)

    print(f"   -> Usuários Totais no Teste: {len(unique_test_users)}")
    if thresh_mode == "per_user":
        print(f"   -> Usuários Ativos com Limiar Personalizado (UEBA): {len(own_threshold_users)}")
    else:
        print(f"   -> Usuários com Limiar Próprio: 0 (Modo Global Ativo)")
    print(f"   -> Fallback Global Utilizado: {global_fallback:.6f}")

    # ---------------------------------------------------------
    # 3. COMPUTAÇÃO E GRAVAÇÃO DAS MÉTRICAS DE AVALIAÇÃO
    # ---------------------------------------------------------
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    # --- NOVO CÁLCULO ROC-AUC ---
    try:
        roc_auc = roc_auc_score(y_test, scores)
    except ValueError:
        roc_auc = 0.5 # Fallback caso haja apenas uma classe na amostra

    print(f"🎯 PR-AUC Global Alcançado : {pr_auc:.4f}")
    print(f"🎯 ROC-AUC Global Alcançado: {roc_auc:.4f}")

    print("\nRelatório de Classificação (Performance de Produção Per-User):")
    print(report_text)

    # Salva o relatório textual
    report_file_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write("=== AVALIAÇÃO EXTERNA DE SEGURANÇA ===\n")
        f.write(f"Data/Hora: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"PR-AUC: {pr_auc:.6f}\n")
        f.write(f"ROC-AUC: {roc_auc:.6f}\n\n")
        f.write(report_text)

    # Coleta de forma resiliente os dados da classe positiva, prevenindo chaves alternativas
    class_1_data = report_dict.get("1", report_dict.get("1.0", report_dict.get("True", {})))

    # Salva o relatório estruturado em JSON para leitura rápida do Runner
    summary_metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1_score": float(class_1_data.get("f1-score", 0.0)),
        "precision": float(class_1_data.get("precision", 0.0)),
        "recall": float(class_1_data.get("recall", 0.0))
    }
    
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(summary_metrics, f, indent=4)

    # Geração opcional de curva Precision-Recall
    if config.get("settings", {}).get("generate_plots", True):
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, color="darkorange", lw=2, label=f"Curva PR (AUC = {pr_auc:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Curva Precision-Recall (Segurança AD)")
            plt.legend(loc="lower left")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as plot_err:
            print(f"⚠️  Aviso: Falha ao desenhar a curva Precision-Recall: {plot_err}")

    print(f"💾 Resultados consolidados gravados na pasta: {output_dir}")
