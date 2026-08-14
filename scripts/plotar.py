import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
    precision_score
)

def find_latest_experiment(base_dir):
    """Busca recursivamente o CSV de predições/resultados mais recente."""
    csv_candidates = glob.glob(os.path.join(base_dir, "**", "*pred*.csv"), recursive=True)
    if not csv_candidates:
        csv_candidates = glob.glob(os.path.join(base_dir, "**", "*eval*.csv"), recursive=True)
    if not csv_candidates:
        csv_candidates = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
        
    if not csv_candidates:
        raise FileNotFoundError(f"Nenhum arquivo CSV de resultados encontrado em: {base_dir}")
    
    # Pega o arquivo modificado mais recentemente
    latest_file = max(csv_candidates, key=os.path.getmtime)
    return latest_file

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Identificação flexível das colunas de rótulo real, predição e score
    y_true_col = next((c for c in ['y_true', 'label', 'labels', 'target', 'is_anomaly'] if c in df.columns), None)
    y_pred_col = next((c for c in ['y_pred', 'prediction', 'pred', 'pred_label'] if c in df.columns), None)
    y_score_col = next((c for c in ['y_score', 'anomaly_score', 'reconstruction_error', 'loss', 'score'] if c in df.columns), None)

    if not y_true_col:
        raise ValueError(f"Coluna de rótulos reais não encontrada no CSV. Colunas disponíveis: {list(df.columns)}")

    y_true = df[y_true_col].values.astype(int)
    
    # Se não tiver y_pred explícito, calcula com base no score ou usa y_true se score ausente
    if y_pred_col:
        y_pred = df[y_pred_col].values.astype(int)
    elif y_score_col:
        threshold = np.percentile(df[y_score_col], 95)
        y_pred = (df[y_score_col].values > threshold).astype(int)
    else:
        y_pred = y_true

    y_score = df[y_score_col].values if y_score_col else y_pred.astype(float)
    
    return y_true, y_pred, y_score, df

def plot_dashboard(y_true, y_pred, y_score, output_path="dashboard_banca.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Avaliação de Desempenho - Detecção de Anomalias (UEBA)', fontsize=16, fontweight='bold', y=0.98)

    # -------------------------------------------------------------
    # 1. Matriz de Confusão Normalizada
    # -------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = ['Normal', 'Anomalia']
    annot = np.array([f"{count}\n({pct:.1f}%)" for count, pct in zip(cm.flatten(), cm_norm.flatten())]).reshape(2, 2)

    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=axes[0, 0],
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, "weight": "bold"})
    axes[0, 0].set_title('Matriz de Confusão (Normalizada / Total)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Predição do Modelo', fontsize=11)
    axes[0, 0].set_ylabel('Rótulo Real', fontsize=11)

    # -------------------------------------------------------------
    # 2. Curva ROC (Receiver Operating Characteristic)
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_val = auc(fpr, tpr)

    axes[0, 1].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC AUC = {roc_auc_val:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Aleatório')
    axes[0, 1].set_title('Curva ROC', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=11)
    axes[0, 1].set_ylabel('Taxa de Verdadeiros Positivos (TPR / Recall)', fontsize=11)
    axes[0, 1].legend(loc='lower right', frameon=True, fontsize=11)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # 3. Curva Precision-Recall (PR-AUC)
    # -------------------------------------------------------------
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc_val = average_precision_score(y_true, y_score)

    axes[1, 0].plot(rec, prec, color='#2ca02c', lw=2.5, label=f'PR-AUC (AP) = {pr_auc_val:.4f}')
    axes[1, 0].set_title('Curva Precision-Recall (Foco em Desbalanceamento)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Recall (Revocação)', fontsize=11)
    axes[1, 0].set_ylabel('Precision (Precisão)', fontsize=11)
    axes[1, 0].legend(loc='lower left', frameon=True, fontsize=11)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # 4. Distribuição dos Scores e Tabela de Métricas
    # -------------------------------------------------------------
    sns.kdeplot(y_score[y_true == 0], ax=axes[1, 1], color='#1f77b4', fill=True, alpha=0.4, label='Normal')
    sns.kdeplot(y_score[y_true == 1], ax=axes[1, 1], color='#d62728', fill=True, alpha=0.4, label='Anomalia')
    axes[1, 1].set_title('Distribuição dos Scores de Anomalia', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Score / Erro de Reconstrução', fontsize=11)
    axes[1, 1].set_ylabel('Densidade', fontsize=11)
    axes[1, 1].legend(loc='upper right', frameon=True, fontsize=11)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    # Cálculo do resumo numérico
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Box com resumo de métricas no gráfico
    metrics_text = f"Precision: {p:.3f} | Recall: {r:.3f} | F1-Score: {f1:.3f} | ROC-AUC: {roc_auc_val:.3f}"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f0f0', edgecolor='#cccccc', lw=1.5))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"\n✅ Painel de gráficos gerado com sucesso: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dashboard de avaliação para a banca.")
    parser.add_argument("--results_dir", type=str, default="results", help="Diretório onde estão os resultados.")
    parser.add_argument("--csv_file", type=str, default=None, help="Caminho direto para o CSV de predições (opcional).")
    parser.add_argument("--save_path", type=str, default="results/dashboard_banca.png", help="Caminho para salvar a imagem.")
    args = parser.parse_args()

    csv_target = args.csv_file if args.csv_file else find_latest_experiment(args.results_dir)
    print(f"📊 Processando resultados de: {csv_target}")

    y_true, y_pred, y_score, _ = load_data(csv_target)
    
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    plot_dashboard(y_true, y_pred, y_score, output_path=args.save_path)
