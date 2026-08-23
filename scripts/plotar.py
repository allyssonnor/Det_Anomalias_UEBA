import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from math import pi

def plot_master_dashboard(csv_path, output_path="results/dashboard_banca.png"):
    df = pd.read_csv(csv_path)
    
    if 'model.sequence_length' not in df.columns and 'sequence_length' in df.columns:
        df.rename(columns={'sequence_length': 'model.sequence_length'}, inplace=True)
    if 'training.threshold_mode' not in df.columns and 'threshold_mode' in df.columns:
        df.rename(columns={'threshold_mode': 'training.threshold_mode'}, inplace=True)

    # Busca especificamente a avaliação isolada do bruteforce
    df_bruteforce = df[df['dataset'] == 'test_bruteforce']
    if not df_bruteforce.empty:
        modelo_alvo = df_bruteforce.iloc[0]
    else:
        # Fallback caso não encontre
        modelo_alvo = df.iloc[-1] 

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Avaliação do Especialista - {modelo_alvo.get('model_preset', 'Modelo')} (Ataque: {modelo_alvo.get('dataset', 'N/A')})", fontsize=20, fontweight='bold', y=0.98)

    cores = {'F1': '#1f77b4', 'Recall': '#2ca02c', 'Precision': '#ff7f0e', 'ROC-AUC': '#9467bd', 'PR-AUC': '#17becf'}

    # 1. Matriz de Confusão
    ax1 = fig.add_subplot(231)
    cm = np.array([[modelo_alvo.get('tn', 0), modelo_alvo.get('fp', 0)], 
                   [modelo_alvo.get('fn', 0), modelo_alvo.get('tp', 0)]])
    
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=ax1, 
                xticklabels=['Normal (Pred)', 'Anomalia (Pred)'], 
                yticklabels=['Normal (Real)', 'Anomalia (Real)'],
                annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title('1. Matriz de Confusão', fontsize=14, fontweight='bold')

    # 2. AUCs
    ax2 = fig.add_subplot(232)
    metricas_auc = ['PR-AUC', 'ROC-AUC']
    valores_auc = [modelo_alvo.get('pr_auc', 0), modelo_alvo.get('roc_auc', 0)]
    
    sns.barplot(x=metricas_auc, y=valores_auc, hue=metricas_auc, palette=[cores['PR-AUC'], cores['ROC-AUC']], legend=False, ax=ax2)
    for i, v in enumerate(valores_auc):
        ax2.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax2.set_title('2. Poder de Discriminação (AUCs)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)

    # 3. Radar
    ax3 = fig.add_subplot(233, polar=True)
    categorias = ['Precision', 'Recall', 'F1-Score', 'PR-AUC', 'ROC-AUC']
    valores = [modelo_alvo.get('precision', 0), modelo_alvo.get('recall', 0), modelo_alvo.get('f1_score', 0), modelo_alvo.get('pr_auc', 0), modelo_alvo.get('roc_auc', 0)]
    valores += [valores[0]]
    angulos = [n / float(len(categorias)) * 2 * pi for n in range(len(categorias))]
    angulos += [angulos[0]]
    ax3.plot(angulos, valores, color='#8c564b', linewidth=2, linestyle='solid')
    ax3.fill(angulos, valores, color='#8c564b', alpha=0.25)
    ax3.set_xticks(angulos[:-1])
    ax3.set_xticklabels(categorias, fontsize=11, fontweight='bold')
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="grey", size=9)
    ax3.set_ylim(0, 1)
    ax3.set_title('3. Perfil Geral de Desempenho', fontsize=14, fontweight='bold', pad=20)

    # 4. Barras
    ax4 = fig.add_subplot(212)
    metricas_base = ['Precision', 'Recall', 'F1-Score']
    valores_base = [modelo_alvo.get('precision', 0), modelo_alvo.get('recall', 0), modelo_alvo.get('f1_score', 0)]
    
    sns.barplot(x=metricas_base, y=valores_base, hue=metricas_base, palette=[cores['Precision'], cores['Recall'], cores['F1']], legend=False, ax=ax4)
    for i, v in enumerate(valores_base):
        ax4.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=14)
        
    # Verifica se é o modelo supervisionado para colocar o título adequado
    if "Supervised" in str(modelo_alvo.get('model_preset', '')):
        texto_limiar = "0.5 (Probabilidade Absoluta)"
    else:
        texto_limiar = modelo_alvo.get('training.threshold_mode', 'N/A')

    ax4.set_title(f"4. Visão Detalhada (Limiar: {texto_limiar})", fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Dashboard gerado com sucesso: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota dashboard para a banca.")
    parser.add_argument("--csv", type=str, required=True, help="Caminho para o csv gerado pelo detectar.")
    parser.add_argument("--save_path", type=str, default="results/dashboard_banca.png")
    args = parser.parse_args()

    plot_master_dashboard(args.csv, output_path=args.save_path)
