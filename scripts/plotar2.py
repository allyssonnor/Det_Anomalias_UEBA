import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from math import pi

def plot_master_dashboard(csv_path, output_path="results/dashboard_arquitetura.png"):
    # 1. Carregar os dados
    df = pd.read_csv(csv_path)
    
    # Tratamento básico de nomes de colunas caso haja variação
    if 'model.sequence_length' not in df.columns and 'sequence_length' in df.columns:
        df.rename(columns={'sequence_length': 'model.sequence_length'}, inplace=True)
    if 'training.threshold_mode' not in df.columns and 'threshold_mode' in df.columns:
        df.rename(columns={'threshold_mode': 'training.threshold_mode'}, inplace=True)

    # Proteção: se rodar num CSV antigo sem a coluna ROC-AUC, cria zerada para não quebrar
    if 'roc_auc' not in df.columns:
        print("⚠️ Coluna 'roc_auc' não encontrada no CSV. Preenchendo com zeros temporariamente.")
        df['roc_auc'] = 0.0

    # Configuração visual geral
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Análise Arquitetural e Trade-offs do Modelo (UEBA)', fontsize=18, fontweight='bold', y=0.98)

    # Cores expandidas para as novas métricas
    cores = {
        'F1': '#1f77b4', 
        'Recall': '#2ca02c', 
        'Precision': '#ff7f0e', 
        'Time': '#d62728',
        'ROC-AUC': '#9467bd',  # Roxo
        'PR-AUC': '#17becf'    # Ciano
    }

    # =========================================================================
    # 1. Gráfico de Barras: Impacto da Sequência (ax1)
    # =========================================================================
    ax1 = fig.add_subplot(221)
    df_seq = df.groupby('model.sequence_length')[['f1_score', 'recall', 'precision', 'roc_auc']].mean().reset_index()
    df_seq_melt = df_seq.melt(id_vars='model.sequence_length', var_name='Métrica', value_name='Score')
    
    sns.barplot(data=df_seq_melt, x='model.sequence_length', y='Score', hue='Métrica', 
                palette=[cores['F1'], cores['Recall'], cores['Precision'], cores['ROC-AUC']], ax=ax1)
    
    ax1.set_title('1. Impacto do Contexto Temporal (Seq. Length)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tamanho da Sequência (Eventos)', fontsize=12)
    ax1.set_ylabel('Score Médio', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(title='', loc='upper left')

    # =========================================================================
    # 2. Eixo Duplo: Desempenho vs Custo Computacional (ax2)
    # =========================================================================
    ax2 = fig.add_subplot(222)
    df_cost = df.groupby('model.sequence_length')[['f1_score', 'elapsed_sec']].mean().reset_index()
    
    # Barra para F1 (Eixo Esquerdo)
    x_pos = np.arange(len(df_cost['model.sequence_length']))
    ax2.bar(x_pos, df_cost['f1_score'], color=cores['F1'], width=0.4, label='F1-Score', alpha=0.8)
    ax2.set_ylabel('Desempenho (F1-Score)', fontsize=12, color=cores['F1'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=cores['F1'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_cost['model.sequence_length'])
    ax2.set_xlabel('Tamanho da Sequência (Eventos)', fontsize=12)
    
    # Linha para Tempo (Eixo Direito)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_pos, df_cost['elapsed_sec'], color=cores['Time'], marker='o', linewidth=2, markersize=8, label='Tempo')
    ax2_twin.set_ylabel('Tempo de Execução (segundos)', fontsize=12, color=cores['Time'], fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=cores['Time'])
    
    # AJUSTE FINO: Zoom dinâmico no eixo Y do tempo para destacar a inclinação
    t_min = df_cost['elapsed_sec'].min()
    t_max = df_cost['elapsed_sec'].max()
    t_pad = (t_max - t_min) * 0.3 if t_max > t_min else 0.5
    ax2_twin.set_ylim(max(0, t_min - t_pad), t_max + t_pad)
    
    ax2.set_title('2. Trade-off: Desempenho vs. Custo', fontsize=14, fontweight='bold')
    # Ajustar legendas combinadas
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # =========================================================================
    # 3. Gráfico de Radar: Perfil do Melhor Modelo (ax3)
    # =========================================================================
    ax3 = fig.add_subplot(223, polar=True)
    
    # Pegar o melhor modelo baseado no F1-Score
    best_row = df.loc[df['f1_score'].idxmax()]
    categorias = ['Precision', 'Recall', 'F1-Score', 'PR-AUC', 'ROC-AUC']
    valores = [best_row['precision'], best_row['recall'], best_row['f1_score'], best_row['pr_auc'], best_row['roc_auc']]
    
    # Fechar o ciclo do radar
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
    
    titulo_radar = f"3. Perfil do Melhor Modelo\n(Seq={best_row['model.sequence_length']} | {best_row['training.threshold_mode']})"
    ax3.set_title(titulo_radar, fontsize=14, fontweight='bold', pad=20)

    # =========================================================================
    # 4. Barras Agrupadas: Global vs Per User em Seq=20 (ax4)
    # =========================================================================
    ax4 = fig.add_subplot(224)
    df_20 = df[df['model.sequence_length'] == 20]
    
    if not df_20.empty:
        df_thresh = df_20.groupby('training.threshold_mode')[['f1_score', 'pr_auc', 'roc_auc']].mean().reset_index()
        df_thresh_melt = df_thresh.melt(id_vars='training.threshold_mode', var_name='Métrica', value_name='Score')
        
        sns.barplot(data=df_thresh_melt, x='training.threshold_mode', y='Score', hue='Métrica', 
                    palette=[cores['F1'], cores['PR-AUC'], cores['ROC-AUC']], ax=ax4)
        
        ax4.set_title('4. Estratégia de Limiar (Apenas Seq=20)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Modo de Threshold', fontsize=12)
        ax4.set_ylabel('Score Médio', fontsize=12)
        ax4.set_ylim(0, max(df_thresh_melt['Score'].max() * 1.2, 0.8))
        ax4.legend(title='', loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'Sem dados para Seq=20', ha='center', va='center', fontsize=12)

    # Ajustes finais e salvar
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Dashboard gerado com sucesso: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota dashboard arquitetural baseado no master_results_table.")
    parser.add_argument("--csv", type=str, required=True, help="Caminho para o master_results_table.csv")
    parser.add_argument("--save_path", type=str, default="results/dashboard_arquitetura.png")
    args = parser.parse_args()

    plot_master_dashboard(args.csv, output_path=args.save_path)
