import argparse
import json
import yaml
import glob
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, auc
from numpy.lib.stride_tricks import sliding_window_view

def main():
    parser = argparse.ArgumentParser(description="Auditoria estrita alinhada ao classification_report com leitura inteligente de YAML (Suporta Debug).")
    parser.add_argument("--output_dir", type=str, required=True, help="Pasta da run (ex: results/LAST_RUN)")
    parser.add_argument("--test_csv", type=str, required=True, help="Caminho do test.jsonl")
    parser.add_argument("--labels_csv", type=str, default=None, help="Caminho do labels.csv (se o gabarito estiver separado)")
    parser.add_argument("--seq_len", type=int, default=None, help="Tamanho da janela (força um valor, ignorando o YAML se preenchido)")
    args = parser.parse_args()
    
    # ---------------------------------------------------------
    # 1. INTELIGÊNCIA DE CONFIGURAÇÃO (ANTES DE CARREGAR OS DADOS)
    # ---------------------------------------------------------
    run_config = {}
    seq_len = args.seq_len
    
    # Procura o YAML na pasta de saída ou na pasta pai (útil para /eval_mixed)
    yaml_files = glob.glob(os.path.join(args.output_dir, "*.yaml"))
    if not yaml_files:
        parent_dir = os.path.dirname(os.path.normpath(args.output_dir))
        yaml_files = glob.glob(os.path.join(parent_dir, "*.yaml"))
        
    if yaml_files:
        config_encontrado = yaml_files[0]
        print(f"🔗 Lendo parâmetros automaticamente de: {config_encontrado}")
        try:
            with open(config_encontrado, "r", encoding="utf-8") as f:
                run_config = yaml.safe_load(f)
                if seq_len is None:
                    seq_len = run_config.get("model", {}).get("sequence_length")
                    if seq_len is not None:
                        print(f"⚙️ sequence_length recuperado do YAML: {seq_len}")
        except Exception as e:
            print(f"⚠️ Erro ao ler o YAML: {e}")

    # Descobre se rodou em modo debug
    sampling_mode = run_config.get("sampling", {}).get("mode", "full")
    test_size = run_config.get("sampling", {}).get("test_size", 1000)
    
    # ---------------------------------------------------------
    # 2. CARREGAMENTO INTELIGENTE DOS DADOS
    # ---------------------------------------------------------
    print(f"📂 Carregando dados de teste: {args.test_csv}")
    
    # Lê apenas o necessário se for modo debug (como o DatasetAdapter faz)
    if sampling_mode == "debug" and test_size > 0:
        print(f"🐛 Modo DEBUG detectado no YAML. Carregando apenas as primeiras {test_size} linhas.")
        if args.test_csv.endswith(".jsonl"):
            test_df = pd.read_json(args.test_csv, lines=True, nrows=test_size)
        else:
            test_df = pd.read_csv(args.test_csv, nrows=test_size)
    else:
        if args.test_csv.endswith(".jsonl"):
            test_df = pd.read_json(args.test_csv, lines=True)
        else:
            test_df = pd.read_csv(args.test_csv)

    test_df.columns = [str(c).strip() for c in test_df.columns]
    col_map = {
        'userid': 'UserID', 'user': 'UserID', 'username': 'UserID',
        'timestamp': 'Time', 'time': 'Time', 'datetime': 'Time',
        'is_anomaly': 'Is_Anomaly', 'target': 'Is_Anomaly', 'label': 'Is_Anomaly'
    }
    
    cols_low = {str(c).lower(): c for c in test_df.columns}
    rename_map = {cols_low[k]: v for k, v in col_map.items() if k in cols_low}
    if rename_map:
        test_df = test_df.rename(columns=rename_map)

    # Injeção de gabarito externo (se necessário)
    if args.labels_csv and os.path.exists(args.labels_csv):
        print(f"🔗 Injetando gabarito externo: {args.labels_csv}")
        labels_df = pd.read_csv(args.labels_csv)
        labels_df.columns = [str(c).strip() for c in labels_df.columns]
        labels_low = {str(c).lower(): c for c in labels_df.columns}
        labels_df = labels_df.rename(columns={labels_low.get(k, k): v for k, v in col_map.items()})
        
        test_df = pd.merge(test_df, labels_df[['UserID', 'Time', 'Is_Anomaly']], on=['UserID', 'Time'], how='left')
        test_df['Is_Anomaly'] = test_df['Is_Anomaly'].fillna(0).astype(int)

    if test_df.columns.duplicated().any():
        test_df = test_df.loc[:, ~test_df.columns.duplicated()]

    # Ordenação (ocorre APÓS o corte do debug, garantindo o mesmo comportamento do pipeline)
    if "UserID" in test_df.columns and "Time" in test_df.columns:
        test_df = test_df.sort_values(["UserID", "Time"]).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. CARREGAMENTO DOS ARTEFATOS
    # ---------------------------------------------------------
    scores_path = os.path.join(args.output_dir, "test_scores.npy")
    indices_path = os.path.join(args.output_dir, "test_indices.npy")
    thresholds_path = os.path.join(args.output_dir, "user_thresholds.json")

    if not (os.path.exists(scores_path) and os.path.exists(indices_path)):
        print(f"❌ Erro: Arquivos .npy não encontrados em {args.output_dir}.")
        return

    scores = np.load(scores_path)
    indices = np.load(indices_path)
    
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)
    global_fallback = thresholds.get("__GLOBAL_FALLBACK__", 0.5)

    target_col = "Is_Anomaly" if "Is_Anomaly" in test_df.columns else "target"
    if target_col not in test_df.columns:
        print(f"❌ Erro Crítico: Coluna de gabarito '{target_col}' não encontrada!")
        return

    # Fallback de dedução matemática da janela (caso o YAML falhe)
    if seq_len is None:
        if len(indices) > 0:
            first_user = test_df.loc[indices[0], "UserID"]
            user_first_idx = test_df[test_df["UserID"] == first_user].index[0]
            seq_len = indices[0] - user_first_idx + 1
            print(f"🔍 YAML ausente. Janela deduzida matematicamente pelos índices: {seq_len}")
        else:
            seq_len = 10
            print("⚠️ Aviso: Usando fallback padrão (sequence_length = 10).")

    # ---------------------------------------------------------
    # 4. REPRODUÇÃO DA LÓGICA DE JANELAMENTO E AVALIAÇÃO
    # ---------------------------------------------------------
    y_true_list = []
    for user_id, group in test_df.groupby("UserID"):
        if len(group) < seq_len:
            continue
        windows_labels = sliding_window_view(group[target_col].values, window_shape=seq_len)
        max_labels = np.max(windows_labels, axis=1)
        y_true_list.append(max_labels)

    if len(y_true_list) > 0:
        y_true = np.concatenate(y_true_list)
    else:
        y_true = np.array([])

    # SOLUÇÃO ROBUSTA: Reconstrução exata via índices em caso de desalinhamento
    if len(y_true) != len(scores):
        print(f"⚠️ Desalinhamento detectado (Rótulos: {len(y_true)} vs Scores: {len(scores)}).")
        print("🔧 Reconstruindo rótulos via mapeamento exato de índices...")
        y_true_reconstruido = []
        for idx in indices:
            start_idx = max(0, idx - seq_len + 1)
            window = test_df.loc[start_idx:idx, target_col].values
            y_true_reconstruido.append(np.max(window))
        y_true = np.array(y_true_reconstruido)
        print("✅ Alinhamento corrigido com sucesso!")

    # Aplicação dos limites
    user_mapping = test_df["UserID"].to_dict()
    test_users = [str(user_mapping.get(idx, "unknown")) for idx in indices]

    sample_thresholds = np.array([
        thresholds[u] if u in thresholds and u != "__GLOBAL_FALLBACK__" else global_fallback 
        for u in test_users
    ])
    
    y_pred = (scores > sample_thresholds).astype(int)

    # ---------------------------------------------------------
    # 5. AUDITORIA STATS
    # ---------------------------------------------------------
    df_evaluated = test_df.loc[indices].copy().reset_index(drop=True)
    df_evaluated["anomaly_score"] = scores
    df_evaluated["threshold_applied"] = sample_thresholds
    df_evaluated["predito_anomalo"] = y_pred

    status_list = []
    for real, pred in zip(y_true, y_pred):
        if real == 1 and pred == 1:
            status_list.append("Verdadeiro Positivo (TP)")
        elif real == 0 and pred == 1:
            status_list.append("Falso Positivo (FP)")
        elif real == 1 and pred == 0:
            status_list.append("Falso Negativo (FN)")
        else:
            status_list.append("Verdadeiro Negativo (TN)")
            
    df_evaluated["status_deteccao"] = status_list

    df_alertas = df_evaluated[df_evaluated["predito_anomalo"] == 1].copy()
    out_csv = os.path.join(args.output_dir, "anomaly_list.csv")
    df_alertas.to_csv(out_csv, index=False)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall_curve, precision_curve)

    print("\n" + "="*60)
    print("📊 RESUMO DA AUDITORIA DE DETECÇÃO (DOUBLE CHECK)")
    print("="*60)
    print(f" 🔹 Janelas Avaliadas            : {len(y_true)}")
    print(f" 🔹 Anomalias Reais no Gabarito  : {int(np.sum(y_true))}")
    print(f" 🔹 Alertas Disparados           : {len(df_alertas)}")
    print("-" * 60)
    print(f" ✅ Verdadeiros Positivos (TP) : {tp}")
    print(f" ⚠️  Falsos Positivos (FP)     : {fp}")
    print(f" ❌ Falsos Negativos (FN)     : {fn}")
    print(f" 🛡️  Verdadeiros Negativos (TN) : {tn}")
    print("-" * 60)
    print(f" 🎯 Precision (Precisão)      : {precision:.4f}")
    print(f" 🎯 Recall (Sensibilidade)    : {recall:.4f}")
    print(f" 🎯 F1-Score                  : {f1:.4f}")
    print(f" 🎯 PR-AUC                    : {pr_auc:.4f}")
    print("="*60)
    print(f"💾 Lista exportada com sucesso: {out_csv}")

if __name__ == "__main__":
    main()