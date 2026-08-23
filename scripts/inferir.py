#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inferência: Carrega um modelo fechado e processa um dataset qualquer.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import yaml

from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor
from core.temporal_builder import TemporalBuilder
from core.models.model_factory import create as create_model

def rodar_inferencia(config_path, dataset_novo_path):
    # 1. Carregar Configuração e Diretórios Salvos
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    output_dir = config.get("output_dir", "results/banca_runs/full/run_seed42/Synthetic_MLP_Bruteforce_Supervised")
    model_path = os.path.join(output_dir, "saved_model")
    meta_path = os.path.join(output_dir, "trainer_meta.pkl")
    thresholds_path = os.path.join(output_dir, "user_thresholds.json")

    print(f"📂 Carregando artefatos do modelo de: {output_dir}")

    # 2. Carregar Metadados (Scalers e Features)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        scalers = meta['scalers']
        feature_columns = meta['feature_columns']

    with open(thresholds_path, "r") as f:
        user_thresholds = json.load(f)
        global_fallback = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)

    # 3. Instanciar e Carregar o Modelo Fechado
    model = create_model(config)
    if hasattr(model, "load"):
        model.load(model_path)
    else:
        raise RuntimeError("O modelo escolhido não implementa o método .load()")

    print("🤖 Modelo carregado e pronto para inferência.")

    # 4. Processar o Dataset Novo (Passando pelo Pipeline Padrão)
    config_infer = config.copy()
    config_infer["data"]["test_path"] = dataset_novo_path # Aponta para o dataset novo
    
    adapter = DatasetAdapter(config_infer)
    _, _, test_raw = adapter.load() # Assume que o adapter lê o path de teste

    processor = FeatureProcessor(config_infer)
    # Em produção, idealmente carregamos o state do FeatureProcessor salvo no treino
    test_scaled = processor.transform(test_raw)

    temporal = TemporalBuilder(config_infer)
    test_df = temporal.transform(test_scaled)

    # 5. Gerar Janelas e Calcular Scores (In-Memory)
    # (Aqui você aplica o janelamento idêntico ao do Trainer)
    sequence_length = config.get("model", {}).get("sequence_length", 20)
    
    X_all, indices_all, users_all = [], [], []
    for user_id, group in test_df.groupby("UserID"):
        # Aplica o scaler correspondente à entidade (ou o global se não existir)
        scaler = scalers.get(str(user_id), list(scalers.values())[0] if scalers else None)
        
        X_num = group.reindex(columns=feature_columns, fill_value=0)
        X_arr = X_num.values.astype(np.float32)
        if scaler:
            X_arr = scaler.transform(X_arr)
            
        if len(X_arr) < sequence_length: continue
        
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(X_arr, window_shape=sequence_length, axis=0)
        windows = np.swapaxes(windows, 1, 2)
        X_all.append(windows)
        
        idx = group.index.values[sequence_length - 1:]
        indices_all.append(idx)
        users_all.append(np.full(len(idx), str(user_id)))

    if not X_all:
        print("⚠️ O dataset novo não possui eventos suficientes para formar janelas temporais.")
        return

    X_test = np.concatenate(X_all)
    indices_test = np.concatenate(indices_all)
    users_test = np.concatenate(users_all)

    # 6. Execução da Inferência Pura
    print("🚀 Executando cálculo de anomalias no dataset externo...")
    scores = model.score(X_test)

    # 7. Aplicação dos Limiares e Emissão de Alertas
    thresholds_aplicados = np.array([
        user_thresholds.get(u, global_fallback) for u in users_test
    ])
    
    alertas = (scores > thresholds_aplicados).astype(int)
    
    print(f"📊 Total de janelas analisadas: {len(scores)}")
    print(f"🚨 Alertas de anomalia disparados: {np.sum(alertas)}")

    # Salva o resultado final para auditoria
    df_resultado = test_df.loc[indices_test].copy().reset_index(drop=True)
    df_resultado["anomaly_score"] = scores
    df_resultado["is_alert"] = alertas
    df_resultado.to_csv("results/resultado_inferencia.csv", index=False)
    print("💾 Relatório de inferência exportado com sucesso para: results/resultado_inferencia.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML de configuração do modelo.")
    parser.add_argument("--data", type=str, required=True, help="Caminho do novo dataset a ser processado.")
    args = parser.parse_args()

    rodar_inferencia(args.config, args.data)
