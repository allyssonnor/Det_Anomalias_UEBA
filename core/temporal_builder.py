# core/temporal_builder.py

import pandas as pd
import numpy as np

class TemporalBuilder:
    def __init__(self, config):
        self.config = config
        temporal_cfg = config.get("temporal", {})
        self.session_timeout = temporal_cfg.get("session_timeout", 3600)
        self.min_events_session = temporal_cfg.get("min_events_session", 2)

    def transform(self, df):
        """
        Organiza o log em sessões temporais contínuas por usuário,
        ordenando cronologicamente para garantir que não haja vazamento do futuro.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        print(f"🕒 [TemporalBuilder] Processando dinâmica temporal (Timeout: {self.session_timeout}s)...")
        
        df_temp = df.copy()

        # 1. ORDENAÇÃO CRONOLÓGICA ABSOLUTA (Anti-Leakage)
        if 'Time' not in df_temp.columns:
            print("⚠️ Coluna 'Time' ausente. Impossível construir sequências. Retornando dataset bruto.")
            return df_temp
            
        df_temp = df_temp.sort_values(by=['UserID', 'Time']).reset_index(drop=True)

        # 2. CALCULAR DELTA TIME E SESSÕES
        # Agrupa por UserID e calcula a diferença de tempo entre o evento atual e o anterior
        df_temp['time_diff'] = df_temp.groupby('UserID')['Time'].diff().fillna(0)
        
        # Identifica o início de uma nova sessão quando o gap for maior que o timeout definido
        df_temp['new_session'] = (df_temp['time_diff'] > self.session_timeout).astype(int)
        
        # Cria um ID único para cada sessão usando a soma cumulativa
        df_temp['session_id'] = df_temp.groupby('UserID')['new_session'].cumsum()
        
        # Opcional: Adiciona o ID do usuário ao session_id para ser globalmente único
        df_temp['session_id'] = df_temp['UserID'].astype(str) + "_" + df_temp['session_id'].astype(str)

        # Limpeza das colunas auxiliares
        df_temp = df_temp.drop(columns=['time_diff', 'new_session'])
        
        print(f"   ✅ Sessões identificadas. Total de sessões únicas: {df_temp['session_id'].nunique()}")

        return df_temp
