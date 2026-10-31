# core/temporal_builder.py

import pandas as pd
import numpy as np


class TemporalBuilder:
    def __init__(self, config):
        self.config = config
        temporal_cfg = config.get("temporal", {})
        self.session_timeout = temporal_cfg.get("session_timeout", 3600)
        # min_events_session: se > 1, sessões menores viram NaN (descartadas
        # quando window_grouping='session'). Default 1 = comportamento anterior.
        self.min_events_session = max(1, int(temporal_cfg.get("min_events_session", 1)))

    def transform(self, df):
        if df is None or df.empty:
            return pd.DataFrame()

        print(f"🕒 [TemporalBuilder] Processando dinâmica temporal (Timeout: {self.session_timeout}s, MinEvents: {self.min_events_session})...")

        df_temp = df.copy()

        if 'Time' not in df_temp.columns:
            print("⚠️ Coluna 'Time' ausente. Retornando dataset bruto.")
            return df_temp

        if not pd.api.types.is_numeric_dtype(df_temp['Time']):
            print("⚠️ Coluna 'Time' não é numérica. Retornando dataset bruto.")
            return df_temp

        # Aviso (sem dropar): valores não-finitos em Time podem contaminar sessões
        n_bad_time = int((~np.isfinite(df_temp['Time'].astype(float))).sum())
        if n_bad_time > 0:
            print(f"   ⚠️ {n_bad_time} eventos com Time inválido (NaN/inf). Considere filtrar antes.")

        # Ordenação estável (determinismo em empates)
        df_temp = df_temp.sort_values(by=['UserID', 'Time'], kind='stable').reset_index(drop=True)

        # Cálculo de delta e sessão
        df_temp['time_diff'] = df_temp.groupby('UserID', sort=False)['Time'].diff().fillna(0)
        df_temp['new_session'] = (df_temp['time_diff'] > self.session_timeout).astype(int)
        df_temp['session_id'] = df_temp.groupby('UserID', sort=False)['new_session'].cumsum()
        df_temp['session_id'] = df_temp['UserID'].astype(str) + "_" + df_temp['session_id'].astype(str)

        # Filtro opcional de sessões pequenas
        if self.min_events_session > 1:
            sizes = df_temp.groupby('session_id')['session_id'].transform('size')
            mask_tiny = sizes < self.min_events_session
            df_temp.loc[mask_tiny, 'session_id'] = None

        df_temp = df_temp.drop(columns=['time_diff', 'new_session'])

        n_sessions = df_temp['session_id'].nunique()
        print(f"   ✅ Sessões identificadas. Total de sessões válidas: {n_sessions}")

        return df_temp