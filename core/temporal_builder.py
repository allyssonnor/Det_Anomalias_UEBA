#NOVA VERSÃO DO TEMPORAL BUILDER
import numpy as np
import pandas as pd

class TemporalBuilder:
    """
    Construtor de contexto temporal para processamento de sequências.
    Gera sessões de utilizador com base em inatividade (timeout) e
    limita o tamanho máximo de cada sessão (chunking).
    """
    def __init__(self, config: dict):
        self.config = config
        # Configurações de hiperparâmetros com fallbacks seguros
        self.sequence_length = self.config.get("model", {}).get("sequence_length", 10)
        
        temporal_cfg = self.config.get("temporal", {})
        self.session_timeout = temporal_cfg.get("session_timeout", 1800)
        self.max_events_session = temporal_cfg.get("max_events_session", 1000)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Validação defensiva do DataFrame de entrada
        if df is None or df.empty:
            return df

        df = df.copy()

        # Garantir a existência das colunas obrigatórias com logs claros
        required_cols = ["Time", "UserID"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ [TemporalBuilder] Dataset em falta das colunas obrigatórias: {missing_cols}. Processamento ignorado.")
            return df

        # 2. Ordenação cronológica estável por utilizador
        df = df.sort_values(["UserID", "Time"]).reset_index(drop=True)

        # 3. Diferença de tempo entre eventos consecutivos do mesmo utilizador
        df["time_diff"] = df.groupby("UserID")["Time"].diff().fillna(0)

        # 4. Identificação de sub-sessões baseadas puramente em timeout
        df["new_session_by_timeout"] = (df["time_diff"] > self.session_timeout).astype(int)
        df["sub_session_id"] = df.groupby("UserID")["new_session_by_timeout"].cumsum()

        # 5. Divisão de sessões excessivamente longas (Chunking)
        # Evita que uma única sessão tenha mais eventos do que o limite definido
        df["chunk_id"] = df.groupby(["UserID", "sub_session_id"]).cumcount() // self.max_events_session

        # 6. Geração de IDs de Sessão Globalmente Únicos (Sem colisões)
        # Uma nova sessão global é demarcada se houver mudança de Utilizador, de sub-sessão ou de chunk
        session_change = (
            (df["UserID"] != df["UserID"].shift()) |
            (df["sub_session_id"] != df["sub_session_id"].shift()) |
            (df["chunk_id"] != df["chunk_id"].shift())
        )
        
        df["session_id"] = session_change.astype(int).cumsum()

        # 7. Limpeza e remoção de colunas auxiliares temporárias
        df.drop(columns=["new_session_by_timeout", "sub_session_id", "chunk_id"], inplace=True)

        return df