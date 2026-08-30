# core/features/feature_processor.py

import pandas as pd
import numpy as np
import logging
import pickle
import os

# =========================================================
# REGISTRY (mantido da versão antiga)
# =========================================================
FEATURE_REGISTRY = {
    "auth_failure": {"required": ["Time", "UserID", "EventID"], "builder": "build_auth_failure"},
    "lateral_movement": {"required": ["Time", "UserID", "LogHost"], "builder": "build_lateral"},
    "temporal_anomaly": {"required": ["Time", "UserID"], "builder": "build_temporal"},
    "volume_anomaly": {"required": ["Time", "UserID"], "builder": "build_volume"},
    "rdp_anomaly": {"required": ["EventID", "UserID"], "builder": "build_rdp"},
    "golden_ticket": {"required": ["EventID"], "builder": "build_golden_ticket"},
    "process_anomaly": {"required": ["ProcessName", "UserID"], "builder": "build_process"},
    "privilege_anomaly": {"required": ["UserID", "ProcessName"], "builder": "build_privilege"}
}


class FeatureProcessor:
    """
    FeatureProcessor UEBA com:
    - Filtro anti-leakage no fit() (apenas dados NORMALES)
    - Frequências por usuário (UEBA)
    - Builders para 8 grupos de features
    """
    def __init__(self, config):
        self.config = config
        self.active_builders = []
        self.features_frozen = False
        self.final_feature_names = []
        self.output_dir = config.get("output_dir")
        
        # Estado histórico (baseline) – agora guarda POR USUÁRIO
        self.historical_state = {}
        self.is_fitted = False

    # =========================================================
    # MÉTODOS AUXILIARES
    # =========================================================
    def _get_enabled_groups(self):
        feat_cfg = self.config.get("features", {})
        enabled = feat_cfg.get("enabled_groups", [])
        if not enabled:
            return set(FEATURE_REGISTRY.keys()) if feat_cfg.get("enabled", True) else set()
        return set(enabled)

    def _detect_features(self, df):
        """Identifica quais builders podem ser usados com as colunas disponíveis."""
        available = set(df.columns)
        enabled = self._get_enabled_groups()
        
        print(f"\n🛠️  CONSTRUINDO VETOR COMPORTAMENTAL (UEBA):")
        for name, spec in FEATURE_REGISTRY.items():
            if name not in enabled:
                continue
            if set(spec["required"]).issubset(available):
                self.active_builders.append(spec["builder"])
                print(f"  ✅ [BUILDER] '{name}' ATIVADA.")
            else:
                missing = set(spec["required"]) - available
                print(f"  ❌ [SKIP] '{name}' ignorada. Faltam: {missing}")
        self.features_frozen = True

    # =========================================================
    # FIT (APENAS NORMALES → ANTI-LEAKAGE)
    # =========================================================
    def fit(self, df):
        """
        Ancora as distribuições normais usando ESTRITAMENTE o conjunto de treinamento
        com eventos NORMALES (Is_Anomaly == 0).
        """
        if df is None or df.empty:
            return

        # 🔥 FILTRO ANTI-LEAKAGE
        target_col = self.config.get("data", {}).get("target_col", "Is_Anomaly")
        if target_col in df.columns:
            df_temp = df[df[target_col] == 0].copy()
            if df_temp.empty:
                print("⚠️ ATENÇÃO: Nenhum evento 'normal' encontrado para fit. Usando base total (Pode gerar viés).")
                df_temp = df.copy()
        else:
            df_temp = df.copy()

        print("🧠 [FeatureProcessor] A processar os parâmetros históricos e codificadores (APENAS NORMALES)...")
        
        # Converte tempo para datetime (necessário para os builders)
        if 'Time' not in df_temp.columns:
            print("⚠️ Coluna 'Time' não encontrada. Impossível extrair features temporais.")
            return
            
        df_temp["datetime"] = pd.to_datetime(df_temp["Time"], unit="s")
        
        # -----------------------------------------------------
        # 1. FREQUÊNCIAS POR USUÁRIO (UEBA)
        # -----------------------------------------------------
        user_counts = df_temp.groupby("UserID").size()

        if "LogHost" in df_temp.columns:
            self.historical_state["loghost_freq"] = (
                df_temp.groupby(["UserID", "LogHost"]).size() / user_counts
            )
        
        if "ProcessName" in df_temp.columns:
            self.historical_state["process_freq"] = (
                df_temp.groupby(["UserID", "ProcessName"]).size() / user_counts
            )
        
        if "IpAddress" in df_temp.columns:
            self.historical_state["ip_freq"] = (
                df_temp.groupby(["UserID", "IpAddress"]).size() / user_counts
            )
            
        if "SubStatus" in df_temp.columns:
            self.historical_state["substatus_freq"] = (
                df_temp.groupby(["UserID", "SubStatus"]).size() / user_counts
            )
            
        if "AuthenticationPackageName" in df_temp.columns:
            self.historical_state["authpkg_freq"] = (
                df_temp.groupby(["UserID", "AuthenticationPackageName"]).size() / user_counts
            )

        # -----------------------------------------------------
        # 2. ESTATÍSTICAS DE VOLUME POR USUÁRIO (para build_volume)
        # -----------------------------------------------------
        if "Time" in df_temp.columns:
            # Média e desvio da hora de atividade (para out_of_hours)
            df_temp["hour"] = df_temp["datetime"].dt.hour
            self.historical_state["hour_mean"] = df_temp.groupby("UserID")["hour"].mean()
            self.historical_state["hour_std"] = df_temp.groupby("UserID")["hour"].std()

            # Estatísticas de volume diário (para event_z)
            df_temp["date"] = df_temp["datetime"].dt.date
            daily_counts = df_temp.groupby(["UserID", "date"]).size()
            self.historical_state["daily_volume_mean"] = daily_counts.groupby("UserID").mean()
            self.historical_state["daily_volume_std"] = daily_counts.groupby("UserID").std()

        self.is_fitted = True
        print("   ✅ Baseline comportamental (UEBA) estabelecido com sucesso.")

    # =========================================================
    # TRANSFORM (APLICA FEATURES)
    # =========================================================
    def transform(self, df):
        """
        Aplica a extração de features usando o baseline histórico (fit) e os builders.
        """
        if df is None or df.empty:
            return df
        
        if not self.is_fitted:
            logging.warning("FeatureProcessor.transform chamado sem .fit() prévio. Possível Leakage.")
            self.fit(df)
            
        df = df.copy()
        df = df.sort_values(["UserID", "Time"]).reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["Time"], unit="s")

        # -----------------------------------------------------
        # 1. BASE STATS (Codificação por frequência POR USUÁRIO)
        # -----------------------------------------------------
        encoders = [
            ("LogHost", "loghost_freq", "LogHost_enc"),
            ("ProcessName", "process_freq", "ProcessName_enc"),
            ("IpAddress", "ip_freq", "IpAddress_enc"),
            ("SubStatus", "substatus_freq", "SubStatus_enc"),
            ("AuthenticationPackageName", "authpkg_freq", "AuthPackage_enc")
        ]

        for col, state_key, enc_col in encoders:
            if col in df.columns and state_key in self.historical_state:
                freq_map = self.historical_state[state_key].reset_index(name=enc_col)
                # Merge preservando a ordem original
                df["_ord"] = range(len(df))
                df = df.merge(freq_map, on=['UserID', col], how='left')
                df = df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
                df[enc_col] = df[enc_col].fillna(0.0)  # novos valores → frequência zero

        # PREVENTIVO: Remove duplicatas se houver (fallback) e reinicia o índice
        if "_ord" in df.columns:
            df = df.drop_duplicates(subset=['_ord']).reset_index(drop=True)

        # -----------------------------------------------------
        # 2. BUILDERS (FEATURES COMPORTAMENTAIS)
        # -----------------------------------------------------
        if not self.features_frozen:
            self._detect_features(df)

        for builder in self.active_builders:
            if hasattr(self, builder):
                df = getattr(self, builder)(df)

        # -----------------------------------------------------
        # 3. LIMPEZA E SELEÇÃO FINAL
        # -----------------------------------------------------
        # Metadados e colunas que NÃO são features (devem ser removidas)
        meta_cols = [
            "datetime", "LogHost", "ProcessName", "UserName", "SourceIP",
            "UserID", "Time", "session_id", "AnomalyType", "EventID", "hour",
            "is_failed", "host_changed", "rdp_event", "admin_activity",
            "event_count", "LogHost_enc", "ProcessName_enc",
            "is_golden_event", "rare_process",
            "Is_Anomaly",  # 🔥 CORREÇÃO 1: Adicionado aqui para NUNCA virar feature!
            # Campos novos (metadados)
            "IpAddress", "SubStatus", "AuthenticationPackageName", "IpAddress_enc",
            "SubStatus_enc", "AuthPackage_enc", "ip_changed", "network_logon",
            "bad_password", "account_locked", "kerberos_auth",
            # Colunas auxiliares dos builders
            "date", "new_session", "time_diff", "sub_session_id", "chunk_id",
            "day_of_week", "time_sin", "time_cos", "is_weekend",
            "_ord"
        ]
        
        # Identifica as features numéricas (excluindo metadados)
        self.final_feature_names = [
            c for c in df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        
        # Salva metadados (opcional)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "feature_metadata.pkl"), "wb") as f:
                pickle.dump({
                    "features": self.final_feature_names,
                    "state": self.historical_state
                }, f)

        print(f"📊 Vetor Final ({len(df)} eventos): {len(self.final_feature_names)} dimensões isoladas.")
        
        # Mantém apenas as features + chaves essenciais
        keep = self.final_feature_names + ["UserID", "Time", "Is_Anomaly", "AnomalyType"]
        keep = list(dict.fromkeys(keep))  # 🔥 CORREÇÃO 2: Remove duplicatas mantendo ordem
        
        return df[[c for c in keep if c in df.columns]].fillna(0)

    # =========================================================
    # BUILDERS (TODOS RESTAURADOS)
    # =========================================================
    def build_temporal(self, df):
        """Detecta atividades fora do horário usual do usuário."""
        df["hour"] = df["datetime"].dt.hour
        df["out_of_hours"] = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
        
        # Desvio do horário médio do usuário (baseline do fit)
        if "hour_mean" in self.historical_state:
            user_mean = self.historical_state["hour_mean"]
            df["hour_dev"] = abs(df["hour"] - df["UserID"].map(user_mean)).fillna(0)
        else:
            df["hour_dev"] = 0.0
            
        return df

    def build_volume(self, df):
        """Detecta picos de volume (event_z) baseado no histórico do usuário."""
        # Contagem de eventos por dia (rolling)
        if "date" in df.columns:
            df["event_count"] = df.groupby(["UserID", "date"]).cumcount() + 1
            
            # Média e desvio diário do usuário (baseline do fit)
            if "daily_volume_mean" in self.historical_state:
                mean_map = self.historical_state["daily_volume_mean"]
                std_map = self.historical_state["daily_volume_std"]
                
                df["vol_mean"] = df["UserID"].map(mean_map).fillna(0)
                df["vol_std"] = df["UserID"].map(std_map).fillna(1)  # evita divisão por zero
                df["event_z"] = ((df["event_count"] - df["vol_mean"]) / (df["vol_std"] + 1e-6)).fillna(0)
            else:
                df["event_z"] = 0.0
                
            # Limpa colunas auxiliares
            df.drop(columns=["vol_mean", "vol_std"], inplace=True, errors="ignore")
            
        return df

    def build_lateral(self, df):
        """Detecta movimento lateral (mudança de host/IP e logons de rede)."""
        if "LogHost_enc" in df.columns:
            df["host_changed"] = df.groupby("UserID")["LogHost_enc"].diff().fillna(0).ne(0).astype(int)
            df["unique_hosts"] = df.groupby("UserID")["host_changed"].transform(
                lambda x: x.expanding().sum().fillna(0) + 1
            )
        
        if "IpAddress_enc" in df.columns:
            df["ip_changed"] = df.groupby("UserID")["IpAddress_enc"].diff().fillna(0).ne(0).astype(int)
            df["unique_ips"] = df.groupby("UserID")["ip_changed"].transform(
                lambda x: x.expanding().sum().fillna(0) + 1
            )
        
        # Logons de rede (EventID 4624 + LogonType 3, ou EventID 4648)
        if "LogonType" in df.columns:
            df["network_logon"] = (
                ((df["EventID"] == 4624) & (df["LogonType"] == 3)) |
                (df["EventID"] == 4648)
            ).astype(int)
            # Taxa cumulativa (expanding) – simula o histórico do usuário
            df["network_logon_rate"] = df.groupby("UserID")["network_logon"].transform(
                lambda x: x.expanding().sum().fillna(0)
            )
            
        return df

    def build_auth_failure(self, df):
        """Detecta falhas de autenticação e códigos de erro específicos."""
        df["is_failed"] = (df["EventID"] == 4625).astype(int)
        df["fail_rate"] = df.groupby("UserID")["is_failed"].transform(
            lambda x: x.expanding().sum().fillna(0)
        )
        
        if "SubStatus" in df.columns:
            sub_str = df["SubStatus"].astype(str).str.upper()
            df["bad_password"] = (
                (df["EventID"] == 4625) & (sub_str.str.contains("0XC000006A"))
            ).astype(int)
            df["account_locked"] = (
                (df["EventID"] == 4625) & (sub_str.str.contains("0XC0000234"))
            ).astype(int)
            
            df["bad_pwd_rate"] = df.groupby("UserID")["bad_password"].transform(
                lambda x: x.expanding().sum().fillna(0)
            )
            df["lock_rate"] = df.groupby("UserID")["account_locked"].transform(
                lambda x: x.expanding().sum().fillna(0)
            )
            
        return df

    def build_rdp(self, df):
        """Detecta acessos RDP (EventID 4624 + LogonType 10)."""
        if "LogonType" in df.columns:
            df["rdp_event"] = ((df["EventID"] == 4624) & (df["LogonType"] == 10)).astype(int)
        else:
            df["rdp_event"] = (df["EventID"] == 4624).astype(int)
            
        df["rdp_rate"] = df.groupby("UserID")["rdp_event"].transform(
            lambda x: x.expanding().sum().fillna(0)
        )
        return df

    def build_golden_ticket(self, df):
        """Detecta Golden Ticket (EventIDs 4672, 4769, e Kerberos)."""
        df["is_golden_event"] = df["EventID"].isin([4672, 4769]).astype(int)
        df["golden_ticket_score"] = df.groupby("UserID")["is_golden_event"].transform(
            lambda x: x.expanding().sum().fillna(0)
        )
        
        if "AuthenticationPackageName" in df.columns:
            df["kerberos_auth"] = (
                df["AuthenticationPackageName"].astype(str).str.lower().str.contains("kerberos")
            ).astype(int)
            df["kerberos_rate"] = df.groupby("UserID")["kerberos_auth"].transform(
                lambda x: x.expanding().sum().fillna(0)
            )
            
        return df

    def build_process(self, df):
        """Detecta execução de processos raros (frequência baixa por usuário)."""
        if "ProcessName_enc" in df.columns:
            df["rare_process"] = (df["ProcessName_enc"] < 0.02).astype(int)
            df["rare_process_rate"] = df.groupby("UserID")["rare_process"].transform(
                lambda x: x.expanding().sum().fillna(0)
            )
        return df

    def build_privilege(self, df):
        """Detecta atividade administrativa incomum."""
        if "ProcessName" in df.columns:
            proc_lower = df["ProcessName"].fillna("").astype(str).str.lower()
            admin_tools = [
                "powershell.exe", "cmd.exe", "wmic.exe", "ntdsutil.exe",
                "psexec.exe", "mmc.exe", "vssadmin.exe"
            ]
            df["admin_activity"] = proc_lower.isin(admin_tools).astype(int)
            df["admin_rate"] = df.groupby("UserID")["admin_activity"].transform(
                lambda x: x.expanding().sum().fillna(0)
            )
        return df
