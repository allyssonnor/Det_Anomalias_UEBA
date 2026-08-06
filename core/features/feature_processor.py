import pandas as pd
import numpy as np
import logging
import pickle
import os

# Versão local do FEATURE_REGISTRY para referência
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
    O Cientista do Esquadrão (Versão Avançada):
    Gera features complexas e lida com os novos campos de autenticação do Windows 
    (LogonType, IpAddress, SubStatus, AuthenticationPackageName).
    Implementa Estado Histórico para eliminar o Data Leakage temporal.
    """
    def __init__(self, config):
        self.config = config
        self.active_builders = []
        self.features_frozen = False
        self.final_feature_names = []
        self.output_dir = config.get("output_dir")
        
        # Estado histórico para confinar as estatísticas ao conjunto de treino
        self.historical_state = {}
        self.is_fitted = False

    def _get_enabled_groups(self):
        feat_cfg = self.config.get("features", {})
        enabled = feat_cfg.get("enabled_groups", [])
        if not enabled:
            return set(FEATURE_REGISTRY.keys()) if feat_cfg.get("enabled", True) else set()
        return set(enabled)

    def _detect_features(self, df):
        available = set(df.columns)
        enabled = self._get_enabled_groups()
        
        print(f"\n🛠️  CONSTRUINDO VETOR COMPORTAMENTAL:")
        for name, spec in FEATURE_REGISTRY.items():
            if name not in enabled: continue
            if set(spec["required"]).issubset(available):
                self.active_builders.append(spec["builder"])
                print(f"  ✅ [BUILDER] '{name}' ATIVADA.")
            else:
                print(f"  ❌ [SKIP] '{name}' ignorada. Faltam: {set(spec['required']) - available}")
        self.features_frozen = True

    def fit(self, df):
        """
        Ancora as distribuições normais usando ESTRITAMENTE o conjunto de treinamento.
        """
        if df is None or df.empty: 
            return
        
        print("🧠 [FeatureProcessor] A memorizar parâmetros históricos e codificadores...")
        df_temp = df.copy()
        df_temp["datetime"] = pd.to_datetime(df_temp["Time"], unit="s")
        
        # 1. Frequências Históricas de Entidades (Codificadores)
        user_counts = df_temp.groupby("UserID").size()

        if "LogHost" in df_temp.columns:
            self.historical_state["loghost_freq"] = df_temp.groupby(["UserID", "LogHost"]).size() / user_counts
        
        if "ProcessName" in df_temp.columns:
            self.historical_state["process_freq"] = df_temp.groupby(["UserID", "ProcessName"]).size() / user_counts

        # --- NOVOS CODIFICADORES DE TEXTO ---
        if "IpAddress" in df_temp.columns:
            self.historical_state["ip_freq"] = df_temp.groupby(["UserID", "IpAddress"]).size() / user_counts
            
        if "SubStatus" in df_temp.columns:
            self.historical_state["substatus_freq"] = df_temp.groupby(["UserID", "SubStatus"]).size() / user_counts
            
        if "AuthenticationPackageName" in df_temp.columns:
            self.historical_state["authpkg_freq"] = df_temp.groupby(["UserID", "AuthenticationPackageName"]).size() / user_counts
            
        self.is_fitted = True

    def transform(self, df):
        """
        Aplica a extração de features usando o conhecimento causal e o estado histórico.
        """
        if df is None or df.empty: 
            return df
        
        if not self.is_fitted:
            logging.warning("FeatureProcessor.transform chamado sem .fit() prévio. Possível Leakage.")
            self.fit(df)
            
        df = df.copy()
        df = df.sort_values(["UserID", "Time"]).reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["Time"], unit="s")

        # 1. Base Stats (Aplicação estrita do mapa histórico)
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
                df["_ord"] = range(len(df))
                df = df.merge(freq_map, on=['UserID', col], how='left').sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
                df[enc_col] = df[enc_col].fillna(0.0)

        # 2. Builders
        if not self.features_frozen:
            self._detect_features(df)

        for builder in self.active_builders:
            if hasattr(self, builder):
                df = getattr(self, builder)(df)

        # 3. Limpeza Rigorosa (Remove textos brutos e lixo temporal)
        meta_cols = [
            "datetime", "LogHost", "ProcessName", "UserName", "SourceIP", "UserID", 
            "Time", "session_id", "AnomalyType", "Is_Anomaly", "EventID", "hour", 
            "is_failed", "host_changed", "rdp_event", "admin_activity", "event_freq", 
            "proc_freq", "fail_count", "fail_mean", "admin_mean", "event_count", 
            "event_mean", "hour_mean", "LogHost_enc", "ProcessName_enc", 
            "is_golden_event", "rare_process", 
            # Novos campos e metadados que não vão para o treino final
            "IpAddress", "SubStatus", "AuthenticationPackageName", "IpAddress_enc",
            "SubStatus_enc", "AuthPackage_enc", "ip_changed", "network_logon", 
            "bad_password", "account_locked", "kerberos_auth"
        ]
        
        self.final_feature_names = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "feature_metadata.pkl"), "wb") as f:
                pickle.dump({"features": self.final_feature_names, "state": self.historical_state}, f)

        print(f"📊 Vetor Final ({len(df)} eventos): {len(self.final_feature_names)} dimensões isoladas.")
        keep = self.final_feature_names + ["UserID", "Time", "Is_Anomaly"]
        return df[[c for c in keep if c in df.columns]].fillna(0)

    # =========================================================
    # BUILDERS (ATUALIZADOS COM NOVOS CAMPOS DO WINDOWS)
    # =========================================================
    def build_rdp(self, df):
        """Avalia acessos RDP. Agora aproveita o LogonType (10) se existir."""
        if "LogonType" in df.columns:
            # Muito mais preciso: Evento Logon + Tipo de Logon exato do RDP
            df["rdp_event"] = ((df["EventID"] == 4624) & (df["LogonType"] == 10)).astype(int)
        else:
            df["rdp_event"] = (df["EventID"] == 4624).astype(int)
            
        df["rdp_rate"] = df.groupby("UserID")["rdp_event"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum().fillna(0))
        return df

    def build_temporal(self, df):
        df["hour"] = df["datetime"].dt.hour
        df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*df["hour"]/24), np.cos(2*np.pi*df["hour"]/24)
        df["out_of_hours"] = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
        
        user_mean_hour = df.groupby("UserID")["hour"].transform(lambda x: x.expanding().mean().shift(1))
        df["hour_dev"] = abs(df["hour"] - user_mean_hour).fillna(0)
        return df

    def build_volume(self, df):
        df["event_count"] = df.groupby("UserID")["Time"].transform(lambda x: x.shift(1).rolling(30, min_periods=1).count().fillna(0))
        
        v_stats = df.groupby("UserID")["event_count"]
        roll_mean = v_stats.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
        roll_std = v_stats.transform(lambda x: x.shift(1).rolling(20, min_periods=1).std())
        
        df["event_z"] = ((df["event_count"] - roll_mean) / (roll_std + 1e-6)).fillna(0)
        return df

    def build_lateral(self, df):
        """Mede movimento lateral. Agora mapeia mudança de IP e LogonType 3."""
        df["host_changed"] = df.groupby("UserID")["LogHost_enc"].diff().fillna(0).ne(0).astype(int)
        df["unique_hosts"] = df.groupby("UserID")["host_changed"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum().fillna(0) + 1)
        
        if "IpAddress_enc" in df.columns:
            df["ip_changed"] = df.groupby("UserID")["IpAddress_enc"].diff().fillna(0).ne(0).astype(int)
            df["unique_ips"] = df.groupby("UserID")["ip_changed"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum().fillna(0) + 1)
            
        if "LogonType" in df.columns:
            # Movimento lateral tipicamente usa logon de rede (3) ou uso explícito de credencial (4648)
            df["network_logon"] = (((df["EventID"] == 4624) & (df["LogonType"] == 3)) | (df["EventID"] == 4648)).astype(int)
            df["network_logon_rate"] = df.groupby("UserID")["network_logon"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum().fillna(0))

        return df

    def build_auth_failure(self, df):
        """Monitoriza falhas. Agora utiliza os códigos de erro hexadecimais (SubStatus)."""
        df["is_failed"] = (df["EventID"] == 4625).astype(int)
        df["fail_rate"] = df.groupby("UserID")["is_failed"].transform(lambda x: x.shift(1).rolling(15, min_periods=1).sum().fillna(0))
        
        if "SubStatus" in df.columns:
            # 0xC000006A (Senha Errada), 0xC0000234 (Conta Bloqueada)
            sub_str = df["SubStatus"].astype(str).str.upper()
            df["bad_password"] = ((df["EventID"] == 4625) & (sub_str.str.contains("0XC000006A"))).astype(int)
            df["account_locked"] = ((df["EventID"] == 4625) & (sub_str.str.contains("0XC0000234"))).astype(int)
            
            df["bad_pwd_rate"] = df.groupby("UserID")["bad_password"].transform(lambda x: x.shift(1).rolling(15, min_periods=1).sum().fillna(0))
            df["lock_rate"] = df.groupby("UserID")["account_locked"].transform(lambda x: x.shift(1).rolling(15, min_periods=1).sum().fillna(0))
            
        return df

    def build_golden_ticket(self, df):
        """Avalia Golden Ticket. Agora usa o pacote de autenticação (Kerberos)."""
        df["is_golden_event"] = df["EventID"].isin([4672, 4769]).astype(int)
        df["golden_ticket_score"] = df.groupby("UserID")["is_golden_event"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum().fillna(0))
        
        if "AuthenticationPackageName" in df.columns:
            df["kerberos_auth"] = (df["AuthenticationPackageName"].astype(str).str.lower().str.contains("kerberos")).astype(int)
            df["kerberos_rate"] = df.groupby("UserID")["kerberos_auth"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum().fillna(0))

        return df

    def build_process(self, df):
        df["rare_process"] = (df["ProcessName_enc"] < 0.02).astype(int)
        df["rare_process_rate"] = df.groupby("UserID")["rare_process"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum().fillna(0))
        return df

    def build_privilege(self, df):
        if "ProcessName" in df.columns:
            proc_lower = df["ProcessName"].fillna("").astype(str).str.lower()
            admin_tools = ["powershell.exe", "cmd.exe", "wmic.exe", "ntdsutil.exe", "psexec.exe", "mmc.exe", "vssadmin.exe"]
            
            df["admin_activity"] = proc_lower.isin(admin_tools).astype(int)
            df["admin_rate"] = df.groupby("UserID")["admin_activity"].transform(lambda x: x.shift(1).rolling(15, min_periods=1).sum().fillna(0))
        return df