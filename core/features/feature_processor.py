# core/features/feature_processor.py
#
# Feature engineering para detector UEBA.
# - Anti-leakage: fit() só usa eventos normais (Is_Anomaly == 0)
# - transform() em modo estrito NÃO reajusta o estado
# - Features "_rate" representam estado no instante t (incluem evento atual);
#   features "_count_*" são janelas temporais causais (olham para trás)
# - Frequências de entidade usam baseline offline do treino
#   (declarado como hipótese: offline training, não detecção online estrita)
# - final_feature_names é congelado após o primeiro transform()
# - Ordenação determinística (kind='stable')

import pandas as pd
import numpy as np
import logging
import pickle
import os

# =========================================================
# REGISTRY
# =========================================================
FEATURE_REGISTRY = {
    "auth_failure":      {"required": ["Time", "UserID", "EventID"], "builder": "build_auth_failure"},
    "lateral_movement":  {"required": ["Time", "UserID", "LogHost", "EventID"], "builder": "build_lateral"},
    "temporal_anomaly":  {"required": ["Time", "UserID"],            "builder": "build_temporal"},
    "volume_anomaly":    {"required": ["Time", "UserID"],            "builder": "build_volume"},
    "rdp_anomaly":       {"required": ["EventID", "UserID"],         "builder": "build_rdp"},
    "golden_ticket":     {"required": ["EventID"],                   "builder": "build_golden_ticket"},
    "process_anomaly":   {"required": ["ProcessName", "UserID"],     "builder": "build_process"},
    "privilege_anomaly": {"required": ["UserID", "ProcessName"],     "builder": "build_privilege"},
}


class FeatureProcessor:
    """
    Constrói o vetor comportamental por evento, com anti-leakage no fit()
    e features causais por usuário.

    Em modo strict=True (default), transform() sem fit() levanta RuntimeError.
    Em inferência, o estado do treino deve ser carregado via
    meta['feature_processor'], nunca reajustado sobre o dataset de teste.
    """

    def __init__(self, config, strict: bool = True):
        self.config = config
        self.active_builders = []
        self.features_frozen = False          # gate de _detect_features
        self.final_feature_names = []         # congelado após primeiro transform
        self.output_dir = config.get("output_dir")

        feature_cfg = config.get("features", {})
        self.rate_window = max(1, int(feature_cfg.get("rate_window", 20)))
        self.excluded_features = set(feature_cfg.get("exclude_features", []))

        self.historical_state = {}
        self.is_fitted = False
        self.strict = strict

    # =========================================================
    # PERSISTÊNCIA DE ESTADO (auditoria)
    # =========================================================
    def save_state(self) -> dict:
        return {
            "historical_state": self.historical_state,
            "active_builders": list(self.active_builders),
            "final_feature_names": list(self.final_feature_names),
            "features_frozen": bool(self.features_frozen),
            "rate_window": int(self.rate_window),
            "is_fitted": bool(self.is_fitted),
        }

    def load_state(self, state: dict) -> None:
        self.historical_state = state["historical_state"]
        self.active_builders = list(state["active_builders"])
        self.final_feature_names = list(state["final_feature_names"])
        self.features_frozen = bool(state["features_frozen"])
        self.rate_window = int(state.get("rate_window", self.rate_window))
        self.is_fitted = bool(state.get("is_fitted", True))

    # =========================================================
    # AUXILIARES
    # =========================================================
    def _get_enabled_groups(self):
        feat_cfg = self.config.get("features", {})
        enabled = feat_cfg.get("enabled_groups", [])
        if not enabled:
            return set(FEATURE_REGISTRY.keys()) if feat_cfg.get("enabled", True) else set()
        return set(enabled)

    def _detect_features(self, df):
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

    def _rolling_rate(self, df, source_col):
        """
        Fração de eventos do tipo `source_col` na janela dos últimos
        `rate_window` eventos. Inclui o evento atual → representa
        'estado no instante t' (declarado na metodologia).
        """
        return df.groupby("UserID", sort=False)[source_col].transform(
            lambda v: v.rolling(self.rate_window, min_periods=1).mean()
        )

    def _time_window_count(self, df, event_col, window_seconds, exclusive=False):
        """
        Conta eventos com event_col == 1 nos `window_seconds` anteriores a cada
        evento, por usuário. Estritamente causal (só olha para trás).

        Semântica para empates de Time: eventos com timestamp idêntico são
        contados em ordem estável (o "próximo" no sort estável vê o "anterior"
        como passado). `exclusive=True` exclui o próprio evento.
        """
        out = pd.Series(0.0, index=df.index, dtype=float)
        if df.empty or event_col not in df.columns or "Time" not in df.columns:
            return out

        for _, idx in df.groupby("UserID", sort=False).groups.items():
            sub = df.loc[idx, ["Time", event_col]].sort_values("Time", kind="stable")
            times = sub["Time"].to_numpy(dtype=float)
            vals = sub[event_col].to_numpy(dtype=float)

            cumsum = np.concatenate([[0.0], np.cumsum(vals)])
            left = np.searchsorted(times, times - window_seconds, side="left")
            sums = cumsum[np.arange(len(times)) + 1] - cumsum[left]

            if exclusive:
                sums = sums - vals

            out.loc[sub.index] = sums

        return out

    # =========================================================
    # FIT (APENAS NORMAIS → anti-leakage)
    # =========================================================
    def fit(self, df):
        if df is None or df.empty:
            return

        target_col = self.config.get("data", {}).get("target_col", "Is_Anomaly")
        if target_col in df.columns:
            df_temp = df[df[target_col] == 0].copy()
            if df_temp.empty:
                if self.strict:
                    raise RuntimeError(
                        "FeatureProcessor.fit: nenhum evento 'Is_Anomaly==0' no treino. "
                        "Recusando ajustar baseline sobre eventos anômalos. "
                        "Verifique o split ou rode com strict=False explicitamente."
                    )
                print("⚠️ ATENÇÃO: Nenhum evento 'normal'. Usando base total (pode gerar viés).")
                df_temp = df.copy()
        else:
            df_temp = df.copy()

        print("🧠 [FeatureProcessor] A processar os parâmetros históricos (APENAS NORMAIS)...")

        if "Time" not in df_temp.columns:
            print("⚠️ Coluna 'Time' não encontrada. Impossível extrair features temporais.")
            return

        df_temp["datetime"] = pd.to_datetime(df_temp["Time"], unit="s")
        user_counts = df_temp.groupby("UserID", sort=False).size()

        # ---------- Frequências por usuário ----------
        if "LogHost" in df_temp.columns:
            self.historical_state["loghost_freq"] = (
                df_temp.groupby(["UserID", "LogHost"]).size() / user_counts
            )
            # Set de hosts por usuário (para user_host_new). Chaves forçadas a str.
            self.historical_state["user_hosts_seen"] = {
                str(uid): set(hosts.dropna().astype(str))
                for uid, hosts in df_temp.groupby("UserID", sort=False)["LogHost"]
            }

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

        # ---------- Baseline temporal por usuário ----------
        df_temp["hour"] = df_temp["datetime"].dt.hour
        self.historical_state["hour_mean"] = df_temp.groupby("UserID")["hour"].mean()
        self.historical_state["hour_std"] = df_temp.groupby("UserID")["hour"].std()

        # ---------- Baseline de volume diário por usuário ----------
        df_temp["date"] = df_temp["datetime"].dt.date
        daily_counts = df_temp.groupby(["UserID", "date"]).size()
        self.historical_state["daily_volume_mean"] = daily_counts.groupby("UserID").mean()
        self.historical_state["daily_volume_std"] = daily_counts.groupby("UserID").std()

        self.is_fitted = True
        print("   ✅ Baseline comportamental (UEBA) estabelecido com sucesso.")

    # =========================================================
    # TRANSFORM
    # =========================================================
    def transform(self, df):
        if df is None or df.empty:
            return df

        if not self.is_fitted:
            if getattr(self, "strict", True):
                raise RuntimeError(
                    "FeatureProcessor.transform() chamado sem .fit() prévio. "
                    "Em inferência, carregue o estado do treino via meta['feature_processor']. "
                    "Em treino, chame .fit(train_df) antes de .transform()."
                )
            logging.warning("FeatureProcessor.transform sem .fit prévio — modo permissivo.")
            self.fit(df)

        df = df.copy()
        df = df.sort_values(["UserID", "Time"], kind="stable").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["Time"], unit="s")

        # ---------- 1. Frequency encoding por usuário ----------
        encoders = [
            ("LogHost",                    "loghost_freq",  "LogHost_enc"),
            ("ProcessName",                "process_freq",  "ProcessName_enc"),
            ("IpAddress",                  "ip_freq",       "IpAddress_enc"),
            ("SubStatus",                  "substatus_freq","SubStatus_enc"),
            ("AuthenticationPackageName",  "authpkg_freq",  "AuthPackage_enc"),
        ]
        for col, state_key, enc_col in encoders:
            if col in df.columns and state_key in self.historical_state:
                freq_map = self.historical_state[state_key].reset_index(name=enc_col)
                df["_ord"] = range(len(df))
                df = df.merge(freq_map, on=["UserID", col], how="left")
                df = df.sort_values("_ord", kind="stable").drop(columns=["_ord"]).reset_index(drop=True)
                df[enc_col] = df[enc_col].fillna(0.0)

        if "_ord" in df.columns:
            df = df.drop_duplicates(subset=["_ord"]).reset_index(drop=True)

        # ---------- 2. Builders ----------
        if not self.features_frozen:
            self._detect_features(df)

        for builder in self.active_builders:
            if hasattr(self, builder):
                df = getattr(self, builder)(df)

        # ---------- 3. Interações cruzadas ----------
        df = self._add_interactions(df)

        # ---------- 4. Seleção final (congelada após 1º transform) ----------
        meta_cols = [
            "datetime", "LogHost", "ProcessName", "UserName", "SourceIP",
            "UserID", "Time", "session_id", "AnomalyType", "EventID", "hour",
            "IpAddress", "SubStatus", "AuthenticationPackageName",
            "date", "new_session", "time_diff", "sub_session_id", "chunk_id",
            "day_of_week", "time_sin", "time_cos", "is_weekend",
            "Is_Anomaly", "_ord",
            "is_failed", "is_success", "bad_password", "bad_username",
            "account_locked", "rdp_event", "admin_activity",
            "network_logon", "is_kerberos_logon", "recent_4768",
            "rare_process",
        ]

        if not self.final_feature_names:
            # Primeira chamada: computa e CONGELA
            self.final_feature_names = [
                c for c in df.columns
                if c not in meta_cols
                and c not in self.excluded_features
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            print(f"📊 Vetor de features CONGELADO: {len(self.final_feature_names)} dimensões.")
        else:
            # Chamadas seguintes: garante que todas as colunas congeladas existem
            # (usuários/datasets novos podem não gerar alguma feature; zera).
            for c in self.final_feature_names:
                if c not in df.columns:
                    df[c] = 0.0

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "feature_metadata.pkl"), "wb") as f:
                pickle.dump({
                    "features": self.final_feature_names,
                    "state": self.historical_state,
                }, f)

        print(f"📊 Vetor Final ({len(df)} eventos): {len(self.final_feature_names)} dimensões.")

        keep = self.final_feature_names + ["UserID", "Time", "Is_Anomaly", "AnomalyType"]
        keep = list(dict.fromkeys(keep))
        return df[[c for c in keep if c in df.columns]].fillna(0)

    # =========================================================
    # BUILDERS
    # =========================================================
    def build_temporal(self, df):
        df["hour"] = df["datetime"].dt.hour

        if "hour_mean" in self.historical_state:
            mean = df["UserID"].map(self.historical_state["hour_mean"])
            std = df["UserID"].map(self.historical_state["hour_std"]).fillna(3.0).clip(lower=0.5)

            raw = (df["hour"] - mean).abs()
            df["hour_dev"] = np.minimum(raw, 24 - raw).fillna(0)  # distância circular
            df["hour_z"] = (df["hour_dev"] / std).fillna(0)
            df["out_of_hours"] = (df["hour_z"] > 2.0).astype(int)
        else:
            df["out_of_hours"] = ((df["hour"] < 7) | (df["hour"] > 21)).astype(int)
            df["hour_dev"] = 0.0
            df["hour_z"] = 0.0

        return df

    def build_auth_failure(self, df):
        df["is_failed"] = (df["EventID"] == 4625).astype(int)
        df["is_success"] = (df["EventID"] == 4624).astype(int)

        df["fail_rate"] = self._rolling_rate(df, "is_failed")
        df["fail_count_5m"] = self._time_window_count(df, "is_failed", 300)
        df["fail_count_15m"] = self._time_window_count(df, "is_failed", 900)
        df["fail_count_60m"] = self._time_window_count(df, "is_failed", 3600)

        prev_fails = self._time_window_count(df, "is_failed", 300, exclusive=True)
        df["fail_then_success"] = ((df["is_success"] == 1) & (prev_fails > 2)).astype(int)

        if "SubStatus" in df.columns:
            sub = df["SubStatus"].astype(str).str.upper()
            df["bad_password"] = ((df["EventID"] == 4625) & sub.str.contains("0XC000006A")).astype(int)
            df["bad_username"] = ((df["EventID"] == 4625) & sub.str.contains("0XC0000064")).astype(int)
            df["account_locked"] = ((df["EventID"] == 4625) & sub.str.contains("0XC0000234")).astype(int)

            df["bad_pwd_rate"] = self._rolling_rate(df, "bad_password")
            df["bad_pwd_count_15m"] = self._time_window_count(df, "bad_password", 900)

            df["lock_rate"] = self._rolling_rate(df, "account_locked")
            df["lock_count_60m"] = self._time_window_count(df, "account_locked", 3600)

        return df

    def build_lateral(self, df):
        if "LogHost" in df.columns:
            prev_host = df.groupby("UserID", sort=False)["LogHost"].shift()
            df["host_changed"] = (
                prev_host.notna()
                & (df["LogHost"].astype(str) != prev_host.astype(str))
            ).astype(int)

            def _cumnunique(x):
                seen, out = set(), []
                for v in x:
                    seen.add(v)
                    out.append(len(seen))
                return out

            df["unique_hosts"] = df.groupby("UserID", sort=False)["LogHost"].transform(_cumnunique)

            if "user_hosts_seen" in self.historical_state:
                seen_map = self.historical_state["user_hosts_seen"]
                df["user_host_new"] = [
                    0 if str(h) in seen_map.get(str(u), set()) else 1
                    for u, h in zip(df["UserID"], df["LogHost"])
                ]

        if "IpAddress" in df.columns:
            prev_ip = df.groupby("UserID", sort=False)["IpAddress"].shift()
            df["ip_changed"] = (
                prev_ip.notna()
                & (df["IpAddress"].astype(str) != prev_ip.astype(str))
            ).astype(int)

            def _cumnunique_ip(x):
                seen, out = set(), []
                for v in x:
                    seen.add(v)
                    out.append(len(seen))
                return out

            df["unique_ips"] = df.groupby("UserID", sort=False)["IpAddress"].transform(_cumnunique_ip)

        if "LogonType" in df.columns:
            df["network_logon"] = (
                ((df["EventID"] == 4624) & (df["LogonType"] == 3))
                | (df["EventID"] == 4648)
            ).astype(int)
            df["network_logon_rate"] = self._rolling_rate(df, "network_logon")
            df["network_logon_count_60m"] = self._time_window_count(df, "network_logon", 3600)

        return df

    def build_volume(self, df):
        if "date" in df.columns:
            df["event_count"] = df.groupby(["UserID", "date"], sort=False).cumcount() + 1

            if "daily_volume_mean" in self.historical_state:
                mean_map = self.historical_state["daily_volume_mean"]
                std_map = self.historical_state["daily_volume_std"]

                df["vol_mean"] = df["UserID"].map(mean_map).fillna(0)
                df["vol_std"] = df["UserID"].map(std_map).fillna(1)
                df["event_z"] = ((df["event_count"] - df["vol_mean"]) / (df["vol_std"] + 1e-6)).fillna(0)
            else:
                df["event_z"] = 0.0

            df.drop(columns=["vol_mean", "vol_std"], inplace=True, errors="ignore")

        return df

    def build_rdp(self, df):
        if "LogonType" in df.columns:
            df["rdp_event"] = ((df["EventID"] == 4624) & (df["LogonType"] == 10)).astype(int)
        else:
            df["rdp_event"] = (df["EventID"] == 4624).astype(int)

        df["rdp_rate"] = self._rolling_rate(df, "rdp_event")
        df["rdp_count_60m"] = self._time_window_count(df, "rdp_event", 3600)
        return df

    def build_golden_ticket(self, df):
        if "AuthenticationPackageName" in df.columns:
            pkg = df["AuthenticationPackageName"].astype(str).str.lower()
            df["is_kerberos_logon"] = (
                (df["EventID"] == 4624) & pkg.str.contains("kerberos")
            ).astype(int)
            df["is_ntlm"] = pkg.str.contains("ntlm").astype(int)

            if "EventID" in df.columns:
                df["recent_4768"] = self._time_window_count(
                    df.assign(_evt4768=(df["EventID"] == 4768).astype(int)),
                    "_evt4768", 600,
                )
                df.drop(columns=["_evt4768"], inplace=True, errors="ignore")

                df["kerberos_logon_without_tgt"] = (
                    (df["is_kerberos_logon"] == 1) & (df["recent_4768"] == 0)
                ).astype(int)

        return df

    def build_process(self, df):
        if "ProcessName_enc" in df.columns:
            df["rare_process"] = (df["ProcessName_enc"] < 0.02).astype(int)
            df["rare_process_rate"] = self._rolling_rate(df, "rare_process")
            df["rare_process_count_60m"] = self._time_window_count(df, "rare_process", 3600)
        return df

    def build_privilege(self, df):
        if "ProcessName" in df.columns:
            proc_lower = df["ProcessName"].fillna("").astype(str).str.lower()
            admin_tools = [
                "powershell.exe", "cmd.exe", "wmic.exe", "ntdsutil.exe",
                "psexec.exe", "mmc.exe", "vssadmin.exe",
            ]
            df["admin_activity"] = proc_lower.isin(admin_tools).astype(int)
            df["admin_rate"] = self._rolling_rate(df, "admin_activity")
            df["admin_count_60m"] = self._time_window_count(df, "admin_activity", 3600)
        return df

    # =========================================================
    # INTERAÇÕES
    # =========================================================
    def _add_interactions(self, df):
        if "out_of_hours" not in df.columns:
            return df

        ooh = df["out_of_hours"].astype(int)
        for feat in ["admin_activity", "rdp_event", "is_success", "bad_password"]:
            if feat in df.columns:
                df[f"ooh_with_{feat}"] = ooh * df[feat].astype(int)

        return df