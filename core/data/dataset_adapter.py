import os
import json
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetAdapter:
    def __init__(self, config):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.sampling_cfg = config.get("sampling", {})
        self.output_dir = config.get("output_dir", "./results/default_run")
        self.target_col = self.data_cfg.get("target_col", "Is_Anomaly")
        self.label_filename = self.data_cfg.get("label_filename", "labels.csv")
        self.mode = self.sampling_cfg.get("mode", "full")

    # =========================================================
    # 🔥 RESTAURADO: Normalização completa (mapeamento de colunas)
    # =========================================================
    def _normalize(self, df):
        if df is None or df.empty:
            return df
        df.columns = [str(c).strip() for c in df.columns]
        
        maps = {
            'userid': 'UserID', 'user': 'UserID', 'username': 'UserID', 'src_user': 'UserID',
            'timestamp': 'Time', 'time': 'Time', 'date': 'Time', 'datetime': 'Time',
            'loghost': 'LogHost', 'pc': 'LogHost', 'host': 'LogHost', 'machine': 'LogHost',
            'eventid': 'EventID', 'id': 'EventID', 'event_id': 'EventID',
            'processname': 'ProcessName', 'process': 'ProcessName', 'proc': 'ProcessName',
            'filename': 'ProcessName'
        }
        
        cols_low = {str(c).lower(): c for c in df.columns}
        rename_map = {cols_low[k]: v for k, v in maps.items() if k in cols_low}
        if rename_map:
            df = df.rename(columns=rename_map)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
        
        if 'UserID' in df.columns:
            df['UserID'] = df['UserID'].astype(str)
        if 'EventID' in df.columns:
            df['EventID'] = pd.to_numeric(df['EventID'], errors='coerce').fillna(0).astype(np.int64)
        
        # Tratamento de tempo (estratégias do config)
        if 'Time' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Time']):
                try:
                    temp_dt = pd.to_datetime(df['Time'], errors='coerce')
                    df['Time'] = temp_dt.astype('int64') / 1e9
                except:
                    pass
            df['Time'] = df['Time'].replace([np.inf, -np.inf], np.nan)
            strat = self.data_cfg.get("time_nan_strategy", "zero")
            if strat == "min":
                min_time = df['Time'].min(skipna=True)
                df['Time'] = df['Time'].fillna(min_time if not pd.isna(min_time) else 0.0)
            elif strat == "zero":
                df['Time'] = df['Time'].fillna(0.0)
        return df

    # =========================================================
    # 🔥 RESTAURADO: Merge com labels.csv
    # =========================================================
    def _apply_auto_labels(self, df, resource_path, specific_label_path=None):
        if df is None or df.empty:
            return df
        if self.target_col in df.columns and df[self.target_col].sum() > 0:
            return df

        label_file = None
        if specific_label_path and os.path.exists(specific_label_path):
            label_file = specific_label_path
        elif os.path.isdir(resource_path):
            candidate = os.path.join(resource_path, self.label_filename)
            if os.path.exists(candidate):
                label_file = candidate
        elif os.path.exists(os.path.join(os.path.dirname(resource_path), self.label_filename)):
            label_file = os.path.join(os.path.dirname(resource_path), self.label_filename)

        if not label_file:
            if self.target_col not in df.columns:
                df[self.target_col] = 0
            return df

        try:
            label_df = pd.read_csv(label_file)
            label_df = self._normalize(label_df)
            label_df[self.target_col] = 1

            df['_t_key'] = (df['Time'].fillna(0.0) * 1000).round(0).astype(np.int64)
            label_df['_t_key'] = (label_df['Time'].fillna(0.0) * 1000).round(0).astype(np.int64)

            keys = ['UserID', '_t_key']
            for col in ['LogHost', 'EventID']:
                if col in df.columns and col in label_df.columns:
                    keys.append(col)

            valid_keys = [k for k in keys if k in label_df.columns and k in df.columns]
            subset = label_df[valid_keys + [self.target_col]].drop_duplicates(subset=valid_keys)
            df = df.merge(subset, on=valid_keys, how='left', suffixes=('', '_lbl'))

            lbl_col = f"{self.target_col}_lbl"
            if lbl_col in df.columns:
                df[self.target_col] = df[lbl_col].fillna(0).astype(int)
                df.drop(columns=[lbl_col], inplace=True)
            elif self.target_col not in df.columns:
                df[self.target_col] = 0

            df.drop(columns=['_t_key'], inplace=True, errors='ignore')
            print(f"🚩 Gabarito '{os.path.basename(label_file)}' aplicado.")
        except Exception as e:
            print(f"❌ Erro nos labels: {e}")
            if self.target_col not in df.columns:
                df[self.target_col] = 0
        return df

    # =========================================================
    # 🔥 RESTAURADO: Leitura robusta com glob
    # =========================================================
    def _load_resource(self, path, sample_size=0):
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        
        files = sorted(glob.glob(os.path.join(path, "*.*"))) if os.path.isdir(path) else [path]
        files = [f for f in files if self.label_filename not in f]

        df_list = []
        rows_acc = 0
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            nrows = None
            if self.mode == "debug" and sample_size > 0:
                nrows = max(0, sample_size - rows_acc)
                if nrows == 0:
                    break
            try:
                if "json" in ext:
                    temp = pd.read_json(f, lines=True, nrows=nrows)
                else:
                    temp = pd.read_csv(f, nrows=nrows)
                if not temp.empty:
                    df_list.append(temp)
                    rows_acc += len(temp)
                if self.mode == "debug" and rows_acc >= sample_size:
                    break
            except:
                continue
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    # =========================================================
    # 🆕 NOVO (bom): Split Dinâmico com Auditoria
    # =========================================================
    def _perform_dynamic_split(self, master_df):
        split_mode = self.sampling_cfg.get("split_strategy", "chronological")
        ratios = self.sampling_cfg.get("split_ratios", {"train": 0.7, "val": 0.2, "test": 0.1})
        train_ratio, val_ratio = ratios.get("train", 0.7), ratios.get("val", 0.2)
        test_ratio = ratios.get("test", 0.1)
        seed = self.config.get("training", {}).get("seed", 42)

        print(f"🔪 Split Dinâmico ({split_mode.upper()}) | {train_ratio}/{val_ratio}/{test_ratio}")

        if split_mode == "chronological":
            if 'Time' in master_df.columns:
                master_df = master_df.sort_values(by='Time').reset_index(drop=True)
            n = len(master_df)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            train_df, val_df, test_df = master_df.iloc[:train_end].copy(), master_df.iloc[train_end:val_end].copy(), master_df.iloc[val_end:].copy()
        else:  # random
            train_val_df, test_df = train_test_split(master_df, test_size=test_ratio, random_state=seed, shuffle=True)
            relative_val_ratio = val_ratio / (train_ratio + val_ratio)
            train_df, val_df = train_test_split(train_val_df, test_size=relative_val_ratio, random_state=seed, shuffle=True)
            # Reordena para manter coerência temporal interna
            for df in [train_df, val_df, test_df]:
                if 'Time' in df.columns:
                    df.sort_values(by='Time', inplace=True)
                df.reset_index(drop=True, inplace=True)

        # Auditoria
        os.makedirs(self.output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(self.output_dir, "train_audit.csv"), index=False)
        val_df.to_csv(os.path.join(self.output_dir, "val_audit.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "test_audit.csv"), index=False)
        print(f"   ✅ Auditoria salva em: {self.output_dir}")
        print(f"   📊 Tamanhos: Treino({len(train_df)}) | Val({len(val_df)}) | Teste({len(test_df)})")
        return train_df, val_df, test_df

    # =========================================================
    # 🔥 RESTAURADO: Alinhamento de colunas comuns
    # =========================================================
    def load(self):
        dataset_path = self.data_cfg.get("dataset_path", "")
        train_path = self.data_cfg.get("train_path", "")
        val_path = self.data_cfg.get("val_path", "")
        test_path = self.data_cfg.get("test_path", "")

        limit_train = self.sampling_cfg.get("train_size", 0)
        limit_val = self.sampling_cfg.get("val_size", 0)
        limit_test = self.sampling_cfg.get("test_size", 0)

        # Modo Smart A: Arquivo Único (Split Dinâmico)
        if dataset_path and os.path.exists(dataset_path):
            master_df = self._load_resource(dataset_path, limit_train)
            master_df = self._normalize(master_df)
            master_df = self._apply_auto_labels(master_df, dataset_path)
            return self._perform_dynamic_split(master_df)

        # Modo Smart B: Arquivos já separados (Clássico)
        print(f"📂 Carregamento Clássico (Modo: {self.mode})")
        train_df = self._load_resource(train_path, limit_train)
        val_df = self._load_resource(val_path, limit_val)
        test_df = self._load_resource(test_path, limit_test)

        train_df = self._normalize(train_df)
        val_df = self._normalize(val_df)
        test_df = self._normalize(test_df)

        train_df = self._apply_auto_labels(train_df, train_path)
        val_df = self._apply_auto_labels(val_df, val_path)
        test_df = self._apply_auto_labels(test_df, test_path)

        # ✅ Alinhamento de colunas (restaurado)
        valid_dfs = [df for df in [train_df, val_df, test_df] if df is not None and not df.empty]
        if valid_dfs:
            common_cols = set(valid_dfs[0].columns)
            for df in valid_dfs[1:]:
                common_cols.intersection_update(df.columns)
            for mandatory in ['UserID', 'Time', self.target_col]:
                if mandatory not in common_cols:
                    common_cols.add(mandatory)
            common_cols = sorted(list(common_cols))
            
            for i, df in enumerate([train_df, val_df, test_df]):
                if df is not None and not df.empty:
                    if self.target_col not in df.columns:
                        df[self.target_col] = 0
                    df = df[common_cols]
                    if i == 0:
                        train_df = df
                    elif i == 1:
                        val_df = df
                    else:
                        test_df = df

        print(f"   ✅ Treino: {len(train_df) if train_df is not None else 0} registros.")
        print(f"   ✅ Val: {len(val_df) if val_df is not None else 0} registros.")
        print(f"   ✅ Teste: {len(test_df) if test_df is not None else 0} registros.")
        return train_df, val_df, test_df
