#NOVA VERSÃO DO DATASET ADAPTER
import pandas as pd
import os
import logging
import glob
import numpy as np

class DatasetAdapter:
    
    def __init__(self, config):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.sampling_cfg = config.get("sampling", {})
        self.mode = self.sampling_cfg.get("mode", "full")
        self.target_col = self.data_cfg.get("target_col", "Is_Anomaly")
        self.label_filename = self.data_cfg.get("label_filename", "labels.csv")

    def _normalize(self, df):
        """Padroniza nomes, tipos e trata tempos com estratégias do config sem disparar warnings."""
        if df.empty: return df
        df.columns = [str(c).strip() for c in df.columns]
        
        maps = {
            'userid': 'UserID', 'user': 'UserID', 'username': 'UserID', 
            'employee_name': 'UserID', 'src_user': 'UserID',
            'timestamp': 'Time', 'time': 'Time', 'date': 'Time', 'datetime': 'Time',
            'loghost': 'LogHost', 'pc': 'LogHost', 'host': 'LogHost', 
            'machine': 'LogHost', 'source_computer': 'LogHost',
            'eventid': 'EventID', 'id': 'EventID', 'event_id': 'EventID',
            'processname': 'ProcessName', 'process': 'ProcessName', 
            'proc': 'ProcessName', 'filename': 'ProcessName'
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
        
        strat = self.data_cfg.get("time_nan_strategy", "zero")
        if 'Time' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Time']):
                try:
                    temp_dt = pd.to_datetime(df['Time'], errors='coerce')
                    df['Time'] = temp_dt.astype('int64') / 1e9
                except Exception as e:
                    logging.error(f"🔥 Erro na conversão de tempo: {e}")
            
            # Limpeza de infinitos
            df['Time'] = df['Time'].replace([np.inf, -np.inf], np.nan)
            
            # Aplicação da estratégia de imputação sem usar inplace=True para evitar FutureWarning
            if strat == "min":
                min_time = df['Time'].min(skipna=True)
                df['Time'] = df['Time'].fillna(min_time if not pd.isna(min_time) else 0.0)
            elif strat == "zero":
                df['Time'] = df['Time'].fillna(0.0)
                
        return df

    def _apply_auto_labels(self, df, resource_path, specific_label_path=None):
        if self.target_col in df.columns and df[self.target_col].sum() > 0:
            return df

        label_file = None
        if specific_label_path and os.path.exists(specific_label_path):
            label_file = specific_label_path
        elif os.path.isdir(resource_path):
            candidate = os.path.join(resource_path, self.label_filename)
            if os.path.exists(candidate): label_file = candidate
        elif os.path.exists(os.path.join(os.path.dirname(resource_path), self.label_filename)):
            label_file = os.path.join(os.path.dirname(resource_path), self.label_filename)

        if not label_file:
            if self.target_col not in df.columns: df[self.target_col] = 0
            return df

        try:
            label_df = self._normalize(pd.read_csv(label_file))
            label_df[self.target_col] = 1
            
            df['Time'] = df['Time'].fillna(0.0)
            df['_t_key'] = (df['Time'] * 1000).round(0).astype(np.int64)
            
            if 'Time' in label_df.columns:
                label_df['Time'] = label_df['Time'].fillna(0.0)
                label_df['_t_key'] = (label_df['Time'] * 1000).round(0).astype(np.int64)
            else:
                label_df['_t_key'] = 0

            keys = ['UserID', '_t_key']
            if 'LogHost' in df.columns and 'LogHost' in label_df.columns: 
                keys.append('LogHost')
            if 'EventID' in df.columns and 'EventID' in label_df.columns: 
                keys.append('EventID')
            
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
            print(f"🚩 Gabarito '{os.path.basename(label_file)}' aplicado com sucesso.")
            
        except Exception as e:
            print(f"❌ Erro nos labels: {e}")
            if self.target_col not in df.columns: df[self.target_col] = 0
            
        return df

    def _load_resource(self, path, sample_size):
        if not path or not os.path.exists(path): return pd.DataFrame()
        files = sorted(glob.glob(os.path.join(path, "*.*"))) if os.path.isdir(path) else [path]
        files = [f for f in files if self.label_filename not in f]
            
        df_list = []
        rows_acc = 0
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            nrows = None
            if self.mode == "debug" and sample_size > 0:
                nrows = max(0, sample_size - rows_acc)
                if nrows == 0: break
            try:
                temp = pd.read_json(f, lines=True, nrows=nrows) if "json" in ext else pd.read_csv(f, nrows=nrows)
                if not temp.empty:
                    df_list.append(temp)
                    rows_acc += len(temp)
                if self.mode == "debug" and rows_acc >= sample_size: break
            except:
                continue
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    def load(self):
        print(f"📂 Carregamento em curso (Modo: {self.mode})")
        results = []
        subsets = [("train_path", "train_size", "train_label_path"),
                   ("val_path", "val_size", "val_label_path"),
                   ("test_path", "test_size", "test_label_path")]
        
        for p_key, s_key, l_key in subsets:
            path = self.data_cfg.get(p_key)
            size = self.sampling_cfg.get(s_key, 0)
            spec_label = self.data_cfg.get(l_key)
            
            df = self._load_resource(path, size)
            df = self._normalize(df)
            df = self._apply_auto_labels(df, path, spec_label)
            
            if not df.empty:
                for col in df.columns:
                    if col != 'Time' and df[col].isna().any():
                        # Substituição do inplace=True para evitar warnings futuras no Pandas
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna('desconhecido')
                
            results.append(df)
            print(f"✅ {p_key} pronto: {len(df)} registros.")
            
        valid_dfs = [df for df in results if not df.empty]
        if valid_dfs:
            original_cols = set(valid_dfs[0].columns)
            common_cols = original_cols.copy()
            
            for df in valid_dfs[1:]:
                common_cols.intersection_update(df.columns)
                
            for mandatory in ['UserID', 'Time', self.target_col]:
                if mandatory not in common_cols:
                    common_cols.add(mandatory)
                
            common_cols = sorted(list(common_cols))
            print(f"🔧 Colunas comuns: {len(common_cols)} (removidas {len(original_cols) - len(common_cols)})")
            
            for i in range(len(results)):
                if not results[i].empty:
                    if self.target_col not in results[i].columns:
                        results[i][self.target_col] = 0
                    results[i] = results[i][common_cols]
                
        return results[0], results[1], results[2]
