# core/trainer.py
# VERSÃO COMPLETA E BLINDADA CONTRA FALHAS SILENCIOSAS E VAZAMENTO DE ESTADO ENTRE SEEDS

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

try:
    from sklearn.ensemble import IsolationForest
    HAS_IFOREST = True
except ImportError:
    HAS_IFOREST = False

class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.sequence_length = config.get("model", {}).get("sequence_length", 10)
        self.scalers = {} 
        self.feature_columns = None
        self.output_dir = config.get("output_dir", "./output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Estado inicial (será resetado a cada execução do run para evitar vazamento de dados)
        self.user_thresholds = {}
        self.global_threshold = 0.0

    # =========================================================
    # MÉTODO AUXILIAR PARA CALCULAR F2-SCORE COM PROTEÇÃO ABSOLUTA
    # =========================================================
    def _compute_f2_score(self, y_true, y_pred):
        """Calcula o F2-Score protegendo contra divisões por zero ou vetores vazios."""
        if y_true is None or len(y_true) == 0:
            return 0.0
            
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Evita divisão por zero retornando zero se não houver predições corretas
        if tp == 0:
            return 0.0
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision == 0 or recall == 0:
            return 0.0
            
        beta = 2.0
        f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        return f2

    def _prepare_and_scale_X(self, group, user_id, fit=False, feature_columns=None):
        if fit and feature_columns is not None:
            self.feature_columns = feature_columns
            
        if self.feature_columns is not None:
            X_num = group.reindex(columns=self.feature_columns, fill_value=0)
        else:
            target = self.config.get("data", {}).get("target_col", "Is_Anomaly")
            drop_cols = ["UserID", "Time", target, "EventID", "datetime", "session_id"]
            X_num = group.drop(columns=[c for c in drop_cols if c in group.columns]).select_dtypes(include=[np.number])
            if fit: 
                self.feature_columns = X_num.columns.tolist()

        X_arr = X_num.values.astype(np.float32)
        
        if X_arr.shape[0] == 0:
            return X_arr

        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_arr)
            self.scalers[str(user_id)] = scaler
        else:
            scaler = self.scalers.get(str(user_id))
            X_scaled = scaler.transform(X_arr) if scaler else X_arr
            
        return X_scaled

    def _build_sequences(self, df, fit=False, return_labels=False, feature_columns=None):
        X_all, indices_all, users_all, labels_all = [], [], [], []
        target = self.config.get("data", {}).get("target_col", "Is_Anomaly")
        
        if return_labels and target not in df.columns:
            raise KeyError(f"❌ Coluna alvo '{target}' não encontrada no DataFrame!")
        
        for user_id, group in df.groupby("UserID"):
            X = self._prepare_and_scale_X(group, user_id, fit=fit, feature_columns=feature_columns)
            if len(X) < self.sequence_length:
                continue
            
            # Janelamento das Features
            windows = sliding_window_view(X, window_shape=self.sequence_length, axis=0)
            windows = np.swapaxes(windows, 1, 2)
            X_all.append(windows)
            
            idx = group.index.values[self.sequence_length - 1:]
            indices_all.append(idx)
            users_all.append(np.full(len(idx), str(user_id)))  # Sempre armazena IDs de usuários como string
            
            if return_labels:
                windows_labels = sliding_window_view(group[target].values, window_shape=self.sequence_length)
                max_labels = np.max(windows_labels, axis=1)
                labels_all.append(max_labels)

        if not X_all:
            return np.array([]), np.array([]), np.array([]), None
        
        X_concat = np.concatenate(X_all)
        indices_concat = np.concatenate(indices_all)
        users_concat = np.concatenate(users_all)
        labels_concat = np.concatenate(labels_all) if labels_all else None
        
        return X_concat, indices_concat, users_concat, labels_concat

    def _calculate_scores(self, X):
        if X is None or len(X) == 0:
            return np.array([])
            
        is_if_model = HAS_IFOREST and isinstance(self.model, IsolationForest)
        model_type_name = type(self.model).__name__.lower()
        
        if is_if_model or "isolationforest" in model_type_name or "iforest" in model_type_name:
            if hasattr(self.model, "decision_function"):
                return -self.model.decision_function(X)
            elif hasattr(self.model, "score_samples"):
                return -self.model.score_samples(X)
            
        if hasattr(self.model, "score"):
            scores = self.model.score(X)
            if self.config.get("model", {}).get("auto_invert_scores", False) and np.mean(scores) < 0:
                return -scores
            return scores
            
        raise ValueError("❌ Erro: O Modelo carregado não possui um método de escoragem válido!")

    def run(self, train_df, val_df, test_df, feature_columns=None):
        # -------------------- RESET DE ESTADO (CRÍTICO PARA MULTI-SEED) --------------------
        self.user_thresholds = {}
        self.global_threshold = 0.0
        
        train_cfg = self.config.get("training", {})
        do_resume = train_cfg.get("resume", False)
        save_model = train_cfg.get("save_model", True)
        thresh_mode = train_cfg.get("threshold_mode", "per_user")
        
        fixed_threshold = train_cfg.get("fixed_threshold", None)
        fixed_percentile = train_cfg.get("fixed_percentile", None)
        
        target = self.config.get("data", {}).get("target_col", "Is_Anomaly")
        
        model_path = os.path.join(self.output_dir, "saved_model")
        meta_path = os.path.join(self.output_dir, "trainer_meta.pkl")

        # -------------------- TREINO --------------------
        if do_resume and os.path.exists(model_path) and os.path.exists(meta_path):
            print("🔄 [Trainer] Checkpoint detectado. Retomando execução...")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.scalers = meta['scalers']
                self.feature_columns = meta['feature_columns']
            if hasattr(self.model, "load"):
                self.model.load(model_path)
        else:
            print(f"🚀 [Trainer] Iniciando treino do zero (Modo: {thresh_mode})...")
            if target in train_df.columns:
                train_df = train_df[train_df[target] == 0].copy()
                print(f"   📊 Treino focado em dados legítimos: {len(train_df)} eventos.")
            else:
                print(f"   ⚠️  Aviso: Coluna '{target}' não foi encontrada no conjunto de treino.")

            X_train, _, _, _ = self._build_sequences(train_df, fit=True, feature_columns=feature_columns)
            
            # Proteção robusta contra janelas vazias no treino
            if X_train is None or len(X_train) == 0:
                raise ValueError(f"❌ Erro Crítico: Conjunto de TREINO gerou 0 janelas para sequence_length={self.sequence_length}. Operação interrompida!")

            if hasattr(self.model, "fit"):
                self.model.fit(X_train)
            
            if save_model:
                if hasattr(self.model, "save"): 
                    self.model.save(model_path)
                with open(meta_path, "wb") as f:
                    pickle.dump({'scalers': self.scalers, 'feature_columns': self.feature_columns}, f)

        # -------------------- VALIDAÇÃO --------------------
        print("\n--- [Trainer] Gerando Janelas de VALIDAÇÃO ---")
        X_val, _, val_users, y_val = self._build_sequences(val_df, fit=False, return_labels=True)
        
        if X_val is None or len(X_val) == 0:
            raise ValueError(f"❌ Erro Crítico: Conjunto de VALIDAÇÃO gerou 0 janelas para sequence_length={self.sequence_length}. Calibração abortada!")

        val_scores = self._calculate_scores(X_val)
        if val_scores is None or len(val_scores) == 0:
            raise ValueError("❌ Erro Crítico: O modelo gerou escores vazios para a validação!")

        if y_val is not None and np.sum(y_val) > 0:
            print(f"   ✅ Validação contém {int(np.sum(y_val))} janelas sinalizadas como anômalas.")
        else:
            print(f"   ❌ ATENÇÃO: Validação livre de anomalias registradas!")

        best_p = 95.0  
        best_f2 = -1.0
        
        print("\n--- [Trainer] Definição de Limiares de Alarme ---")

        # =========================================================
        # OPÇÃO 1: Limiar Absoluto Forçado
        # =========================================================
        if fixed_threshold is not None:
            print(f"⚖️ [YAML] Limiar absoluto fixado pelo usuário em: {fixed_threshold}")
            self.global_threshold = float(fixed_threshold)
            self.user_thresholds = {"__GLOBAL_FALLBACK__": self.global_threshold}
            
            if y_val is not None and len(y_val) > 0:
                y_pred = (val_scores > self.global_threshold).astype(int)
                best_f2 = self._compute_f2_score(y_val, y_pred)
                print(f"   📊 F2-Score medido para este limiar: {best_f2:.4f}")
                
        # =========================================================
        # OPÇÃO 2: Percentil Fixo Forçado
        # =========================================================
        elif fixed_percentile is not None:
            print(f"⚖️ [YAML] Percentil fixado pelo usuário em: {fixed_percentile}%")
            best_p = float(fixed_percentile)
            self.global_threshold = float(np.percentile(val_scores, best_p)) if val_scores.size > 0 else 0.5
            self.user_thresholds = {"__GLOBAL_FALLBACK__": self.global_threshold}
            
            if thresh_mode == "per_user":
                for uid in np.unique(val_users):
                    user_mask = (val_users == uid)
                    if np.sum(user_mask) >= 5: 
                        self.user_thresholds[str(uid)] = float(np.percentile(val_scores[user_mask], best_p))
            
            if y_val is not None and len(y_val) > 0:
                if thresh_mode == "per_user":
                    user_thresh_arr = np.array([self.user_thresholds.get(str(uid), self.global_threshold) for uid in val_users])
                    y_pred = (val_scores > user_thresh_arr).astype(int)
                else:
                    y_pred = (val_scores > self.global_threshold).astype(int)
                
                best_f2 = self._compute_f2_score(y_val, y_pred)
                print(f"   📊 F2-Score medido para este percentil: {best_f2:.4f}")

        # =========================================================
        # OPÇÃO 3: Dinâmico pelo F2-Score (Automático)
        # =========================================================
        else:
            if y_val is not None and np.sum(y_val) > 0:
                print(f"🧠 [AUTO] Otimizando curva de decisão dinamicamente com F2-Score...")
                for p in np.arange(50.0, 99.6, 0.5):
                    t_glob = np.percentile(val_scores, p)
                    if thresh_mode == "per_user":
                        u_t_dict = {str(uid): np.percentile(val_scores[val_users==uid], p) 
                                     for uid in np.unique(val_users) if np.sum(val_users==uid) >= 5}
                        user_thresh_arr = np.array([u_t_dict.get(str(uid), t_glob) for uid in val_users])
                        y_pred = (val_scores > user_thresh_arr).astype(int)
                    else:
                        y_pred = (val_scores > t_glob).astype(int)
                    
                    f2 = self._compute_f2_score(y_val, y_pred)
                    if f2 > best_f2:
                        best_f2, best_p = f2, p
                print(f"✅ [AUTO] Otimização concluída. Percentil ideal: {best_p:.1f}% (F2-Score: {best_f2:.4f})")
            else:
                best_p = 99.0
                print(f"⚠️ [AUTO] Validação limpa. Adotando percentil fixo de segurança: {best_p}%")

            self.global_threshold = float(np.percentile(val_scores, best_p)) if val_scores.size > 0 else 0.5
            self.user_thresholds = {"__GLOBAL_FALLBACK__": self.global_threshold}
            
            if thresh_mode == "per_user":
                for uid in np.unique(val_users):
                    user_mask = (val_users == uid)
                    if np.sum(user_mask) >= 5: 
                        self.user_thresholds[str(uid)] = float(np.percentile(val_scores[user_mask], best_p))
        
        # Salva o arquivo JSON local de thresholds
        with open(os.path.join(self.output_dir, "user_thresholds.json"), "w") as f:
            json.dump(self.user_thresholds, f, indent=4)

        # -------------------- TESTE --------------------
        if test_df is not None and not test_df.empty:
            print("\n--- [Trainer] Gerando Janelas de TESTE ---")
            X_test, test_indices, _, _ = self._build_sequences(test_df, fit=False)
            
            if X_test is None or len(X_test) == 0:
                raise ValueError(f"❌ Erro Crítico: Conjunto de TESTE gerou 0 janelas para sequence_length={self.sequence_length}!")

            test_scores = self._calculate_scores(X_test)
            if test_scores is None or len(test_scores) == 0:
                raise ValueError("❌ Erro Crítico: O modelo gerou escores vazios para o teste!")
            
            np.save(os.path.join(self.output_dir, "test_scores.npy"), test_scores)
            np.save(os.path.join(self.output_dir, "test_indices.npy"), test_indices)
        else:
            print("\n--- [Trainer] Avaliação de TESTE ignorada (test_df não fornecido) ---")

        print("\n--- [Trainer] Execução Finalizada com Sucesso ---")
        print(f"📊 Limiar de Fallback Global: {self.global_threshold:.6f}")
        print(f"📊 Limiares Personalizados de Usuários: {len(self.user_thresholds) - 1} cadastrados.")

        return {
            "val_f2": float(best_f2) if best_f2 != -1 else -1.0, 
            "global_threshold": self.global_threshold
        }