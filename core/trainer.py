# core/trainer.py

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
    def _checkpoint_uses_current_protocol(self, meta_path):
        """Reject checkpoints created before validation was kept out of training."""
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            return meta.get("training_protocol") == "train_val_independent_v3_benign_early_stop"
        except Exception:
            return False

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.sequence_length = config.get("model", {}).get("sequence_length", 10)
        self.scalers = {}
        self.feature_columns = None
        self.output_dir = config.get("output_dir", "./output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.user_thresholds = {}
        self.global_threshold = 0.0

    @staticmethod
    def _select_training_events(train_df, target="Is_Anomaly"):
        """Return benign training events used to fit per-user feature scalers."""
        if target not in train_df.columns:
            return train_df.copy()
        return train_df.loc[train_df[target] == 0].copy()

    def _build_benign_training_sequences(self, train_df, feature_columns=None):
        """Fit scalers on benign train events and keep only all-benign train windows.

        Windows are built in the original event order before filtering, so removing an
        attack cannot make events on either side appear adjacent. The model's internal
        early-stopping split is consequently drawn only from benign training windows.
        The external mixed validation partition remains reserved for threshold tuning.
        """
        target = self.config.get("data", {}).get("target_col", "Is_Anomaly")
        benign_events = self._select_training_events(train_df, target=target)
        for user_id, group in benign_events.groupby("UserID"):
            self._prepare_and_scale_X(
                group, user_id, fit=True, feature_columns=feature_columns
            )

        X_all, indices, users, labels = self._build_sequences(
            train_df, fit=False, return_labels=True
        )
        if labels is not None:
            benign_windows = labels == 0
            X_all = X_all[benign_windows]
            indices = indices[benign_windows]
            users = users[benign_windows]
        return X_all, indices, users, labels

    # =========================================================
    # F2-SCORE (COM PROTEÃ‡ÃƒO)
    # =========================================================
    def _compute_f2_score(self, y_true, y_pred):
        if y_true is None or len(y_true) == 0:
            return 0.0
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if tp == 0:
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision == 0 or recall == 0:
            return 0.0
        beta = 2.0
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    def _personalized_thresholds(self, scores, users, labels, percentile, min_windows=5):
        """Calibrate only users with enough validation windows and positive examples."""
        if labels is None:
            return {}
        scores = np.asarray(scores)
        users = np.asarray(users)
        labels = np.asarray(labels)
        thresholds = {}
        for uid in np.unique(users):
            user_mask = users == uid
            if np.sum(user_mask) >= min_windows and np.any(labels[user_mask] == 1):
                thresholds[str(uid)] = float(np.percentile(scores[user_mask], percentile))
        return thresholds

    # =========================================================
    # PREPARAÃ‡ÃƒO E ESCALONAMENTO
    # =========================================================
    def _prepare_and_scale_X(self, group, user_id, fit=False, feature_columns=None):
        if fit and feature_columns is not None:
            self.feature_columns = feature_columns

        if self.feature_columns is not None:
            X_num = group.reindex(columns=self.feature_columns, fill_value=0)
        else:
            target = self.config.get("data", {}).get("target_col", "Is_Anomaly")
            drop_cols = ["UserID", "Time", target, "EventID", "datetime", "session_id", "AnomalyType"]
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

    # =========================================================
    # CONSTRUÃ‡ÃƒO DE JANELAS (SLIDING WINDOWS)
    # =========================================================
    def _build_sequences(self, df, fit=False, return_labels=False, feature_columns=None):
        X_all, indices_all, users_all, labels_all = [], [], [], []
        target = self.config.get("data", {}).get("target_col", "Is_Anomaly")

        for user_id, group in df.groupby("UserID"):
            X = self._prepare_and_scale_X(group, user_id, fit=fit, feature_columns=feature_columns)
            if len(X) < self.sequence_length:
                continue

            windows = sliding_window_view(X, window_shape=self.sequence_length, axis=0)
            windows = np.swapaxes(windows, 1, 2)
            X_all.append(windows)

            idx = group.index.values[self.sequence_length - 1:]
            indices_all.append(idx)
            users_all.append(np.full(len(idx), str(user_id)))

            if return_labels and target in df.columns:
                windows_labels = sliding_window_view(group[target].values, window_shape=self.sequence_length)
                labels_all.append(np.max(windows_labels, axis=1))

        if not X_all:
            return np.array([]), np.array([]), np.array([]), None

        return (
            np.concatenate(X_all),
            np.concatenate(indices_all),
            np.concatenate(users_all),
            np.concatenate(labels_all) if labels_all else None
        )

    # =========================================================
    # CÃLCULO DE SCORES (PADRONIZADO)
    # =========================================================
    def _calculate_scores(self, X):
        if X is None or len(X) == 0:
            return np.array([])

        is_if_model = HAS_IFOREST and isinstance(self.model, IsolationForest)
        model_type_name = type(self.model).__name__.lower()

        # Trata Isolation Forest
        if is_if_model or "isolationforest" in model_type_name or "iforest" in model_type_name:
            if hasattr(self.model, "decision_function"):
                return -self.model.decision_function(X)
            elif hasattr(self.model, "score_samples"):
                return -self.model.score_samples(X)

        # Trata modelos customizados com .score()
        if hasattr(self.model, "score"):
            scores = self.model.score(X)
            if self.config.get("model", {}).get("auto_invert_scores", False) and np.mean(scores) < 0:
                return -scores
            return scores

        raise ValueError("âŒ Erro: O Modelo nÃ£o possui um mÃ©todo de escoragem vÃ¡lido!")

    # =========================================================
    # MÃ‰TODO PRINCIPAL: RUN (COM CORREÃ‡ÃƒO DE ÃNDICES)
    # =========================================================
    def run(self, train_df, val_df, test_df, feature_columns=None):
        """
        Executa o treino, calibraÃ§Ã£o de limiares e preparaÃ§Ã£o para avaliaÃ§Ã£o.
        CORREÃ‡ÃƒO: reset_index(drop=True) garantido para evitar duplicatas.
        """
        # ðŸ”¥ CORREÃ‡ÃƒO CRÃTICA: Garante Ã­ndices Ãºnicos em todos os DataFrames
        # Resolve o erro "cannot reindex on an axis with duplicate labels"
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        if test_df is not None:
            test_df = test_df.reset_index(drop=True)

        # Reset de estado (importante para multi-seed)
        self.user_thresholds = {}
        self.global_threshold = 0.0

        train_cfg = self.config.get("training", {})
        do_resume = train_cfg.get("resume", False)
        save_model = train_cfg.get("save_model", True)
        thresh_mode = train_cfg.get("threshold_mode", "per_user")

        is_supervised = train_cfg.get("supervised", False)
        target_attack = train_cfg.get("target_attack_type", None)

        fixed_threshold = train_cfg.get("fixed_threshold", None)
        fixed_percentile = train_cfg.get("fixed_percentile", None)
        target = self.config.get("data", {}).get("target_col", "Is_Anomaly")

        model_path = os.path.join(self.output_dir, "saved_model")
        meta_path = os.path.join(self.output_dir, "trainer_meta.pkl")

        # -------------------- TREINO (com checkpoint) --------------------
        if do_resume and os.path.exists(model_path) and os.path.exists(meta_path) and self._checkpoint_uses_current_protocol(meta_path):
            print("ðŸ”„ [Trainer] Checkpoint detectado. Retomando execuÃ§Ã£o...")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.scalers = meta['scalers']
                self.feature_columns = meta['feature_columns']
            if hasattr(self.model, "load"):
                self.model.load(model_path)
        else:
            print(f"ðŸš€ [Trainer] Iniciando treino do zero (Modo: {thresh_mode})...")

            # O treino supervisionado usa somente rÃ³tulos da partiÃ§Ã£o de treino.
            # A validaÃ§Ã£o fica reservada Ã  calibraÃ§Ã£o e nunca fornece exemplos ao modelo.
            if is_supervised:
                X_train, _, _, y_train = self._build_sequences(
                    train_df, fit=True, return_labels=True,
                    feature_columns=feature_columns
                )
            else:
                X_train, _, _, y_train = self._build_benign_training_sequences(
                    train_df, feature_columns=feature_columns
                )
                if y_train is not None:
                    y_train = y_train[y_train == 0]
            if len(X_train) == 0:
                raise ValueError("Nenhuma janela de treino elegível foi gerada para o protocolo escolhido.")

            if y_train is None:
                y_train = np.zeros(len(X_train))

            if is_supervised and len(np.unique(y_train)) < 2:
                raise ValueError(
                    "Treino supervisionado contÃ©m uma Ãºnica classe. "
                    "NÃ£o use a validaÃ§Ã£o para obter positivos; forneÃ§a ataques "
                    "na partiÃ§Ã£o de treino ou retire o modelo supervisionado "
                    "deste protocolo confirmatÃ³rio."
                )

            # Fit do modelo
            if hasattr(self.model, "fit"):
                if is_supervised:
                    self.model.fit(X_train, y=y_train)
                else:
                    self.model.fit(X_train)

            # Salva modelo e metadados
            if save_model:
                os.makedirs(model_path, exist_ok=True)
                if hasattr(self.model, "save"):
                    self.model.save(model_path)
                with open(meta_path, "wb") as f:
                    pickle.dump({'scalers': self.scalers, 'feature_columns': self.feature_columns, 'training_protocol': 'train_val_independent_v3_benign_early_stop'}, f)
                print(f"ðŸ’¾ [Trainer] Modelo e metadados salvos com sucesso em: {model_path}")

        # -------------------- VALIDAÃ‡ÃƒO (calibraÃ§Ã£o de limiares) --------------------
        print("\n--- [Trainer] Gerando Janelas de VALIDAÃ‡ÃƒO ---")

        # Filtro para o modo especialista (apenas normal + ataque alvo)
        if target_attack and "AnomalyType" in val_df.columns:
            print(f"   ðŸ” [ESPECIALISTA] Filtrando validaÃ§Ã£o para '{target_attack}'...")
            mask = (val_df[target] == 0) | (val_df["AnomalyType"] == target_attack)
            val_df = val_df[mask].copy()

        X_val, _, val_users, y_val = self._build_sequences(val_df, fit=False, return_labels=True)
        if X_val is None or len(X_val) == 0:
            raise ValueError("âŒ Erro CrÃ­tico: Conjunto de VALIDAÃ‡ÃƒO gerou 0 janelas.")

        val_scores = self._calculate_scores(X_val)

        if y_val is not None and np.sum(y_val) > 0:
            print(f"   âœ… ValidaÃ§Ã£o contÃ©m {int(np.sum(y_val))} janelas anÃ´malas.")
        else:
            print(f"   âŒ ATENÃ‡ÃƒO: ValidaÃ§Ã£o livre de anomalias registradas!")

        best_p, best_f2 = 95.0, -1.0
        print("\n--- [Trainer] DefiniÃ§Ã£o de Limiares de Alarme ---")

        # ---------- LIMIAR FIXO (Absoluto ou Percentil) ----------
        if fixed_threshold is not None:
            self.global_threshold = float(fixed_threshold)
            self.user_thresholds = {"__GLOBAL_FALLBACK__": self.global_threshold}
            print(f"âš–ï¸ [YAML] Limiar absoluto fixado em: {fixed_threshold}")

        elif fixed_percentile is not None:
            best_p = float(fixed_percentile)
            self.global_threshold = float(np.percentile(val_scores, best_p)) if val_scores.size > 0 else 0.5
            self.user_thresholds = {"__GLOBAL_FALLBACK__": self.global_threshold}
            if thresh_mode == "per_user":
                self.user_thresholds.update(
                    self._personalized_thresholds(val_scores, val_users, y_val, best_p)
                )
            print(f"âš–ï¸ [YAML] Percentil fixado em: {best_p}%")

        # ---------- LIMIAR OTIMIZADO (F2-Score) ----------
        else:
            if y_val is not None and np.sum(y_val) > 0:
                print(f"ðŸ§  [AUTO] Otimizando curva de decisÃ£o dinamicamente com F2-Score...")
                calibration_cfg = train_cfg.get("threshold_search", {})
                percentile_min = float(calibration_cfg.get("percentile_min", 0.0))
                percentile_max = float(calibration_cfg.get("percentile_max", 99.5))
                percentile_step = float(calibration_cfg.get("percentile_step", 0.5))
                if not (0 <= percentile_min < percentile_max <= 100) or percentile_step <= 0:
                    raise ValueError("ConfiguraÃ§Ã£o invÃ¡lida em training.threshold_search.")
                for p in np.arange(percentile_min, percentile_max + percentile_step / 2, percentile_step):
                    t_glob = np.percentile(val_scores, p)
                    if thresh_mode == "per_user":
                        u_t_dict = self._personalized_thresholds(
                            val_scores, val_users, y_val, p
                        )
                        y_pred = (val_scores > np.array([u_t_dict.get(str(uid), t_glob) for uid in val_users])).astype(int)
                    else:
                        y_pred = (val_scores > t_glob).astype(int)

                    f2 = self._compute_f2_score(y_val, y_pred)
                    if f2 > best_f2:
                        best_f2, best_p = f2, p
                print(f"âœ… [AUTO] OtimizaÃ§Ã£o concluÃ­da. Percentil ideal: {best_p:.1f}% (F2-Score: {best_f2:.4f})")
            else:
                best_p = 99.0
                print(f"âš ï¸ [AUTO] ValidaÃ§Ã£o limpa. Adotando percentil fixo de seguranÃ§a: {best_p}%")

            self.global_threshold = float(np.percentile(val_scores, best_p)) if val_scores.size > 0 else 0.5
            self.user_thresholds = {"__GLOBAL_FALLBACK__": self.global_threshold}
            if thresh_mode == "per_user":
                self.user_thresholds.update(
                    self._personalized_thresholds(val_scores, val_users, y_val, best_p)
                )

        # Salva limiares
        with open(os.path.join(self.output_dir, "user_thresholds.json"), "w") as f:
            json.dump(self.user_thresholds, f, indent=4)

        print("\n--- [Trainer] ExecuÃ§Ã£o Finalizada com Sucesso ---")
        return {
            "val_f2": float(best_f2) if best_f2 != -1 else -1.0,
            "global_threshold": self.global_threshold
        }





