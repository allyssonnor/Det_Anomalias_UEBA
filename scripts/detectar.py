#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import yaml
import copy
import json
import argparse
import time
import psutil
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import traceback
import gc
import shutil
import pickle

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from core.data.dataset_adapter import DatasetAdapter
    from core.features.feature_processor import FeatureProcessor
    from core.temporal_builder import TemporalBuilder
    from core.models.model_factory import create as create_model
    from core.trainer import Trainer
    from scripts.evaluate_performance import run_external_evaluation
except ImportError as e:
    print(f"❌ Erro de Importação: {e}")
    sys.exit(1)


class ExperimentRunner:
    def __init__(self, experiment_config_path):
        actual_exp_path = self._resolve_path(experiment_config_path)
        print(f"📖 Lendo configuração unificada: {actual_exp_path}")
        with open(actual_exp_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.seeds = self.config.get("settings", {}).get("seeds", [42])

        self.isolated_root = self.config.get("data", {}).get("isolated_tests_dir", None)
        if self.isolated_root:
            self.isolated_root = self._resolve_path(self.isolated_root)

        self.sampling_mode = self.config.get("sampling", {}).get("mode", "full")
        base_root = self.config.get("settings", {}).get("results_root", "results")
        self.results_root = os.path.join(base_root, self.sampling_mode)

        cache_root = self.config.get("settings", {}).get("cache_root", "data/GLOBAL_CACHE_DETECTOR")
        self.cache_root = os.path.abspath(os.path.join(root_path, cache_root))
        os.makedirs(self.cache_root, exist_ok=True)

    def _resolve_path(self, path):
        if not path: return path
        search_paths = [
            path,
            os.path.join(root_path, path),
            os.path.join("/content/drive/MyDrive/Project_Detector", path)
        ]
        for p in search_paths:
            if p and os.path.exists(p): return os.path.abspath(p)
        return path

    def _apply_overrides(self, config, overrides):
        if not overrides: return config
        for key, val in overrides.items():
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val
        return config

    def _validate_dataframe(self, df, name):
        if df is None or df.empty:
            print(f"⚠️ [WARNING] {name} está VAZIO!")
            return False
        if 'Time' not in df.columns:
            print(f"⚠️ [WARNING] {name} não tem coluna 'Time'")
            return False
        return True
    
    def _evaluate_subset(self, trainer, temporal, test_df, output_dir, config, subset_name, verbose=False, precomputed_test=None):
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.perf_counter()
        
        if precomputed_test is not None:
            X_test, indices, users, y_test = precomputed_test
        else:
            X_test, indices, users, y_test = trainer._build_sequences(test_df, fit=False, return_labels=True)

        if len(X_test) == 0:
            if verbose: print(f"   ⚠️ Nenhuma sequência válida para {subset_name}.")
            return {"pr_auc": 0.0, "roc_auc": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, 
                    "tp": 0, "fp": 0, "tn": 0, "fn": 0, "elapsed_sec": 0.0, "mem_peak_mb": 0.0}

        scores = trainer._calculate_scores(X_test)
        
        global_fallback = trainer.global_threshold
        thresholds_aplicados = np.array([trainer.user_thresholds.get(str(u), global_fallback) for u in users])
        y_pred = (scores > thresholds_aplicados).astype(int)

        np.save(os.path.join(output_dir, "scores.npy"), scores)
        np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(output_dir, "indices.npy"), indices)
        if y_test is not None:
            np.save(os.path.join(output_dir, "y_true.npy"), y_test)

        source_thresholds = os.path.join(trainer.output_dir, "user_thresholds.json")
        if os.path.exists(source_thresholds):
            shutil.copy(source_thresholds, os.path.join(output_dir, "user_thresholds.json"))

        eval_config = copy.deepcopy(config)
        eval_config["output_dir"] = output_dir
        
        try:
            metrics = run_external_evaluation(eval_config, test_df=test_df, scores=scores, indices=indices, verbose=verbose)
        except Exception as e:
            if verbose: print(f"   ❌ Erro na avaliação de {subset_name}: {e}")
            metrics = {"pr_auc": 0.0, "roc_auc": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}

        end_time = time.perf_counter()
        try:
            mem_peak_mb = psutil.Process().memory_info().peak_wset / (1024*1024) if sys.platform == 'win32' else psutil.Process().memory_info().rss / (1024*1024)
        except:
            mem_peak_mb = 0.0

        return {**metrics, "elapsed_sec": end_time - start_time, "mem_peak_mb": mem_peak_mb}

    def run(self):
        datasets = self.config.get("datasets", [])
        model_presets = self.config.get("model_presets", [])
        hparam_grid = self.config.get("hyperparameter_grid", {})

        grid_keys = list(hparam_grid.keys())
        grid_values = list(hparam_grid.values())
        grid_combinations = list(product(*grid_values)) if grid_values else [()]

        total_runs = len(datasets) * len(model_presets) * max(1, len(grid_combinations)) * len(self.seeds)

        print("\n" + "="*70)
        print("🚀 RESUMO DA MARATONA DE EXPERIMENTOS")
        print("="*70)
        print(f" 🔹 Modo de Execução : {self.sampling_mode.upper()}")
        print(f" 🔹 Diretório Base   : {self.results_root}")
        print(f" 🔹 Cache Global     : {self.cache_root}")
        print(f" 🔹 Datasets         : {len(datasets)}")
        print(f" 🔹 Modelos (Presets): {len(model_presets)}")
        print(f" 🔹 Combinações Grid : {max(1, len(grid_combinations))}")
        print(f" 🔹 Sementes (Seeds) : {len(self.seeds)} {self.seeds}")
        print(f" 🎯 TOTAL DE RUNS    : {total_runs}")
        print("="*70 + "\n")

        for seed in self.seeds:
            print(f"\n{'#'*70}\n🌱 INICIANDO TREINO (SEED {seed})\n{'#'*70}\n")
            
            config_run_base = copy.deepcopy(self.config)
            config_run_base.setdefault("training", {})["seed"] = seed

            self.base_output_dir = os.path.join(self.results_root, f"run_seed{seed}")
            os.makedirs(self.base_output_dir, exist_ok=True)
            self.results_log = []
            master_csv_path = os.path.join(self.base_output_dir, "master_results_table.csv")

            # =========================================================
            # 🔥 PLANEJAMENTO: Gerar lista de tarefas para esta seed
            # =========================================================
            tarefas = []
            for ds in datasets:
                ds_name = ds.get("name", "unknown_ds")
                for mp in model_presets:
                    mp_name = mp.get("name", "unknown_model")
                    for combo in grid_combinations:
                        current_hparams = dict(zip(grid_keys, combo))
                        tarefas.append((ds_name, mp_name, current_hparams, combo))

            tarefas_pendentes = []
            if os.path.exists(master_csv_path):
                try:
                    existing_df = pd.read_csv(master_csv_path)
                    existing_df = existing_df[existing_df["seed"] == seed]
                    subset_cols = ["dataset", "model_preset", "training.threshold_mode", "model.sequence_length"]
                    available_subset = [col for col in subset_cols if col in existing_df.columns]
                    
                    if available_subset:
                        tarefas_df = pd.DataFrame([
                            {
                                "dataset": t[0],
                                "model_preset": t[1],
                                "training.threshold_mode": t[2].get("training.threshold_mode"),
                                "model.sequence_length": t[2].get("model.sequence_length")
                            }
                            for t in tarefas
                        ])
                        
                        merged = tarefas_df.merge(
                            existing_df[available_subset],
                            on=available_subset,
                            how="left",
                            indicator=True
                        )
                        tarefas_para_rodar = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
                        
                        for _, row in tarefas_para_rodar.iterrows():
                            ds_name = row["dataset"]
                            mp_name = row["model_preset"]
                            current_hparams = {}
                            for k in grid_keys:
                                if k == "training.threshold_mode":
                                    current_hparams[k] = row["training.threshold_mode"]
                                elif k == "model.sequence_length":
                                    current_hparams[k] = row["model.sequence_length"]
                                else:
                                    for t_orig in tarefas:
                                        if t_orig[0] == ds_name and t_orig[1] == mp_name and t_orig[2].get("training.threshold_mode") == row["training.threshold_mode"] and t_orig[2].get("model.sequence_length") == row["model.sequence_length"]:
                                            current_hparams = t_orig[2]
                                            break
                            combo_orig = None
                            for t_orig in tarefas:
                                if t_orig[0] == ds_name and t_orig[1] == mp_name and t_orig[2] == current_hparams:
                                    combo_orig = t_orig[3]
                                    break
                            tarefas_pendentes.append((ds_name, mp_name, current_hparams, combo_orig))
                        
                        print(f"📊 Planejamento: {len(tarefas_pendentes)} tarefas pendentes de {len(tarefas)} totais.")
                    else:
                        tarefas_pendentes = tarefas
                except Exception as e:
                    print(f"⚠️ Erro ao ler CSV para planejamento: {e}. Executando todas as tarefas.")
                    tarefas_pendentes = tarefas
            else:
                tarefas_pendentes = tarefas

            if not tarefas_pendentes:
                print(f"✅ Seed {seed}: todas as tarefas já foram executadas. Pulando.")
                continue

            # =========================================================
            # PROCESSAMENTO: Executar apenas as tarefas pendentes
            # =========================================================
            for ds in datasets:
                ds_name = ds.get("name", "unknown_ds")
                config_ds = self._apply_overrides(copy.deepcopy(config_run_base), ds.get("overrides", {}))

                tarefas_ds = [t for t in tarefas_pendentes if t[0] == ds_name]
                if not tarefas_ds:
                    continue

                # =========================================================
                # ETAPA 1: CACHE GLOBAL DE FEATURES (N1)
                # =========================================================
                split_strategy = config_ds.get("sampling", {}).get("split_strategy", "chronological")
                cache_dir = os.path.join(self.cache_root, ds_name, f"seed_{seed}", split_strategy)
                os.makedirs(cache_dir, exist_ok=True)
                
                cache_t_train = os.path.join(cache_dir, "t_train.parquet")
                cache_t_val = os.path.join(cache_dir, "t_val.parquet")
                cache_t_test = os.path.join(cache_dir, "t_test.parquet")
                cache_meta = os.path.join(cache_dir, "processor_meta.pkl")
                cache_iso = os.path.join(cache_dir, "isolated_datasets.pkl")

                run_isolated = config_ds.get("settings", {}).get("run_isolated_eval", True)

                if all(os.path.exists(p) for p in [cache_t_train, cache_t_val, cache_t_test, cache_meta]):
                    print(f"\n📦 [CACHE GLOBAL] Carregando features pré-computadas para '{ds_name}' (Seed:{seed}, Split:{split_strategy.upper()})...")
                    t_train = pd.read_parquet(cache_t_train)
                    t_val = pd.read_parquet(cache_t_val)
                    t_test = pd.read_parquet(cache_t_test)
                    
                    with open(cache_meta, "rb") as f:
                        meta_cache = pickle.load(f)
                        valid_features = meta_cache["valid_features"]
                        processor = meta_cache["processor"]
                        temporal = meta_cache["temporal"]

                    isolated_datasets = {}
                    if run_isolated and os.path.exists(cache_iso):
                        with open(cache_iso, "rb") as f:
                            isolated_datasets = pickle.load(f)
                else:
                    print(f"\n⚙️ [PROCESSAMENTO] Extraindo features do zero para '{ds_name}' e gravando no CACHE GLOBAL...")
                    adapter = DatasetAdapter(config_ds)
                    train_raw, val_raw, test_raw = adapter.load()

                    if not self._validate_dataframe(train_raw, "train") or not self._validate_dataframe(test_raw, "test"):
                        print(f"⚠️ Pulando dataset {ds_name}: Dataframes vazios.")
                        continue

                    processor = FeatureProcessor(config_ds)
                    if hasattr(processor, 'fit'): processor.fit(train_raw)
                    
                    train_scaled = processor.transform(train_raw)
                    val_scaled = processor.transform(val_raw)
                    test_scaled = processor.transform(test_raw)
                    
                    valid_features = [c for c in processor.final_feature_names if c in train_scaled.columns and c in val_scaled.columns]

                    isolated_datasets = {}
                    test_size = config_ds.get("sampling", {}).get("test_size", 0)
                    isolated_targets = config_ds.get("settings", {}).get("isolated_targets", [])
                    
                    if run_isolated and self.isolated_root and os.path.isdir(self.isolated_root):
                        for item in os.listdir(self.isolated_root):
                            if isolated_targets and item not in isolated_targets:
                                continue
                                
                            item_path = os.path.join(self.isolated_root, item)
                            if os.path.isdir(item_path) and item.startswith("test_"):
                                test_file = os.path.join(item_path, f"{item}.jsonl")
                                if os.path.exists(test_file):
                                    iso_raw = adapter._load_resource(test_file, test_size)
                                    iso_raw = adapter._normalize(iso_raw)
                                    iso_raw = adapter._apply_auto_labels(iso_raw, test_file)
                                    if iso_raw is not None and not iso_raw.empty:
                                        isolated_datasets[item] = processor.transform(iso_raw)

                    temporal = TemporalBuilder(config_ds)
                    t_train = temporal.transform(train_scaled)
                    t_val = temporal.transform(val_scaled)
                    t_test = temporal.transform(test_scaled)
                    
                    t_train.to_parquet(cache_t_train)
                    t_val.to_parquet(cache_t_val)
                    t_test.to_parquet(cache_t_test)
                    
                    with open(cache_meta, "wb") as f:
                        pickle.dump({
                            "valid_features": valid_features,
                            "processor": processor,
                            "temporal": temporal
                        }, f)
                        
                    if isolated_datasets:
                        with open(cache_iso, "wb") as f:
                            pickle.dump(isolated_datasets, f)
                            
                    del train_raw, val_raw, test_raw, train_scaled, val_scaled, test_scaled
                    gc.collect()

                # =========================================================
                # 🔥 OTIMIZAÇÃO: N2 Hot Cache em memória
                # =========================================================
                n2_bundles = {}  # chave: (seq_len, is_sup) -> bundle

                def get_n2_bundle(seq_len, is_sup):
                    """Carrega ou cria o N2 uma única vez por (seq_len, is_sup)."""
                    key = (seq_len, is_sup)
                    if key in n2_bundles:
                        return n2_bundles[key]

                    cache_seq_dir = os.path.join(cache_dir, f"seq_{seq_len}_sup_{is_sup}")
                    
                    if os.path.exists(cache_seq_dir) and os.path.exists(os.path.join(cache_seq_dir, "meta.pkl")):
                        print(f"\n   ⚡ [CACHE N2] Carregando matrizes de janelas (Seq={seq_len}, Sup={is_sup})...")
                        X_train = np.load(os.path.join(cache_seq_dir, "X_train.npy"))
                        y_train = np.load(os.path.join(cache_seq_dir, "y_train.npy"))
                        X_val = np.load(os.path.join(cache_seq_dir, "X_val.npy"))
                        val_users = np.load(os.path.join(cache_seq_dir, "val_users.npy"), allow_pickle=True)
                        y_val = np.load(os.path.join(cache_seq_dir, "y_val.npy"))
                        
                        X_test = np.load(os.path.join(cache_seq_dir, "X_test.npy"))
                        test_indices = np.load(os.path.join(cache_seq_dir, "test_indices.npy"))
                        test_users = np.load(os.path.join(cache_seq_dir, "test_users.npy"), allow_pickle=True)
                        y_test = np.load(os.path.join(cache_seq_dir, "y_test.npy"))
                        
                        with open(os.path.join(cache_seq_dir, "meta.pkl"), "rb") as f:
                            seq_meta = pickle.load(f)
                            scalers_dict = seq_meta["scalers"]
                            feat_cols = seq_meta["feature_columns"]
                    else:
                        print(f"\n   ⚙️ [PROCESSAMENTO] Fatiando matrizes de janelas (Seq={seq_len}, Sup={is_sup}) e salvando no cache...")
                        os.makedirs(cache_seq_dir, exist_ok=True)
                        
                        dummy_trainer = Trainer(config_final, None)  # será sobrescrito pelo config do modelo, mas só usamos _build_sequences
                        target = config_ds.get("data", {}).get("target_col", "Is_Anomaly")
                        if not is_sup:
                            X_train, _, _, y_train = dummy_trainer._build_benign_training_sequences(
                                t_train, feature_columns=valid_features
                            )
                            if y_train is not None:
                                y_train = y_train[y_train == 0]
                        else:
                            X_train, _, _, y_train = dummy_trainer._build_sequences(
                                t_train, fit=True, return_labels=True,
                                feature_columns=valid_features
                            )
                        if y_train is None: y_train = np.zeros(len(X_train))
                        
                        X_val, _, val_users, y_val = dummy_trainer._build_sequences(t_val, fit=False, return_labels=True)
                        if y_val is None: y_val = np.zeros(len(X_val))
                        
                        X_test, test_indices, test_users, y_test = dummy_trainer._build_sequences(t_test, fit=False, return_labels=True)
                        if y_test is None: y_test = np.zeros(len(X_test))
                        
                        scalers_dict = dummy_trainer.scalers
                        feat_cols = dummy_trainer.feature_columns
                        
                        np.save(os.path.join(cache_seq_dir, "X_train.npy"), X_train)
                        np.save(os.path.join(cache_seq_dir, "y_train.npy"), y_train)
                        np.save(os.path.join(cache_seq_dir, "X_val.npy"), X_val)
                        np.save(os.path.join(cache_seq_dir, "val_users.npy"), val_users)
                        np.save(os.path.join(cache_seq_dir, "y_val.npy"), y_val)
                        np.save(os.path.join(cache_seq_dir, "X_test.npy"), X_test)
                        np.save(os.path.join(cache_seq_dir, "test_indices.npy"), test_indices)
                        np.save(os.path.join(cache_seq_dir, "test_users.npy"), test_users)
                        np.save(os.path.join(cache_seq_dir, "y_test.npy"), y_test)
                        
                        with open(os.path.join(cache_seq_dir, "meta.pkl"), "wb") as f:
                            pickle.dump({"scalers": scalers_dict, "feature_columns": feat_cols}, f)

                    bundle = {
                        "X_train": X_train, "y_train": y_train,
                        "X_val": X_val, "val_users": val_users, "y_val": y_val,
                        "X_test": X_test, "test_indices": test_indices, "test_users": test_users, "y_test": y_test,
                        "scalers": scalers_dict, "feat_cols": feat_cols
                    }
                    n2_bundles[key] = bundle
                    return bundle

                # =========================================================
                # LOOP DE MODELOS (APENAS OS PENDENTES)
                # =========================================================
                for mp in model_presets:
                    mp_name = mp.get("name", "unknown_model")
                    tarefas_mp = [t for t in tarefas_ds if t[1] == mp_name]
                    if not tarefas_mp:
                        continue
                                              
                    for combo in grid_combinations:
                        # 🔥 CORREÇÃO: usa any() em vez de set()
                        # pois combos podem conter listas (ex: hidden_layer_sizes=[32, 16])
                        # e tuplas com listas não são hashable.
                        if not any(t[3] == combo for t in tarefas_mp):
                            continue           

                        current_hparams = dict(zip(grid_keys, combo))
                        
                        config_final = self._apply_overrides(copy.deepcopy(config_ds), mp.get("overrides", {}))
                        for k, v in current_hparams.items():
                            config_final = self._apply_overrides(config_final, {k: v})

                        seq_len = config_final.get("model", {}).get("sequence_length", 10)
                        is_sup = config_final.get("training", {}).get("supervised", False)
                        
                        hparam_str = "_".join([str(v) for v in combo])
                        modelo_str = f"{ds_name}_{mp_name}_{hparam_str}" if hparam_str else f"{ds_name}_{mp_name}"
                        modelo_str = modelo_str.replace("/", "-")
                        
                        run_dir = os.path.join(self.base_output_dir, modelo_str)
                        os.makedirs(run_dir, exist_ok=True)
                        config_final["output_dir"] = run_dir

                        print(f"--- 🏋️ Treinando: {modelo_str} ---")

                        # 🔥 Obtém o bundle N2 do cache em memória
                        bundle = get_n2_bundle(seq_len, is_sup)
                        
                        X_train = bundle["X_train"]
                        y_train = bundle["y_train"]
                        X_val = bundle["X_val"]
                        val_users = bundle["val_users"]
                        y_val = bundle["y_val"]
                        X_test = bundle["X_test"]
                        test_indices = bundle["test_indices"]
                        test_users = bundle["test_users"]
                        y_test = bundle["y_test"]
                        scalers_dict = bundle["scalers"]
                        feat_cols = bundle["feat_cols"]

                        # =========================================================
                        # TREINO E AVALIAÇÃO
                        # =========================================================
                        try:
                            model = create_model(config_final)
                            trainer = Trainer(config_final, model)
                            
                            trainer.scalers = scalers_dict
                            trainer.feature_columns = feat_cols
                            
                            # 🔥 Lê a métrica de otimização do YAML (com fallback para f2)
                            optim_metric = config_final.get("training", {}).get("optimization_metric", "f2")
                            
                            results = trainer.run(t_train, t_val, t_test, feature_columns=valid_features)                        

                            config_save_path = os.path.join(run_dir, "config_used.yaml")
                            with open(config_save_path, "w", encoding="utf-8") as f:
                                yaml.dump(config_final, f)
                                
                            meta_path = os.path.join(run_dir, "trainer_meta.pkl")
                            if os.path.exists(meta_path):
                                with open(meta_path, "rb") as f:
                                    meta_trainer = pickle.load(f)
                                meta_trainer['feature_processor'] = processor
                                meta_trainer['temporal_builder'] = temporal
                                with open(meta_path, "wb") as f:
                                    pickle.dump(meta_trainer, f)

                            run_mixed = config_final.get("settings", {}).get("run_mixed_eval", True)
                            if run_mixed:
                                mixed_output = os.path.join(run_dir, "eval_mixed")
                                print(f"   🧪 Avaliando conjunto de teste misto...")
                                
                                precomp_test_bundle = (X_test, test_indices, test_users, y_test)
                                mixed_metrics = self._evaluate_subset(
                                    trainer, temporal, t_test, mixed_output, config_final, 
                                    subset_name=ds_name, verbose=True, precomputed_test=precomp_test_bundle
                                )
                                
                                log_entry = {"seed": seed, "dataset": ds_name, "model_preset": mp_name, **current_hparams, **results, **mixed_metrics}
                                self.results_log.append(log_entry)
                            else:
                                print(f"   ⏭️ Avaliação mista ignorada.")

                            if isolated_datasets and run_isolated:
                                print(f"   🧪 Avaliando {len(isolated_datasets)} ataques isolados...")
                                for iso_name, iso_scaled in isolated_datasets.items():
                                    iso_df = temporal.transform(iso_scaled.copy())
                                    iso_out = os.path.join(run_dir, f"eval_{iso_name}")
                                    iso_metrics = self._evaluate_subset(trainer, temporal, iso_df, iso_out, config_final, subset_name=iso_name, verbose=True)
                                    self.results_log.append({"seed": seed, "dataset": iso_name, "model_preset": mp_name, **current_hparams, **results, **iso_metrics})

                            print(f"✅ Execução {modelo_str} finalizada e salva.\n")

                        except Exception as e:
                            print(f"❌ ERRO CRÍTICO no modelo {mp_name} (combo: {combo}, seed: {seed}):")
                            traceback.print_exc()
                            print(f"   ⚠️ Pulando este modelo e continuando com os próximos...\n")
                            continue                            
                            
                        # =========================================================
                        # SALVAMENTO COM APPEND + MERGE
                        # =========================================================
                        new_df = pd.DataFrame(self.results_log)

                        if os.path.exists(master_csv_path) and not new_df.empty:
                            try:
                                existing_df = pd.read_csv(master_csv_path)

                                # 🔥 Deduplicação dinâmica
                                # Chave de identidade = colunas fixas + TODAS as colunas
                                # de hiperparâmetros (n_estimators, max_depth, hidden_layer_sizes, etc.)
                                # Excluímos métricas e metadados do critério de dedup.
                                colunas_identidade = [
                                    "seed", "dataset", "model_preset",
                                    "training.threshold_mode", "model.sequence_length",
                                ]

                                metricas_e_meta = {
                                    "val_score", "val_f2", "global_threshold", "best_percentile", "prevalence", "lift_pr",
                                    "pr_auc", "roc_auc", "f1_score", "precision", "recall",
                                    "tp", "fp", "tn", "fn", "elapsed_sec", "mem_peak_mb",
                                    "source_file", "split_strategy", "supervised",
                                }

                                for col in new_df.columns:
                                    if col not in colunas_identidade and col not in metricas_e_meta:
                                        colunas_identidade.append(col)

                                available_subset = [
                                    col for col in colunas_identidade
                                    if col in existing_df.columns and col in new_df.columns
                                ]

                                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                                # 🔥 CORREÇÃO: converte listas/tuplas em strings antes do drop_duplicates,
                                # pois listas não são hashable e quebram o drop_duplicates.
                                for col in combined_df.columns:
                                    if combined_df[col].apply(
                                        lambda x: isinstance(x, (list, tuple))
                                    ).any():
                                        combined_df[col] = combined_df[col].apply(
                                            lambda x: str(x)
                                            if isinstance(x, (list, tuple))
                                            else x
                                        )

                                if available_subset:
                                    combined_df = combined_df.drop_duplicates(
                                        subset=available_subset, keep="last"
                                    )

                                combined_df = combined_df.drop_duplicates()
                                combined_df.to_csv(master_csv_path, index=False)
                                print(f"💾 Resultados mesclados com sucesso (Total: {len(combined_df)} linhas).")

                            except Exception as e:
                                print(f"⚠️ Erro ao mesclar CSV existente: {e}. Salvando apenas os novos resultados.")
                                new_df.to_csv(master_csv_path, index=False)
                        else:
                            new_df.to_csv(master_csv_path, index=False)    
                            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True, help="Caminho para o arquivo YAML único.")
    args = parser.parse_args()

    runner = ExperimentRunner(args.exp_config)
    runner.run()
