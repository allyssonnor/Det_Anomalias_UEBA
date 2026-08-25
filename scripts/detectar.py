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
import pickle # Adicionado o import crítico para manipulação de metadados

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
    def __init__(self, experiment_config_path, run_type="full"):
        actual_exp_path = self._resolve_path(experiment_config_path)
        print(f"📖 Lendo configuração unificada: {actual_exp_path}")
        with open(actual_exp_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.seeds = self.config.get("settings", {}).get("seeds", [42])

        self.isolated_root = self.config.get("data", {}).get("isolated_tests_dir", None)
        if self.isolated_root:
            self.isolated_root = self._resolve_path(self.isolated_root)
            print(f"📁 Raiz de testes isolados: {self.isolated_root}")

        base_root = self.config.get("settings", {}).get("results_root", "results/thesis_runs_optimized")
        self.results_root = os.path.join(base_root, run_type)
        print(f"📁 Diretório de resultados: {self.results_root}")

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
    
    def _evaluate_subset(self, trainer, temporal, test_df, output_dir, config, subset_name, verbose=False):
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.perf_counter()
        
        # Pega também os usuários e rótulos para extração fiel
        X_test, indices, users, y_test = trainer._build_sequences(test_df, fit=False, return_labels=True)

        if len(X_test) == 0:
            if verbose: print(f"   ⚠️ Nenhuma sequência válida para {subset_name}.")
            return {"pr_auc": 0.0, "roc_auc": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, 
                    "tp": 0, "fp": 0, "tn": 0, "fn": 0, "elapsed_sec": 0.0, "mem_peak_mb": 0.0}

        # Calcula os scores brutos
        scores = trainer._calculate_scores(X_test)
        
        # Calcula as predições usando os limiares do modelo
        global_fallback = trainer.global_threshold
        thresholds_aplicados = np.array([trainer.user_thresholds.get(str(u), global_fallback) for u in users])
        y_pred = (scores > thresholds_aplicados).astype(int)

        # SALVA AS MATRIZES (.npy) PARA O EXTRATOR
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

        for seed in self.seeds:
            print(f"\n{'#'*70}\n🌱 INICIANDO TREINO (SEED {seed})\n{'#'*70}\n")
            
            config_run_base = copy.deepcopy(self.config)
            config_run_base.setdefault("training", {})["seed"] = seed

            self.base_output_dir = os.path.join(self.results_root, f"run_seed{seed}")
            os.makedirs(self.base_output_dir, exist_ok=True)
            self.results_log = []
            master_csv_path = os.path.join(self.base_output_dir, "master_results_table.csv")

            for ds in datasets:
                ds_name = ds.get("name", "unknown_ds")
                config_ds = self._apply_overrides(copy.deepcopy(config_run_base), ds.get("overrides", {}))

                adapter = DatasetAdapter(config_ds)
                train_raw, val_raw, test_raw = adapter.load()

                if not self._validate_dataframe(train_raw, "train") or not self._validate_dataframe(test_raw, "test"):
                    raise ValueError("Dataframes de entrada vazios ou inválidos.")

                processor = FeatureProcessor(config_ds)
                if hasattr(processor, 'fit'): processor.fit(train_raw)
                train_scaled, val_scaled, test_scaled = processor.transform(train_raw), processor.transform(val_raw), processor.transform(test_raw)
                valid_features = [c for c in processor.final_feature_names if c in train_scaled.columns and c in val_scaled.columns]

                # Precarrega dados isolados se o YAML mandar
                isolated_datasets = {}
                run_isolated = config_ds.get("settings", {}).get("run_isolated_eval", True)
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

                for mp in model_presets:
                    mp_name = mp.get("name", "unknown_model")
                    for combo in grid_combinations:
                        current_hparams = dict(zip(grid_keys, combo))
                        run_name = f"{ds_name}_{mp_name}"
                        run_dir = os.path.join(self.base_output_dir, run_name)
                        
                        config_final = self._apply_overrides(copy.deepcopy(config_ds), mp.get("overrides", {}))
                        for k, v in current_hparams.items():
                            config_final = self._apply_overrides(config_final, {k: v})
                        
                        config_final["output_dir"] = run_dir

                        temporal = TemporalBuilder(config_final)
                        t_train, t_val, t_test = temporal.transform(train_scaled), temporal.transform(val_scaled), temporal.transform(test_scaled)

                        model = create_model(config_final)
                        trainer = Trainer(config_final, model)
                        
                        # Treino
                        results = trainer.run(t_train, t_val, t_test, feature_columns=valid_features)

                        # =========================================================
                        # SALVAMENTO DO PIPELINE DE ENGENHARIA E CONFIGURAÇÃO (ARQUITETURA MLOPS)
                        # =========================================================
                        config_save_path = os.path.join(run_dir, "config_used.yaml")
                        with open(config_save_path, "w", encoding="utf-8") as f:
                            yaml.dump(config_final, f)
                            
                        meta_path = os.path.join(run_dir, "trainer_meta.pkl")
                        if os.path.exists(meta_path):
                            with open(meta_path, "rb") as f:
                                meta = pickle.load(f)
                            
                            # Acopla os objetos já ajustados (fitted) para inferência futura
                            meta['feature_processor'] = processor
                            meta['temporal_builder'] = temporal
                            
                            with open(meta_path, "wb") as f:
                                pickle.dump(meta, f)
                        # =========================================================

                        # =========================================================
                        # AVALIAÇÃO PRINCIPAL (FLAG RUN_MIXED_EVAL)
                        # =========================================================
                        run_mixed = config_final.get("settings", {}).get("run_mixed_eval", True)
                        
                        if run_mixed:
                            mixed_output = os.path.join(run_dir, "eval_mixed")
                            print(f"\n   🧪 Avaliando conjunto de teste misto...")
                            mixed_metrics = self._evaluate_subset(trainer, temporal, t_test, mixed_output, config_final, subset_name=ds_name, verbose=True)
                            
                            log_entry = {"seed": seed, "dataset": ds_name, "model_preset": mp_name, **current_hparams, **results, **mixed_metrics}
                            self.results_log.append(log_entry)
                        else:
                            print(f"\n   ⏭️ Avaliação do conjunto misto ignorada (run_mixed_eval=False no YAML). Indo direto para os isolados.")

                        # =========================================================
                        # AVALIAÇÕES ISOLADAS
                        # =========================================================
                        if isolated_datasets and run_isolated:
                            print(f"\n   🧪 Avaliando {len(isolated_datasets)} ataques isolados...")
                            for iso_name, iso_scaled in isolated_datasets.items():
                                iso_df = temporal.transform(iso_scaled.copy())
                                iso_out = os.path.join(run_dir, f"eval_{iso_name}")
                                iso_metrics = self._evaluate_subset(trainer, temporal, iso_df, iso_out, config_final, subset_name=iso_name, verbose=True)
                                
                                self.results_log.append({"seed": seed, "dataset": iso_name, "model_preset": mp_name, **current_hparams, **results, **iso_metrics})

                        pd.DataFrame(self.results_log).to_csv(master_csv_path, index=False)
                        print(f"✅ Execução {run_name} finalizada e salva.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True, help="Caminho para o arquivo YAML único.")
    args = parser.parse_args()

    runner = ExperimentRunner(args.exp_config)
    runner.run()
