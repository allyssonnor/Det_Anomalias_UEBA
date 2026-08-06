#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script otimizado para maratona de experimentos UEBA.
Características:
- Pasta fixa por seed (permite resume robusto)
- Processamento de features cacheado por dataset/seed
- TemporalBuilder executado dentro do grid (suporta variação de sequence_length)
- Monitoramento de tempo e pico de memória
- Extração defensiva de métricas (JSON + fallback TXT)
- Ressincronização de logs em modo resume
- Gravação do YAML de configuração usado em cada execução
- Suporte a --run_type (full/debug) para organizar os resultados em subpastas
- [CORRIGIDO] Identificação do nome real do dataset (CERT, LANL, Synthetic) nas métricas principais.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # Silencia warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'      # Força GPU (altere para '-1' se quiser CPU)

import sys
import yaml
import copy
import json
import argparse
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import traceback
import gc
import shutil

# =========================================================
# RESOLUÇÃO DE AMBIENTE: Core e Scripts
# =========================================================
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
    print(f"Diretório Raiz detectado: {root_path}")
    sys.exit(1)


class ExperimentRunner:
    def __init__(self, experiment_config_path, run_type="full"):
        """
        Inicializa o executor de experimentos.
        """
        actual_exp_path = self._resolve_path(experiment_config_path)
        print(f"📖 Lendo plano de tese: {actual_exp_path}")
        with open(actual_exp_path, "r", encoding='utf-8') as f:
            self.exp_plan = yaml.safe_load(f)

        # Carrega configuração base (pode ser o próprio plano ou um arquivo separado)
        base_path = self._resolve_path(self.exp_plan.get("base_config", "config.yaml"))
        print(f"📖 Lendo config base: {base_path}")
        with open(base_path, "r", encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)

        # Mescla blocos soltos do plano (sampling, features, etc.) na base
        self._merge_global_configs()

        # Sementes e diretório raiz
        self.seeds = self.exp_plan.get("settings", {}).get("seeds", [42])

        # Raiz dos datasets de teste isolados (gerados pelo orquestrador)
        self.isolated_root = self.exp_plan.get("settings", {}).get(
            "isolated_tests_root", None
        )
        if self.isolated_root:
            self.isolated_root = self._resolve_path(self.isolated_root)
            print(f"📁 Raiz de testes isolados: {self.isolated_root}")

        # Diretório de resultados com subpasta do tipo de execução
        base_root = self.exp_plan.get("settings", {}).get(
            "results_root", "results/thesis_runs_optimized"
        )
        self.results_root = os.path.join(base_root, run_type)
        print(f"📁 Diretório de resultados definido como: {self.results_root}")

    # ---------------------------------------------------------
    #  MESCLAGEM DE CONFIGURAÇÕES (merge profundo)
    # ---------------------------------------------------------
    def _deep_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _merge_global_configs(self):
        control_keys = {'base_config', 'datasets', 'model_presets',
                        'hyperparameter_grid', 'settings', 'experiment'}
        for key, value in self.exp_plan.items():
            if key not in control_keys and isinstance(value, dict):
                if key not in self.base_config:
                    self.base_config[key] = {}
                self.base_config[key] = self._deep_update(self.base_config[key], value)
                print(f"🔄 Bloco global '{key}' mesclado na base_config.")

    # ---------------------------------------------------------
    #  UTILITÁRIOS DE CAMINHO E OVERRIDES
    # ---------------------------------------------------------
    def _resolve_path(self, path):
        if not path:
            return path
        search_paths = [
            path,
            os.path.join(root_path, path),
            os.path.join(root_path, "config", os.path.basename(path)),
            os.path.join("/content/drive/MyDrive/Project_Detector", path),
            os.path.join("/content/drive/MyDrive/Project_Detector/config",
                         os.path.basename(path))
        ]
        for p in search_paths:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        print(f"⚠️ Caminho não encontrado, usado como está: {path}")
        return path

    def _apply_overrides(self, config, overrides):
        if not overrides:
            return config
        for key, val in overrides.items():
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val
        return config

    # ---------------------------------------------------------
    #  VALIDAÇÃO DE DATAFRAMES
    # ---------------------------------------------------------
    def _validate_dataframe(self, df, name, config):
        if df is None or df.empty:
            print(f"⚠️ [WARNING] {name} está VAZIO!")
            if name == "train_df" and config.get("data", {}).get("train_path"):
                old_path = config["data"]["train_path"]
                new_path = self._resolve_path(old_path)
                if new_path != old_path:
                    print(f"🔄 Tentando corrigir caminho: {old_path} -> {new_path}")
                    config["data"]["train_path"] = new_path
            return False
        if 'Time' not in df.columns:
            print(f"⚠️ [WARNING] {name} não tem coluna 'Time'")
            return False
        return True

    # ---------------------------------------------------------
    #  EXTRAÇÃO DE MÉTRICAS (DEFENSIVA: JSON + TXT)
    # ---------------------------------------------------------
    def _extract_metrics(self, output_dir):
        metrics = {"pr_auc": 0.0, "f1_score": 0.0,
                   "precision": 0.0, "recall": 0.0}
        json_path = os.path.join(output_dir, "classification_report.json")
        json_success = False

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                metrics["pr_auc"] = data.get("pr_auc", 0.0)
                class_1 = data.get("1.0") or data.get("1") or {}
                if class_1:
                    metrics["precision"] = class_1.get("precision", 0.0)
                    metrics["recall"] = class_1.get("recall", 0.0)
                    metrics["f1_score"] = class_1.get("f1-score") or \
                                          class_1.get("f1_score", 0.0)
                else:
                    metrics["precision"] = data.get("precision", 0.0)
                    metrics["recall"] = data.get("recall", 0.0)
                    metrics["f1_score"] = data.get("f1_score") or \
                                          data.get("f1-score", 0.0)
                if metrics["precision"] > 0.0 or metrics["recall"] > 0.0:
                    json_success = True
            except Exception as e:
                print(f"⚠️ Erro ao processar JSON: {e}")

        if not json_success:
            report_path = os.path.join(output_dir, "classification_report.txt")
            if os.path.exists(report_path):
                try:
                    with open(report_path, "r", encoding='utf-8',
                              errors='ignore') as f:
                        content = f.read()
                    content = content.replace('\xa0', ' ')
                    lines = content.split('\n')
                    for line in lines:
                        if "PR-AUC:" in line or "pr_auc:" in line:
                            parts = line.split(":")
                            if len(parts) >= 2:
                                metrics["pr_auc"] = float(parts[1].strip())
                                break
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            label = parts[0].strip()
                            if label in ["1", "1.0"]:
                                metrics["precision"] = float(parts[1])
                                metrics["recall"] = float(parts[2])
                                metrics["f1_score"] = float(parts[3])
                                break
                except Exception as e:
                    print(f"⚠️ Erro ao extrair métricas do TXT: {e}")
        return metrics

    # ---------------------------------------------------------
    #  VERIFICAÇÃO DE EXECUÇÃO CONCLUÍDA
    # ---------------------------------------------------------
    def _is_run_completed(self, run_dir):
        json_path = os.path.join(run_dir, "classification_report.json")
        txt_path = os.path.join(run_dir, "classification_report.txt")
        return os.path.exists(json_path) or os.path.exists(txt_path)

    # ---------------------------------------------------------
    #  GRAVAÇÃO DO YAML USADO NA EXECUÇÃO
    # ---------------------------------------------------------
    def _save_config_yaml(self, config, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config_used.yaml")
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # ---------------------------------------------------------
    #  CARREGAMENTO DE DATASET ISOLADO
    # ---------------------------------------------------------
    def _load_isolated_dataset(self, adapter, filepath):
        try:
            if hasattr(adapter, '_load_resource'):
                df = adapter._load_resource(filepath, 0)
                df = adapter._normalize(df)
                df = adapter._apply_auto_labels(df, filepath)
                return df
            else:
                df = pd.read_json(filepath, lines=True)
                if 'Time' not in df.columns:
                    raise ValueError("Arquivo sem coluna 'Time'")
                if 'Is_Anomaly' not in df.columns:
                    if 'AttackID' in df.columns:
                        df['Is_Anomaly'] = df['AttackID'].notna().astype(int)
                    else:
                        df['Is_Anomaly'] = 0
                return df
        except Exception as e:
            print(f"⚠️ Erro ao carregar {filepath}: {e}")
            return None

    # ---------------------------------------------------------
    #  AVALIAÇÃO DE UM SUBSET (MIXED OU ISOLADO)
    # ---------------------------------------------------------
    def _evaluate_subset(self, trainer, temporal, test_df, output_dir, config,
                         subset_name, resume=False):
        os.makedirs(output_dir, exist_ok=True)

        if resume and self._is_run_completed(output_dir):
            print(f"   ⏭️ [RESUME] Avaliação já realizada: {subset_name}")
            metrics = self._extract_metrics(output_dir)
            return {**metrics, "elapsed_sec": 0.0, "mem_peak_mb": 0.0}

        start_time = time.perf_counter()
        process = psutil.Process()

        X_test, indices, _, _ = trainer._build_sequences(test_df, fit=False)
        if len(X_test) == 0:
            print(f"   ⚠️ Nenhuma sequência válida para {subset_name}.")
            return {"pr_auc": 0.0, "f1_score": 0.0,
                    "precision": 0.0, "recall": 0.0,
                    "elapsed_sec": 0.0, "mem_peak_mb": 0.0}

        scores = trainer.model.score(X_test)
        np.save(os.path.join(output_dir, "test_scores.npy"), scores)
        np.save(os.path.join(output_dir, "test_indices.npy"), indices)

        source_thresholds = os.path.join(trainer.output_dir, "user_thresholds.json")
        target_thresholds = os.path.join(output_dir, "user_thresholds.json")
        if os.path.exists(source_thresholds):
            shutil.copy(source_thresholds, target_thresholds)

        eval_config = copy.deepcopy(config)
        eval_config["output_dir"] = output_dir
        try:
            run_external_evaluation(eval_config, test_df=test_df)
        except Exception as e:
            print(f"   ❌ Erro na avaliação de {subset_name}: {e}")
            return {"pr_auc": 0.0, "f1_score": 0.0,
                    "precision": 0.0, "recall": 0.0,
                    "elapsed_sec": 0.0, "mem_peak_mb": 0.0}

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time

        try:
            if sys.platform == 'win32':
                mem_peak_mb = process.memory_info().peak_wset / (1024 * 1024)
            else:
                import resource
                ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mem_peak_mb = (ru_maxrss / (1024 * 1024)) if sys.platform == 'darwin' \
                              else (ru_maxrss / 1024)
        except Exception:
            mem_peak_mb = process.memory_info().rss / (1024 * 1024)

        metrics = self._extract_metrics(output_dir)
        return {**metrics, "elapsed_sec": elapsed_sec, "mem_peak_mb": mem_peak_mb}

    # ---------------------------------------------------------
    #  PIPELINE SINGLE RUN COM CACHE DE FEATURES (CORRIGIDO)
    # ---------------------------------------------------------
    def _run_single_pipeline_with_cached_data(self, config, run_name, ds_name,
                                              train_scaled, val_scaled, test_scaled,
                                              valid_features, isolated_datasets=None,
                                              resume=False):
        """
        Executa uma combinação: treina o modelo uma vez e avalia todos os subsets.
        CORREÇÃO APLICADA: Atribui ds_name (CERT, LANL, Synthetic) em vez de 'mixed'.
        """
        print(f"\n{'='*70}\n🚀 [EXP] {run_name} (Dataset: {ds_name})\n{'='*70}")
        base_output_dir = os.path.join(self.base_output_dir, run_name)
        os.makedirs(base_output_dir, exist_ok=True)

        self._save_config_yaml(config, base_output_dir)

        start_time = time.perf_counter()
        process = psutil.Process()

        temporal = TemporalBuilder(config)
        train_df = temporal.transform(train_scaled.copy())
        val_df = temporal.transform(val_scaled.copy())
        test_df = temporal.transform(test_scaled.copy())

        model = create_model(config)
        trainer = Trainer(config, model)
        results = trainer.run(train_df, val_df, None, feature_columns=valid_features)

        all_metrics = {}

        # 1. Avaliar o dataset misto/completo com o NOME REAL do dataset
        mixed_output = os.path.join(base_output_dir, "eval_mixed")
        print(f"   🧪 Avaliando conjunto de teste de {ds_name}...")
        mixed_metrics = self._evaluate_subset(
            trainer, temporal, test_df, mixed_output, config,
            subset_name=ds_name, resume=resume
        )
        
        # O nome do dataset na chave agora é exatamente ds_name (ex: CERT, LANL, Synthetic)
        all_metrics[ds_name] = {
            **results,
            **mixed_metrics
        }

        # 2. Avaliar datasets de cenários de ataques isolados (se houver)
        if isolated_datasets:
            for iso_name, iso_scaled in isolated_datasets.items():
                iso_output = os.path.join(base_output_dir, f"eval_{iso_name}")
                print(f"   🧪 Avaliando sub-cenário isolado: {iso_name}...")
                iso_df = temporal.transform(iso_scaled.copy())
                iso_metrics = self._evaluate_subset(
                    trainer, temporal, iso_df, iso_output, config,
                    subset_name=iso_name, resume=resume
                )
                all_metrics[iso_name] = iso_metrics

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time

        try:
            if sys.platform == 'win32':
                mem_peak_mb = process.memory_info().peak_wset / (1024 * 1024)
            else:
                import resource
                ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mem_peak_mb = (ru_maxrss / (1024 * 1024)) if sys.platform == 'darwin' \
                              else (ru_maxrss / 1024)
        except Exception:
            mem_peak_mb = process.memory_info().rss / (1024 * 1024)

        for subset_name in all_metrics.keys():
            if all_metrics[subset_name].get("elapsed_sec", 0.0) == 0.0:
                all_metrics[subset_name]["elapsed_sec"] = elapsed_sec
            if all_metrics[subset_name].get("mem_peak_mb", 0.0) == 0.0:
                all_metrics[subset_name]["mem_peak_mb"] = mem_peak_mb

        del temporal, model, trainer, train_df, val_df, test_df
        gc.collect()

        return all_metrics

    # ---------------------------------------------------------
    #  MÉTODO RUN PRINCIPAL (COM CACHE E RESUME)
    # ---------------------------------------------------------
    def run(self, resume=False):
        datasets = self.exp_plan.get("datasets", [])
        model_presets = self.exp_plan.get("model_presets", [])
        hparam_grid = self.exp_plan.get("hyperparameter_grid", {})

        grid_keys = list(hparam_grid.keys())
        grid_values = list(hparam_grid.values())
        grid_combinations = list(product(*grid_values)) if grid_values else [()]

        total_runs_per_seed = len(datasets) * len(model_presets) * len(grid_combinations)
        total_runs = total_runs_per_seed * len(self.seeds)

        print(f"\n{'='*70}")
        print(f"🏁 INICIANDO MARATONA OTIMIZADA DE EXPERIMENTOS")
        print(f"🌱 Sementes: {self.seeds}")
        print(f"📊 Total de execuções de treino: {total_runs} "
              f"({len(self.seeds)} seeds × {total_runs_per_seed} combinações)")
        print(f"🔄 Modo Resume: {'ATIVO' if resume else 'DESATIVADO'}")
        print(f"📁 Pasta raiz de saída: {self.results_root}")
        if self.isolated_root:
            print(f"📁 Raiz de testes isolados: {self.isolated_root}")
        print(f"{'='*70}\n")

        for seed_idx, seed in enumerate(self.seeds, 1):
            print(f"\n{'#'*70}")
            print(f"🌱 SEED {seed} ({seed_idx}/{len(self.seeds)})")
            print(f"{'#'*70}\n")

            config_base = copy.deepcopy(self.base_config)
            if "training" not in config_base:
                config_base["training"] = {}
            config_base["training"]["seed"] = seed
            config_base["seed"] = seed

            self.base_output_dir = os.path.join(self.results_root,
                                                f"marathon_seed{seed}")
            os.makedirs(self.base_output_dir, exist_ok=True)

            self.results_log = []
            master_csv_path = os.path.join(self.base_output_dir,
                                           "master_results_table.csv")

            if resume and os.path.exists(master_csv_path):
                try:
                    df_existing = pd.read_csv(master_csv_path)
                    self.results_log = df_existing.to_dict(orient='records')
                    print(f"📂 [RESUME] Carregado CSV com {len(self.results_log)} "
                          f"registros para seed {seed}.")
                except Exception as e:
                    print(f"⚠️ Erro ao carregar CSV anterior ({e}). "
                          "Iniciando do zero.")

            current_run = 0

            # -------------------------------------------------
            #  LOOP POR DATASET (PROCESSAMENTO DE FEATURES 1x)
            # -------------------------------------------------
            for ds in datasets:
                ds_name = ds.get("name", "unknown_ds")
                ds_overrides = ds.get("overrides", {})

                config_ds = copy.deepcopy(config_base)
                config_ds = self._apply_overrides(config_ds, ds_overrides)

                print(f"\n📦 [PREPARAÇÃO] Dataset: {ds_name} (Seed {seed})")
                adapter = DatasetAdapter(config_ds)
                train_raw, val_raw, test_raw = adapter.load()

                if not self._validate_dataframe(train_raw, "train_df", config_ds):
                    raise ValueError("train_df vazio/inválido. Verifique caminhos.")
                if not self._validate_dataframe(val_raw, "val_df", config_ds):
                    raise ValueError("val_df vazio/inválido.")
                if not self._validate_dataframe(test_raw, "test_df", config_ds):
                    raise ValueError("test_df vazio/inválido.")

                processor = FeatureProcessor(config_ds)
                if hasattr(processor, 'fit'):
                    processor.fit(train_raw)
                train_scaled = processor.transform(train_raw)
                val_scaled = processor.transform(val_raw)
                test_scaled = processor.transform(test_raw)

                valid_features = [c for c in processor.final_feature_names
                                  if c in train_scaled.columns
                                  and c in val_scaled.columns
                                  and c in test_scaled.columns]

                # -------------------------------------------------
                #  CARREGAR DATASETS ISOLADOS (se houver)
                # -------------------------------------------------
                isolated_datasets = {}
                if self.isolated_root and os.path.isdir(self.isolated_root):
                    for item in os.listdir(self.isolated_root):
                        item_path = os.path.join(self.isolated_root, item)
                        if os.path.isdir(item_path) and item.startswith("test_"):
                            test_file = os.path.join(item_path, f"{item}.jsonl")
                            if os.path.exists(test_file):
                                print(f"   📦 Carregando subset isolado: {item}")
                                iso_raw = self._load_isolated_dataset(adapter, test_file)
                                if iso_raw is not None and not iso_raw.empty:
                                    iso_scaled = processor.transform(iso_raw)
                                    isolated_datasets[item] = iso_scaled
                                else:
                                    print(f"   ⚠️ Subset {item} vazio ou falhou.")
                            else:
                                print(f"   ⚠️ Arquivo não encontrado: {test_file}")

                del train_raw, val_raw, test_raw, processor, adapter
                gc.collect()

                # -------------------------------------------------
                #  LOOP POR MODELO E HIPERPARÂMETROS
                # -------------------------------------------------
                for mp in model_presets:
                    mp_name = mp.get("name", "unknown_model")
                    mp_overrides = mp.get("overrides", {})

                    for combo in grid_combinations:
                        current_run += 1
                        hparam_desc = ""
                        current_hparams = {}
                        param_name_map = {
                            'training.epochs': 'max_iter',
                            'model.sequence_length': 'seq_len',
                        }
                        for k, v in zip(grid_keys, combo):
                            key_short = k.split('.')[-1]
                            key_short = param_name_map.get(k, key_short)
                            hparam_desc += f"_{key_short}{v}"
                            current_hparams[k] = v

                        run_name = f"{ds_name}_{mp_name}{hparam_desc}"
                        run_dir = os.path.join(self.base_output_dir, run_name)
                        pct = (current_run / total_runs_per_seed) * 100

                        print(f"\n📊 [PROGRESSO SEED {seed}] "
                              f"Rodada {current_run}/{total_runs_per_seed} ({pct:.1f}%)")
                        print(f"🎯 Dataset: {ds_name} | Modelo: {mp_name} "
                              f"| HParams: {current_hparams}")

                        mixed_dir = os.path.join(run_dir, "eval_mixed")
                        if resume and self._is_run_completed(mixed_dir):
                            print(f"⏭️ [IGNORANDO] Execução já realizada: {run_name}")

                            already_logged = any(
                                log.get("dataset") == ds_name and
                                log.get("model_preset") == mp_name and
                                all(log.get(k) == v for k, v in current_hparams.items())
                                for log in self.results_log
                            )
                            if not already_logged:
                                try:
                                    print(f"🔄 Ressincronizando métricas de {run_name}")
                                    metrics = self._extract_metrics(mixed_dir)
                                    log_entry = {
                                        "seed": seed,
                                        "timestamp": datetime.fromtimestamp(
                                            os.path.getmtime(mixed_dir)
                                        ).isoformat(),
                                        "dataset": ds_name, # Usando ds_name real
                                        "model_preset": mp_name,
                                        **current_hparams,
                                        **metrics,
                                        "elapsed_sec": 0.0,
                                        "mem_peak_mb": 0.0
                                    }
                                    self.results_log.append(log_entry)
                                    
                                    for iso_name in isolated_datasets.keys():
                                        iso_dir = os.path.join(run_dir, f"eval_{iso_name}")
                                        if os.path.exists(iso_dir):
                                            iso_metrics = self._extract_metrics(iso_dir)
                                            log_iso = {
                                                "seed": seed,
                                                "timestamp": datetime.fromtimestamp(
                                                    os.path.getmtime(iso_dir)
                                                ).isoformat(),
                                                "dataset": iso_name,
                                                "model_preset": mp_name,
                                                **current_hparams,
                                                **iso_metrics,
                                                "elapsed_sec": 0.0,
                                                "mem_peak_mb": 0.0
                                            }
                                            self.results_log.append(log_iso)
                                    pd.DataFrame(self.results_log).to_csv(
                                        master_csv_path, index=False
                                    )
                                except Exception as err:
                                    print(f"⚠️ Erro ao ressincronizar: {err}")
                            continue

                        # -------------------------------------------------
                        #  EXECUÇÃO EFETIVA (COM CACHE)
                        # -------------------------------------------------
                        config_run = copy.deepcopy(config_ds)
                        config_run = self._apply_overrides(config_run, mp_overrides)
                        for k, v in zip(grid_keys, combo):
                            config_run = self._apply_overrides(config_run, {k: v})

                        try:
                            results_map = self._run_single_pipeline_with_cached_data(
                                config_run, run_name, ds_name,
                                train_scaled, val_scaled, test_scaled,
                                valid_features,
                                isolated_datasets=isolated_datasets,
                                resume=resume
                            )

                            for subset_name, metrics in results_map.items():
                                log_entry = {
                                    "seed": seed,
                                    "timestamp": datetime.now().isoformat(),
                                    "dataset": subset_name,
                                    "model_preset": mp_name,
                                    **current_hparams,
                                    **metrics
                                }
                                self.results_log.append(log_entry)

                            pd.DataFrame(self.results_log).to_csv(
                                master_csv_path, index=False
                            )
                            print(f"✅ [{current_run}/{total_runs_per_seed}] "
                                  f"Concluído: {run_name} | "
                                  f"Subsets avaliados: {len(results_map)}")

                        except Exception as e:
                            print(f"❌ Falha crítica em {run_name}: {e}")
                            traceback.print_exc()
                            error_log = os.path.join(self.base_output_dir,
                                                     "errors.log")
                            with open(error_log, "a") as f:
                                f.write(f"{datetime.now().isoformat()} | "
                                        f"{run_name} | {str(e)}\n")
                                f.write(traceback.format_exc())
                                f.write("\n" + "="*80 + "\n")

                del train_scaled, val_scaled, test_scaled
                gc.collect()

            self._plot_summary()

        print(f"\n{'='*70}")
        print(f"✅ MARATONA CONCLUÍDA OU RETOMADA COM SUCESSO!")
        print(f"📁 Resultados salvos em: {self.results_root}")
        print(f"{'='*70}")

    # ---------------------------------------------------------
    #  GRÁFICOS DE RESUMO
    # ---------------------------------------------------------
    def _plot_summary(self):
        if not self.results_log:
            print("⚠️ Sem dados para gerar gráficos.")
            return
        try:
            df = pd.DataFrame(self.results_log)
            if 'pr_auc' not in df.columns or 'f1_score' not in df.columns:
                print("⚠️ Métricas 'pr_auc' ou 'f1_score' ausentes.")
                return

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            pivot_pr = df.pivot_table(index='model_preset', columns='dataset',
                                      values='pr_auc', aggfunc='mean')
            pivot_pr.plot(kind='bar', ax=axes[0])
            axes[0].set_ylabel("Média PR-AUC")
            axes[0].set_title("Comparativo de PR-AUC por Subset")

            pivot_f1 = df.pivot_table(index='model_preset', columns='dataset',
                                      values='f1_score', aggfunc='mean')
            pivot_f1.plot(kind='bar', ax=axes[1])
            axes[1].set_ylabel("Média F1-Score")
            axes[1].set_title("Comparativo de F1-Score por Subset")

            plt.tight_layout()
            plt.savefig(os.path.join(self.base_output_dir,
                                     "thesis_comparison_chart.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("📈 Gráfico comparativo gerado com sucesso!")
        except Exception as e:
            print(f"⚠️ Falha ao gerar gráfico: {e}")


# ---------------------------------------------------------
#  PONTO DE ENTRADA
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str,
                        default="config/experiment_config.yaml",
                        help="Caminho para o arquivo YAML de configuração.")
    parser.add_argument("--resume", action="store_true",
                        help="Retoma execuções interrompidas.")
    parser.add_argument("--run_type", type=str, default="full",
                        choices=["full", "debug"],
                        help="Tipo de execução: 'full' (dados completos) ou 'debug' (amostra reduzida). "
                             "Define a subpasta de saída dentro de results_root.")

    args = parser.parse_args()

    runner = ExperimentRunner(args.exp_config, run_type=args.run_type)
    runner.run(resume=args.resume)