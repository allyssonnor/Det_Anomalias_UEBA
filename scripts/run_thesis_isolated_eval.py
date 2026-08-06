import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import yaml
import shutil
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import gc

# =========================================================
# RESOLUÇÃO DE AMBIENTE E CAMINHOS
# =========================================================
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

def resolve_path(path):
    """Garante que o ficheiro/diretório é encontrado, quer no Colab quer localmente."""
    if os.path.isabs(path) and os.path.exists(path):
        return path

    search_paths = [
        path,
        os.path.join(root_path, path),
        os.path.join("/content/drive/MyDrive/Project_Detector", path)
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"❌ Ficheiro ou diretório não encontrado: {path}\nLocais procurados: {search_paths}")

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

def extract_pr_auc_from_report(report_path):
    """Vasculha o arquivo de relatório inteiro em busca do PR-AUC."""
    if not os.path.exists(report_path):
        return 0.0
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                normalized_line = line.upper().replace(" ", "").replace("-", "")
                if "PRAUC" in normalized_line:
                    for delimiter in [":", "="]:
                        if delimiter in line:
                            parts = line.split(delimiter)
                            val_str = "".join(c for c in parts[1] if c.isdigit() or c == ".").strip()
                            return round(float(val_str), 4)
    except Exception as e:
        print(f"⚠️ Erro ao analisar classificação de texto para obter PR-AUC: {e}")
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Avaliação UEBA de Ataques Isolados")
    parser.add_argument("--config", type=str, default="config/ultima_run2.yaml")
    parser.add_argument("--tests_dir", type=str, default=None, help="Ignora o YAML e força este diretório")
    args = parser.parse_args()

    # 1. Carregar Configuração Base
    actual_config_path = resolve_path(args.config)
    print(f"📖 A ler configuração base: {actual_config_path}")
    
    with open(actual_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # 2. Resolução Dinâmica do Diretório de Testes Isolados
    yaml_tests_dir = base_config.get("data", {}).get("isolated_tests_dir")
    
    if args.tests_dir:
        raw_tests_dir = args.tests_dir
        print(f"⚠️ Aviso: A usar diretório de testes passado por argumento: {raw_tests_dir}")
    elif yaml_tests_dir:
        raw_tests_dir = yaml_tests_dir
        print(f"📂 Diretório de testes lido do YAML: {raw_tests_dir}")
    else:
        raw_tests_dir = "/content/drive/MyDrive/Project_Detector/data/synthetic"
        print(f"⚠️ Aviso: 'isolated_tests_dir' não encontrado no YAML. Usando fallback: {raw_tests_dir}")

    tests_dir_resolved = resolve_path(raw_tests_dir)

    # =========================================================
    # 🔥 ALTERAÇÃO AQUI: LEITURA PRIORIZANDO O HYPERPARAMETER_GRID
    # =========================================================
    t_modes_raw = base_config.get("hyperparameter_grid", {}).get("training.threshold_mode", None)
    
    # Se não estiver no grid, tenta ler da raiz (fallback)
    if t_modes_raw is None:
        t_modes_raw = base_config.get("training", {}).get("threshold_mode", "per_user")
    
    # Garante que seja uma lista
    threshold_modes = t_modes_raw if isinstance(t_modes_raw, list) else [t_modes_raw]
    
    print(f"🔧 Modos de threshold a avaliar: {threshold_modes}")

    # Lista de ataques a avaliar (deve bater com os nomes configurados no Gerador)
    attacks = [
        "bruteforce", "pass_the_hash", "lateral_movement", "out_of_hours",
        "credential_stuffing", "unusual_admin", "volume_spike",
        "rdp_anomaly", "golden_ticket", "service_abuse"
    ]

    for mode in threshold_modes:
        print(f"\n{'#'*80}")
        print(f"🔄 A INICIAR CICLO DE AVALIAÇÃO PARA MODO DE LIMIAR: {mode.upper()}")
        print(f"{'#'*80}")

        config = copy.deepcopy(base_config)
        config["training"]["threshold_mode"] = mode

        marathon_output = config.get("output_dir", "results/isolated_eval_marathon")
        marathon_output = os.path.join(marathon_output, f"mode_{mode}")
        base_model_output = os.path.join(marathon_output, "base_model")
        config["output_dir"] = base_model_output
        os.makedirs(base_model_output, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"🧠 FASE 1: TREINO DO MODELO GENERALISTA ({mode.upper()})")
        print(f"{'='*70}")

        adapter = DatasetAdapter(config)
        train_df, val_df, _ = adapter.load()

        if train_df is None or train_df.empty:
            print("❌ ERRO: O dataset de treino está vazio.")
            return

        processor = FeatureProcessor(config)
        train_df = processor.transform(train_df)
        val_df = processor.transform(val_df)

        temporal = TemporalBuilder(config)
        train_df = temporal.transform(train_df)
        val_df = temporal.transform(val_df)

        model = create_model(config)
        trainer = Trainer(config, model)
        valid_features = [c for c in processor.final_feature_names if c in train_df.columns and c in val_df.columns]
        trainer.run(train_df, val_df, val_df.copy(), feature_columns=valid_features)
        
        print(f"\n✅ Modelo treinado e congelado para o modo {mode}!")

        print(f"\n{'='*70}")
        print(f"🎯 FASE 2: AVALIAÇÃO CONTRA ATAQUES ISOLADOS ({mode.upper()})")
        print(f"{'='*70}")

        results_log = []

        for attack in attacks:
            # =========================================================
            # AJUSTE CRÍTICO: Caminho Limpo Baseado no Gerador
            # =========================================================
            test_file = os.path.join(
                tests_dir_resolved, 
                f"test_{attack}", 
                f"test_{attack}.jsonl"
            )
            
            if not os.path.exists(test_file):
                print(f"⚠️ Ficheiro não encontrado, a saltar: {test_file}")
                continue

            print(f"\n🔬 A avaliar cenário: {attack.upper()}")
            
            attack_output_dir = os.path.join(marathon_output, f"eval_{attack}")
            os.makedirs(attack_output_dir, exist_ok=True)

            try:
                source_thresholds = os.path.join(base_model_output, "user_thresholds.json")
                target_thresholds = os.path.join(attack_output_dir, "user_thresholds.json")
                shutil.copy(source_thresholds, target_thresholds)
            except Exception as e:
                print(f"❌ ERRO CRÍTICO: Falha ao copiar thresholds para {attack}. ({e})")
                continue

            # Processar Teste
            test_df = adapter._load_resource(test_file, 0)
            test_df = adapter._normalize(test_df)
            test_df = adapter._apply_auto_labels(test_df, test_file)

            test_df = processor.transform(test_df)
            test_df = temporal.transform(test_df)

            if test_df.empty or len(test_df) < trainer.sequence_length:
                print(f"⚠️ Eventos insuficientes para {attack}.")
                continue

            X_test, indices, _, _ = trainer._build_sequences(test_df, fit=False)
            
            if len(X_test) == 0:
                print(f"⚠️ Nenhuma sequência válida para {attack}.")
                continue
                
            scores = trainer.model.score(X_test)

            np.save(os.path.join(attack_output_dir, "test_scores.npy"), scores)
            np.save(os.path.join(attack_output_dir, "test_indices.npy"), indices)

            eval_config = copy.deepcopy(config)
            eval_config["output_dir"] = attack_output_dir
            
            try:
                run_external_evaluation(eval_config, test_df=test_df)
            except Exception as e:
                print(f"❌ ERRO no avaliador externo para {attack}: {e}")
                continue

            report_path = os.path.join(attack_output_dir, "classification_report.txt")
            pr_auc = extract_pr_auc_from_report(report_path)

            results_log.append({
                "Ataque": attack.upper(),
                "PR-AUC": pr_auc
            })
            
            print(f"✅ {attack.upper()} -> PR-AUC: {pr_auc:.4f}")

        # Relatório Final para o Modo Atual
        if results_log:
            print(f"\n{'='*70}")
            print(f"🏆 RESUMO DA SENSIBILIDADE ({mode.upper()})")
            print(f"{'='*70}")
            df_results = pd.DataFrame(results_log)
            print(df_results.to_string(index=False))
            
            df_results.to_csv(os.path.join(marathon_output, "sensibilidade_ataques.csv"), index=False)
            
            plt.figure(figsize=(12, 6))
            df_results.sort_values("PR-AUC", ascending=True, inplace=True)
            plt.barh(df_results["Ataque"], df_results["PR-AUC"], color="skyblue")
            plt.xlabel("Métrica PR-AUC")
            plt.title(f"Sensibilidade do Modelo UEBA ({mode.upper()})")
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(marathon_output, f"grafico_sensibilidade_{mode}.png"), dpi=300)

        # Força limpeza total de memória
        try:
            import keras
            keras.backend.clear_session()
        except:
            pass
        gc.collect()

    print(f"\n🚀 TUDO CONCLUÍDO! Verifique as subpastas em 'results/LAST_RUN'.")

if __name__ == "__main__":
    main()