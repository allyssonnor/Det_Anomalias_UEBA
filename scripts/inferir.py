🚀 Executando inferência e gerando dashboard para a banca...
======================================================================
🔍 INFERÊNCIA DE ALTA FIDELIDADE E ZERO VAZAMENTO
======================================================================
📂 Diretório de artefatos recuperado: /content/Det_Anomalias_UEBA/results/colab_cpu_runs/full/run_seed42/Synthetic_Isolation_Forest_CPU
✅ Configuração oficial do treino (config_used.yaml) carregada com sucesso.
✅ Pipeline de features recuperado: 19 features esperadas.
🔗 Carregando pesos do modelo de: /content/Det_Anomalias_UEBA/results/colab_cpu_runs/full/run_seed42/Synthetic_Isolation_Forest_CPU/saved_model/model.pkl
🤖 Wrapper nativo blindado recuperado: IsolationForestModel
🎯 Forçando leitura do novo dataset alvo: data/output_thesis_marathon/synthetic/test/test.jsonl
📂 Carregamento em curso (Modo: full)
✅ train_path pronto: 0 registros.
✅ val_path pronto: 0 registros.
✅ test_path pronto: 211185 registros.
🔧 Colunas comuns: 22 (removidas 0)
✅ FeatureProcessor histórico recuperado dos metadados (Zero Leakage garantido).
📊 Vetor Final (211185 eventos): 19 dimensões isoladas.
⚙️ Extraindo janelas exatas...
🚀 Executando inferência probabilística delegada ao motor oficial...

============================================================
📊 RESULTADOS DA INFERÊNCIA PURA (Zero VAZAMENTO)
============================================================
 🔹 Janelas Avaliadas            : 209385
 🔹 Anomalias Reais (Gabarito)   : 29713
 🔹 Alertas Disparados           : 94016
------------------------------------------------------------
 ✅ Verdadeiros Positivos (TP) : 22277
 ⚠️  Falsos Positivos (FP)     : 71739
 ❌ Falsos Negativos (FN)     : 7436
 🛡️  Verdadeiros Negativos (TN) : 107933
------------------------------------------------------------
 🎯 Precision                  : 0.2369
 🎯 Recall                     : 0.7497
 🎯 F1-Score                   : 0.3601
 🎯 PR-AUC                     : 0.3600
 🎯 ROC-AUC                    : 0.7406
============================================================

📊 Gerando Dashboard Visual...
✅ Dashboard salvo com sucesso em: results/dashboard_inferencia_banca.png

🏆 DASHBOARD DE INFERÊNCIA EXECUTADO COM SUCESSO:
