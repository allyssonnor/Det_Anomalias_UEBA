#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolida todos os arquivos master_results_table.csv encontrados recursivamente
em uma pasta raiz, gerando um único CSV com todos os resultados.

Uso:
    python scripts/consolidate_results.py --root results --output resultados_consolidados.csv

Argumentos:
    --root    : Pasta raiz onde procurar (padrão: results)
    --output  : Caminho do arquivo de saída (padrão: resultados_consolidados.csv)
    --add_path: Adiciona coluna com o caminho de origem (padrão: True)
"""

import os
import sys
import glob
import pandas as pd
import argparse
from datetime import datetime

def consolidate_results(root_dir, output_file, add_path=True):
    """
    Procura recursivamente por master_results_table.csv e consolida em um único CSV.
    """
    if not os.path.exists(root_dir):
        print(f"❌ Erro: Pasta '{root_dir}' não encontrada.")
        return

    # Padrão de busca recursiva
    pattern = os.path.join(root_dir, "**", "master_results_table.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"⚠️ Nenhum arquivo 'master_results_table.csv' encontrado em '{root_dir}'.")
        return

    print(f"🔍 Encontrados {len(files)} arquivos.")

    # Lista para armazenar os DataFrames
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if add_path:
                # Adiciona coluna com o caminho relativo (para rastreabilidade)
                rel_path = os.path.relpath(f, root_dir)
                df["source_file"] = rel_path
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Erro ao ler {f}: {e}")

    if not dfs:
        print("❌ Nenhum dado válido foi lido.")
        return

    # Concatena todos os DataFrames
    df_all = pd.concat(dfs, ignore_index=True)

    # Remove duplicatas (caso haja) baseado em todas as colunas
    df_all = df_all.drop_duplicates()

    # Ordena por dataset, modelo, seed (se existirem)
    sort_cols = []
    for col in ["dataset", "model_preset", "seed"]:
        if col in df_all.columns:
            sort_cols.append(col)
    if sort_cols:
        df_all = df_all.sort_values(by=sort_cols).reset_index(drop=True)

    # Salva
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df_all.to_csv(output_file, index=False)

    print(f"\n✅ Consolidação concluída!")
    print(f"📊 Total de linhas: {len(df_all)}")
    print(f"📁 Arquivo salvo em: {output_file}")
    print(f"📌 Colunas disponíveis: {list(df_all.columns)}")

def main():
    parser = argparse.ArgumentParser(description="Consolida todos os master_results_table.csv de uma pasta.")
    parser.add_argument("--root", type=str, default="results",
                        help="Pasta raiz para busca recursiva (padrão: results)")
    parser.add_argument("--output", type=str, default="resultados_consolidados.csv",
                        help="Caminho do arquivo de saída (padrão: resultados_consolidados.csv)")
    parser.add_argument("--no-path", action="store_true",
                        help="Não adiciona a coluna 'source_file' com o caminho de origem")
    args = parser.parse_args()

    consolidate_results(
        root_dir=args.root,
        output_file=args.output,
        add_path=not args.no_path
    )

if __name__ == "__main__":
    main()
