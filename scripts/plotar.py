#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera matrizes de confusão e resumo de métricas da execução da demo."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GROUP_COLUMNS = [
    "dataset",
    "model_preset",
    "training.threshold_mode",
    "model.sequence_length",
]
CONFUSION_COLUMNS = ["tn", "fp", "fn", "tp"]
METRIC_LABELS = {
    "precision": "Precisão",
    "recall": "Recall",
    "f1_score": "F1",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "prevalence": "Prevalência",
    "lift_pr": "Lift PR",
}


def _scenario_label(dataset):
    return "Brute force" if "bruteforce" in str(dataset).casefold() else "Generalista"


def _split_label(dataset):
    value = str(dataset).casefold()
    if "chronological" in value or "chrono" in value:
        return "Cronológico"
    if "random" in value:
        return "Aleatório"
    return str(dataset)


def _load_results(root, csv_path=None, source="training"):
    if source == "inference":
        paths = sorted(Path(root).rglob("classification_report.json"))
        if not paths:
            raise FileNotFoundError(
                f"Nenhum classification_report.json encontrado em {root!s}. "
                "Execute primeiro a célula de inferência."
            )

        # Os relatórios no diretório raiz de cada modelo vêm do treinamento e têm
        # apenas métricas. A demo deve plotar os relatórios completos de inference_demo.
        required = set(
            GROUP_COLUMNS
            + CONFUSION_COLUMNS
            + ["seed", "precision", "recall", "f1_score", "pr_auc",
               "roc_auc", "prevalence", "lift_pr"]
        )
        frames = []
        ignored = 0
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as report_file:
                    report = json.load(report_file)
            except (OSError, json.JSONDecodeError):
                ignored += 1
                continue
            if not isinstance(report, dict) or not required.issubset(report):
                ignored += 1
                continue
            frames.append(pd.DataFrame([report]))

        if not frames:
            raise ValueError(
                f"Nenhum relatório completo de inferência encontrado em {root!s}. "
                "Rode a célula 3 depois de preparar os modelos."
            )
        if ignored:
            print(f"Relatórios de treino/incompletos ignorados: {ignored}")
        df = pd.concat(frames, ignore_index=True)
    else:
        if csv_path:
            paths = [Path(csv_path)]
        else:
            paths = sorted(Path(root).rglob("master_results_table.csv"))
        if not paths:
            raise FileNotFoundError(
                f"Nenhum master_results_table.csv encontrado em {root!s}."
            )

        frames = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"CSV não encontrado: {path}")
            frames.append(pd.read_csv(path))
        df = pd.concat(frames, ignore_index=True)

    aliases = {
        "threshold_mode": "training.threshold_mode",
        "sequence_length": "model.sequence_length",
    }
    for old_name, new_name in aliases.items():
        if new_name not in df.columns and old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    missing = [
        column for column in GROUP_COLUMNS + CONFUSION_COLUMNS + ["seed"]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            "Os resultados não têm as colunas necessárias para o resumo: "
            + ", ".join(missing)
        )
    return df


def _summarize(df):
    metric_columns = [
        metric for metric in METRIC_LABELS if metric in df.columns
    ]
    required_metrics = ["precision", "recall", "f1_score", "pr_auc", "roc_auc"]
    missing_metrics = [metric for metric in required_metrics if metric not in metric_columns]
    if missing_metrics:
        raise ValueError(
            "Métricas ausentes nos resultados: " + ", ".join(missing_metrics)
        )

    summary_rows = []
    grouped = df.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    for keys, group in grouped:
        row = dict(zip(GROUP_COLUMNS, keys))
        row["split"] = _split_label(row["dataset"])
        row["scenario"] = _scenario_label(row["dataset"])
        row["n_runs"] = int(len(group))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    split_order = {"Cronológico": 0, "Aleatório": 1}
    summary["_split_order"] = summary["split"].map(split_order).fillna(2)
    summary.sort_values(
        ["_split_order", "model_preset", "training.threshold_mode", "model.sequence_length"],
        inplace=True,
    )
    summary.drop(columns="_split_order", inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary, metric_columns


def _plot_confusion_matrices(df, summary, output_path):
    count = len(summary)
    if count == 0:
        raise ValueError("Nenhuma combinação de modelo e protocolo para plotar.")
    columns = min(4, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.7 * rows))
    axes = np.atleast_1d(axes).ravel()

    for index, row in summary.iterrows():
        subset = df
        for column in GROUP_COLUMNS:
            subset = subset[subset[column].astype(str) == str(row[column])]
        totals = {
            column: int(pd.to_numeric(subset[column], errors="coerce").fillna(0).sum())
            for column in CONFUSION_COLUMNS
        }
        matrix = np.array([
            [totals["tn"], totals["fp"]],
            [totals["fn"], totals["tp"]],
        ])
        ax = axes[index]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=ax,
            xticklabels=["Normal", "Anomalia"],
            yticklabels=["Normal", "Anomalia"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_title(
            f"{row['scenario']} · {row['split']} · {row['model_preset']}\n"
            f"limiar {row['training.threshold_mode']} · "
            f"janela {row['model.sequence_length']}\n"
            f"contagens de {row['n_seeds']} semente(s)",
            fontsize=10,
        )

    for ax in axes[count:]:
        ax.set_visible(False)

    fig.suptitle(
        "Demo da banca — matrizes de confusão no teste",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _short_label(row):
    split = "Crono" if row["split"] == "Cronológico" else (
        "Rand" if row["split"] == "Aleatório" else row["split"]
    )
    model = "RF" if "forest" in str(row["model_preset"]).casefold() else str(row["model_preset"])
    threshold = "global" if str(row["training.threshold_mode"]).casefold() == "global" else "per_user"
    scenario = "BF" if row["scenario"] == "Brute force" else "Geral"
    return f"{scenario} · {split}\n{model}\n{threshold}"


def _plot_metrics(summary, metric_columns, output_path):
    plot_metrics = [
        metric for metric in
        ["precision", "recall", "f1_score", "pr_auc", "roc_auc", "prevalence", "lift_pr"]
        if metric in metric_columns
    ]
    labels = [_short_label(row) for _, row in summary.iterrows()]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(
        len(plot_metrics),
        1,
        figsize=(max(16, 2.3 * len(summary)), 2.8 * len(plot_metrics)),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, plot_metrics):
        means = pd.to_numeric(summary[f"{metric}_mean"], errors="coerce").to_numpy()
        errors = pd.to_numeric(summary[f"{metric}_std"], errors="coerce").fillna(0).to_numpy()
        ax.errorbar(
            x,
            means,
            yerr=errors,
            fmt="o",
            color="#1f77b4",
            ecolor="#7f7f7f",
            capsize=4,
            markersize=6,
        )
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)
        if metric != "lift_pr":
            ax.set_ylim(0, 1.08)
        for xpos, mean, error in zip(x, means, errors):
            label = f"{mean:.3f}" if np.isfinite(mean) else "n/d"
            ax.annotate(
                label,
                (xpos, mean + error if np.isfinite(mean) else 0),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=9)
    axes[-1].set_xlabel("Partição · modelo · modo de limiar")
    one_seed = summary["n_seeds"].max() == 1
    title = (
        "Demo da banca — métricas da inferência (uma semente)"
        if one_seed
        else "Demo da banca — médias no teste (barras: desvio-padrão entre sementes)"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_demo_plots(root, output_dir, csv_path=None, source="training"):
    df = _load_results(root, csv_path, source=source)
    summary, metric_columns = _summarize(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion_path = output_dir / "matrizes_confusao_demo.png"
    metrics_path = output_dir / "metricas_demo.png"
    summary_path = output_dir / "resumo_metricas_demo.csv"

    _plot_confusion_matrices(df, summary, confusion_path)
    _plot_metrics(summary, metric_columns, metrics_path)
    summary.to_csv(summary_path, index=False)
    print(f"Execuções agregadas: {len(df)}")
    print(f"Combinações avaliadas: {len(summary)}")
    print(f"Matrizes de confusão: {confusion_path}")
    print(f"Gráficos de métricas: {metrics_path}")
    print(f"Resumo por combinação: {summary_path}")


def generate_batch_inference_plots(csv_path, output_dir, confusion_percentiles=(90, 95)):
    """Gera matrizes, curvas e comparações com baselines triviais da inferência batch."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV de inferência não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {
        "source_dir", "inference_dataset", "model_preset", "split_strategy",
        "threshold_mode", "percentile", "precision", "recall", "f1_score",
        "pr_auc", "roc_auc", "lift_pr", "prevalence", "n_windows",
        "n_anomalies_true", "n_alerts", "tp", "fp", "tn", "fn",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Colunas ausentes no CSV batch: " + ", ".join(missing))
    if df.empty:
        raise ValueError(f"CSV de inferência sem resultados: {csv_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce")
    df["label"] = df["source_dir"].map(lambda value: Path(str(value)).name)
    dataset_name = str(df["inference_dataset"].iloc[0])
    confusion_paths = {}

    # Matrizes para dois pontos operacionais: P90 amplia o orçamento de alertas;
    # P95 mantém a referência de baixo volume já usada na demo.
    for percentile in confusion_percentiles:
        operating = df[df["percentile"] == percentile].copy()
        if operating.empty:
            raise ValueError(
                f"O CSV não contém o percentil {percentile} para gerar matrizes."
            )
        operating.sort_values("source_dir", inplace=True)
        count = len(operating)
        columns = min(4, count)
        rows = math.ceil(count / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for index, (_, row) in enumerate(operating.iterrows()):
            matrix = np.array([
                [int(row["tn"]), int(row["fp"])],
                [int(row["fn"]), int(row["tp"])],
            ])
            ax = axes[index]
            sns.heatmap(
                matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                ax=ax, xticklabels=["Normal", "Anomalia"],
                yticklabels=["Normal", "Anomalia"],
                annot_kws={"size": 10, "weight": "bold"},
            )
            ax.set_xlabel("Classe predita")
            ax.set_ylabel("Classe real")
            ax.set_title(
                f"{row['label']}\n{row['model_preset']} · {row['split_strategy']} · "
                f"{row['threshold_mode']} · P{int(row['percentile'])}",
                fontsize=8,
            )
        for ax in axes[count:]:
            ax.set_visible(False)
        fig.suptitle(
            f"Inferência em {dataset_name} — matrizes no percentil {percentile}",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        confusion_path = output_dir / f"matrizes_confusao_test_prev_5pct_p{percentile}.png"
        fig.savefig(confusion_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        confusion_paths[int(percentile)] = confusion_path

    # Curvas das métricas de decisão ao longo da varredura de percentis.
    label_by_source = {}
    for _, row in df.drop_duplicates("source_dir").sort_values("source_dir").iterrows():
        short_scenario = "BF" if "Bruteforce" in str(row["source_dir"]) else "Generalista"
        short_split = "Chron" if str(row["split_strategy"]).lower().startswith("chron") else "Random"
        short_model = "RF" if "forest" in str(row["model_preset"]).lower() else "MLP"
        label_by_source[row["source_dir"]] = (
            f"{short_scenario} · {short_split} · {short_model} · {row['threshold_mode']}"
        )
    source_order = sorted(df["source_dir"].dropna().unique())
    metrics = [("precision", "Precisão"), ("recall", "Recall"), ("f1_score", "F1")]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    palette = sns.color_palette("husl", n_colors=max(1, len(source_order)))
    for ax, (metric, title) in zip(axes, metrics):
        for color, source in zip(palette, source_order):
            part = df[df["source_dir"] == source].sort_values("percentile")
            if not part.empty:
                ax.plot(
                    part["percentile"], part[metric], marker="o", linewidth=1.5,
                    markersize=3, color=color, label=label_by_source[source],
                )
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    axes[-1].set_xlabel("Percentil do limiar calculado no dataset de destino")
    fig.suptitle(f"Métricas por percentil — {dataset_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.78, 0.96])
    curves_path = output_dir / "metricas_por_percentil_test_prev_5pct.png"
    fig.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Baselines: sempre normal, sempre anomalia e seleção aleatória com o mesmo
    # orçamento de alertas do modelo. O último é a comparação mais justa para F1.
    baseline_rows = []
    selected = df[df["percentile"].isin(confusion_percentiles)].copy()
    for _, row in selected.iterrows():
        n = int(row["n_windows"])
        positives = int(row["n_anomalies_true"])
        prevalence = positives / n if n else 0.0
        alert_rate = int(row["n_alerts"]) / n if n else 0.0
        random_f1 = (
            2 * prevalence * alert_rate / (prevalence + alert_rate)
            if prevalence + alert_rate > 0 else 0.0
        )
        all_positive_f1 = (
            2 * prevalence / (1 + prevalence) if prevalence > 0 else 0.0
        )
        baseline_rows.append({
            "source_dir": row["source_dir"],
            "scenario_model_split_threshold": label_by_source[row["source_dir"]],
            "inference_dataset": row["inference_dataset"],
            "percentile": int(row["percentile"]),
            "n_windows": n,
            "n_anomalies_true": positives,
            "prevalence": prevalence,
            "alert_rate_model": alert_rate,
            "model_precision": row["precision"],
            "model_recall": row["recall"],
            "model_f1": row["f1_score"],
            "random_same_alert_precision_expected": prevalence,
            "random_same_alert_recall_expected": alert_rate,
            "random_same_alert_f1_expected": random_f1,
            "always_normal_accuracy": 1 - prevalence,
            "always_normal_f1": 0.0,
            "always_anomaly_precision": prevalence,
            "always_anomaly_recall": 1.0,
            "always_anomaly_f1": all_positive_f1,
            "pr_auc": row["pr_auc"],
            "random_ranking_pr_auc_baseline": prevalence,
            "lift_pr": row["lift_pr"],
            "beats_random_same_alert_f1": bool(row["f1_score"] > random_f1),
            "beats_always_anomaly_f1": bool(row["f1_score"] > all_positive_f1),
            "beats_random_ranking": bool(row["lift_pr"] > 1.0),
        })
    baseline = pd.DataFrame(baseline_rows)
    baseline_path = output_dir / "comparacao_baselines_test_prev_5pct.csv"
    baseline.to_csv(baseline_path, index=False)

    # F1 do modelo versus seletor aleatório com o mesmo número de alertas.
    # O score aleatório é uma expectativa, não uma execução aleatória específica.
    fig, axes = plt.subplots(
        1, len(confusion_percentiles),
        figsize=(max(15, len(source_order) * 0.95), 10),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, percentile in zip(axes, confusion_percentiles):
        part = baseline[baseline["percentile"] == percentile].sort_values(
            "model_f1", ascending=True
        )
        y = np.arange(len(part))
        height = 0.38
        ax.barh(y - height / 2, part["model_f1"], height=height,
                color="#2878a5", label="Modelo")
        ax.barh(y + height / 2, part["random_same_alert_f1_expected"],
                height=height, color="#c8c8c8", label="Aleatório, mesmo nº de alertas")
        ax.set_yticks(y)
        ax.set_yticklabels(part["scenario_model_split_threshold"], fontsize=7)
        ax.set_xlim(left=0)
        ax.set_xlabel("F1 da classe anômala")
        ax.set_title(f"P{percentile}")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Cenário · partição · modelo · modo de treino")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Comparação com baseline trivial de alertas aleatórios no mesmo orçamento",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    baseline_plot_path = output_dir / "baseline_alertas_aleatorios_test_prev_5pct.png"
    fig.savefig(baseline_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Métricas de ranking; Lift_PR=1 equivale ao baseline sem habilidade (PR-AUC=prevalência).
    ranking = df.sort_values("percentile").drop_duplicates("source_dir").copy()
    ranking.sort_values("source_dir", inplace=True)
    summary_cols = [
        "source_dir", "inference_dataset", "model_preset", "split_strategy",
        "threshold_mode", "seed", "n_windows", "n_anomalies_true", "prevalence",
        "pr_auc", "lift_pr", "roc_auc",
    ]
    summary_cols = [column for column in summary_cols if column in ranking.columns]
    summary = ranking[summary_cols].reset_index(drop=True)
    summary_path = output_dir / "resumo_inferencia_test_prev_5pct.csv"
    summary.to_csv(summary_path, index=False)

    x = np.arange(len(summary))
    short_labels = [
        f"{label_by_source[row['source_dir']]} "
        for _, row in summary.iterrows()
    ]
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(summary) * 1.1), 9), sharex=True)
    for index, (ax, metric, title) in enumerate(zip(
        axes, ["pr_auc", "lift_pr", "roc_auc"], ["PR-AUC", "Lift_PR", "ROC-AUC"]
    )):
        values = pd.to_numeric(summary[metric], errors="coerce").to_numpy()
        bars = ax.bar(x, values, color="#3986a8")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
        if metric == "lift_pr":
            ax.axhline(1.0, color="#b22222", linestyle="--", linewidth=1.5,
                       label="Baseline sem habilidade (Lift_PR=1)")
            ax.legend(loc="upper right", fontsize=8)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.3f}", (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7,
                )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(short_labels, rotation=55, ha="right", fontsize=7)
    fig.suptitle(f"Ranking versus baseline aleatório — {dataset_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ranking_path = output_dir / "ranking_lift_test_prev_5pct.png"
    fig.savefig(ranking_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Dataset de inferência: {dataset_name}")
    print(f"Resultados batch: {len(df)} linhas; modelos: {df['source_dir'].nunique()}")
    for percentile, path in confusion_paths.items():
        print(f"Matrizes P{percentile}: {path}")
    print(f"Curvas por percentil: {curves_path}")
    print(f"Comparação com baseline: {baseline_plot_path}")
    print(f"PR-AUC, Lift_PR e ROC-AUC (baseline Lift=1): {ranking_path}")
    print(f"Resumo dos baselines: {baseline_path}")
    print(f"Resumo de ranking: {summary_path}")
    return {
        "confusion": confusion_paths.get(95),
        "confusion_by_percentile": confusion_paths,
        "curves": curves_path,
        "ranking": ranking_path,
        "baseline_plot": baseline_plot_path,
        "baseline": baseline_path,
        "summary": summary_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera gráficos agregados dos resultados da demo da banca."
    )
    parser.add_argument(
        "--root",
        default="results/demo_banca/full",
        help="Pasta onde estão os resultados por semente.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV de resultados único; se omitido, busca os CSVs sob --root.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/demo_banca",
        help="Pasta para salvar os gráficos e o resumo CSV.",
    )
    parser.add_argument(
        "--source",
        choices=["training", "inference"],
        default="training",
        help="Fonte: CSVs do treinamento ou relatórios da inferência pré-computada.",
    )
    args = parser.parse_args()
    generate_demo_plots(args.root, args.out_dir, args.csv, args.source)
