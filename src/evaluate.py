"""Step 5: pull the four logged runs from MLflow and build comparison
artifacts: a summary CSV and grouped bar charts of the key metrics.
"""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import mlflow

import config

import os
EXPERIMENT_PATH = os.environ.get("MLFLOW_EXPERIMENT_PATH",
                                 "/Users/mwaseem@uchicago.edu/athletes-mlops-a2")
REPORTS = config.ROOT / "reports"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    REPORTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("databricks")
    exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    # keep only the four labeled experiment runs, drop tests / partials
    runs = runs[runs["params.feature_version"].notna()].copy()
    runs["label"] = runs["params.feature_version"] + "_C" + runs["params.C"]

    cols = {f"metrics.{m}": m for m in METRICS}
    summary = runs[["label", "params.feature_version", "params.C"]
                   + list(cols)].rename(columns=cols)
    summary = summary.rename(columns={"params.feature_version": "feature_version",
                                      "params.C": "C"})
    summary = summary.sort_values(["feature_version", "C"]).reset_index(drop=True)

    csv_path = REPORTS / "experiment_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[evaluate] wrote {csv_path}")
    print(summary.to_string(index=False))

    # grouped bar chart across the four runs
    ax = summary.set_index("label")[METRICS].plot(
        kind="bar", figsize=(10, 6), ylim=(0.7, 0.95), rot=0)
    ax.set_title("Metric comparison across feature versions and C")
    ax.set_ylabel("score")
    ax.legend(loc="lower right", ncol=len(METRICS))
    plt.tight_layout()
    fig_path = REPORTS / "metric_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[evaluate] wrote {fig_path}")

    # focused ROC-AUC chart, the metric that separates the versions
    # focused ROC-AUC chart, the metric that separates the versions
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.bar(summary["label"], summary["roc_auc"], color="#4c72b0")
    ax2.set_ylim(0.85, 0.93)
    ax2.set_title("ROC-AUC by experiment")
    ax2.set_ylabel("roc_auc")
    for i, v in enumerate(summary["roc_auc"]):
        ax2.text(i, v + 0.001, f"{v:.3f}", ha="center")
    plt.tight_layout()
    roc_path = REPORTS / "roc_auc_comparison.png"
    fig2.savefig(roc_path, dpi=150)
    print(f"[evaluate] wrote {roc_path}")


if __name__ == "__main__":
    main()