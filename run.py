#!/usr/bin/env python
"""Run the full athletes MLOps pipeline end to end (cross-platform).

Runs the same six stages as run.sh, but works on any OS with Python (no bash
required). Each stage runs as a subprocess; the script stops immediately if any
stage returns a non-zero exit code.

Prerequisites:
  1. Dependencies installed (conda env or pip).
  2. A .env file in the repo root with Databricks credentials:
       MLFLOW_TRACKING_URI=databricks
       DATABRICKS_HOST=https://YOUR-WORKSPACE.cloud.databricks.com
       DATABRICKS_TOKEN=dapiXXXXXXXXXXXX
       MLFLOW_EXPERIMENT_PATH=/Users/you@school.edu/athletes-mlops-a2

Usage:
    python run.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # use the same interpreter that launched this script

# (label, command, working directory)
STAGES = [
    ("1/6 Clean raw data",              [PY, "src/clean.py"],           ROOT),
    ("2/6 Build feature source parquet",[PY, "src/featurize.py"],       ROOT),
    ("3/6 Register Feast definitions",  ["feast", "apply"],  ROOT / "feature_repo"),
    ("4/6 Verify Feast retrieval",      [PY, "src/load_features.py"],   ROOT),
    ("5/6 Run four experiments",        [PY, "src/run_experiments.py"], ROOT),
    ("6/6 Build summary and charts",    [PY, "src/evaluate.py"],        ROOT),
]


def main():
    for label, cmd, cwd in STAGES:
        print(f"\n==> [{label}]  ({' '.join(cmd)})", flush=True)
        result = subprocess.run(cmd, cwd=str(cwd))
        if result.returncode != 0:
            print(f"\nStage failed: {label} (exit code {result.returncode}). "
                  f"Stopping.", file=sys.stderr)
            sys.exit(result.returncode)

    print("\nDone. See reports/ for outputs, and your Databricks MLflow "
          "experiment for the four tracked runs.")


if __name__ == "__main__":
    main()