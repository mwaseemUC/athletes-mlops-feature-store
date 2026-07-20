#!/usr/bin/env bash
# Run the full athletes MLOps pipeline end to end.
#
# Prerequisites:
#   1. Conda env active (conda activate athletes-mlops) OR deps installed via pip
#   2. A .env file in the repo root with your Databricks credentials:
#        MLFLOW_TRACKING_URI=databricks
#        DATABRICKS_HOST=https://YOUR-WORKSPACE.cloud.databricks.com
#        DATABRICKS_TOKEN=dapiXXXXXXXXXXXX
#   3. The experiment path in src/train.py and src/evaluate.py set to your
#      Databricks user path (e.g. /Users/you@uchicago.edu/athletes-mlops-a2)
#
# Usage:
#   bash run.sh
#
# Exits immediately if any stage fails.

set -euo pipefail

echo "==> [1/6] Clean raw data"
python src/clean.py

echo "==> [2/6] Build feature source parquet"
python src/featurize.py

echo "==> [3/6] Register Feast feature definitions"
cd feature_repo
feast apply
cd ..

echo "==> [4/6] Verify Feast retrieval"
python src/load_features.py

echo "==> [5/6] Run all four experiments (logs to Databricks MLflow)"
python src/run_experiments.py

echo "==> [6/6] Build comparison summary and charts"
python src/evaluate.py

echo ""
echo "Done. See reports/ for outputs, and your Databricks MLflow experiment"
echo "for the four tracked runs."