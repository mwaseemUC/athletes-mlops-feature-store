"""Step 4: train one experiment = one feature version x one hyperparameter config.

Pulls features from Feast (via load_features), builds a version-aware sklearn
Pipeline (ColumnTransformer preprocessing + LogisticRegression), evaluates on a
hold-out split, and logs params, metrics, and artifacts to Databricks MLflow.
"""
import matplotlib
matplotlib.use("Agg")  # no GUI backend, we only save figures
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay,
)

import config
from load_features import load_training_frame

from dotenv import load_dotenv
load_dotenv()

import os
EXPERIMENT_PATH = os.environ.get("MLFLOW_EXPERIMENT_PATH",
                                 "/Users/mwaseem@uchicago.edu/athletes-mlops-a2")

# which columns are numeric vs categorical, per feature version
VERSION_COLS = {
    "v1": (config.NUMERIC_V1, config.CATEGORICAL_V1),
    "v2": (config.NUMERIC_V2, config.CATEGORICAL_V2),
}


def build_pipeline(version: str, C: float) -> Pipeline:
    numeric, categorical = VERSION_COLS[version]

    numeric_pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # min_frequency caps rare survey categories so v2 stays tractable
    categorical_pre = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pre, numeric),
        ("cat", categorical_pre, categorical),
    ])
    clf = LogisticRegression(C=C, max_iter=1000, random_state=config.RANDOM_STATE)
    return Pipeline([("preprocess", pre), ("clf", clf)])


def run(version: str, C: float):
    df = load_training_frame(version)
    numeric, categorical = VERSION_COLS[version]
    X = df[numeric + categorical]
    y = df[config.TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline(version, C)
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba),
    }

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_PATH)
    run_name = f"{version}_C{C}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("feature_version", version)
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("n_features_in", X.shape[1])
        mlflow.log_param("n_train", len(X_tr))
        mlflow.set_tag("assignment", "adsp31021-a2")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(pipe, X_te, y_te, ax=ax)
        ax.set_title(f"Confusion Matrix ({run_name})")
        os.makedirs("artifacts", exist_ok=True)
        plot_path = os.path.join("artifacts", f"cm_{run_name}.png")
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(plot_path)

        mlflow.sklearn.log_model(
            pipe, name="model",
            serialization_format="cloudpickle",
            input_example=X_te.iloc[:5],
        )

    print(f"[train] {run_name}: " +
          " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


if __name__ == "__main__":
    # quick single-run smoke test
    run("v1", 1.0)