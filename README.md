# Athletes MLOps Pipeline with a Feature Store

**ADSP 31021 Machine Learning Operations, Assignment #2 (Feature Store)**

**GitHub repository:** https://github.com/mwaseemUC/athletes-mlops-feature-store

**Author:** Mohammed Waseem (mwaseem@uchicago.edu)

An end-to-end, reproducible ML workflow on the CrossFit Open `athletes.csv`
dataset. Features are defined and served through a **Feast** feature store with
two versioned feature definitions. A single algorithm (**LogisticRegression**)
is trained across two feature versions and two hyperparameter configurations,
producing four tracked experiments. All runs are logged to **MLflow on
Databricks** (parameters, metrics, artifacts, and the serialized model).

---

## 1. Problem framing

**Task.** Binary classification: predict whether an athlete is a *high total-lift*
performer.

**Target (`high_total_lift`).** `total_lift = clean&jerk + snatch + deadlift +
back squat`. The label is `1` if an athlete's `total_lift` is at or above the
population **median**, else `0`. A median split gives a balanced target
(positive class share ≈ 0.50), so accuracy is a meaningful metric.

**Leakage guard.** Because the target is derived from the four strength lifts,
those four columns are **never used as features**. The model predicts strength
tier from demographics and training-lifestyle signal only.

---

## 2. Repository structure

```
athletes-mlops-feature-store/
├── README.md
├── requirements.txt
├── environment.yml
├── run.py                    # runs the full pipeline (cross-platform)
├── run_pipeline.ipynb        # runs the full pipeline (Colab / Jupyter path)
├── .gitignore
├── .env                      # (git-ignored) Databricks credentials
├── data/
│   ├── raw/athletes.csv      # original dataset
│   └── processed/            # (git-ignored) cleaned parquet
├── feature_repo/             # Feast project
│   ├── feature_store.yaml
│   ├── feature_defs.py       # entity + two feature views (v1, v2)
│   └── data/                 # feature parquet + Feast registry / online store
├── src/
│   ├── config.py             # paths, column lists, constants
│   ├── clean.py              # ingestion + cleaning -> processed parquet
│   ├── featurize.py          # builds the feature source parquet + target
│   ├── load_features.py      # retrieves training data from Feast by version
│   ├── train.py              # trains one experiment, logs to MLflow
│   ├── run_experiments.py    # runs all four experiment combinations
│   └── evaluate.py           # pulls runs from MLflow, builds comparison outputs
└── reports/
    ├── experiment_comparison.md   # standalone experiment comparison summary
    ├── experiment_summary.csv     # metrics table (pulled from MLflow runs)
    ├── metric_comparison.png
    └── roc_auc_comparison.png
```

---

## 3. MLOps platform choice

**MLflow on Databricks Managed MLflow** was selected for experiment tracking and
model logging, and **Feast** for the feature store.

- **MLflow / Databricks**: the experiment tracking server is fully managed, so
  every run's parameters, metrics, artifacts, and model are centralized and
  comparable in one UI with no local server to maintain. This directly supports
  the assignment's requirement to document parameters, feature versions, and
  results.
- **Feast**: a lightweight, open-source feature store that runs fully local
  (SQLite registry + Parquet offline store), with first-class support for
  versioned feature definitions via separate feature views. No cloud
  infrastructure is required, which keeps the project reproducible from a clean
  environment.

---

## 4. Data ingestion and cleaning

Ingestion and cleaning happen in `src/clean.py` (the raw-file read at the top of
the script is the ingestion step). Cleaning rules:

- Drop rows missing any required survey/lift field (`region, age, weight,
  height, howlong, gender, eat, background, experience, schedule`, and the four
  lifts).
- Drop unused / sparse columns (affiliate, team, name, and the sparse
  conditioning benchmarks such as `fran`, `helen`, etc.).
- Range filters: `weight < 1500`, `age >= 18`, `48 < height < 96`, and
  gender-specific deadlift ceilings (Male ≤ 1105, Female ≤ 636). Per-lift upper
  bounds on `candj`, `snatch`, `backsq`.
- Replace survey noise (`"Decline to answer|"`) with NaN and drop the affected
  rows.
- **`athlete_id` is retained** (Assignment 1 dropped it) because Feast requires
  an entity key.

**Result:** 423,006 raw rows → **30,190** cleaned rows. The reduction reflects
strict completeness requirements on the survey and lift fields.

### Assumptions

- Rows with any missing required field are dropped rather than imputed, to keep
  the training set fully observed.
- Values outside plausible human ranges are treated as data-entry errors.
- Gender is restricted to `Male` / `Female` (the only well-supported values).
- A synthetic constant `event_timestamp` is added so Feast can perform its
  point-in-time join. The dataset is a static snapshot, so a single timestamp is
  appropriate.

---

## 5. Feature store and feature versioning

Features are served from Feast. Both versions read from the **same** source
parquet (`feature_repo/data/athlete_features.parquet`) but expose different
feature sets through two feature views:

| Feature version | Feature view | Features |
|-----------------|--------------|----------|
| **v1 (baseline)** | `athlete_features_v1` | `age, height, weight, gender` |
| **v2 (engineered)** | `athlete_features_v2` | v1 + `bmi`, `age_bin`, `region`, `experience`, `schedule`, `howlong`, `eat` |

**Difference between versions.** v1 is minimal demographics. v2 adds a
body-composition feature (`bmi`), a binned age band (`age_bin`), and
training-lifestyle survey signal (region, experience level, training schedule,
how long training, and eating habits). This isolates the effect of feature
enrichment while the algorithm is held constant.

Retrieval uses `store.get_historical_features(...)` in `src/load_features.py`,
which joins the requested feature view onto an entity dataframe
(`athlete_id + event_timestamp + label`), the point-in-time-correct way a
feature store serves training data.

---

## 6. Modeling pipeline

A single scikit-learn `Pipeline` (`src/train.py`) wraps preprocessing and the
estimator so the entire transform is reproducible and leakage-free:

- **Numeric branch:** median imputation → `StandardScaler`.
- **Categorical branch:** most-frequent imputation →
  `OneHotEncoder(handle_unknown="ignore", min_frequency=25)`. The
  `min_frequency` cap folds rare survey categories into an infrequent bucket so
  v2's one-hot matrix stays tractable.
- **Estimator:** `LogisticRegression(max_iter=1000)`, the same algorithm across
  all four experiments (no AutoML, no automated algorithm selection).

Evaluation is on a stratified 80/20 hold-out split (`random_state=42`).

---

## 7. Experiments and results

Four experiments: **2 feature versions × 2 hyperparameter configurations**
(`C = 0.1` and `C = 10.0`), same algorithm throughout.

| Experiment | Feature version | C | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------------|-----------------|-----|----------|-----------|--------|--------|---------|
| v1_C0.1 | v1 | 0.1 | 0.803 | 0.749 | 0.913 | 0.823 | 0.878 |
| v1_C10.0 | v1 | 10.0 | 0.803 | 0.749 | 0.912 | 0.822 | 0.878 |
| v2_C0.1 | v2 | 0.1 | 0.837 | 0.806 | 0.888 | 0.845 | **0.919** |
| v2_C10.0 | v2 | 10.0 | 0.836 | 0.807 | 0.884 | 0.844 | 0.918 |

A standalone writeup of this comparison, with the full findings and evidence
screenshots, is in **`reports/experiment_comparison.md`**. The metrics table is
also saved as `reports/experiment_summary.csv`, and the charts as
`reports/metric_comparison.png` and `reports/roc_auc_comparison.png`.

### Findings

- **Feature version is the dominant factor.** Moving v1 → v2 raises ROC-AUC from
  0.878 to 0.919 and improves every metric, driven by the engineered and
  lifestyle features.
- **The hyperparameter barely matters.** `C = 0.1` vs `C = 10.0` changes metrics
  only in the third/fourth decimal, indicating the model sits in a stable
  regularization regime for this data.
- **Best model:** `v2` at `C = 0.1` (ROC-AUC 0.919), essentially tied with
  `v2 C=10.0`.
- v1 shows high recall with lower precision. With only demographics it tends to
  over-predict the high-tier class.

---

## 8. Setup and reproduction

### Environment

Conda (Python 3.11):

```bash
conda env create -f environment.yml
conda activate athletes-mlops
```

or pip:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Databricks credentials

Create a `.env` file in the repo root (git-ignored, never commit it):

```
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://YOUR-WORKSPACE.cloud.databricks.com
DATABRICKS_TOKEN=dapiXXXXXXXXXXXX
MLFLOW_EXPERIMENT_PATH=/Users/YOUR-EMAIL@school.edu/athletes-mlops-a2
```

The experiment path must point to a folder under your own Databricks user. The
scripts read it from the `MLFLOW_EXPERIMENT_PATH` environment variable (with a
fallback default), so no source edits are needed, just set it in `.env` (shell
path) or enter it when prompted (notebook path).

### Run the pipeline

There are two ways to run the full pipeline. Both execute the same `src/`
scripts and produce identical results.

#### Option A: script runner (local, any OS)

Best for a local run with the environment active and a `.env` file in place.
The Python runner works on any OS (Windows, macOS, Linux) with no bash required,
and stops immediately if any stage fails:

```bash
python run.py
```

`run.py` runs all six stages in order. To run them by hand instead:

```bash
python src/clean.py            # 1. clean raw data -> processed parquet
python src/featurize.py        # 2. build feature source parquet (target, bmi, age_bin)
cd feature_repo && feast apply && cd ..   # 3. register Feast definitions
python src/load_features.py    # 4. verify Feast retrieval
python src/run_experiments.py  # 5. run the four experiments (logs to Databricks)
python src/evaluate.py         # 6. build comparison summary + charts
```

#### Option B: notebook (Google Colab or local Jupyter)

Open `run_pipeline.ipynb`. In Colab it clones the repo, installs dependencies,
prompts for your Databricks credentials securely (via `getpass`, nothing is
saved to the notebook), runs all six stages, and displays the summary table and
charts inline. Locally, open it from the repo root and run the cells top to
bottom (skip the clone and install cells if your env is already set up).

The notebook is convenient because it does not require a committed `.env`: a
fresh clone has no credentials, so you enter them at runtime.

Outputs land in `reports/` either way. All four runs appear in the Databricks
MLflow experiment, comparable side by side by their `feature_version` and `C`
parameters.

---

## 9. Deliverables checklist

- [x] GitHub repository with working code
- [x] README with setup and execution instructions
- [x] Feature store integration (Feast) with two versioned feature definitions
- [x] Four experiments (2 feature versions × 2 hyperparameters, one algorithm)
- [x] Experiment tracking on MLflow / Databricks (params, metrics, artifacts, model)
- [x] Experiment comparison summary (`reports/experiment_comparison.md`)
- [x] Model evaluation metrics and visualizations (`reports/*.png`)
- [x] Dependency management (`requirements.txt`, `environment.yml`)

***NOTE**: Claude Sonnet 5 was used to assist in this assignment*