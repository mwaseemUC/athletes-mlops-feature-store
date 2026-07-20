from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "athletes.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_PARQUET = PROCESSED_DIR / "athletes_clean.parquet"

FEATURE_REPO_DIR = ROOT / "feature_repo"
FEAST_SOURCE_PARQUET = FEATURE_REPO_DIR / "data" / "athlete_features.parquet"

RANDOM_STATE = 42
ENTITY_KEY = "athlete_id"
EVENT_TS = "event_timestamp"
TARGET = "high_total_lift"

# Strength lifts define the target -> never used as features (leakage guard)
STRENGTH_LIFTS = ["candj", "snatch", "deadlift", "backsq"]

FEATURES_V1 = ["age", "height", "weight", "gender"]
FEATURES_V2 = ["age", "height", "weight", "gender", "bmi", "age_bin",
               "region", "experience", "schedule", "howlong", "eat"]

# numeric vs categorical split, used by the modeling pipeline in step 4
NUMERIC_V1 = ["age", "height", "weight"]
CATEGORICAL_V1 = ["gender"]
NUMERIC_V2 = ["age", "height", "weight", "bmi"]
CATEGORICAL_V2 = ["gender", "age_bin", "region", "experience",
                  "schedule", "howlong", "eat"]