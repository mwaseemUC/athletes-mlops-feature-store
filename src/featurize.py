"""Step 2b: build the feature source parquet Feast serves.
Holds entity key, event timestamp, target, and the union of v1 and v2
feature columns. Strength lifts are excluded (they define the target).
"""
import pandas as pd
import config

AGE_BINS = [17, 25, 35, 45, 90]
AGE_LABELS = ["u25", "25_34", "35_44", "45p"]


def main():
    config.FEAST_SOURCE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.CLEAN_PARQUET)

    for c in config.STRENGTH_LIFTS + ["age", "height", "weight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["total_lift"] = df[config.STRENGTH_LIFTS].sum(axis=1)
    median_total = df["total_lift"].median()
    df[config.TARGET] = (df["total_lift"] >= median_total).astype(int)

    df["bmi"] = 703.0 * df["weight"] / (df["height"] ** 2)
    df["age_bin"] = pd.cut(df["age"], bins=AGE_BINS,
                           labels=AGE_LABELS).astype("object")
    df["gender"] = df["gender"].astype(str).str.strip().str.title()
    df[config.EVENT_TS] = pd.Timestamp("2021-01-01")

    keep = [config.ENTITY_KEY, config.EVENT_TS, config.TARGET]
    feat_cols = sorted(set(config.FEATURES_V1) | set(config.FEATURES_V2))
    out = df[keep + feat_cols].copy()
    out["age"] = out["age"].astype("int64")
    for c in ["height", "weight", "bmi"]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.FEAST_SOURCE_PARQUET, index=False)
    print(f"[featurize] {out.shape[0]:,} rows -> {config.FEAST_SOURCE_PARQUET}")
    print(f"[featurize] median total_lift={median_total:.1f} | "
          f"positive class share={df[config.TARGET].mean():.3f}")
    print("[featurize] nulls per feature:")
    print(out[feat_cols].isna().sum())


if __name__ == "__main__":
    main()