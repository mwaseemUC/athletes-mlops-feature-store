"""Step 3 retrieval: pull a training set from Feast by feature version.

get_historical_features joins the requested feature view onto an entity
dataframe (athlete_id + timestamp + label), which is the point-in-time
correct way a feature store serves training data.
"""
import pandas as pd
from feast import FeatureStore
import config

# maps a version string to its feature view and that view's feature columns
VIEW_FEATURES = {
    "v1": ("athlete_features_v1", config.FEATURES_V1),
    "v2": ("athlete_features_v2", config.FEATURES_V2),
}


def load_training_frame(version: str) -> pd.DataFrame:
    if version not in VIEW_FEATURES:
        raise ValueError(f"unknown version {version!r}, use one of {list(VIEW_FEATURES)}")
    view, feats = VIEW_FEATURES[version]

    # entity df: keys + timestamp + the label we carried into the parquet
    base = pd.read_parquet(config.FEAST_SOURCE_PARQUET)
    entity_df = base[[config.ENTITY_KEY, config.EVENT_TS, config.TARGET]].copy()
    entity_df = entity_df.rename(columns={config.ENTITY_KEY: "athlete_id"})

    store = FeatureStore(repo_path=str(config.FEATURE_REPO_DIR))
    refs = [f"{view}:{f}" for f in feats]
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=refs,
    ).to_df()

    print(f"[load_features] {version}: {training_df.shape[0]:,} rows, "
          f"features={feats}")
    return training_df


if __name__ == "__main__":
    for v in ["v1", "v2"]:
        df = load_training_frame(v)
        print(df.head(3).to_string())
        print(f"target balance: {df[config.TARGET].mean():.3f}\n")


