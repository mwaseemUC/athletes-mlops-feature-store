"""Feast definitions: one entity, two feature views = two feature versions.

v1 (athlete_features_v1): baseline demographics only.
v2 (athlete_features_v2): v1 plus BMI, age band, and training-lifestyle
survey fields. Same underlying source, different feature definitions.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

athlete = Entity(
    name="athlete",
    join_keys=["athlete_id"],
    value_type=ValueType.INT64,
    description="A CrossFit Open athlete",
)

source = FileSource(
    name="athlete_features_source",
    path="data/athlete_features.parquet",
    timestamp_field="event_timestamp",
)

# ---- Feature version 1: baseline demographics ----
athlete_features_v1 = FeatureView(
    name="athlete_features_v1",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32),
        Field(name="gender", dtype=String),
    ],
    source=source,
    online=True,
)

# ---- Feature version 2: engineered + lifestyle ----
athlete_features_v2 = FeatureView(
    name="athlete_features_v2",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32),
        Field(name="gender", dtype=String),
        Field(name="bmi", dtype=Float32),
        Field(name="age_bin", dtype=String),
        Field(name="region", dtype=String),
        Field(name="experience", dtype=String),
        Field(name="schedule", dtype=String),
        Field(name="howlong", dtype=String),
        Field(name="eat", dtype=String),
    ],
    source=source,
    online=True,
)