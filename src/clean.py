"""Step 2a: clean raw athletes.csv. Adapted from Assignment 1 cleaning.
Change vs A1: athlete_id is retained (needed as the Feast entity key),
and output is parquet to preserve dtypes.
"""
import numpy as np
import pandas as pd
import config


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(config.RAW_CSV, low_memory=False)
    data.columns = [c.strip().lower() for c in data.columns]
    print(f"[clean] raw: {data.shape[0]:,} rows x {data.shape[1]} cols")

    data = data.dropna(subset=[
        'region', 'age', 'weight', 'height', 'howlong', 'gender', 'eat',
        'background', 'experience', 'schedule', 'deadlift', 'candj',
        'snatch', 'backsq'])
    # A1 dropped athlete_id here; we keep it. Drop only what we don't use.
    data = data.drop(columns=[
        'affiliate', 'team', 'name', 'fran', 'helen', 'grace',
        'filthy50', 'fgonebad', 'run400', 'run5k', 'pullups', 'train'],
        errors='ignore')
    data = data[data['weight'] < 1500]
    data = data[data['gender'] != '--']
    data = data[data['age'] >= 18]
    data = data[(data['height'] < 96) & (data['height'] > 48)]
    data = data[
        ((data['gender'] == 'Male') & (data['deadlift'] <= 1105)) |
        ((data['gender'] == 'Female') & (data['deadlift'] <= 636))]
    data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]
    data = data.replace({'Decline to answer|': np.nan})
    data = data.dropna(subset=[
        'background', 'experience', 'schedule', 'howlong', 'eat'])

    data['athlete_id'] = pd.to_numeric(data['athlete_id'], errors='coerce')
    data = data.dropna(subset=['athlete_id'])
    data['athlete_id'] = data['athlete_id'].astype(int)
    data = data.drop_duplicates(subset=['athlete_id'])

    data.to_parquet(config.CLEAN_PARQUET, index=False)
    print(f"[clean] cleaned: {data.shape[0]:,} rows x {data.shape[1]} cols "
          f"-> {config.CLEAN_PARQUET}")


if __name__ == "__main__":
    main()