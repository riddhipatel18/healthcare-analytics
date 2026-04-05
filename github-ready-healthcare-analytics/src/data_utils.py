from __future__ import annotations

import pandas as pd

from .config import DATA_PATH, FEATURES, TARGET_COL


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].copy()


def get_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL].copy()


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_ratio": (df[FEATURES].isna().mean() * 100).round(2).to_dict(),
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
    }
