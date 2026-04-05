from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
)
from .data_utils import get_feature_frame, get_target, load_dataset


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def get_model_candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def extract_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = []
        feature_names.extend(NUMERIC_FEATURES)
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
            for cat in cats:
                feature_names.append(f"{col}_{cat}")
        return feature_names


def train_and_save_model() -> dict:
    df = load_dataset()
    X = get_feature_frame(df)
    y = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    candidates = get_model_candidates()
    leaderboard = []
    pipelines = {}

    for name, model in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        leaderboard.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_test, preds, average="weighted")), 4),
        })
        pipelines[name] = pipeline

    leaderboard = sorted(leaderboard, key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    best_model_name = leaderboard[0]["model"]
    best_pipeline = pipelines[best_model_name]

    preds = best_pipeline.predict(X_test)
    probs = best_pipeline.predict_proba(X_test)
    labels = sorted(y.unique().tolist())
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=labels)

    feature_importance = []
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    names = extract_feature_names(preprocessor)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        feature_importance = [
            {"feature": names[i], "importance": round(float(importances[i]), 6)}
            for i in top_idx
        ]

    metadata = {
        "features": FEATURES,
        "target": TARGET_COL,
        "target_classes": labels,
        "leaderboard": leaderboard,
        "best_model": best_model_name,
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "missing_ratio": (df[FEATURES].isna().mean() * 100).round(2).to_dict(),
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
        "sample_prediction_preview": [
            {
                "actual": actual,
                "predicted": pred,
                "confidence": round(float(prob.max()), 4),
            }
            for actual, pred, prob in zip(y_test.iloc[:10], preds[:10], probs[:10])
        ],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "metadata": metadata}, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def load_model_artifacts():
    artifacts = joblib.load(MODEL_PATH)
    return artifacts["pipeline"], artifacts["metadata"]


def predict_dataframe(pipeline, input_df: pd.DataFrame) -> pd.DataFrame:
    preds = pipeline.predict(input_df)
    probs = pipeline.predict_proba(input_df)
    result = input_df.copy()
    result["predicted_condition"] = preds
    result["prediction_confidence"] = probs.max(axis=1)
    for idx, cls in enumerate(pipeline.classes_):
        result[f"prob_{cls.lower()}"] = probs[:, idx]
    return result
