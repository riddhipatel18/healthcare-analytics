from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.model_utils import train_and_save_model
from src.config import MODEL_PATH
import os

from src.config import APP_SUBTITLE, APP_TITLE, FEATURES, MODEL_PATH, NUMERIC_FEATURES
from src.data_utils import load_dataset
from src.model_utils import load_model_artifacts, predict_dataframe
from src.visuals import (
    plot_category_share,
    plot_condition_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_histogram,
    plot_missing_values,
    plot_prediction_probabilities,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")


@st.cache_data
def cached_dataset():
    return load_dataset()


@st.cache_resource
def cached_artifacts():
    if not MODEL_PATH.exists():
        from src.model_utils import train_and_save_model
        train_and_save_model()
    return load_model_artifacts()

def show_overview(df: pd.DataFrame, metadata: dict):
    st.subheader("Project Summary")
    st.write(
        "This dashboard predicts patient condition categories from structured medical attributes and provides EDA, performance monitoring, and batch scoring in a GitHub- and Streamlit-ready format."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Features Used", len(FEATURES))
    c3.metric("Best Model", metadata["best_model"].replace("_", " ").title())
    c4.metric("Macro F1", f"{metadata['test_macro_f1']:.3f}")

    left, right = st.columns((1.2, 1))
    with left:
        st.plotly_chart(plot_condition_distribution(df), use_container_width=True)
    with right:
        st.plotly_chart(plot_missing_values(metadata["missing_ratio"]), use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


def show_eda(df: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")
    categorical_cols = ["gender", "smoking_status", "condition"]

    c1, c2 = st.columns(2)
    with c1:
        selected_numeric = st.selectbox("Select numeric feature", NUMERIC_FEATURES)
        st.plotly_chart(plot_histogram(df, selected_numeric), use_container_width=True)
    with c2:
        selected_categorical = st.selectbox("Select categorical feature", categorical_cols)
        st.plotly_chart(plot_category_share(df, selected_categorical), use_container_width=True)

    st.plotly_chart(plot_correlation_heatmap(df, NUMERIC_FEATURES), use_container_width=True)

    summary = df.groupby("condition")[NUMERIC_FEATURES].agg(["mean", "median", "min", "max"]).round(2)
    st.subheader("Condition-wise Summary")
    st.dataframe(summary, use_container_width=True)


def show_model_performance(metadata: dict):
    st.subheader("Model Performance")
    leaderboard = pd.DataFrame(metadata["leaderboard"])
    leaderboard["model"] = leaderboard["model"].str.replace("_", " ").str.title()
    st.dataframe(leaderboard, use_container_width=True)

    st.plotly_chart(
        plot_confusion_matrix(metadata["confusion_matrix"], metadata["target_classes"]),
        use_container_width=True,
    )

    report_df = pd.DataFrame(metadata["classification_report"]).transpose().round(3)
    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True)

    if metadata.get("feature_importance"):
        st.plotly_chart(plot_feature_importance(metadata["feature_importance"]), use_container_width=True)


def show_single_prediction(pipeline):
    st.subheader("Single Patient Prediction")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
        with c2:
            blood_pressure = st.number_input("Blood Pressure", min_value=50.0, max_value=250.0, value=120.0, step=0.1)
            glucose_levels = st.number_input("Glucose Levels", min_value=50.0, max_value=300.0, value=140.0, step=0.1)
        with c3:
            gender = st.selectbox("Gender", ["male", "female"])
            smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])

        submit = st.form_submit_button("Predict Condition")

    if submit:
        input_df = pd.DataFrame([
            {
                "age": age,
                "bmi": bmi,
                "blood_pressure": blood_pressure,
                "glucose_levels": glucose_levels,
                "gender": gender,
                "smoking_status": smoking_status,
            }
        ])
        result = predict_dataframe(pipeline, input_df)
        st.success(
            f"Predicted Condition: {result.loc[0, 'predicted_condition']} ({result.loc[0, 'prediction_confidence']:.2%} confidence)"
        )
        st.plotly_chart(plot_prediction_probabilities(result.iloc[0]), use_container_width=True)

    st.info("Educational use only — not a substitute for professional medical diagnosis.")


def show_batch_prediction(pipeline):
    st.subheader("Batch Prediction")
    st.write("Upload a CSV with columns: age, bmi, blood_pressure, glucose_levels, gender, smoking_status")

    sample_df = pd.DataFrame([
        {"age": 55, "bmi": 31.2, "blood_pressure": 145.0, "glucose_levels": 180.0, "gender": "male", "smoking_status": "Smoker"},
        {"age": 34, "bmi": 23.7, "blood_pressure": 118.0, "glucose_levels": 95.0, "gender": "female", "smoking_status": "Non-Smoker"},
    ])
    st.download_button(
        "Download Sample Input CSV",
        sample_df.to_csv(index=False),
        file_name="sample_batch_input.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV for scoring", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing_columns = [col for col in FEATURES if col not in batch_df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return

        predictions = predict_dataframe(pipeline, batch_df[FEATURES])
        st.dataframe(predictions, use_container_width=True)
        st.plotly_chart(
            predictions["predicted_condition"].value_counts().rename_axis("condition").reset_index(name="count").pipe(
                lambda x: __import__("plotly.express").express.bar(x, x="condition", y="count", color="condition", title="Batch Prediction Summary")
            ),
            use_container_width=True,
        )
        st.download_button(
            "Download Predictions",
            predictions.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


def main():
    df = cached_dataset()
    pipeline, metadata = cached_artifacts()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    overview_tab, eda_tab, model_tab, single_tab, batch_tab = st.tabs([
        "Overview",
        "EDA",
        "Model Performance",
        "Single Prediction",
        "Batch Prediction",
    ])

    with overview_tab:
        show_overview(df, metadata)
    with eda_tab:
        show_eda(df)
    with model_tab:
        show_model_performance(metadata)
    with single_tab:
        show_single_prediction(pipeline)
    with batch_tab:
        show_batch_prediction(pipeline)


if __name__ == "__main__":
    main()
