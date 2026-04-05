from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


def plot_condition_distribution(df: pd.DataFrame):
    chart_df = df["condition"].value_counts().reset_index()
    chart_df.columns = ["condition", "count"]
    fig = px.bar(chart_df, x="condition", y="count", color="condition", text="count", title="Condition Distribution")
    fig.update_layout(showlegend=False)
    return fig


def plot_missing_values(missing_ratio: dict):
    missing_df = pd.DataFrame({"feature": list(missing_ratio.keys()), "missing_percent": list(missing_ratio.values())})
    return px.bar(
        missing_df,
        x="missing_percent",
        y="feature",
        orientation="h",
        color="missing_percent",
        color_continuous_scale="Reds",
        title="Missing Value Percentage",
    )


def plot_histogram(df: pd.DataFrame, feature: str):
    return px.histogram(
        df,
        x=feature,
        color="condition",
        marginal="box",
        nbins=40,
        barmode="overlay",
        title=f"Distribution of {feature}",
    )


def plot_category_share(df: pd.DataFrame, column: str):
    chart_df = df[column].value_counts(dropna=False).reset_index()
    chart_df.columns = [column, "count"]
    return px.pie(chart_df, names=column, values="count", title=f"{column} Share")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]):
    corr = df[numeric_cols].corr(numeric_only=True).round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.astype(str).values,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(title="Correlation Heatmap")
    return fig


def plot_confusion_matrix(confusion_matrix_data: list[list[int]], labels: list[str]):
    cm = np.array(confusion_matrix_data)
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        annotation_text=cm.astype(str),
        colorscale="Viridis",
        showscale=True,
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def plot_feature_importance(feature_importance: list[dict]):
    fi = pd.DataFrame(feature_importance)
    return px.bar(
        fi.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Teal",
        title="Top Feature Importances",
    )


def plot_prediction_probabilities(output_row: pd.Series):
    prob_cols = [col for col in output_row.index if col.startswith("prob_")]
    plot_df = pd.DataFrame({
        "Condition": [col.replace("prob_", "").title() for col in prob_cols],
        "Probability": [output_row[col] for col in prob_cols],
    })
    fig = px.bar(plot_df, x="Condition", y="Probability", color="Condition", title="Prediction Probabilities")
    fig.update_yaxes(range=[0, 1])
    return fig
