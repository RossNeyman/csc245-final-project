"""Linear regression utilities."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS, TARGET_COLUMNS

@st.cache_resource(show_spinner=False)
def train_regression_model(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def predict_targets(model: Pipeline, sample: pd.DataFrame) -> Dict[str, float]:
    preds = model.predict(sample)[0]
    return {target: float(value) for target, value in zip(TARGET_COLUMNS, preds)}
