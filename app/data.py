"""Data helpers for the Streamlit estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "AI_Developer_Performance_Extended_1000.csv"
FEATURE_COLUMNS: List[str] = [
    "Hours_Coding",
    "Coffee_Intake",
    "Stress_Level",
    "AI_Usage_Hours",
    "Sleep_Hours",
]
TARGET_COLUMNS: List[str] = [
    "Lines_of_Code",
    "Bugs_Fixed",
    "Task_Success_Rate",
    "Commits",
]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_feature_columns() -> List[str]:
    return FEATURE_COLUMNS.copy()


def get_target_columns() -> List[str]:
    return TARGET_COLUMNS.copy()


def get_feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        stats[column] = {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "median": float(df[column].median()),
        }
    return stats
