"""Streamlit UI for the linear regression estimator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .data import FEATURE_COLUMNS, get_feature_stats

PRIMARY_GRADIENT = "radial-gradient(circle at top, #140c1f 0%, #221129 40%, #360f28 100%)"
ACCENT = "#f472b6"
TEXT_LIGHT = "#f8fafc"


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {PRIMARY_GRADIENT};
                color: {TEXT_LIGHT};
                font-family: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif;
            }}
            .block-container {{
                padding-top: 3rem;
                max-width: 1100px;
            }}
            .input-panel {{
                background: rgba(15, 23, 42, 0.85);
                border-radius: 20px;
                padding: 1.5rem;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 30px 50px rgba(0, 0, 0, 0.45);
            }}
            .result-card {{
                background: rgba(255, 255, 255, 0.12);
                border-radius: 16px;
                padding: 1rem 1.5rem;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }}
            .result-card p {{
                margin: 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _gather_inputs(df: pd.DataFrame) -> pd.DataFrame:
    stats = get_feature_stats(df)
    with st.form("prediction_form"):
        st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
        columns = st.columns(2)
        with columns[0]:
            hours = st.slider(
                "Hours coding",
                min_value=float(stats["Hours_Coding"]["min"]),
                max_value=float(stats["Hours_Coding"]["max"]),
                value=float(stats["Hours_Coding"]["median"]),
                step=0.5,
            )
            sleep = st.slider(
                "Sleep hours",
                min_value=float(stats["Sleep_Hours"]["min"]),
                max_value=float(stats["Sleep_Hours"]["max"]),
                value=float(stats["Sleep_Hours"]["median"]),
                step=0.25,
            )
        with columns[1]:
            ai_max = float(hours)
            ai_min = min(float(stats["AI_Usage_Hours"]["min"]), ai_max)
            ai_default = min(float(stats["AI_Usage_Hours"]["median"]), ai_max)
            ai_usage = st.slider(
                "AI usage hours",
                min_value=ai_min,
                max_value=ai_max,
                value=ai_default,
                step=0.25,
                help="The number of hours you will be using AI tools to code. AI usage cannot exceed total coding hours.",
            )
            coffee = st.number_input(
                "Coffee intake (cups)",
                min_value=int(stats["Coffee_Intake"]["min"]),
                max_value=int(1000),
                value=int(stats["Coffee_Intake"]["median"]),
                step=1,
            )
        stress_default = int(max(1, min(100, stats["Stress_Level"]["median"])))
        stress = st.slider("Stress level", min_value=1, max_value=100, value=stress_default)
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict outcomes", use_container_width=True)

    if not submitted:
        return pd.DataFrame()

    data = {
        "Hours_Coding": hours,
        "Coffee_Intake": float(coffee),
        "Stress_Level": float(stress),
        "AI_Usage_Hours": ai_usage,
        "Sleep_Hours": sleep,
    }
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def _render_prediction_cards(predictions: dict[str, float]) -> None:
    labels = {
        "Lines_of_Code": "Lines of code",
        "Bugs_Fixed": "Bugs fixed",
        "Task_Success_Rate": "Task success rate",
        "Commits": "Commits",
    }
    grid = st.columns(4)
    for col, key in zip(grid, labels):
        value = predictions[key]
        suffix = "%" if key == "Task_Success_Rate" else ""
        formatted = f"{value:.1f}{suffix}" if key == "Task_Success_Rate" else f"{value:.0f}"
        col.markdown(
            f"<div class='result-card'><p style='text-transform:uppercase;font-size:0.75rem;color:#cbd5ff'>{labels[key]}</p><p style='font-size:2rem;font-weight:600;color:{ACCENT}'>{formatted}</p></div>",
            unsafe_allow_html=True,
        )


def render_predictor(df: pd.DataFrame, model, predict_fn) -> None:
    st.title("Developer Productivity Estimator")
    st.caption("Enter the context of your work session to predict your performance")

    sample = _gather_inputs(df)
    if sample.empty:
        return

    predictions = predict_fn(model, sample)
    _render_prediction_cards(predictions)
