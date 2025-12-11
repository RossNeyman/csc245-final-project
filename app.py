import streamlit as st

from app.data import load_data
from app.model import predict_targets, train_regression_model
from app.ui import apply_theme, render_predictor


def main() -> None:
    st.set_page_config(page_title="AI Dev Outcome Estimator", page_icon="🧠", layout="centered")
    apply_theme()

    data = load_data()
    model = train_regression_model(data)
    render_predictor(data, model, predict_targets)


if __name__ == "__main__":
    main()
