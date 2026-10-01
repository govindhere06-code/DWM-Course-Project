"""Cached model outputs for the performance / prediction pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR, TARGET
from src.data import load_split
from ui.data import load_model

RESULT_FILES = {
    "Baselines": "baseline_results.csv",
    "Imbalance strategies": "imbalance_comparison.csv",
    "Tuning": "tuning_results.csv",
    "Ensembles": "ensemble_comparison.csv",
}
METRIC_COLS = {"pr_auc_mean": "PR-AUC", "roc_auc_mean": "ROC-AUC", "recall_mean": "Recall",
               "precision_mean": "Precision", "f1_mean": "F1", "accuracy_mean": "Accuracy"}


def model_ready() -> bool:
    pipeline, _ = load_model()
    if pipeline is None:
        st.warning("No trained model found. Run `python -m src.evaluate` in `backend/` "
                   "(or `make evaluate`) and reload.", icon=":material/warning:")
        return False
    return True


@st.cache_data(show_spinner="Scoring the test set…")
def test_predictions() -> pd.DataFrame:
    """Test rows with true label and predicted churn probability."""
    pipeline, _ = load_model()
    _, X_test, _, y_test = load_split()
    out = X_test.copy()
    out[TARGET] = y_test.values
    out["churn_probability"] = pipeline.predict_proba(X_test)[:, 1]
    return out


@st.cache_data
def results_table(name: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / RESULT_FILES[name]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keep = [c for c in ["model", "type", "strategy", "selected", "kept_defaults",
                        "baseline_pr_auc", "tuned_pr_auc"] if c in df.columns]
    view = df[keep].copy()
    for col, label in METRIC_COLS.items():
        if col in df.columns:
            view[label] = df[col]
    if "fit_time_s" in df.columns:
        view["Fit time (s)"] = df["fit_time_s"]
    return view.sort_values("PR-AUC", ascending=False, ignore_index=True)


@st.cache_data
def permutation_table() -> pd.DataFrame | None:
    path = REPORTS_DIR / "permutation_importance.csv"
    return pd.read_csv(path) if path.exists() else None
