"""Cached data / model access. All heavy lifting is delegated to backend `src`."""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

from src.config import METADATA_PATH, PIPELINE_PATH, REPORTS_DIR
from src.data import load_raw
from src.features import AGE_BINS, AGE_LABELS

# Columns offered in the explorer (identifiers excluded: no analytical meaning).
ID_COLS = ["RowNumber", "CustomerId", "Surname"]
CONTINUOUS = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender", "AgeGroup", "NumOfProducts", "Tenure",
               "HasCrCard", "IsActiveMember", "BalanceZero"]
NUMERIC_FOR_CORR = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
LABELS = {0: "Retained", 1: "Churned"}


@st.cache_data(show_spinner="Loading customer data…")
def load_customers() -> pd.DataFrame:
    """Raw dataset plus a few display columns (AgeGroup, BalanceZero, Status)."""
    df = load_raw()
    df["AgeGroup"] = pd.Categorical(pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS),
                                    categories=AGE_LABELS, ordered=True)
    df["BalanceZero"] = (df["Balance"] == 0).astype(int)
    df["Status"] = df["Exited"].map(LABELS)
    return df


@st.cache_resource(show_spinner="Loading churn model…")
def load_model():
    """Fitted pipeline + metadata, or (None, None) if the model has not been trained."""
    if not PIPELINE_PATH.exists():
        return None, None
    from src.predict import load_model as _load

    return _load()


@st.cache_data
def load_metadata() -> dict | None:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else None


@st.cache_data
def load_key_findings() -> str:
    """The 'Key findings' section of reports/eda_summary.md."""
    path = REPORTS_DIR / "eda_summary.md"
    if not path.exists():
        return "_`reports/eda_summary.md` not found — run the EDA notebook first._"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## Key findings\s*(.*?)(?=\n## |\Z)", text, flags=re.S)
    return match.group(1).strip() if match else text
