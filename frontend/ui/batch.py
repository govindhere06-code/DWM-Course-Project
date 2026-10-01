"""Batch scoring helpers, kept free of Streamlit calls so they are easy to test."""
from __future__ import annotations

import io

import pandas as pd

from src.config import FEATURE_COLS, TARGET
from src.predict import MissingColumnsError, predict

NEW_COLS = ["churn_probability", "churn_prediction", "risk_band"]


class BatchInputError(ValueError):
    """User-facing problem with an uploaded file."""


def read_csv(data: bytes | io.BytesIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else data)
    except Exception as exc:  # parser errors, encoding, empty file
        raise BatchInputError(f"Could not read the file as CSV ({exc.__class__.__name__}).") from exc
    if df.empty:
        raise BatchInputError("The file has no data rows.")
    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Original columns + churn_probability, churn_prediction, risk_band, sorted by risk."""
    try:
        scored = df.join(predict(df), how="left")
    except MissingColumnsError as exc:
        missing = str(exc).split(": ", 1)[-1]
        raise BatchInputError(
            f"Missing required columns: **{missing}**. The file needs these columns: "
            f"{', '.join(FEATURE_COLS)} (`{TARGET}` and ID columns are optional).") from exc
    except (ValueError, TypeError) as exc:
        raise BatchInputError(
            "Some values could not be scored — check that numeric columns contain only "
            "numbers and Geography / Gender use the original labels "
            "(France / Germany / Spain, Female / Male).") from exc
    return scored.sort_values("churn_probability", ascending=False)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
