"""Score raw customer rows with the saved churn pipeline.

    from src.predict import predict
    predict(raw_df)  # -> churn_probability, churn_prediction, risk_band

Input rows use the raw CSV schema; ID columns (RowNumber, CustomerId, Surname)
and Exited are optional and ignored — the pipeline does all feature
engineering itself. Demo: ``python -m src.predict``.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from numpy.typing import ArrayLike

from src.config import (
    FEATURE_COLS,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    METADATA_PATH,
    PIPELINE_PATH,
    RAW_DATA,
)

RISK_BANDS = ["Low", "Medium", "High"]


class MissingColumnsError(ValueError):
    """Raised when input rows lack required raw feature columns."""


@lru_cache(maxsize=1)
def load_model() -> tuple[Pipeline, dict]:
    """Fitted pipeline and metadata (cached after the first call)."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"{PIPELINE_PATH} not found — run `python -m src.evaluate` to train and save it"
        )
    return joblib.load(PIPELINE_PATH), json.loads(METADATA_PATH.read_text())


def risk_band(probability: ArrayLike) -> np.ndarray:
    """Low (< 0.3), Medium (0.3 – 0.6), High (> 0.6)."""
    p = np.asarray(probability, dtype=float)
    return np.select([p < LOW_RISK_MAX, p <= MEDIUM_RISK_MAX], RISK_BANDS[:2], RISK_BANDS[2])


def validate_input(df: pd.DataFrame) -> None:
    """Raise MissingColumnsError listing any required raw columns that are absent."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise MissingColumnsError(f"Missing required columns: {', '.join(missing)}")


def predict(df_raw: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Churn probability, 0/1 prediction (at the tuned threshold) and risk band.

    Returns a DataFrame aligned to ``df_raw.index`` with columns
    ``churn_probability``, ``churn_prediction``, ``risk_band``.
    """
    if isinstance(df_raw, dict):
        df_raw = pd.DataFrame([df_raw])
    validate_input(df_raw)
    pipeline, meta = load_model()
    threshold = meta["threshold"] if threshold is None else threshold
    proba = pipeline.predict_proba(df_raw[FEATURE_COLS])[:, 1]
    return pd.DataFrame(
        {
            "churn_probability": proba.round(4),
            "churn_prediction": (proba >= threshold).astype(int),
            "risk_band": risk_band(proba),
        },
        index=df_raw.index,
    )


if __name__ == "__main__":
    raw = pd.read_csv(RAW_DATA).sample(5, random_state=7)  # raw rows, ID columns included
    _, meta = load_model()
    print(f"Model: {meta['model_name']}, threshold {meta['threshold']}\n")
    print("Input (raw CSV rows with ID columns):")
    print(raw.to_string(index=False))
    scored = raw[["CustomerId", "Surname", "Exited"]].join(predict(raw))
    print("\nPredictions:")
    print(scored.to_string(index=False))
