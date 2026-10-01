"""Batch scoring helpers, kept free of Streamlit calls so they are easy to test."""

from __future__ import annotations

import io

import pandas as pd

from src.config import FEATURE_COLS, TARGET
from src.predict import MissingColumnsError, predict

NEW_COLS = ["churn_probability", "churn_prediction", "risk_band"]
UPLOAD_TYPES = ["csv", "xlsx"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class BatchInputError(ValueError):
    """User-facing problem with an uploaded file."""


def _check_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise BatchInputError("The file has no data rows.")
    return df


def read_csv(data: bytes | io.BytesIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else data)
    except Exception as exc:  # parser errors, encoding, empty file
        raise BatchInputError(
            f"Could not read the file as CSV ({exc.__class__.__name__})."
        ) from exc
    return _check_rows(df)


def read_excel(data: bytes | io.BytesIO) -> pd.DataFrame:
    """First worksheet of an .xlsx workbook; header row = column names."""
    try:
        df = pd.read_excel(io.BytesIO(data) if isinstance(data, bytes) else data, sheet_name=0)
    except Exception as exc:  # not a valid workbook, password-protected, ...
        raise BatchInputError(
            f"Could not read the file as an Excel workbook ({exc.__class__.__name__})."
        ) from exc
    return _check_rows(df.dropna(how="all"))  # Excel often carries fully blank rows


def read_upload(data: bytes, filename: str) -> pd.DataFrame:
    """Dispatch on the file extension: .csv or .xlsx."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "csv":
        return read_csv(data)
    if ext == "xlsx":
        return read_excel(data)
    raise BatchInputError(f"Unsupported file type '.{ext}'. Upload a CSV or Excel (.xlsx) file.")


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Original columns + churn_probability, churn_prediction, risk_band, sorted by risk."""
    try:
        scored = df.join(predict(df), how="left")
    except MissingColumnsError as exc:
        missing = str(exc).split(": ", 1)[-1]
        raise BatchInputError(
            f"Missing required columns: **{missing}**. The file needs these columns: "
            f"{', '.join(FEATURE_COLS)} (`{TARGET}` and ID columns are optional)."
        ) from exc
    except (ValueError, TypeError) as exc:
        raise BatchInputError(
            "Some values could not be scored — check that numeric columns contain only "
            "numbers and Geography / Gender use the original labels "
            "(France / Germany / Spain, Female / Male)."
        ) from exc
    return scored.sort_values("churn_probability", ascending=False)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Scored customers") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()
