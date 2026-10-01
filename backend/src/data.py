"""Load, validate, clean and split the raw churn dataset."""

from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes
from sklearn.model_selection import train_test_split

from src.config import (
    DROP_COLS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA,
    TARGET,
    TEST_DATA,
    TEST_SIZE,
    TRAIN_DATA,
)


class DataValidationError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


# Expected column -> dtype kind ("int", "numeric" or "string").
# Balance/EstimatedSalary are "numeric" so an all-integer file still validates.
EXPECTED_SCHEMA = {
    "RowNumber": "int",
    "CustomerId": "int",
    "Surname": "string",
    "CreditScore": "int",
    "Geography": "string",
    "Gender": "string",
    "Age": "int",
    "Tenure": "int",
    "Balance": "numeric",
    "NumOfProducts": "int",
    "HasCrCard": "int",
    "IsActiveMember": "int",
    "EstimatedSalary": "numeric",
    "Exited": "int",
}

# Inclusive valid ranges for numeric columns.
VALUE_RANGES = {
    "Age": (18, 100),
    "CreditScore": (300, 900),
    "NumOfProducts": (1, 4),
    "Tenure": (0, 10),
}

_KIND_CHECKS = {
    "int": ptypes.is_integer_dtype,
    "numeric": ptypes.is_numeric_dtype,
    "string": lambda s: ptypes.is_string_dtype(s) or ptypes.is_object_dtype(s),
}


def validate(df: pd.DataFrame) -> dict:
    """Check schema, dtypes, nulls, target values and ranges.

    Raises DataValidationError on the first failed check; otherwise returns a
    summary dict (shape, null counts, duplicates, class balance).
    """
    missing = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    if extra:
        raise DataValidationError(f"Unexpected columns: {extra}")

    bad_dtypes = {
        col: str(df[col].dtype)
        for col, kind in EXPECTED_SCHEMA.items()
        if not _KIND_CHECKS[kind](df[col])
    }
    if bad_dtypes:
        expected = {c: EXPECTED_SCHEMA[c] for c in bad_dtypes}
        raise DataValidationError(f"Unexpected dtypes {bad_dtypes}; expected kinds {expected}")

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise DataValidationError(f"Null values found: {nulls.to_dict()}")

    bad_target = set(df[TARGET].unique()) - {0, 1}
    if bad_target:
        raise DataValidationError(f"'{TARGET}' must only contain 0/1, found {sorted(bad_target)}")

    for col, (lo, hi) in VALUE_RANGES.items():
        out_of_range = int((~df[col].between(lo, hi)).sum())
        if out_of_range:
            raise DataValidationError(f"'{col}' has {out_of_range} values outside [{lo}, {hi}]")

    counts = df[TARGET].value_counts().sort_index()
    return {
        "shape": df.shape,
        "null_counts": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "churn_rate": float(df[TARGET].mean()),
    }


def load_raw(path: Path | str = RAW_DATA) -> pd.DataFrame:
    """Read the raw CSV. Never modifies the file on disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop identifier columns and separate features from the target.

    Rows are never removed: outliers are plausible real customers (see
    reports/preprocessing_decisions.md).
    """
    if df.duplicated().any():
        raise DataValidationError(f"{int(df.duplicated().sum())} duplicate rows found")
    X = df.drop(columns=DROP_COLS + [TARGET], errors="ignore")
    y = df[TARGET].astype(int)
    return X, y


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split. Optionally writes train.csv / test.csv.

    The split happens before any fitting, so no statistic from the test set
    can leak into preprocessing or modelling.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{TARGET: y_train}).to_csv(TRAIN_DATA, index=False)
        X_test.assign(**{TARGET: y_test}).to_csv(TEST_DATA, index=False)
    return X_train, X_test, y_train, y_test


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Read the saved train/test CSVs (run ``python -m src.data`` first)."""
    train, test = pd.read_csv(TRAIN_DATA), pd.read_csv(TEST_DATA)
    return (train.drop(columns=TARGET), test.drop(columns=TARGET), train[TARGET], test[TARGET])


if __name__ == "__main__":
    raw = load_raw()
    summary = validate(raw)
    X, y = prepare(raw)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Validated {summary['shape']}; features {X.shape[1]}: {list(X.columns)}")
    print(f"Train {X_train.shape}, churn rate {y_train.mean():.4f} -> {TRAIN_DATA}")
    print(f"Test  {X_test.shape}, churn rate {y_test.mean():.4f} -> {TEST_DATA}")
