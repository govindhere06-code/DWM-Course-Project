import pandas as pd
import pytest

from src.data import DataValidationError, load_raw, validate


@pytest.fixture(scope="module")
def raw():
    return load_raw()


# ------------------------------------------------------------- load_raw
def test_load_raw_returns_dataframe(raw):
    assert isinstance(raw, pd.DataFrame)
    assert raw.shape == (10000, 14)


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


# ------------------------------------------------------------- validate
def test_validate_summary(raw):
    summary = validate(raw)
    assert summary["shape"] == (10000, 14)
    assert summary["duplicate_rows"] == 0
    assert sum(summary["null_counts"].values()) == 0
    assert summary["class_counts"] == {0: 7963, 1: 2037}
    assert summary["churn_rate"] == pytest.approx(0.2037, abs=1e-4)


def test_missing_column_raises(raw):
    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate(raw.drop(columns=["Age"]))


def test_extra_column_raises(raw):
    with pytest.raises(DataValidationError, match="Unexpected columns"):
        validate(raw.assign(Foo=1))


def test_bad_dtype_raises(raw):
    df = raw.copy()
    df["Age"] = df["Age"].astype(str)
    with pytest.raises(DataValidationError, match="Unexpected dtypes"):
        validate(df)


def test_null_raises(raw):
    df = raw.copy()
    df.loc[0, "Balance"] = None
    with pytest.raises(DataValidationError, match="Null values"):
        validate(df)


def test_bad_target_raises(raw):
    df = raw.copy()
    df.loc[0, "Exited"] = 2
    with pytest.raises(DataValidationError, match="must only contain 0/1"):
        validate(df)


@pytest.mark.parametrize(
    "col, value",
    [("Age", 17), ("Age", 101), ("CreditScore", 299), ("CreditScore", 901),
     ("NumOfProducts", 0), ("NumOfProducts", 5), ("Tenure", -1), ("Tenure", 11)],
)
def test_out_of_range_raises(raw, col, value):
    df = raw.copy()
    df.loc[0, col] = value
    with pytest.raises(DataValidationError, match=f"'{col}' has 1 values outside"):
        validate(df)
