import pandas as pd
import pytest

from src.config import DROP_COLS, TARGET
from src.data import DataValidationError, load_raw, prepare, split, validate


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
    [
        ("Age", 17),
        ("Age", 101),
        ("CreditScore", 299),
        ("CreditScore", 901),
        ("NumOfProducts", 0),
        ("NumOfProducts", 5),
        ("Tenure", -1),
        ("Tenure", 11),
    ],
)
def test_out_of_range_raises(raw, col, value):
    df = raw.copy()
    df.loc[0, col] = value
    with pytest.raises(DataValidationError, match=f"'{col}' has 1 values outside"):
        validate(df)


# ------------------------------------------------------ prepare / split
def test_prepare_drops_ids_and_target(raw):
    X, y = prepare(raw)
    assert not set(DROP_COLS + [TARGET]) & set(X.columns)
    assert X.shape == (10000, 10)
    assert y.name == TARGET and len(y) == 10000


def test_prepare_keeps_all_rows(raw):
    X, _ = prepare(raw)
    assert len(X) == len(raw)  # outliers are kept


def test_prepare_rejects_duplicates(raw):
    with pytest.raises(DataValidationError, match="duplicate"):
        prepare(pd.concat([raw, raw.iloc[[0]]]))


def test_split_sizes_and_stratification(raw):
    X, y = prepare(raw)
    X_train, X_test, y_train, y_test = split(X, y, save=False)
    assert (len(X_train), len(X_test)) == (8000, 2000)
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.01)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_is_reproducible(raw):
    X, y = prepare(raw)
    a = split(X, y, save=False)[1].index
    b = split(X, y, save=False)[1].index
    assert a.equals(b)
