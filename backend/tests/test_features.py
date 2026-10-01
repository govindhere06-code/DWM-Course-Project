import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import DROP_COLS, RANDOM_STATE
from src.data import load_raw, prepare, split
from src.features import (
    ENGINEERED_COLS,
    FeatureEngineer,
    build_pipeline,
    build_preprocessor,
    get_feature_names,
)


@pytest.fixture
def three_rows():
    """Hand-built rows; row 3 has EstimatedSalary = 0 and Tenure = 0 to test guards."""
    return pd.DataFrame(
        {
            "CreditScore": [600, 750, 500],
            "Geography": ["France", "Germany", "Spain"],
            "Gender": ["Female", "Male", "Female"],
            "Age": [40, 60, 25],
            "Tenure": [3, 10, 0],
            "Balance": [0.0, 100_000.0, 50_000.0],
            "NumOfProducts": [2, 1, 3],
            "HasCrCard": [1, 0, 1],
            "IsActiveMember": [1, 0, 0],
            "EstimatedSalary": [50_000.0, 200_000.0, 0.0],
        }
    )


@pytest.fixture(scope="module")
def splits():
    X, y = prepare(load_raw())
    return split(X, y, save=False)


@pytest.fixture
def engineered(three_rows):
    return FeatureEngineer().fit_transform(three_rows)


# ------------------------------------------------- individual features
def test_balance_zero(engineered):
    assert engineered["BalanceZero"].tolist() == [1, 0, 0]


def test_balance_salary_ratio(engineered):
    # row 3: salary 0 -> guarded to 0 instead of inf
    assert engineered["BalanceSalaryRatio"].tolist() == pytest.approx([0.0, 0.5, 0.0])


def test_tenure_by_age(engineered):
    assert engineered["TenureByAge"].tolist() == pytest.approx([3 / 40, 10 / 60, 0.0])


def test_credit_score_given_age(engineered):
    assert engineered["CreditScoreGivenAge"].tolist() == pytest.approx([15.0, 12.5, 20.0])


def test_products_per_tenure(engineered):
    assert engineered["ProductsPerTenure"].tolist() == pytest.approx([0.5, 1 / 11, 3.0])


def test_age_group(engineered):
    assert engineered["AgeGroup"].tolist() == ["40-49", "60+", "18-29"]


def test_age_group_boundaries(three_rows):
    df = pd.concat([three_rows.iloc[[0]]] * 8, ignore_index=True)
    df["Age"] = [18, 29, 30, 39, 49, 50, 59, 60]
    out = FeatureEngineer().fit_transform(df)
    assert out["AgeGroup"].tolist() == [
        "18-29",
        "18-29",
        "30-39",
        "30-39",
        "40-49",
        "50-59",
        "50-59",
        "60+",
    ]


def test_is_senior(engineered):
    assert engineered["IsSenior"].tolist() == [0, 1, 0]


# ------------------------------------------------- transformer contract
def test_adds_exactly_engineered_columns(three_rows, engineered):
    added = [c for c in engineered.columns if c not in three_rows.columns]
    assert added == ENGINEERED_COLS
    assert len(engineered) == len(three_rows)


def test_drops_id_columns_if_present(three_rows):
    raw_like = three_rows.assign(
        RowNumber=[1, 2, 3], CustomerId=[11, 12, 13], Surname=["A", "B", "C"], Exited=[0, 1, 0]
    )
    out = FeatureEngineer().fit_transform(raw_like)
    assert not set(DROP_COLS + ["Exited"]) & set(out.columns)


def test_is_stateless(three_rows, splits):
    X_train = splits[0]
    fitted_on_train = FeatureEngineer().fit(X_train).transform(three_rows)
    fitted_on_rows = FeatureEngineer().fit(three_rows).transform(three_rows)
    pd.testing.assert_frame_equal(fitted_on_train, fitted_on_rows)


def test_does_not_mutate_input(three_rows):
    before = three_rows.copy()
    FeatureEngineer().fit_transform(three_rows)
    pd.testing.assert_frame_equal(three_rows, before)


def test_no_nan_or_inf_on_full_dataset():
    X, _ = prepare(load_raw())
    out = FeatureEngineer().fit_transform(X)
    numeric = out.select_dtypes("number")
    assert not out.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


def test_get_feature_names_out(three_rows):
    fe = FeatureEngineer().fit(three_rows)
    assert list(fe.get_feature_names_out()) == list(fe.transform(three_rows).columns)


# ------------------------------------------------------- preprocessor
@pytest.mark.parametrize("scale, n_cols", [(True, 21), (False, 23)])
def test_preprocessor_train_test_consistent(splits, scale, n_cols):
    X_train, X_test, y_train, _ = splits
    model = (
        LogisticRegression(max_iter=1000)
        if scale
        else DecisionTreeClassifier(random_state=RANDOM_STATE)
    )
    pipe = build_pipeline(model, scale=scale)
    pipe.fit(X_train, y_train)
    Xt_train = pipe[:-1].transform(X_train)
    Xt_test = pipe[:-1].transform(X_test)
    assert Xt_train.shape == (len(X_train), n_cols)
    assert Xt_test.shape == (len(X_test), n_cols)
    assert np.isfinite(Xt_test).all()
    assert len(get_feature_names(pipe)) == n_cols


def test_preprocessor_fitted_on_train_only(splits):
    X_train, X_test, y_train, _ = splits
    pipe = build_pipeline(LogisticRegression(max_iter=1000), scale=True)
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"]
    train_feats = FeatureEngineer().fit_transform(X_train)
    test_feats = FeatureEngineer().fit_transform(X_test)
    np.testing.assert_allclose(scaler.mean_[1], train_feats["Age"].mean())
    assert scaler.mean_[1] != pytest.approx(test_feats["Age"].mean(), abs=1e-9)
    # scaled train features are centred; test features are not exactly
    Xt_train = pipe[:-1].transform(X_train)
    assert abs(Xt_train[:, 1].mean()) < 1e-9


def test_gender_binary_encoding(three_rows):
    pre = build_preprocessor(scale=False)
    out = pre.fit_transform(FeatureEngineer().fit_transform(three_rows))
    names = list(pre.get_feature_names_out())
    assert out[:, names.index("Gender")].tolist() == [0.0, 1.0, 0.0]


def test_onehot_drop_first_only_for_linear():
    linear = dict((n, t) for n, t, _ in build_preprocessor(scale=True).transformers)["onehot"]
    tree = dict((n, t) for n, t, _ in build_preprocessor(scale=False).transformers)["onehot"]
    assert linear.drop == "first" and tree.drop is None
    assert linear.handle_unknown == tree.handle_unknown == "ignore"


def test_unknown_category_does_not_crash(splits, three_rows):
    X_train, _, y_train, _ = splits
    pipe = build_pipeline(LogisticRegression(max_iter=1000), scale=True).fit(X_train, y_train)
    unseen = three_rows.assign(Geography=["Italy", "France", "Spain"])
    with pytest.warns(UserWarning, match="unknown categories"):
        proba = pipe.predict_proba(unseen)  # unseen level -> all-zero one-hot
    assert proba.shape == (3, 2)


def test_sampler_only_runs_during_fit(splits):
    X_train, X_test, y_train, _ = splits
    pipe = build_pipeline(
        LogisticRegression(max_iter=1000), scale=True, sampler=SMOTE(random_state=RANDOM_STATE)
    )
    assert list(pipe.named_steps) == ["features", "preprocess", "sampler", "model"]
    pipe.fit(X_train, y_train)
    # predicting on test must not resample: one prediction per test row
    assert len(pipe.predict(X_test)) == len(X_test)
