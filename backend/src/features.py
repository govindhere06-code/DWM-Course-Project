"""Feature engineering and preprocessing pipeline.

Chain used by every model:

    FeatureEngineer -> preprocessor (ColumnTransformer) -> [sampler] -> model

built with ``imblearn.pipeline.Pipeline`` so a sampler such as SMOTE is only
applied when fitting, i.e. on training folds, never on validation/test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.config import DROP_COLS, TARGET

AGE_BINS = [0, 29, 39, 49, 59, np.inf]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
SENIOR_AGE = 60

ENGINEERED_COLS = [
    "BalanceZero",
    "BalanceSalaryRatio",
    "TenureByAge",
    "CreditScoreGivenAge",
    "ProductsPerTenure",
    "AgeGroup",
    "IsSenior",
]

# Column groups after FeatureEngineer, as consumed by the preprocessor.
SCALED_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "TenureByAge",
    "CreditScoreGivenAge",
    "ProductsPerTenure",
]
# Extremely right-skewed (median 0.75, max ~10,600 because some salaries are tiny):
# log1p before scaling so a handful of rows don't dominate distance/linear models.
HEAVY_TAIL_COLS = ["BalanceSalaryRatio"]
ONEHOT_COLS = ["Geography", "AgeGroup"]
GENDER_COL = ["Gender"]
BINARY_PASSTHROUGH = ["HasCrCard", "IsActiveMember", "BalanceZero", "IsSenior"]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, returning 0 where den is 0 so the output never holds inf/NaN."""
    num = num.astype(float)
    den = den.astype(float)
    out = np.divide(num, den, out=np.zeros(len(num)), where=den != 0)
    return pd.Series(out, index=num.index)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer that adds domain features to raw customer rows.

    Nothing is learned in ``fit``; every value is computed from the row itself,
    so it cannot leak information between train and test. Identifier columns
    (and the target, if present) are dropped so raw CSV rows can be scored.

    Added features:
    - BalanceZero: 1 if Balance == 0 — 36% of customers; a distinct, lower-churn group.
    - BalanceSalaryRatio: Balance / EstimatedSalary — savings relative to income.
    - TenureByAge: Tenure / Age — share of adult life spent with the bank (loyalty).
    - CreditScoreGivenAge: CreditScore / Age — creditworthiness relative to life stage.
    - ProductsPerTenure: NumOfProducts / (Tenure + 1) — products bought fast may signal mis-selling.
    - AgeGroup: 18-29 / 30-39 / 40-49 / 50-59 / 60+ — churn is non-monotonic in age (peak 50-59).
    - IsSenior: 1 if Age >= 60 — churn drops again for the oldest customers.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        """Record the input columns; nothing is learned from the data."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop ID/target columns and append the engineered features."""
        X = X.drop(columns=DROP_COLS + [TARGET], errors="ignore").copy()
        X["BalanceZero"] = (X["Balance"] == 0).astype(int)
        X["BalanceSalaryRatio"] = _safe_divide(X["Balance"], X["EstimatedSalary"])
        X["TenureByAge"] = _safe_divide(X["Tenure"], X["Age"])
        X["CreditScoreGivenAge"] = _safe_divide(X["CreditScore"], X["Age"])
        X["ProductsPerTenure"] = _safe_divide(X["NumOfProducts"], X["Tenure"] + 1)
        X["AgeGroup"] = pd.cut(X["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
        X["IsSenior"] = (X["Age"] >= SENIOR_AGE).astype(int)
        return X

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        """Output column names: kept inputs followed by the engineered features."""
        base = self.feature_names_in_ if input_features is None else input_features
        kept = [c for c in base if c not in DROP_COLS + [TARGET]]
        return np.asarray(kept + ENGINEERED_COLS, dtype=object)


def build_preprocessor(scale: bool = True, drop_first: bool | None = None) -> ColumnTransformer:
    """ColumnTransformer for the FeatureEngineer output.

    scale=True  -> StandardScaler on continuous/ratio features (LogReg, KNN, SVM);
                   heavy-tailed ratios get log1p first (stateless, so no leakage).
    scale=False -> numeric features passed through untouched (tree models).
    drop_first defaults to ``scale``: drop one one-hot level for linear models
    (avoids perfect collinearity), keep all levels for trees.
    """
    drop_first = scale if drop_first is None else drop_first
    numeric = StandardScaler() if scale else "passthrough"
    heavy_tail = (
        make_pipeline(
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()
        )
        if scale
        else "passthrough"
    )
    onehot = OneHotEncoder(
        handle_unknown="ignore", drop="first" if drop_first else None, sparse_output=False
    )
    gender = OrdinalEncoder(categories=[["Female", "Male"]])  # Female=0, Male=1
    return ColumnTransformer(
        transformers=[
            ("num", numeric, SCALED_COLS),
            ("heavy_tail", heavy_tail, HEAVY_TAIL_COLS),
            ("onehot", onehot, ONEHOT_COLS),
            ("gender", gender, GENDER_COL),
            ("binary", "passthrough", BINARY_PASSTHROUGH),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    model: BaseEstimator, scale: bool = True, sampler: BaseEstimator | None = None
) -> Pipeline:
    """FeatureEngineer -> preprocessor -> [sampler] -> model as an imblearn Pipeline."""
    steps = [("features", FeatureEngineer()), ("preprocess", build_preprocessor(scale))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Final column names fed to the model (for importance plots and SHAP)."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from src.config import RANDOM_STATE
    from src.data import load_raw, prepare, split

    X, y = prepare(load_raw())
    X_train, X_test, y_train, y_test = split(X, y, save=False)
    demos = {
        True: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        False: RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    for scale, model in demos.items():
        pipe = build_pipeline(model, scale=scale)
        pipe.fit(X_train, y_train)
        Xt = pipe[:-1].transform(X_test)  # everything except the model
        names = get_feature_names(pipe)
        print(
            f"\n{type(model).__name__} (scale={scale}): X_test {X_test.shape} -> {Xt.shape}, "
            f"NaN={np.isnan(Xt).sum()}, inf={np.isinf(Xt).sum()}"
        )
        print(names)
