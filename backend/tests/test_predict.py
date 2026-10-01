import numpy as np
import pandas as pd
import pytest

from src.config import PIPELINE_PATH, RAW_DATA, TEST_DATA
from src.predict import MissingColumnsError, predict, risk_band

needs_model = pytest.mark.skipif(
    not PIPELINE_PATH.exists(),
    reason="trained pipeline missing — run `python -m src.evaluate`",
)


def test_risk_band_edges():
    bands = risk_band([0.0, 0.29, 0.3, 0.45, 0.6, 0.61, 1.0])
    assert bands.tolist() == ["Low", "Low", "Medium", "Medium", "Medium", "High", "High"]


@needs_model
def test_predict_five_raw_test_rows():
    rows = pd.read_csv(TEST_DATA).head(5)
    out = predict(rows)
    assert list(out.columns) == ["churn_probability", "churn_prediction", "risk_band"]
    assert len(out) == 5 and out.index.equals(rows.index)
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["churn_prediction"]) <= {0, 1}
    assert set(out["risk_band"]) <= {"Low", "Medium", "High"}


@needs_model
def test_predict_accepts_id_columns():
    raw = pd.read_csv(RAW_DATA).head(5)  # RowNumber, CustomerId, Surname, Exited present
    with_ids = predict(raw)
    without_ids = predict(raw.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"]))
    pd.testing.assert_frame_equal(with_ids, without_ids)


@needs_model
def test_predict_missing_columns_lists_them():
    rows = pd.read_csv(TEST_DATA).head(2).drop(columns=["Age", "Balance"])
    with pytest.raises(MissingColumnsError, match="Age, Balance"):
        predict(rows)


@needs_model
def test_high_risk_profile_scores_higher_than_low_risk():
    base = {"CreditScore": 650, "Tenure": 5, "Balance": 100_000.0, "HasCrCard": 1,
            "EstimatedSalary": 100_000.0}
    high = predict({**base, "Geography": "Germany", "Gender": "Female", "Age": 50,
                    "IsActiveMember": 0, "NumOfProducts": 3})
    low = predict({**base, "Geography": "France", "Gender": "Male", "Age": 30,
                   "IsActiveMember": 1, "NumOfProducts": 2})
    assert high["churn_probability"].iloc[0] > low["churn_probability"].iloc[0] + 0.3
    assert high["risk_band"].iloc[0] == "High" and low["risk_band"].iloc[0] == "Low"


@needs_model
def test_threshold_override():
    rows = pd.read_csv(TEST_DATA).head(50)
    assert predict(rows, threshold=0.0)["churn_prediction"].eq(1).all()
    assert predict(rows, threshold=1.01)["churn_prediction"].eq(0).all()
