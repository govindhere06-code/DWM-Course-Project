import numpy as np
import pandas as pd
import pytest

from src.config import PIPELINE_PATH
from src.explain import describe_value, raw_group, top_reasons


@pytest.mark.parametrize(
    "feature, group",
    [
        ("AgeGroup_50-59", "Age"),
        ("IsSenior", "Age"),
        ("Age", "Age"),
        ("Geography_Germany", "Geography"),
        ("BalanceZero", "Balance"),
        ("BalanceSalaryRatio", "Balance / salary ratio"),
        ("Gender", "Gender"),
    ],
)
def test_raw_group(feature, group):
    assert raw_group(feature) == group


def test_top_reasons_text():
    row = pd.Series(
        {"Age": 52, "IsActiveMember": 0, "NumOfProducts": 2, "Geography": "Germany", "Balance": 0.0}
    )
    contrib = pd.Series(
        {"Age": 0.9, "NumOfProducts": -0.8, "IsActiveMember": 0.3, "Geography": 0.1}
    )
    assert top_reasons(contrib, row) == [
        "Age 52 increases churn risk",
        "2 products decreases churn risk",
        "Inactive member increases churn risk",
    ]
    assert describe_value("Balance", row) == "Zero balance"


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="trained pipeline missing")
def test_explain_customer_is_additive():
    from src.explain import explain_customer

    customer = {
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 50,
        "Tenure": 5,
        "Balance": 100_000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 100_000.0,
    }
    out = explain_customer(customer)
    p = out["probability"]
    logit = np.log(p / (1 - p))
    assert out["base_value"] + out["contributions"].sum() == pytest.approx(logit, abs=1e-3)
    assert len(out["reasons"]) == 3
    assert not any(c.startswith(("AgeGroup_", "Geography_")) for c in out["contributions"].index)
