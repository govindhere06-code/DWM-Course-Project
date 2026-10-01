"""Headless tests of the dashboard pages with streamlit.testing.AppTest."""
import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[1]
BACKEND = FRONTEND.parent / "backend"
for p in (BACKEND, FRONTEND):
    sys.path.insert(0, str(p))

from ui.data import CATEGORICAL, CONTINUOUS  # noqa: E402

PAGES = FRONTEND / "views"
TIMEOUT = 60


def run_page(name: str) -> AppTest:
    at = AppTest.from_file(str(PAGES / name), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def metrics(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def charts(at: AppTest) -> list[str]:
    """Serialized Plotly figures, to detect that charts changed."""
    return [c.proto.spec for c in at.get("plotly_chart")]


# --------------------------------------------------------------- overview
def test_overview_unfiltered_kpis():
    at = run_page("overview.py")
    m = metrics(at)
    assert m["Total customers"] == "10,000"
    assert m["Churned customers"] == "2,037"
    assert m["Churn rate"] == "20.4%"
    assert len(at.get("plotly_chart")) == 4


@pytest.mark.parametrize("widget, key, value", [
    ("multiselect", "f_geo", ["Germany"]),
    ("multiselect", "f_gender", ["Female"]),
    ("slider", "f_age", (40, 59)),
    ("radio", "f_active", "Inactive only"),
])
def test_each_filter_updates_every_kpi_and_chart(widget, key, value):
    at = run_page("overview.py")
    before_m, before_c = metrics(at), charts(at)
    getattr(at, widget)(key=key).set_value(value).run()
    assert not at.exception
    after_m, after_c = metrics(at), charts(at)
    changed = [k for k in before_m if before_m[k] != after_m[k]]
    # Inactive filter makes "Inactive members" 100% — still a change; every KPI must move.
    assert set(changed) == set(before_m), f"unchanged KPIs: {set(before_m) - set(changed)}"
    # Gender / geography filters collapse that chart to one bar but must still re-render.
    assert all(b != a for b, a in zip(before_c, after_c))


def test_filter_matches_german_numbers():
    at = run_page("overview.py")
    at.multiselect(key="f_geo").set_value(["Germany"]).run()
    m = metrics(at)
    assert m["Total customers"] == "2,509"
    assert m["Churn rate"] == "32.4%"


def test_empty_filter_shows_info_not_error():
    at = run_page("overview.py")
    at.multiselect(key="f_geo").set_value([]).run()
    assert not at.exception
    assert any("No customers match" in i.value for i in at.info)
    assert len(at.get("plotly_chart")) == 0


# ----------------------------------------------------------- eda explorer
@pytest.mark.parametrize("feature", CONTINUOUS + CATEGORICAL)
def test_eda_explorer_every_feature(feature):
    at = run_page("eda_explorer.py")
    at.selectbox(key="eda_feature").set_value(feature).run()
    assert not at.exception, feature
    assert len(at.get("plotly_chart")) >= 3  # 2 in the feature tab + corr + scatter


def test_eda_explorer_correlation_and_scatter_options():
    at = run_page("eda_explorer.py")
    at.radio(key="eda_corr_method").set_value("spearman").run()
    for x in at.selectbox(key="eda_x").options:
        at.selectbox(key="eda_x").set_value(x).run()
        assert not at.exception, x
    at.selectbox(key="eda_y").set_value("CreditScore").run()
    assert not at.exception


def test_eda_explorer_narrow_filters():
    at = run_page("eda_explorer.py")
    at.radio(key="f_active").set_value("Active only").run()  # IsActiveMember becomes constant
    assert not at.exception
    assert any("Constant under current filters" in c.value for c in at.caption)
    at.multiselect(key="f_gender").set_value([]).run()
    assert any("No customers match" in i.value for i in at.info)


def test_eda_findings_expander():
    at = run_page("eda_explorer.py")
    text = " ".join(md.value for md in at.markdown)
    assert "German customers churn twice as often" in text


# ------------------------------------------------------- model performance
def _cm_values(at):
    fig = [c for c in at.get("plotly_chart") if "Confusion matrix" in c.proto.spec][0]
    return json.loads(fig.proto.spec)["data"][0]["z"]


def test_model_performance_renders():
    at = run_page("model_performance.py")
    assert len(at.dataframe) == 4  # one comparison table per tab
    assert len(at.get("plotly_chart")) >= 4  # matrix, ROC, PR, importance
    labels = {m.label for m in at.metric}
    assert {"Precision", "Recall", "F1", "Accuracy", "Customers flagged"} <= labels


def test_threshold_slider_updates_matrix_and_metrics():
    at = run_page("model_performance.py")
    before_cm, before_m = _cm_values(at), metrics(at)
    at.slider(key="perf_threshold").set_value(0.7).run()
    assert not at.exception
    after_cm, after_m = _cm_values(at), metrics(at)
    assert before_cm != after_cm
    assert before_m["Recall"] != after_m["Recall"]
    assert after_m["Precision"] > before_m["Precision"]  # stricter threshold -> more precise
    assert sum(map(sum, after_cm)) == 2000


# ----------------------------------------------------------- single predict
def _probability(at):
    gauge = [c for c in at.get("plotly_chart") if '"indicator"' in c.proto.spec][0]
    return json.loads(gauge.proto.spec)["data"][0]["value"] / 100


def test_predict_page_presets_order_risk():
    at = run_page("predict_customer.py")
    at.button[1].click().run()  # High-risk example
    assert not at.exception
    high = _probability(at)
    high_md = " ".join(m.value for m in at.markdown)
    at.button[2].click().run()  # Low-risk example
    low = _probability(at)
    assert high > 0.6 > 0.3 > low
    assert "High risk" in high_md and "increases churn risk" in high_md
    assert len(at.get("plotly_chart")) == 2  # gauge + waterfall


def test_predict_page_reacts_to_inputs():
    at = run_page("predict_customer.py")
    base = _probability(at)
    at.slider(key="in_NumOfProducts").set_value(4).run()
    assert _probability(at) > base


# ---------------------------------------------------------- batch predict
def test_batch_page_sample_end_to_end():
    at = run_page("batch_prediction.py")
    assert any("Upload a CSV" in i.value for i in at.info)
    at.button[0].click().run()  # "Use sample: test.csv"
    assert not at.exception
    m = metrics(at)
    assert m["Customers scored"] == "2,000"
    assert float(m["ROC-AUC"]) > 0.85  # Exited present -> evaluation section
    assert len(at.get("download_button")) == 1


def test_batch_helpers_scored_columns():
    from src.config import TEST_DATA
    from ui.batch import NEW_COLS, read_csv, score, to_csv_bytes

    scored = score(read_csv(TEST_DATA.read_bytes()))
    assert all(c in scored.columns for c in NEW_COLS)
    assert scored["churn_probability"].is_monotonic_decreasing
    roundtrip = read_csv(to_csv_bytes(scored))
    assert list(roundtrip.columns[-3:]) == NEW_COLS and len(roundtrip) == 2000


def test_batch_helpers_friendly_errors():
    from src.config import TEST_DATA
    from ui.batch import BatchInputError, read_csv, score

    df = read_csv(TEST_DATA.read_bytes()).drop(columns=["Age", "Geography"])
    with pytest.raises(BatchInputError, match=r"Missing required columns: \*\*Geography, Age"):
        score(df)
    with pytest.raises(BatchInputError, match="no data rows"):
        read_csv(b"CreditScore,Age\n")
    bad = read_csv(TEST_DATA.read_bytes()).head(3).assign(Age=["old", "x", "y"])
    with pytest.raises(BatchInputError, match="could not be scored"):
        score(bad)
