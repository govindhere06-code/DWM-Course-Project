"""Headless tests of the dashboard pages with streamlit.testing.AppTest."""
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


# ------------------------------------------------------------ placeholders
@pytest.mark.parametrize("page", ["model_performance.py", "predict_customer.py",
                                  "batch_prediction.py"])
def test_placeholder_pages_render(page):
    at = run_page(page)
    assert any("Coming in T07" in i.value for i in at.info)
