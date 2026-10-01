"""EDA Explorer: per-feature distributions, correlations and a free scatter plot."""
import streamlit as st

from ui.charts import (
    box_by_status,
    churn_rate_bar,
    correlation_heatmap,
    count_by_status,
    histogram_by_status,
    scatter_by_status,
)
from ui.data import CATEGORICAL, CONTINUOUS, NUMERIC_FOR_CORR, load_customers, load_key_findings
from ui.filters import require_rows, sidebar_filters

df = sidebar_filters(load_customers())

st.title("EDA Explorer")
st.caption("Explore any feature against churn. Respects the sidebar filters.")

with st.expander("Key findings from the EDA (reports/eda_summary.md)", icon=":material/lightbulb:"):
    st.markdown(load_key_findings())

if require_rows(df):
    tab_dist, tab_corr, tab_scatter = st.tabs(
        [":material/bar_chart: Feature vs churn", ":material/grid_on: Correlation",
         ":material/scatter_plot: Scatter"])

    with tab_dist:
        feature = st.selectbox("Feature", CONTINUOUS + CATEGORICAL, key="eda_feature",
                               help="Continuous features show distributions; categorical and "
                                    "discrete features show churn rate per level.")
        if feature in CONTINUOUS:
            normalise = st.toggle("Normalise each group to 100%", value=True,
                                  help="Compare shapes even though churners are only ~20%.")
            left, right = st.columns([2, 1])
            left.plotly_chart(histogram_by_status(df, feature, normalise),
                              width="stretch")
            right.plotly_chart(box_by_status(df, feature), width="stretch")
            stats = (df.groupby("Status")[feature]
                     .agg(["count", "mean", "median", "std", "min", "max"])
                     .reindex(["Retained", "Churned"]).dropna(how="all"))
            st.dataframe(stats.style.format("{:,.1f}"), width="stretch")
        else:
            rate = df["Exited"].mean() * 100
            left, right = st.columns(2)
            left.plotly_chart(churn_rate_bar(df, feature, rate), width="stretch")
            right.plotly_chart(count_by_status(df, feature), width="stretch")

    with tab_corr:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                          format_func=str.capitalize, key="eda_corr_method")
        cols = [c for c in NUMERIC_FOR_CORR if df[c].nunique() > 1]
        dropped = sorted(set(NUMERIC_FOR_CORR) - set(cols))
        if len(cols) < 2:
            st.info("Not enough varying numeric columns under the current filters.")
        else:
            st.plotly_chart(correlation_heatmap(df, cols, method), width="stretch")
            if dropped:
                st.caption(f"Constant under current filters, so left out: {', '.join(dropped)}.")
            st.caption("Features are nearly uncorrelated; the strongest link to churn is Age. "
                       "NumOfProducts' U-shaped effect is invisible to a linear correlation.")

    with tab_scatter:
        numeric = CONTINUOUS + ["Tenure", "NumOfProducts"]
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", numeric, index=numeric.index("Age"), key="eda_x")
        y = c2.selectbox("Y axis", numeric, index=numeric.index("Balance"), key="eda_y")
        st.plotly_chart(scatter_by_status(df, x, y), width="stretch")
