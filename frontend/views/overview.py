"""Overview: KPI cards and churn-rate breakdowns, all driven by the sidebar filters."""

import streamlit as st

from ui.charts import churn_rate_bar
from ui.data import load_customers
from ui.filters import require_rows, sidebar_filters

full = load_customers()
df = sidebar_filters(full)

st.title("Overview")
st.caption("Who is leaving the bank? Every number and chart reacts to the sidebar filters.")

if require_rows(df):
    overall_rate = full["Exited"].mean() * 100
    rate = df["Exited"].mean() * 100
    churned = df[df["Exited"] == 1]
    retained = df[df["Exited"] == 0]

    def money(series):
        return f"{series.mean():,.0f}" if len(series) else "—"

    row1 = st.columns(3)
    row1[0].metric("Total customers", f"{len(df):,}", help="Customers matching the current filters")
    row1[1].metric("Churned customers", f"{len(churned):,}")
    filtered = len(df) < len(full)
    row1[2].metric(
        "Churn rate",
        f"{rate:.1f}%",
        delta=f"{rate - overall_rate:+.1f} pts vs all customers" if filtered else None,
        delta_color="inverse",
    )
    row2 = st.columns(3)
    row2[0].metric(
        "Avg balance — churned", money(churned["Balance"]), help="Dataset currency is not specified"
    )
    row2[1].metric("Avg balance — retained", money(retained["Balance"]))
    row2[2].metric("Inactive members", f"{(df['IsActiveMember'] == 0).mean() * 100:.1f}%")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            churn_rate_bar(df, "Geography", rate, "Churn rate by geography"), width="stretch"
        )
        st.plotly_chart(
            churn_rate_bar(df, "NumOfProducts", rate, "Churn rate by number of products"),
            width="stretch",
        )
        st.caption("3–4 product groups are small (326 customers in total) — read with care.")
    with right:
        st.plotly_chart(churn_rate_bar(df, "Gender", rate, "Churn rate by gender"), width="stretch")
        st.plotly_chart(
            churn_rate_bar(df, "AgeGroup", rate, "Churn rate by age group"), width="stretch"
        )
    st.caption("Dashed line = churn rate of the filtered customers. Hover a bar for counts.")
