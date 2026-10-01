"""Predict a Customer: live churn probability, risk band and SHAP explanation."""

import time

import pandas as pd
import streamlit as st

from src.config import LOW_RISK_MAX, MEDIUM_RISK_MAX
from src.data import VALUE_RANGES
from src.explain import explain_customer
from src.predict import risk_band
from ui.charts import gauge, waterfall
from ui.data import load_customers, load_metadata, load_model
from ui.model import model_ready

st.title("Predict a Customer")
st.caption("Set a customer's profile — the prediction and explanation update instantly.")

BADGE = {
    "Low": ("green", ":material/check_circle:"),
    "Medium": ("orange", ":material/warning:"),
    "High": ("red", ":material/error:"),
}


@st.cache_data
def defaults() -> dict:
    """Median for numeric features, mode for categorical ones."""
    df = load_customers()
    return {
        "CreditScore": int(df["CreditScore"].median()),
        "Geography": df["Geography"].mode()[0],
        "Gender": df["Gender"].mode()[0],
        "Age": int(df["Age"].median()),
        "Tenure": int(df["Tenure"].median()),
        "Balance": float(round(df["Balance"].median(), -2)),
        "NumOfProducts": int(df["NumOfProducts"].mode()[0]),
        "HasCrCard": int(df["HasCrCard"].mode()[0]),
        "IsActiveMember": int(df["IsActiveMember"].mode()[0]),
        "EstimatedSalary": float(round(df["EstimatedSalary"].median(), -2)),
    }


PRESETS = {
    "High-risk example": {
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 50,
        "IsActiveMember": 0,
        "NumOfProducts": 3,
    },
    "Low-risk example": {
        "Geography": "France",
        "Gender": "Male",
        "Age": 30,
        "IsActiveMember": 1,
        "NumOfProducts": 2,
    },
}


def apply_preset(values: dict) -> None:
    for key, value in {**defaults(), **values}.items():
        st.session_state[f"in_{key}"] = value


if model_ready():
    meta = load_metadata()
    pipeline, _ = load_model()
    data = load_customers()
    if "in_Age" not in st.session_state:
        apply_preset({})

    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    b1.button(
        "Typical customer",
        on_click=apply_preset,
        args=({},),
        width="stretch",
        help="Median / most common value for every feature",
    )
    b2.button(
        "High-risk example",
        on_click=apply_preset,
        args=(PRESETS["High-risk example"],),
        width="stretch",
    )
    b3.button(
        "Low-risk example",
        on_click=apply_preset,
        args=(PRESETS["Low-risk example"],),
        width="stretch",
    )

    form, result = st.columns([1, 1.5], gap="large")
    with form:
        st.markdown("##### Customer profile")
        c1, c2 = st.columns(2)
        c1.selectbox("Geography", sorted(data["Geography"].unique()), key="in_Geography")
        c2.selectbox("Gender", sorted(data["Gender"].unique()), key="in_Gender")
        st.slider("Age", *VALUE_RANGES["Age"], key="in_Age")
        st.slider("Credit score", *VALUE_RANGES["CreditScore"], key="in_CreditScore")
        c1, c2 = st.columns(2)
        c1.slider("Tenure (years)", *VALUE_RANGES["Tenure"], key="in_Tenure")
        c2.slider("Number of products", *VALUE_RANGES["NumOfProducts"], key="in_NumOfProducts")
        st.number_input(
            "Balance", min_value=0.0, max_value=300_000.0, step=1_000.0, key="in_Balance"
        )
        st.number_input(
            "Estimated salary",
            min_value=0.0,
            max_value=250_000.0,
            step=1_000.0,
            key="in_EstimatedSalary",
        )
        c1, c2 = st.columns(2)
        c1.radio(
            "Active member?",
            [1, 0],
            format_func=lambda v: "Yes" if v else "No",
            key="in_IsActiveMember",
            horizontal=True,
        )
        c2.radio(
            "Has credit card?",
            [1, 0],
            format_func=lambda v: "Yes" if v else "No",
            key="in_HasCrCard",
            horizontal=True,
        )

    customer = {k: st.session_state[f"in_{k}"] for k in defaults()}
    t0 = time.perf_counter()
    result_ = explain_customer(pd.DataFrame([customer]), pipeline)
    elapsed = time.perf_counter() - t0
    p = result_["probability"]
    band = str(risk_band(p))
    threshold = meta["threshold"]

    with result:
        st.plotly_chart(gauge(p, threshold, LOW_RISK_MAX, MEDIUM_RISK_MAX), width="stretch")
        color, icon = BADGE[band]
        verdict = "likely to churn" if p >= threshold else "likely to stay"
        st.markdown(f"#### :{color}-badge[{icon} {band} risk] &nbsp; {verdict}")
        st.caption(
            f"Bands: Low < {LOW_RISK_MAX:.0%} ≤ Medium ≤ {MEDIUM_RISK_MAX:.0%} < High · "
            f"blue tick on the gauge = decision threshold {threshold:.0%}"
        )

        st.markdown("##### Top 3 reasons")
        for reason, value in zip(
            result_["reasons"], result_["contributions"].head(3), strict=False
        ):
            arrow = (
                ":orange[:material/trending_up:]"
                if value > 0
                else ":blue[:material/trending_down:]"
            )
            st.markdown(f"{arrow} **{reason}**")

    st.plotly_chart(waterfall(result_["contributions"], result_["base_value"]), width="stretch")
    st.caption(
        "SHAP contributions in log-odds, summed back to the raw feature (e.g. the age "
        "bands count towards Age). The dashed line is the model's average output; "
        f"bars add up to this customer's score. Computed in {elapsed * 1000:.0f} ms."
    )
