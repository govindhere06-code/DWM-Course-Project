"""Predict a Customer page — placeholder until T07."""
import streamlit as st

from ui.data import load_metadata

st.title("Predict a Customer")
st.info("Coming in T07 Part B — form for the 10 raw features, churn probability gauge, risk band and a SHAP explanation with the top 3 reasons.", icon=":material/construction:")

meta = load_metadata()
if meta:
    st.caption(f"Model ready: **{meta['model_name']}**, decision threshold "
               f"{meta['threshold']}, test ROC-AUC {meta['test_metrics']['roc_auc']:.3f}.")
else:
    st.warning("No trained model found. Run `python -m src.evaluate` in backend/ first.")
