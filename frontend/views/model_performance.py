"""Model Performance page — placeholder until T07."""
import streamlit as st

from ui.data import load_metadata

st.title("Model Performance")
st.info("Coming in T07 Part A — model comparison table, ROC/PR curves, live threshold slider, feature importance and SHAP summary.", icon=":material/construction:")

meta = load_metadata()
if meta:
    st.caption(f"Model ready: **{meta['model_name']}**, decision threshold "
               f"{meta['threshold']}, test ROC-AUC {meta['test_metrics']['roc_auc']:.3f}.")
else:
    st.warning("No trained model found. Run `python -m src.evaluate` in backend/ first.")
