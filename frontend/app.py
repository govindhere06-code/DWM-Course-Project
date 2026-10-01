"""Bank Customer Churn dashboard — Streamlit entry point.

Run from the project root:  streamlit run frontend/app.py   (or `make dashboard`)
"""
import sys
from pathlib import Path

import streamlit as st

# Make the backend package (`src`) and the frontend helpers (`ui`) importable.
FRONTEND_DIR = Path(__file__).resolve().parent
BACKEND_DIR = FRONTEND_DIR.parent / "backend"
for path in (BACKEND_DIR, FRONTEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(page_title="Bank Churn Dashboard", page_icon=":material/insights:",
                   layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/eda_explorer.py", title="EDA Explorer", icon=":material/query_stats:"),
    st.Page("views/model_performance.py", title="Model Performance", icon=":material/monitoring:"),
    st.Page("views/predict_customer.py", title="Predict a Customer", icon=":material/person_search:"),
    st.Page("views/batch_prediction.py", title="Batch Prediction", icon=":material/upload_file:"),
]

with st.sidebar:
    st.markdown("## :material/account_balance: Bank Churn")
    st.caption("10,000 customers · target: `Exited`")

st.navigation(pages).run()
