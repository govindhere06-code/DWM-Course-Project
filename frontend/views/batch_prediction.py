"""Batch Prediction: upload a CSV or Excel file, score every customer, download the results."""

import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import TARGET, TEST_DATA
from src.evaluate import metrics_at
from ui.batch import (
    NEW_COLS,
    UPLOAD_TYPES,
    XLSX_MIME,
    BatchInputError,
    read_upload,
    score,
    to_csv_bytes,
    to_excel_bytes,
)
from ui.charts import risk_band_bar
from ui.data import load_metadata
from ui.model import model_ready

st.title("Batch Prediction")
st.caption(
    "Upload customers as a CSV or Excel (.xlsx) file with the same columns as "
    "`Churn_Modelling.csv` (Excel: first sheet, header in row 1). "
    "ID columns and `Exited` are optional."
)

if model_ready():
    meta = load_metadata()
    up_col, sample_col = st.columns([3, 1], vertical_alignment="bottom")
    uploaded = up_col.file_uploader("Customer file (CSV or Excel)", type=UPLOAD_TYPES)
    if sample_col.button(
        "Use sample file",
        width="stretch",
        help="Score test.csv: the 2,000-customer held-out test set",
    ):
        st.session_state["batch_source"] = ("test.csv", TEST_DATA.read_bytes())
    if st.query_params.get("sample") == "1" and "batch_source" not in st.session_state:
        st.session_state["batch_source"] = ("test.csv", TEST_DATA.read_bytes())  # demo link
    if uploaded is not None:
        st.session_state["batch_source"] = (uploaded.name, uploaded.getvalue())

    source = st.session_state.get("batch_source")
    if source is None:
        st.info(
            "Upload a CSV / Excel file or use the sample file to get started.",
            icon=":material/upload:",
        )
    else:
        name, raw_bytes = source
        try:
            scored = score(read_upload(raw_bytes, name))
        except BatchInputError as err:
            st.error(f"**{name}**: {err}", icon=":material/error:")
        else:
            n_high = int((scored["risk_band"] == "High").sum())
            k = st.columns(4)
            k[0].metric("Customers scored", f"{len(scored):,}")
            k[1].metric(
                "Predicted churners",
                f"{int(scored['churn_prediction'].sum()):,}",
                help=f"Probability ≥ threshold {meta['threshold']}",
            )
            k[2].metric("High risk", f"{n_high:,}")
            k[3].metric("Mean churn probability", f"{scored['churn_probability'].mean():.1%}")

            # Full-width chart, then table: side by side they get too cramped on narrow screens.
            st.plotly_chart(risk_band_bar(scored["risk_band"], height=300), width="stretch")
            display = scored[NEW_COLS + [c for c in scored.columns if c not in NEW_COLS]]
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                height=360,
                column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Churn probability", format="percent", min_value=0, max_value=1
                    ),
                    "churn_prediction": st.column_config.CheckboxColumn("Predicted churn"),
                    "risk_band": st.column_config.TextColumn("Risk band"),
                },
            )
            stem = name.rsplit(".", 1)[0]
            with st.container(horizontal=True):  # buttons sized to their labels, wrap if needed
                st.download_button(
                    "Download scored CSV",
                    to_csv_bytes(scored),
                    file_name=f"{stem}_scored.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    type="primary",
                )
                st.download_button(
                    "Download scored Excel",
                    to_excel_bytes(scored),
                    file_name=f"{stem}_scored.xlsx",
                    mime=XLSX_MIME,
                    icon=":material/table_view:",
                )

            if TARGET in scored.columns and scored[TARGET].nunique() == 2:
                st.divider()
                st.subheader(f"Accuracy on this file (`{TARGET}` present)")
                y, p = scored[TARGET].to_numpy(), scored["churn_probability"].to_numpy()
                m = metrics_at(y, p, meta["threshold"])
                c = st.columns(5)
                c[0].metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}")
                c[1].metric("PR-AUC", f"{average_precision_score(y, p):.3f}")
                c[2].metric("Recall", f"{m['recall']:.3f}")
                c[3].metric("Precision", f"{m['precision']:.3f}")
                c[4].metric("F1", f"{m['f1']:.3f}")
