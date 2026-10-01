"""Model Performance: model comparison, test-set curves, live threshold, importances."""

import json

import streamlit as st

from src.config import FIGURES_DIR, MODELS_DIR
from src.evaluate import metrics_at
from ui.charts import confusion_matrix_fig, importance_bar, roc_pr_figs
from ui.data import load_metadata
from ui.model import RESULT_FILES, model_ready, permutation_table, results_table, test_predictions

st.title("Model Performance")

if model_ready():
    meta = load_metadata()
    final = json.loads((MODELS_DIR / "final_model.json").read_text())
    st.caption(
        f"Final model: **{meta['model_name']}** · trained {meta['trained_at'][:10]} · "
        f"tuned threshold **{meta['threshold']}** (F1-optimal on out-of-fold train "
        f"predictions) · all curves below are on the untouched 2,000-row test set."
    )

    # ------------------------------------------------------ comparison table
    st.subheader("Model comparison (5-fold CV on train)")
    tabs = st.tabs(list(RESULT_FILES))
    for tab, name in zip(tabs, RESULT_FILES, strict=False):
        with tab:
            table = results_table(name)
            if table is None:
                st.info(f"`{RESULT_FILES[name]}` not found — run `python -m src.train`.")
                continue
            if name == "Baselines":
                is_final = table.index == 0  # best baseline (sorted by PR-AUC)
            else:
                is_final = (
                    (table["model"] == final["model"]) & (table["strategy"] == final["strategy"])
                ).to_numpy()

            def highlight(row, mask=is_final):
                style = "background-color: rgba(235, 104, 52, 0.22); font-weight: 600"
                return [style if mask[row.name] else ""] * len(row)

            num_cols = table.select_dtypes("number").columns
            st.dataframe(
                table.style.apply(highlight, axis=1).format("{:.3f}", subset=num_cols),
                width="stretch",
                hide_index=True,
            )
    st.caption(
        f"Highlighted: **{final['model']} + {final['strategy']}**, the final model "
        f"(best tuned single model; ensembles gained only {final['ensemble_gain']:+.4f} "
        "PR-AUC, below the 0.005 bar). On the Baselines tab the best baseline is highlighted."
    )

    # --------------------------------------------------- threshold explorer
    st.divider()
    st.subheader("Decision threshold — test set")
    preds = test_predictions()
    y, proba = preds["Exited"].to_numpy(), preds["churn_probability"].to_numpy()

    st.session_state.setdefault("perf_threshold", float(meta["threshold"]))
    threshold = st.slider(
        "Threshold: customers with churn probability ≥ threshold are " "flagged as churners",
        0.05,
        0.95,
        step=0.01,
        key="perf_threshold",
    )
    m = metrics_at(y, proba, threshold)
    ref = metrics_at(y, proba, meta["threshold"])

    cols = st.columns(5)
    for col, key, label in zip(
        cols,
        ["precision", "recall", "f1", "accuracy"],
        ["Precision", "Recall", "F1", "Accuracy"],
        strict=False,
    ):
        delta = m[key] - ref[key]
        col.metric(
            label, f"{m[key]:.3f}", delta=f"{delta:+.3f} vs tuned" if abs(delta) > 1e-9 else None
        )
    cols[4].metric(
        "Customers flagged",
        f"{m['tp'] + m['fp']:,}",
        help="Predicted churners = true positives + false positives",
    )

    left, right = st.columns(2)
    left.plotly_chart(
        confusion_matrix_fig(m["tn"], m["fp"], m["fn"], m["tp"], height=400), width="stretch"
    )
    roc, pr = roc_pr_figs(y, proba, threshold, height=400)
    pr_tab, roc_tab = right.tabs(["Precision-Recall curve", "ROC curve"])
    pr_tab.plotly_chart(pr, width="stretch")
    roc_tab.plotly_chart(roc, width="stretch")
    st.caption(
        f"Caught **{m['tp']} of {m['tp'] + m['fn']}** churners, with **{m['fp']}** false "
        f"alarms. Business cost (missed churner = 5 × false alarm): **{m['cost']:,.0f}**. "
        "Lower thresholds catch more churners at the price of more false alarms."
    )

    # ----------------------------------------------------- explainability
    st.divider()
    st.subheader("What drives the model")
    perm = permutation_table()
    left, right = st.columns(2)
    with left:
        if perm is not None:
            st.plotly_chart(
                importance_bar(
                    perm,
                    "importance_mean",
                    "importance_std",
                    "Drop in test PR-AUC when shuffled",
                    "Permutation importance (raw features, test set)",
                ),
                width="stretch",
            )
        bar = FIGURES_DIR / "explain_shap_bar.png"
        if bar.exists():
            st.image(str(bar), caption="Mean |SHAP| per model feature", width="stretch")
    with right:
        bee = FIGURES_DIR / "explain_shap_beeswarm.png"
        if bee.exists():
            st.image(
                str(bee),
                caption="SHAP summary: each dot is a test customer; red = high "
                "feature value; right = pushes towards churn",
                width="stretch",
            )
        else:
            st.info("SHAP figures not found — run `python -m src.explain`.")
