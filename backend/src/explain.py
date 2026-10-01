"""Explainability for the final pipeline: importances and SHAP.

Run ``python -m src.evaluate`` first (it saves models/churn_pipeline.joblib),
then ``python -m src.explain``. Figures go to reports/figures/explain_*.png.
"""
from __future__ import annotations

import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src import eda
from src.config import (
    METADATA_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from src.data import load_split
from src.features import get_feature_names

BAR_COLOR = "#2a78d6"
TOP_N = 15


# ------------------------------------------------------------- helpers
def load_pipeline():
    """The fitted final pipeline and its metadata."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"{PIPELINE_PATH} missing — run `python -m src.evaluate` first")
    return joblib.load(PIPELINE_PATH), json.loads(METADATA_PATH.read_text())


def model_input(pipeline, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> the exact matrix the model sees (feature engineering + preprocessing)."""
    steps = pipeline.named_steps
    Xt = steps["preprocess"].transform(steps["features"].transform(X_raw))
    # passthrough + encoded columns come back as an object array; the model sees floats
    return pd.DataFrame(np.asarray(Xt, dtype=float), columns=get_feature_names(pipeline),
                        index=X_raw.index)


def get_explainer(pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(pipeline.named_steps["model"])


def _hbar(series: pd.Series, err: pd.Series | None, title: str, xlabel: str,
          name: str) -> plt.Figure:
    series = series.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(series) + 1.4))
    ax.barh(series.index, series.values, color=BAR_COLOR, height=0.65,
            xerr=None if err is None else err.reindex(series.index).values,
            error_kw={"elinewidth": 0.8, "ecolor": eda.MUTED_TEXT, "capsize": 2})
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return eda.save_figure(fig, name)


# ------------------------------------------------------- importances
def builtin_importance(pipeline) -> pd.Series:
    """Model's own (gain-based) feature importance over the model features."""
    model = pipeline.named_steps["model"]
    imp = pd.Series(model.feature_importances_, index=get_feature_names(pipeline))
    imp = imp.sort_values(ascending=False)
    _hbar(imp.head(TOP_N), None, f"XGBoost built-in importance (gain) — top {TOP_N}",
          "Normalised gain", "explain_builtin_importance")
    return imp


def permutation_importances(pipeline, X_test, y_test, n_repeats: int = 20) -> pd.DataFrame:
    """Drop in test PR-AUC when each RAW input column is shuffled.

    Permuting raw columns (not engineered ones) keeps derived features
    consistent, e.g. shuffling Age also moves AgeGroup, TenureByAge, IsSenior.
    """
    res = permutation_importance(pipeline, X_test, y_test, scoring="average_precision",
                                 n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    df = pd.DataFrame({"importance_mean": res.importances_mean,
                       "importance_std": res.importances_std},
                      index=X_test.columns).sort_values("importance_mean", ascending=False)
    df.round(4).to_csv(REPORTS_DIR / "permutation_importance.csv", index_label="feature")
    _hbar(df["importance_mean"], df["importance_std"],
          f"Permutation importance on test set ({n_repeats} repeats)",
          "Decrease in PR-AUC when the column is shuffled", "explain_permutation_importance")
    return df


# ---------------------------------------------------------------- SHAP
def shap_explanation(pipeline, X_raw: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-odds of churn) for raw rows."""
    Xm = model_input(pipeline, X_raw)
    return get_explainer(pipeline)(Xm)


def plot_shap_summary(sv: shap.Explanation) -> None:
    plt.figure()
    shap.plots.beeswarm(sv, max_display=TOP_N, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 7)
    plt.title("SHAP summary — impact on churn log-odds (test set)")
    eda.save_figure(fig, "explain_shap_beeswarm")
    plt.close(fig)

    plt.figure()
    shap.plots.bar(sv, max_display=TOP_N, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 7)
    plt.title("Mean |SHAP value| — global importance (test set)")
    eda.save_figure(fig, "explain_shap_bar")
    plt.close(fig)


def top_shap_features(sv: shap.Explanation, n: int = 4) -> list[str]:
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n]
    return [sv.feature_names[i] for i in order]


def plot_shap_dependence(sv: shap.Explanation, features: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, feat in zip(axes.flat, features):
        shap.plots.scatter(sv[:, feat], color=sv, ax=ax, show=False, alpha=0.6, dot_size=10)
        ax.axhline(0, color=eda.NEUTRAL_COLOR, linewidth=0.8, linestyle="--")
        ax.set_title(f"SHAP dependence — {feat}")
    fig.tight_layout()
    return eda.save_figure(fig, "explain_shap_dependence")


def plot_waterfall(sv_row: shap.Explanation, title: str, name: str) -> plt.Figure:
    plt.figure()
    shap.plots.waterfall(sv_row, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6.5)
    plt.title(title)
    return eda.save_figure(fig, name)


def explain_single(row: pd.DataFrame | pd.Series | dict, pipeline=None, top_k: int | None = None):
    """Explain one customer.

    Returns (waterfall figure, [(feature, feature value, SHAP contribution), ...])
    sorted by absolute contribution. Contributions are in log-odds of churn;
    positive values push towards churn.
    """
    pipeline = pipeline or load_pipeline()[0]
    if isinstance(row, dict):
        row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        row = row.to_frame().T
    row = row.drop(columns=[TARGET], errors="ignore")
    sv = shap_explanation(pipeline, row)[0]
    proba = float(pipeline.predict_proba(row)[:, 1][0])
    contribs = sorted(zip(sv.feature_names, sv.data, sv.values),
                      key=lambda t: abs(t[2]), reverse=True)
    contribs = [(f, float(v), float(c)) for f, v, c in contribs[:top_k]]
    plt.figure()
    shap.plots.waterfall(sv, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6.5)
    plt.title(f"Why this customer scores {proba:.0%} churn probability")
    fig.tight_layout()
    return fig, contribs


# --------------------------------------------- human-readable explanations
# Model features that are exact functions of ONE raw column are summed back into
# it, so contributions read in business terms. Ratios mix two raw columns and
# stay separate under a friendly name.
RATIO_LABELS = {
    "BalanceSalaryRatio": "Balance / salary ratio",
    "TenureByAge": "Tenure relative to age",
    "CreditScoreGivenAge": "Credit score relative to age",
    "ProductsPerTenure": "Products per year of tenure",
}


def raw_group(model_feature: str) -> str:
    """Name of the raw column (or ratio label) a model feature belongs to."""
    if model_feature.startswith("AgeGroup_") or model_feature == "IsSenior":
        return "Age"
    if model_feature.startswith("Geography_"):
        return "Geography"
    if model_feature == "BalanceZero":
        return "Balance"
    return RATIO_LABELS.get(model_feature, model_feature)


def grouped_contributions(sv_row: shap.Explanation) -> pd.Series:
    """SHAP values of one row summed per raw feature, sorted by |contribution|."""
    s = pd.Series(sv_row.values, index=sv_row.feature_names)
    grouped = s.groupby(s.index.map(raw_group)).sum()
    return grouped.reindex(grouped.abs().sort_values(ascending=False).index)


def describe_value(group: str, row: pd.Series) -> str:
    """Short human description of a customer's value for a feature group."""
    match group:
        case "Age":
            return f"Age {int(row['Age'])}"
        case "Geography":
            return f"Customer in {row['Geography']}"
        case "Gender":
            return str(row["Gender"])
        case "IsActiveMember":
            return "Active member" if row["IsActiveMember"] == 1 else "Inactive member"
        case "HasCrCard":
            return "Has a credit card" if row["HasCrCard"] == 1 else "No credit card"
        case "NumOfProducts":
            n = int(row["NumOfProducts"])
            return f"{n} product" + ("s" if n != 1 else "")
        case "Balance":
            return "Zero balance" if row["Balance"] == 0 else f"Balance {row['Balance']:,.0f}"
        case "CreditScore":
            return f"Credit score {int(row['CreditScore'])}"
        case "Tenure":
            return f"Tenure {int(row['Tenure'])} years"
        case "EstimatedSalary":
            return f"Salary {row['EstimatedSalary']:,.0f}"
        case _:
            return group


def top_reasons(contributions: pd.Series, row: pd.Series, k: int = 3) -> list[str]:
    """Plain-English sentences for the k strongest drivers of one prediction."""
    reasons = []
    for group, value in contributions.head(k).items():
        direction = "increases" if value > 0 else "decreases"
        reasons.append(f"{describe_value(group, row)} {direction} churn risk")
    return reasons


def explain_customer(row: pd.DataFrame | pd.Series | dict, pipeline=None) -> dict:
    """Probability, grouped SHAP contributions and top-3 reasons for one customer."""
    pipeline = pipeline or load_pipeline()[0]
    if isinstance(row, dict):
        row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        row = row.to_frame().T
    row = row.drop(columns=[TARGET], errors="ignore")
    sv = shap_explanation(pipeline, row)[0]
    contributions = grouped_contributions(sv)
    return {
        "probability": float(pipeline.predict_proba(row)[:, 1][0]),
        "base_value": float(sv.base_values),
        "contributions": contributions,
        "reasons": top_reasons(contributions, row.iloc[0]),
    }


# ------------------------------------------------------------------ main
def _pick_example(proba: np.ndarray, mask: np.ndarray) -> int:
    """Positional index of the row in ``mask`` whose probability is the group median."""
    idx = np.flatnonzero(mask)
    return int(idx[np.argsort(proba[idx])[len(idx) // 2]])


def main() -> None:
    plt.switch_backend("Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pipeline, meta = load_pipeline()
    _, X_test, _, y_test = load_split()
    threshold = meta["threshold"]
    print(f"Explaining {meta['model_name']} on {len(X_test)} test rows")

    builtin = builtin_importance(pipeline)
    print("\nBuilt-in importance (top 8):\n" + builtin.head(8).round(4).to_string())
    perm = permutation_importances(pipeline, X_test, y_test)
    print("\nPermutation importance (PR-AUC drop):\n" + perm.round(4).to_string())

    sv = shap_explanation(pipeline, X_test)
    plot_shap_summary(sv)
    top4 = top_shap_features(sv, 4)
    plot_shap_dependence(sv, top4)
    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=sv.feature_names)
    mean_abs.sort_values(ascending=False).round(4).to_csv(
        REPORTS_DIR / "shap_mean_abs.csv", header=["mean_abs_shap"], index_label="feature")
    print("\nTop SHAP features:", top4)

    proba = pipeline.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()
    churner = _pick_example(proba, (y == 1) & (proba >= threshold))
    stayer = _pick_example(proba, (y == 0) & (proba < threshold))
    for i, label, name in [(churner, "churner", "explain_shap_waterfall_churner"),
                           (stayer, "non-churner", "explain_shap_waterfall_nonchurner")]:
        plot_waterfall(sv[i], f"Test {label} — predicted churn probability {proba[i]:.0%}", name)
        plt.close("all")
        print(f"{label}: row {X_test.index[i]}, p = {proba[i]:.3f}, "
              f"{X_test.iloc[i].to_dict()}")

    fig, contribs = explain_single(X_test.iloc[0], pipeline, top_k=3)
    plt.close(fig)
    print("\nexplain_single demo (top 3):", contribs)


if __name__ == "__main__":
    main()
