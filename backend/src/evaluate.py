"""Final model: threshold selection, one-time test evaluation, artifacts.

Steps (``python -m src.evaluate`` from backend/):
1. Rebuild the model chosen in T04 (models/final_model.json + best_params.json).
2. Out-of-fold predictions on TRAIN -> choose decision thresholds
   (F1-max, Recall >= 0.75, and cost-optimal with FN = 5 x FP).
   The test set is never used to pick a threshold.
3. Refit on the full train set and evaluate ONCE on test.
4. Save the fitted pipeline, threshold.json, metadata.json and final_metrics.json.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict

from src import eda
from src.config import (
    COST_FN,
    COST_FP,
    FEATURE_COLS,
    FINAL_METRICS_PATH,
    METADATA_PATH,
    MODELS_DIR,
    PIPELINE_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    THRESHOLD_PATH,
)
from src.data import load_split
from src.features import get_feature_names
from src.train import BEST_PARAMS_JSON, FINAL_CHOICE_JSON, get_cv, load_tuned_pipeline

TARGET_RECALL = 0.75
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order
LIBRARIES = [
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "lightgbm",
    "pandas",
    "numpy",
    "shap",
    "joblib",
]


# ------------------------------------------------------------- building
def build_final_pipeline(y_train: pd.Series) -> tuple[Pipeline, dict]:
    """Unfitted pipeline for the model chosen in T04."""
    choice = json.loads(FINAL_CHOICE_JSON.read_text())
    if choice["type"] != "single":
        raise NotImplementedError("T04 kept a single model; ensembles are not rebuilt here")
    return load_tuned_pipeline(choice["params_key"], y_train), choice


# ------------------------------------------------------------ thresholds
def metrics_at(y_true: ArrayLike, proba: ArrayLike, threshold: float) -> dict:
    """Threshold-dependent metrics plus confusion counts and business cost."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "cost": float(COST_FN * fn + COST_FP * fp),
    }


def threshold_curve(y_true: ArrayLike, proba: ArrayLike) -> pd.DataFrame:
    """Precision / recall / F1 / cost for thresholds 0.01 … 0.99."""
    rows = [metrics_at(y_true, proba, t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    return pd.DataFrame(rows)


def choose_thresholds(curve: pd.DataFrame) -> dict:
    """F1-max, Recall >= TARGET_RECALL with best precision, and cost-optimal."""
    f1_row = curve.loc[curve["f1"].idxmax()]
    meets = curve[curve["recall"] >= TARGET_RECALL]
    recall_row = meets.loc[meets["precision"].idxmax()]
    cost_row = curve.loc[curve["cost"].idxmin()]
    return {
        "f1_optimal": float(f1_row["threshold"]),
        f"recall_{TARGET_RECALL}": float(recall_row["threshold"]),
        "cost_optimal": float(cost_row["threshold"]),
    }


def plot_threshold_curve(curve: pd.DataFrame, chosen: dict) -> plt.Figure:
    """Precision/recall/F1 and business cost vs threshold, chosen thresholds marked."""
    fig, (ax, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    for col, color in zip(["precision", "recall", "f1"], SERIES, strict=False):
        ax.plot(
            curve["threshold"],
            curve[col],
            color=color,
            linewidth=2,
            label=col.capitalize() if col != "f1" else "F1",
        )
    styles = {
        "f1_optimal": ("F1-max", "-"),
        f"recall_{TARGET_RECALL}": (f"Recall ≥ {TARGET_RECALL}", "--"),
        "cost_optimal": ("Cost-optimal", ":"),
    }
    for key, t in chosen.items():
        label, ls = styles[key]
        ax.axvline(
            t, color=eda.NEUTRAL_COLOR, linestyle=ls, linewidth=1.2, label=f"{label} ({t:.2f})"
        )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (out-of-fold, train)")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="lower left",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
    )
    ax.set_title("Precision / Recall / F1 vs threshold (out-of-fold, train)")

    ax_c.plot(curve["threshold"], curve["cost"], color=SERIES[0], linewidth=2)
    t = chosen["cost_optimal"]
    best = curve.loc[curve["threshold"] == t, "cost"].iloc[0]
    ax_c.scatter([t], [best], color=SERIES[1], s=50, zorder=3)
    ax_c.annotate(
        f"min cost {best:,.0f} at {t:.2f}",
        (t, best),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
        color=eda.TEXT_COLOR,
    )
    ax_c.set_xlabel("Decision threshold")
    ax_c.set_ylabel(f"Cost = {COST_FN:g}×FN + {COST_FP:g}×FP")
    ax_c.set_title("Business cost vs threshold (out-of-fold, train)")
    fig.tight_layout()
    return eda.save_figure(fig, "threshold_tuning")


# -------------------------------------------------------- test figures
def plot_confusion_matrices(y_true: ArrayLike, pred: ArrayLike, threshold: float) -> plt.Figure:
    """Confusion matrix as counts and normalised by the true class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, norm, title in [
        (axes[0], None, "counts"),
        (axes[1], "true", "normalised by true class"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            pred,
            labels=[0, 1],
            display_labels=["Retained", "Churned"],
            normalize=norm,
            cmap=eda.CHURN_RATE_CMAP,
            colorbar=False,
            values_format=".2f" if norm else "d",
            ax=ax,
        )
        ax.set_title(f"Confusion matrix — {title}")
        ax.grid(False)
    fig.suptitle(f"Test set, threshold = {threshold:.2f}", fontweight="bold")
    fig.tight_layout()
    return eda.save_figure(fig, "test_confusion_matrix")


def plot_roc_pr(y_true: ArrayLike, proba: ArrayLike, ops: dict) -> plt.Figure:
    """ROC and precision-recall curves with the operating points in ``ops`` marked."""
    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax_r.plot(
        fpr,
        tpr,
        color=SERIES[0],
        linewidth=2,
        label=f"Model (AUC = {roc_auc_score(y_true, proba):.3f})",
    )
    ax_r.plot(
        [0, 1],
        [0, 1],
        color=eda.NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1,
        label="Random (AUC = 0.500)",
    )
    ax_r.set_xlabel("False positive rate")
    ax_r.set_ylabel("True positive rate (recall)")
    ax_r.set_title("ROC curve — test set")
    ax_r.legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, proba)
    base = y_true.mean()
    ax_p.plot(
        rec,
        prec,
        color=SERIES[0],
        linewidth=2,
        label=f"Model (AP = {average_precision_score(y_true, proba):.3f})",
    )
    ax_p.axhline(
        base,
        color=eda.NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1,
        label=f"No skill (AP = {base:.3f})",
    )
    for (name, m), color in zip(ops.items(), SERIES[1:], strict=False):
        ax_p.scatter(
            m["recall"],
            m["precision"],
            s=60,
            color=color,
            zorder=3,
            edgecolor="white",
            linewidth=1.5,
            label=f"{name} (t = {m['threshold']:.2f})",
        )
    ax_p.set_xlabel("Recall")
    ax_p.set_ylabel("Precision")
    ax_p.set_ylim(0, 1.02)
    ax_p.set_title("Precision-Recall curve — test set")
    ax_p.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return eda.save_figure(fig, "test_roc_pr_curves")


def plot_calibration(y_true: ArrayLike, proba: ArrayLike) -> plt.Figure:
    """Reliability curve (quantile bins) plus a histogram of predicted probabilities."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(6.5, 7), height_ratios=[3, 1], sharex=True)
    ax.plot(
        [0, 1],
        [0, 1],
        color=eda.NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1,
        label="Perfectly calibrated",
    )
    ax.plot(
        mean_pred,
        frac_pos,
        color=SERIES[0],
        marker="o",
        markersize=7,
        linewidth=2,
        label=f"Model (Brier = {brier_score_loss(y_true, proba):.3f})",
    )
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration curve — test set (10 quantile bins)")
    ax.legend(loc="upper left")
    ax_h.hist(proba, bins=40, color=SERIES[0], edgecolor="white", linewidth=0.5)
    ax_h.set_xlabel("Predicted churn probability")
    ax_h.set_ylabel("Customers")
    fig.tight_layout()
    return eda.save_figure(fig, "test_calibration_curve")


# ----------------------------------------------------------------- main
def _round(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def main() -> dict:
    """Choose thresholds, evaluate once on test and save every artifact."""
    plt.switch_backend("Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    X_train, X_test, y_train, y_test = load_split()
    pipe, choice = build_final_pipeline(y_train)
    print(
        f"Final model: {choice['model']} + {choice['strategy']} "
        f"(CV PR-AUC {choice['cv_pr_auc']})"
    )

    # 1. thresholds from out-of-fold predictions on train only
    oof = cross_val_predict(pipe, X_train, y_train, cv=get_cv(), method="predict_proba", n_jobs=-1)[
        :, 1
    ]
    curve = threshold_curve(y_train, oof)
    chosen = choose_thresholds(curve)
    curve.round(4).to_csv(REPORTS_DIR / "threshold_curve_oof.csv", index=False)
    plot_threshold_curve(curve, chosen)
    print("Thresholds chosen on OOF train predictions:", chosen)

    # 2. refit on full train, evaluate once on test
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    threshold = chosen["f1_optimal"]

    ops = {
        "Default 0.5": metrics_at(y_test, proba, 0.5),
        "F1-max": metrics_at(y_test, proba, chosen["f1_optimal"]),
        f"Recall≥{TARGET_RECALL}": metrics_at(y_test, proba, chosen[f"recall_{TARGET_RECALL}"]),
        "Cost-optimal": metrics_at(y_test, proba, chosen["cost_optimal"]),
    }
    ranking = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
    }

    pred = (proba >= threshold).astype(int)
    names = ["Retained", "Churned"]
    report = classification_report(y_test, pred, target_names=names, digits=3)
    report_05 = classification_report(
        y_test, (proba >= 0.5).astype(int), target_names=names, digits=3
    )
    (REPORTS_DIR / "classification_report.txt").write_text(
        f"Test set, threshold = {threshold:.2f} (F1-optimal on OOF train)\n\n{report}\n"
        f"Default threshold 0.5:\n\n{report_05}",
        encoding="utf-8",
    )
    plot_confusion_matrices(y_test, pred, threshold)
    plot_roc_pr(y_test, proba, {k: v for k, v in ops.items() if k != "Default 0.5"})
    plot_calibration(y_test, proba)

    final_metrics = {
        "model": f"{choice['model']} + {choice['strategy']}",
        "evaluated_on": f"test set ({len(y_test)} rows, churn rate {y_test.mean():.4f})",
        "threshold_source": "out-of-fold predictions on the train set",
        "chosen_threshold": threshold,
        "threshold_free": _round(ranking),
        "at_default_0.5": _round(ops["Default 0.5"]),
        "at_f1_optimal": _round(ops["F1-max"]),
        f"at_recall_{TARGET_RECALL}": _round(ops[f"Recall≥{TARGET_RECALL}"]),
        "at_cost_optimal": _round(ops["Cost-optimal"]),
        "cost_assumption": f"FN = {COST_FN:g}, FP = {COST_FP:g}",
        "cv_train": {"pr_auc": choice["cv_pr_auc"], "roc_auc": choice["cv_roc_auc"]},
    }
    FINAL_METRICS_PATH.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    # 3. artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, PIPELINE_PATH)
    THRESHOLD_PATH.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "selected_by": "F1-optimal on out-of-fold train predictions",
                "alternatives": chosen,
            },
            indent=2,
        )
    )
    params = json.loads(BEST_PARAMS_JSON.read_text())[choice["params_key"]]["params"]
    metadata = {
        "model_name": f"{choice['model']} + {choice['strategy']}",
        "estimator": type(pipe.named_steps["model"]).__name__,
        "params": params,
        "threshold": threshold,
        "risk_bands": {"low": "< 0.3", "medium": "0.3 – 0.6", "high": "> 0.6"},
        "test_metrics": {
            **_round(ranking),
            **{
                k: v
                for k, v in _round(ops["F1-max"]).items()
                if k in ("accuracy", "precision", "recall", "f1")
            },
        },
        "cv_metrics": final_metrics["cv_train"],
        "input_features": FEATURE_COLS,
        "model_features": get_feature_names(pipe),
        "n_train": len(y_train),
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "library_versions": {lib: version(lib) for lib in LIBRARIES},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(
        f"\nTest ROC-AUC {ranking['roc_auc']:.4f} | PR-AUC {ranking['pr_auc']:.4f} | "
        f"Brier {ranking['brier']:.4f}"
    )
    table = pd.DataFrame(ops).T[
        ["threshold", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn", "cost"]
    ]
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        f"\nSaved {PIPELINE_PATH.name}, {THRESHOLD_PATH.name}, {METADATA_PATH.name}, "
        f"{FINAL_METRICS_PATH.name}"
    )
    return final_metrics


if __name__ == "__main__":
    main()
