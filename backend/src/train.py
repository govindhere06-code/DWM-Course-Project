"""Model selection: baselines, imbalance strategies, tuning and ensembles.

All evaluation is 5-fold stratified CV on the TRAIN split only; the test split
is untouched until T05. Every candidate is a full pipeline

    FeatureEngineer -> preprocessor -> [sampler] -> model

so scalers, encoders and samplers are re-fitted inside each training fold.

Usage (from backend/):
    python -m src.train                    # all stages
    python -m src.train --stage baseline   # baseline | imbalance | tune | ensemble
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src import eda
from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.data import load_split
from src.features import build_pipeline

BASELINE_CSV = REPORTS_DIR / "baseline_results.csv"
IMBALANCE_CSV = REPORTS_DIR / "imbalance_comparison.csv"
TUNING_CSV = REPORTS_DIR / "tuning_results.csv"
ENSEMBLE_CSV = REPORTS_DIR / "ensemble_comparison.csv"
BEST_PARAMS_JSON = MODELS_DIR / "best_params.json"
FINAL_CHOICE_JSON = MODELS_DIR / "final_model.json"

N_ITER = 40
ENSEMBLE_MARGIN = 0.005  # ensemble must beat best single model by more than this
TIE_TOLERANCE = 0.01     # PR-AUC gap treated as a tie (~1 standard error of a 5-fold mean)
N_JOBS = -1

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
METRICS = list(SCORING)
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}

# Categorical series colours (fixed order, validated palette).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

STRATEGIES = ["none", "class_weight", "smote", "smoteenn"]
STRATEGY_LABELS = {"none": "No handling", "class_weight": "Class weight",
                   "smote": "SMOTE", "smoteenn": "SMOTEENN"}


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """GradientBoosting with 'balanced' sample weights computed inside fit.

    sklearn's GradientBoosting has no ``class_weight`` parameter; doing the
    weighting inside ``fit`` keeps it per-fold and lets the model sit in
    pipelines, searches and ensembles without routing ``sample_weight``.
    """

    def fit(self, X, y, sample_weight=None, monitor=None):
        weights = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            weights = weights * sample_weight
        return super().fit(X, y, sample_weight=weights, monitor=monitor)


# ------------------------------------------------------------- registry
# name -> (needs scaling, supports class weighting)
MODEL_SPECS = {
    "Dummy": (True, False),
    "LogisticRegression": (True, True),
    "KNN": (True, False),
    "GaussianNB": (True, True),
    "DecisionTree": (False, True),
    "SVM": (True, True),
    "RandomForest": (False, True),
    "GradientBoosting": (False, True),
    "XGBoost": (False, True),
    "LightGBM": (False, True),
}


def make_model(name: str, weighted: bool, pos_weight: float):
    """Instantiate a model with default params; ``weighted`` turns on class weighting."""
    cw = "balanced" if weighted else None
    rs = RANDOM_STATE
    if name == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "LogisticRegression":
        return LogisticRegression(class_weight=cw, max_iter=2000, random_state=rs)
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "GaussianNB":
        # Equal priors are the NB equivalent of balanced class weights.
        return GaussianNB(priors=[0.5, 0.5] if weighted else None)
    if name == "DecisionTree":
        return DecisionTreeClassifier(class_weight=cw, random_state=rs)
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, class_weight=cw, random_state=rs)
    if name == "RandomForest":
        return RandomForestClassifier(class_weight=cw, n_jobs=1, random_state=rs)
    if name == "GradientBoosting":
        cls = BalancedGradientBoostingClassifier if weighted else GradientBoostingClassifier
        return cls(random_state=rs)
    if name == "XGBoost":
        return XGBClassifier(scale_pos_weight=pos_weight if weighted else 1.0,
                             eval_metric="logloss", n_jobs=1, random_state=rs)
    if name == "LightGBM":
        return LGBMClassifier(class_weight=cw, n_jobs=1, verbosity=-1, random_state=rs)
    raise KeyError(name)


def make_sampler(strategy: str):
    if strategy == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if strategy == "smoteenn":
        return SMOTEENN(random_state=RANDOM_STATE)
    return None


def make_pipeline(name: str, strategy: str, pos_weight: float):
    """Full pipeline for a model + imbalance strategy.

    Weighting and resampling are never combined, to avoid double-correcting.
    """
    scale, supports_weight = MODEL_SPECS[name]
    if strategy == "class_weight" and not supports_weight:
        raise ValueError(f"{name} does not support class weighting")
    model = make_model(name, weighted=strategy == "class_weight", pos_weight=pos_weight)
    return build_pipeline(model, scale=scale, sampler=make_sampler(strategy))


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def positive_weight(y) -> float:
    """negatives / positives — XGBoost's scale_pos_weight (≈ 3.9 here)."""
    return float((y == 0).sum() / (y == 1).sum())


def evaluate_cv(pipeline, X, y) -> dict:
    """Mean ± std of every metric over the shared stratified folds, plus fit time."""
    res = cross_validate(pipeline, X, y, cv=get_cv(), scoring=SCORING, n_jobs=N_JOBS)
    out = {}
    for m in METRICS:
        out[f"{m}_mean"] = res[f"test_{m}"].mean()
        out[f"{m}_std"] = res[f"test_{m}"].std()
    out["fit_time_s"] = res["fit_time"].mean()
    return out


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    """Readable 'mean ± std' view of a results table."""
    view = df[[c for c in df.columns if not c.endswith(("_mean", "_std"))]].copy()
    for m in METRICS:
        view[METRIC_LABELS[m]] = (df[f"{m}_mean"].map("{:.3f}".format) + " ± "
                                  + df[f"{m}_std"].map("{:.3f}".format))
    return view


# ================================================ Part A: baseline models
def run_baselines(X, y) -> pd.DataFrame:
    """Default params, class weighting wherever the model supports it."""
    pw = positive_weight(y)
    rows = []
    for name, (_, supports_weight) in MODEL_SPECS.items():
        strategy = "class_weight" if supports_weight else "none"
        t0 = time.perf_counter()
        scores = evaluate_cv(make_pipeline(name, strategy, pw), X, y)
        rows.append({"model": name, "strategy": strategy, **scores})
        print(f"  {name:<20} PR-AUC {scores['pr_auc_mean']:.3f}  "
              f"recall {scores['recall_mean']:.3f}  ({time.perf_counter() - t0:.0f}s)")
    df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)
    df.round(4).to_csv(BASELINE_CSV, index=False)
    plot_baselines(df)
    return df


def plot_baselines(df: pd.DataFrame) -> plt.Figure:
    metrics = ["accuracy", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(df))
    width = 0.16
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (m, color) in enumerate(zip(metrics, SERIES_COLORS)):
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, df[f"{m}_mean"], width * 0.92, yerr=df[f"{m}_std"],
               color=color, label=METRIC_LABELS[m], capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": eda.MUTED_TEXT})
    ax.set_xticks(x, df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-fold CV score (mean ± std)")
    ax.set_title("Baseline models — sorted by PR-AUC (train set, 5-fold stratified CV)")
    ax.legend(ncol=5, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return eda.save_figure(fig, "baseline_comparison")


# ======================================== Part B: imbalance strategies
def top_models(df: pd.DataFrame, n: int = 3) -> list[str]:
    """Top-n non-dummy models by CV PR-AUC."""
    ranked = df[df["model"] != "Dummy"].sort_values("pr_auc_mean", ascending=False)
    return ranked["model"].head(n).tolist()


def run_imbalance(X, y, models: list[str]) -> pd.DataFrame:
    pw = positive_weight(y)
    rows = []
    for name in models:
        for strategy in STRATEGIES:
            if strategy == "class_weight" and not MODEL_SPECS[name][1]:
                continue
            scores = evaluate_cv(make_pipeline(name, strategy, pw), X, y)
            rows.append({"model": name, "strategy": strategy, **scores})
            print(f"  {name:<18} {strategy:<13} PR-AUC {scores['pr_auc_mean']:.3f}  "
                  f"recall {scores['recall_mean']:.3f}  F1 {scores['f1_mean']:.3f}")
    df = pd.DataFrame(rows)
    df["selected"] = False
    for name, group in df.groupby("model"):
        df.loc[pick_strategy(group), "selected"] = True
    df.round(4).to_csv(IMBALANCE_CSV, index=False)
    plot_imbalance(df)
    return df


def pick_strategy(group: pd.DataFrame):
    """Index of the best strategy: highest PR-AUC; within TIE_TOLERANCE, higher recall."""
    best = group["pr_auc_mean"].max()
    contenders = group[group["pr_auc_mean"] >= best - TIE_TOLERANCE]
    return contenders["recall_mean"].idxmax()


def plot_imbalance(df: pd.DataFrame) -> plt.Figure:
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    width = 0.2
    for ax, metric in zip(axes, ["pr_auc", "recall"]):
        x = np.arange(len(models))
        for i, (strategy, color) in enumerate(zip(STRATEGIES, SERIES_COLORS)):
            sub = df[df["strategy"] == strategy].set_index("model").reindex(models)
            offset = (i - (len(STRATEGIES) - 1) / 2) * width
            ax.bar(x + offset, sub[f"{metric}_mean"], width * 0.92,
                   yerr=sub[f"{metric}_std"], color=color, capsize=2,
                   label=STRATEGY_LABELS[strategy],
                   error_kw={"elinewidth": 0.8, "ecolor": eda.MUTED_TEXT})
        ax.set_xticks(x, models)
        ax.set_ylim(0, 1)
        ax.set_title(f"{METRIC_LABELS[metric]} by imbalance strategy")
        ax.set_ylabel(f"CV {METRIC_LABELS[metric]} (mean ± std)")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(STRATEGIES), loc="upper center",
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return eda.save_figure(fig, "imbalance_comparison")


# ============================================ Part C: hyperparameter tuning
def search_space(name: str, strategy: str) -> dict:
    """Randomized-search distributions, keyed for the pipeline's ``model`` step."""
    spaces = {
        "LogisticRegression": {"C": loguniform(1e-3, 1e2)},
        "SVM": {"C": loguniform(1e-1, 1e2), "gamma": loguniform(1e-3, 1)},
        "KNN": {"n_neighbors": randint(5, 80), "weights": ["uniform", "distance"]},
        "DecisionTree": {"max_depth": randint(3, 15), "min_samples_leaf": randint(1, 50)},
        "RandomForest": {
            "n_estimators": randint(100, 601), "max_depth": [None, 6, 8, 10, 12, 16],
            "min_samples_split": randint(2, 21), "min_samples_leaf": randint(1, 11),
            "max_features": ["sqrt", "log2", 0.5],
        },
        "GradientBoosting": {
            "n_estimators": randint(100, 601), "learning_rate": loguniform(0.01, 0.3),
            "max_depth": randint(2, 7), "subsample": uniform(0.6, 0.4),
            "min_samples_leaf": randint(1, 51), "max_features": [None, "sqrt"],
        },
        "XGBoost": {
            "n_estimators": randint(100, 601), "max_depth": randint(3, 9),
            "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 11),
            "gamma": uniform(0, 5),
        },
        "LightGBM": {
            "n_estimators": randint(100, 601), "num_leaves": randint(15, 128),
            "learning_rate": loguniform(0.01, 0.3), "min_child_samples": randint(5, 101),
            "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
            "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": loguniform(1e-3, 10),
        },
    }
    space = {f"model__{k}": v for k, v in spaces[name].items()}
    if strategy == "smote":
        space["sampler__k_neighbors"] = randint(3, 11)
    return space


def _jsonable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif not isinstance(v, (int, float, str, bool, type(None))):
            v = repr(v)
        out[k] = v
    return out


def run_tuning(X, y, combos: list[tuple[str, str]], baseline_scores: dict) -> pd.DataFrame:
    """RandomizedSearchCV per combo; keeps defaults if the search does not beat them."""
    pw = positive_weight(y)
    rows, best_params = [], {}
    for name, strategy in combos:
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            make_pipeline(name, strategy, pw), search_space(name, strategy),
            n_iter=N_ITER, scoring=SCORING, refit="pr_auc", cv=get_cv(),
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
        )
        search.fit(X, y)
        i = search.best_index_
        cvr = search.cv_results_
        tuned = {f"{m}_mean": cvr[f"mean_test_{m}"][i] for m in METRICS}
        tuned |= {f"{m}_std": cvr[f"std_test_{m}"][i] for m in METRICS}
        base = baseline_scores[(name, strategy)]
        improved = tuned["pr_auc_mean"] >= base["pr_auc_mean"]
        params = _jsonable(search.best_params_) if improved else {}
        best_params[f"{name}|{strategy}"] = {
            "model": name, "strategy": strategy, "params": params,
            "cv_pr_auc": tuned["pr_auc_mean"] if improved else base["pr_auc_mean"],
            "baseline_cv_pr_auc": base["pr_auc_mean"],
            "kept_defaults": not improved,
        }
        final = tuned if improved else base
        rows.append({"model": name, "strategy": strategy, "baseline_pr_auc": base["pr_auc_mean"],
                     "tuned_pr_auc": tuned["pr_auc_mean"], "kept_defaults": not improved,
                     "search_time_s": time.perf_counter() - t0,
                     **{k: final[k] for k in final if k.endswith(("_mean", "_std"))}})
        print(f"  {name:<18} {strategy:<13} baseline {base['pr_auc_mean']:.4f} -> "
              f"tuned {tuned['pr_auc_mean']:.4f} ({time.perf_counter() - t0:.0f}s)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_JSON.write_text(json.dumps(best_params, indent=2))
    df = pd.DataFrame(rows)
    df.round(4).to_csv(TUNING_CSV, index=False)
    return df


def load_tuned_pipeline(key: str, y) -> object:
    """Rebuild an (unfitted) tuned pipeline from best_params.json."""
    entry = json.loads(BEST_PARAMS_JSON.read_text())[key]
    pipe = make_pipeline(entry["model"], entry["strategy"], positive_weight(y))
    return pipe.set_params(**entry["params"])


# ================================================== Part D: ensembles
def run_ensembles(X, y, tuning: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = [f"{r.model}|{r.strategy}" for r in tuning.itertuples()]
    estimators = [(k.split("|")[0], load_tuned_pipeline(k, y)) for k in keys]
    candidates = {
        "SoftVoting": VotingClassifier(estimators, voting="soft", n_jobs=1),
        "Stacking": StackingClassifier(
            estimators, final_estimator=LogisticRegression(max_iter=2000),
            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            stack_method="predict_proba", n_jobs=1,
        ),
    }
    rows = []
    for _, r in tuning.iterrows():
        rows.append({"model": r["model"], "type": "single", "strategy": r["strategy"],
                     **{c: r[c] for c in r.index if c.endswith(("_mean", "_std"))}})
    for name, ens in candidates.items():
        t0 = time.perf_counter()
        scores = evaluate_cv(ens, X, y)
        rows.append({"model": name, "type": "ensemble", "strategy": "mixed", **scores})
        print(f"  {name:<18} PR-AUC {scores['pr_auc_mean']:.4f} ({time.perf_counter() - t0:.0f}s)")
    df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)
    df.round(4).to_csv(ENSEMBLE_CSV, index=False)

    best_single = df[df["type"] == "single"].iloc[0]
    best_ens = df[df["type"] == "ensemble"].iloc[0]
    gain = best_ens["pr_auc_mean"] - best_single["pr_auc_mean"]
    use_ensemble = gain > ENSEMBLE_MARGIN
    winner = best_ens if use_ensemble else best_single
    choice = {
        "model": winner["model"],
        "type": winner["type"],
        "strategy": winner["strategy"],
        "params_key": None if use_ensemble else f"{winner['model']}|{winner['strategy']}",
        "ensemble_members": keys if use_ensemble else None,
        "cv_pr_auc": round(float(winner["pr_auc_mean"]), 4),
        "cv_recall": round(float(winner["recall_mean"]), 4),
        "cv_roc_auc": round(float(winner["roc_auc_mean"]), 4),
        "best_single": best_single["model"],
        "best_single_pr_auc": round(float(best_single["pr_auc_mean"]), 4),
        "best_ensemble": best_ens["model"],
        "best_ensemble_pr_auc": round(float(best_ens["pr_auc_mean"]), 4),
        "ensemble_gain": round(float(gain), 4),
        "rule": f"ensemble kept only if it beats the best single model by > {ENSEMBLE_MARGIN} PR-AUC",
    }
    FINAL_CHOICE_JSON.write_text(json.dumps(choice, indent=2))
    return df, choice


# ================================================================ runner
def main(stage: str = "all") -> None:
    plt.switch_backend("Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # "±" on Windows consoles
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    X, _, y, _ = load_split()
    print(f"Train set {X.shape}, churn rate {y.mean():.4f}, "
          f"scale_pos_weight {positive_weight(y):.2f}")

    if stage in ("all", "baseline"):
        print("\n[A] Baseline models")
        base = run_baselines(X, y)
        print(_fmt(base).to_string(index=False))

    if stage in ("all", "imbalance"):
        base = pd.read_csv(BASELINE_CSV)
        print("\n[B] Imbalance strategies for", top_models(base))
        imb = run_imbalance(X, y, top_models(base))
        print(_fmt(imb).to_string(index=False))

    if stage in ("all", "tune"):
        imb = pd.read_csv(IMBALANCE_CSV)
        selected = imb[imb["selected"]].sort_values("pr_auc_mean", ascending=False)
        combos = list(zip(selected["model"], selected["strategy"]))
        print("\n[C] Tuning", combos)
        baseline_scores = {(r["model"], r["strategy"]): r for _, r in imb.iterrows()}
        tun = run_tuning(X, y, combos, baseline_scores)
        print(tun[["model", "strategy", "baseline_pr_auc", "tuned_pr_auc",
                   "kept_defaults", "search_time_s"]].round(4).to_string(index=False))

    if stage in ("all", "ensemble"):
        print("\n[D] Ensembles")
        ens, choice = run_ensembles(X, y, pd.read_csv(TUNING_CSV))
        print(_fmt(ens).to_string(index=False))
        print("\nFinal model choice:", json.dumps(choice, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all",
                        choices=["all", "baseline", "imbalance", "tune", "ensemble"])
    main(parser.parse_args().stage)
