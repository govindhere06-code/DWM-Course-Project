"""Exploratory data analysis: tables, statistical tests and figures.

Every plotting function saves a PNG (dpi=150) to ``reports/figures/`` and
returns the matplotlib Figure. Table functions return a DataFrame and, where
the ticket asks for it, write a CSV to ``reports/``.

Run everything with ``python -m src.eda``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.config import (
    BINARY_COLS,
    CONTINUOUS_COLS,
    DISCRETE_COLS,
    DROP_COLS,
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from src.data import load_raw

# ----------------------------------------------------------------- style
RETAINED_COLOR = "#2a78d6"  # blue
CHURNED_COLOR = "#eb6834"  # orange
NEUTRAL_COLOR = "#8a8984"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT = "#52514e"
GRID_COLOR = "#e4e3df"

CHURN_PALETTE = {0: RETAINED_COLOR, 1: CHURNED_COLOR}
CHURN_LABELS = {0: "Retained", 1: "Churned"}

# Single-hue sequential map (light -> dark orange) for churn-rate heatmaps.
CHURN_RATE_CMAP = LinearSegmentedColormap.from_list(
    "churn_rate", ["#fdf1ea", "#f6b48f", CHURNED_COLOR, "#a8401a"]
)
# Diverging map: blue <- neutral grey -> orange, for correlations.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "corr", [RETAINED_COLOR, "#c7d9f0", "#f2f1ed", "#f6c4a8", CHURNED_COLOR]
)

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]
AGE_BINS = [17, 29, 39, 49, 59, np.inf]
AGE_LABELS = ["18–29", "30–39", "40–49", "50–59", "60+"]
DPI = 150


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": MUTED_TEXT,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
        }
    )


_apply_style()


def save_figure(fig: plt.Figure, name: str) -> plt.Figure:
    """Save ``fig`` as reports/figures/<name>.png (dpi 150) and return it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=DPI)
    return fig


def _churn_legend(ax: plt.Axes, **kwargs) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=CHURN_PALETTE[k]) for k in (0, 1)]
    ax.legend(handles, [CHURN_LABELS[0], CHURN_LABELS[1]], **kwargs)


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an ordered categorical ``AgeBand`` column."""
    out = df.copy()
    out["AgeBand"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return out


def churn_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers, churners and churn rate (%) per level of ``col``."""
    g = df.groupby(col, observed=True)[TARGET]
    return pd.DataFrame(
        {
            "customers": g.size(),
            "churned": g.sum(),
            "churn_rate_pct": (g.mean() * 100).round(1),
        }
    )


# ======================================================= Part B: univariate
def summary_table(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """dtype, unique count, min/max/mean/median/std, skewness, kurtosis per feature."""
    features = df.drop(columns=DROP_COLS, errors="ignore")
    rows = []
    for col in features.columns:
        s = features[col]
        row = {"feature": col, "dtype": str(s.dtype), "n_unique": s.nunique()}
        if pd.api.types.is_numeric_dtype(s):
            row.update(
                min=s.min(),
                max=s.max(),
                mean=s.mean(),
                median=s.median(),
                std=s.std(),
                skewness=s.skew(),
                kurtosis=s.kurt(),
            )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("feature").round(3)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORTS_DIR / "feature_summary.csv")
    return table


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of Retained vs Churned with count and % labels."""
    counts = df[TARGET].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        [CHURN_LABELS[k] for k in counts.index],
        counts.values,
        color=[CHURN_PALETTE[k] for k in counts.index],
        width=0.55,
    )
    for bar, n, p in zip(bars, counts.values, pct.values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 120,
            f"{n:,}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
        )
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("Customers")
    ax.set_title("Target distribution — Exited")
    ax.grid(axis="x", visible=False)
    return save_figure(fig, "target_distribution")


def plot_continuous_distribution(df: pd.DataFrame, col: str) -> plt.Figure:
    """Histogram + KDE and a boxplot side by side for one continuous feature."""
    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    sns.histplot(
        df[col],
        bins=40,
        kde=True,
        color=RETAINED_COLOR,
        edgecolor="white",
        linewidth=0.5,
        ax=ax_h,
        line_kws={"linewidth": 2},
    )
    ax_h.set_title(f"{col} — histogram + KDE")
    ax_h.set_ylabel("Customers")
    sns.boxplot(
        x=df[col],
        color="#c7d9f0",
        linecolor=RETAINED_COLOR,
        linewidth=1.2,
        flierprops={
            "markersize": 3,
            "markerfacecolor": RETAINED_COLOR,
            "markeredgecolor": RETAINED_COLOR,
        },
        ax=ax_b,
    )
    ax_b.set_title(f"{col} — boxplot")
    fig.tight_layout()
    return save_figure(fig, f"distribution_{col.lower()}")


def plot_categorical_counts(df: pd.DataFrame) -> plt.Figure:
    """Count plots for categorical / discrete features (2 × 3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES, strict=False):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=RETAINED_COLOR, width=0.65)
        for x, n in zip(counts.index.astype(str), counts.values, strict=False):
            ax.text(x, n, f"{n:,}", ha="center", va="bottom", fontsize=8, color=MUTED_TEXT)
        ax.set_title(col)
        ax.set_ylabel("Customers")
        ax.set_ylim(0, counts.max() * 1.12)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Categorical & discrete feature counts", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "categorical_counts")


def outlier_report(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """IQR-rule outlier counts per continuous column. Report only — nothing is removed."""
    rows = []
    for col in CONTINUOUS_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_low = int((df[col] < lo).sum())
        n_high = int((df[col] > hi).sum())
        rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lo,
                "upper_fence": hi,
                "n_below": n_low,
                "n_above": n_high,
                "n_outliers": n_low + n_high,
                "pct_outliers": round((n_low + n_high) / len(df) * 100, 2),
            }
        )
    report = pd.DataFrame(rows).set_index("feature").round(2)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report.to_csv(REPORTS_DIR / "outlier_report.csv")
    return report


# ======================================================== Part C: bivariate
def _churn_rate_bars(ax: plt.Axes, table: pd.DataFrame, overall: float, title: str) -> None:
    labels = table.index.astype(str)
    ax.bar(labels, table["churn_rate_pct"], color=CHURNED_COLOR, width=0.65)
    ax.axhline(overall, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, zorder=1)
    crowded = len(table) > 6  # many bars: drop the n= line to avoid collisions
    for x, r, n in zip(labels, table["churn_rate_pct"], table["customers"], strict=False):
        text = f"{r:.0f}%" if crowded else f"{r:.1f}%\nn={n:,}"
        ax.text(
            x,
            r + 1,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED_TEXT,
            zorder=3,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )
    ax.set_ylim(0, min(115, table["churn_rate_pct"].max() * 1.3 + 5))
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def plot_churn_rate_by_category(df: pd.DataFrame) -> plt.Figure:
    """Churn-rate bar per level for each categorical / discrete feature."""
    overall = df[TARGET].mean() * 100
    cols = ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard", "Tenure"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, cols, strict=False):
        _churn_rate_bars(ax, churn_rate_table(df, col), overall, col)
    fig.suptitle(f"Churn rate by feature (dashed line = overall {overall:.1f}%)", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "churn_rate_by_category")


def plot_continuous_vs_churn(df: pd.DataFrame) -> plt.Figure:
    """Overlaid KDE and violin plot of each continuous feature split by Exited."""
    fig, axes = plt.subplots(len(CONTINUOUS_COLS), 2, figsize=(13, 14))
    plot_df = df.assign(Status=df[TARGET].map(CHURN_LABELS))
    status_palette = {CHURN_LABELS[k]: v for k, v in CHURN_PALETTE.items()}
    for (ax_k, ax_v), col in zip(axes, CONTINUOUS_COLS, strict=False):
        for k in (0, 1):
            sns.kdeplot(
                df.loc[df[TARGET] == k, col],
                ax=ax_k,
                fill=True,
                alpha=0.25,
                linewidth=2,
                color=CHURN_PALETTE[k],
                clip=(df[col].min(), df[col].max()),
                label=CHURN_LABELS[k],
            )
        ax_k.set_title(f"{col} — density by churn")
        ax_k.legend(loc="best")
        sns.violinplot(
            data=plot_df,
            x="Status",
            y=col,
            hue="Status",
            palette=status_palette,
            order=["Retained", "Churned"],
            inner="quartile",
            linewidth=1,
            cut=0,
            saturation=1,
            legend=False,
            ax=ax_v,
        )
        ax_v.set_title(f"{col} — violin by churn")
        ax_v.set_xlabel("")
    fig.tight_layout()
    return save_figure(fig, "continuous_vs_churn")


def plot_age_band_churn(df: pd.DataFrame) -> plt.Figure:
    """Churn rate per age band (18–29, 30–39, 40–49, 50–59, 60+)."""
    overall = df[TARGET].mean() * 100
    table = churn_rate_table(add_age_band(df), "AgeBand")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _churn_rate_bars(ax, table, overall, "Churn rate by age band")
    ax.set_xlabel("Age band")
    return save_figure(fig, "churn_rate_by_age_band")


def plot_balance_zero_churn(df: pd.DataFrame) -> plt.Figure:
    """Churn rate for Balance == 0 vs Balance > 0."""
    overall = df[TARGET].mean() * 100
    labelled = df.assign(BalanceGroup=np.where(df["Balance"] == 0, "Balance = 0", "Balance > 0"))
    table = churn_rate_table(labelled, "BalanceGroup")
    fig, ax = plt.subplots(figsize=(6, 4.8))
    _churn_rate_bars(ax, table, overall, "Churn rate: zero vs non-zero balance")
    return save_figure(fig, "churn_rate_balance_zero")


def statistical_tests(df: pd.DataFrame, alpha: float = 0.05, save: bool = True) -> pd.DataFrame:
    """Chi-square (categorical) and Mann-Whitney U (continuous) vs Exited."""
    rows = []
    for col in CATEGORICAL_FEATURES:
        ct = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        rows.append(
            {
                "feature": col,
                "test": "chi-square",
                "statistic": chi2,
                "dof": dof,
                "p_value": p,
                "effect_size": cramers_v,
                "effect_measure": "Cramer's V",
            }
        )
    for col in CONTINUOUS_COLS:
        churned = df.loc[df[TARGET] == 1, col]
        retained = df.loc[df[TARGET] == 0, col]
        u, p = stats.mannwhitneyu(churned, retained, alternative="two-sided")
        # Rank-biserial correlation: >0 means churners tend to have higher values.
        rbc = 2 * u / (len(churned) * len(retained)) - 1
        rows.append(
            {
                "feature": col,
                "test": "Mann-Whitney U",
                "statistic": u,
                "dof": np.nan,
                "p_value": p,
                "effect_size": rbc,
                "effect_measure": "rank-biserial r",
            }
        )
    table = pd.DataFrame(rows)
    table["significant"] = np.where(table["p_value"] < alpha, "yes", "no")
    table = table.sort_values("p_value").reset_index(drop=True)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORTS_DIR / "statistical_tests.csv", index=False)
    return table


# ===================================================== Part D: multivariate
def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = CONTINUOUS_COLS + DISCRETE_COLS + BINARY_COLS
    return df[cols]


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlation of numeric features + target (lower triangle)."""
    corr = df[CONTINUOUS_COLS + DISCRETE_COLS + BINARY_COLS + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=2,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )
    ax.grid(False)
    ax.set_title("Correlation heatmap — numeric features + Exited")
    return save_figure(fig, "correlation_heatmap")


def plot_churn_heatmap(df: pd.DataFrame, row: str, col: str) -> plt.Figure:
    """Churn-rate (%) heatmap of ``row`` × ``col`` with counts in each cell."""
    rate = (
        df.pivot_table(index=row, columns=col, values=TARGET, aggfunc="mean", observed=True) * 100
    )
    count = df.pivot_table(index=row, columns=col, values=TARGET, aggfunc="size", observed=True)
    annot = rate.round(1).astype(str) + "%\nn=" + count.astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(7, 1.2 * len(rate) + 1.5))
    sns.heatmap(
        rate,
        annot=annot,
        fmt="",
        cmap=CHURN_RATE_CMAP,
        vmin=0,
        vmax=100,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Churn rate (%)"},
        ax=ax,
    )
    ax.set_title(f"Churn rate — {row} × {col}")
    return save_figure(fig, f"churn_heatmap_{row.lower()}_x_{col.lower()}")


def plot_pairplot(df: pd.DataFrame, n: int = 2000) -> sns.PairGrid:
    """Pairplot of continuous features coloured by Exited (random sample)."""
    sample = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    sample = sample.assign(Status=sample[TARGET].map(CHURN_LABELS))
    grid = sns.pairplot(
        sample,
        vars=CONTINUOUS_COLS,
        hue="Status",
        hue_order=["Retained", "Churned"],
        palette={CHURN_LABELS[k]: v for k, v in CHURN_PALETTE.items()},
        plot_kws={"s": 10, "alpha": 0.5, "linewidth": 0},
        diag_kws={"common_norm": False, "linewidth": 1.5},
        corner=True,
        height=2.3,
    )
    grid.figure.suptitle(
        f"Pairplot of continuous features (sample of {len(sample):,})", y=1.02, fontweight="bold"
    )
    save_figure(grid.figure, "pairplot_continuous")
    return grid


def vif_table(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Variance Inflation Factor for each numeric feature (with intercept).

    VIF_i = 1 / (1 - R²_i), where R²_i comes from regressing feature i on all
    the other numeric features. Rule of thumb: > 5 moderate, > 10 high.
    """
    X = _numeric_frame(df).astype(float)
    rows = []
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        vif = np.inf if r2 >= 1 else 1 / (1 - r2)
        rows.append({"feature": col, "r_squared": r2, "vif": vif})
    table = pd.DataFrame(rows).sort_values("vif", ascending=False)
    table["concern"] = pd.cut(
        table["vif"], [0, 5, 10, np.inf], labels=["low", "moderate", "high"]
    ).astype(str)
    table = table.round(3).reset_index(drop=True)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORTS_DIR / "vif.csv", index=False)
    return table


# ================================================================ runner
def run_all(df: pd.DataFrame | None = None) -> dict:
    """Produce every table and figure. Returns the tables keyed by name."""
    df = load_raw() if df is None else df
    tables = {
        "summary": summary_table(df),
        "outliers": outlier_report(df),
        "stat_tests": statistical_tests(df),
        "vif": vif_table(df),
    }
    figures = [plot_target_distribution(df)]
    figures += [plot_continuous_distribution(df, c) for c in CONTINUOUS_COLS]
    figures += [
        plot_categorical_counts(df),
        plot_churn_rate_by_category(df),
        plot_continuous_vs_churn(df),
        plot_age_band_churn(df),
        plot_balance_zero_churn(df),
        plot_correlation_heatmap(df),
        plot_churn_heatmap(df, "Geography", "Gender"),
        plot_churn_heatmap(df, "NumOfProducts", "IsActiveMember"),
        plot_pairplot(df).figure,
    ]
    for fig in figures:
        plt.close(fig)
    print(f"Saved {len(figures)} figures to {FIGURES_DIR}")
    print(f"Saved feature_summary, outlier_report, statistical_tests, vif CSVs to {REPORTS_DIR}")
    return tables


if __name__ == "__main__":
    plt.switch_backend("Agg")  # headless: only write PNGs
    run_all()
