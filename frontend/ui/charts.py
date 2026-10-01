"""Plotly chart builders with consistent churn colours."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RETAINED = "#2a78d6"
CHURNED = "#eb6834"
NEUTRAL = "#8a8984"
STATUS_COLORS = {"Retained": RETAINED, "Churned": CHURNED}
STATUS_ORDER = ["Retained", "Churned"]
# Diverging blue <- grey -> orange for correlations (neutral midpoint).
DIVERGING = [
    [0.0, RETAINED],
    [0.25, "#a9c7ee"],
    [0.5, "#efeee9"],
    [0.75, "#f6b48f"],
    [1.0, CHURNED],
]


def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=40, t=50, b=10),  # r: room for the vertical toolbar
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        # vertical toolbar down the right edge, so it never covers the title on narrow charts
        modebar=dict(orientation="v"),
        hoverlabel=dict(font_size=13),
        **kwargs,
    )
    return fig


def churn_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["Exited"]
    out = pd.DataFrame(
        {"customers": g.size(), "churned": g.sum(), "churn_rate": g.mean() * 100}
    ).reset_index()
    out[col] = out[col].astype(str)
    return out


def churn_rate_bar(
    df: pd.DataFrame,
    col: str,
    overall: float | None = None,
    title: str | None = None,
    height: int = 340,
) -> go.Figure:
    """Churn rate (%) per level of ``col``; counts in the hover and bar labels."""
    t = churn_rate_table(df, col)
    fig = px.bar(
        t,
        x=col,
        y="churn_rate",
        text=t["churn_rate"].map("{:.1f}%".format),
        custom_data=["customers", "churned"],
        title=title or f"Churn rate by {col}",
    )
    fig.update_traces(
        marker_color=CHURNED,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"<b>{col}: %{{x}}</b><br>Churn rate: %{{y:.1f}}%"
            "<br>Churned: %{customdata[1]:,} of %{customdata[0]:,}"
            "<extra></extra>"
        ),
    )
    if overall is not None:
        fig.add_hline(y=overall, line_dash="dash", line_color=NEUTRAL, line_width=1)
    fig.update_yaxes(title="Churn rate (%)", range=[0, max(t["churn_rate"].max() * 1.2, 5)])
    fig.update_xaxes(title=None, type="category")
    return _layout(fig, height)


def count_by_status(df: pd.DataFrame, col: str, height: int = 340) -> go.Figure:
    """Customer counts per level of ``col``, stacked by churn status."""
    t = df.groupby([col, "Status"], observed=True).size().rename("customers").reset_index()
    t[col] = t[col].astype(str)
    fig = px.bar(
        t,
        x=col,
        y="customers",
        color="Status",
        color_discrete_map=STATUS_COLORS,
        category_orders={"Status": STATUS_ORDER},
        title=f"Customers by {col} and churn status",
    )
    fig.update_traces(
        hovertemplate=f"<b>{col}: %{{x}}</b><br>%{{fullData.name}}: " "%{y:,}<extra></extra>"
    )
    fig.update_xaxes(title=None, type="category")
    fig.update_yaxes(title="Customers")
    return _layout(fig, height, bargap=0.25)


def histogram_by_status(
    df: pd.DataFrame, col: str, normalise: bool = True, bins: int = 40, height: int = 380
) -> go.Figure:
    """Overlaid histograms of ``col`` for retained vs churned customers."""
    fig = px.histogram(
        df,
        x=col,
        color="Status",
        color_discrete_map=STATUS_COLORS,
        category_orders={"Status": STATUS_ORDER},
        nbins=bins,
        barmode="overlay",
        opacity=0.6,
        histnorm="percent" if normalise else None,
        title=f"Distribution of {col} by churn status",
    )
    fig.update_yaxes(title="% of group" if normalise else "Customers")
    fig.update_traces(marker_line_width=0)
    return _layout(fig, height)


def box_by_status(df: pd.DataFrame, col: str, height: int = 380) -> go.Figure:
    fig = px.box(
        df,
        x="Status",
        y=col,
        color="Status",
        color_discrete_map=STATUS_COLORS,
        category_orders={"Status": STATUS_ORDER},
        points=False,
        title=f"{col} spread by churn status",
    )
    fig.update_xaxes(title=None)
    return _layout(fig, height, showlegend=False)


def correlation_heatmap(
    df: pd.DataFrame, cols: list[str], method: str = "pearson", height: int = 520
) -> go.Figure:
    corr = df[cols].corr(method=method)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=DIVERGING,
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"{method.capitalize()} correlation",
    )
    fig.update_traces(hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>", xgap=2, ygap=2)
    return _layout(fig, height, coloraxis_colorbar=dict(title="r"))


def scatter_by_status(
    df: pd.DataFrame, x: str, y: str, max_points: int = 3000, height: int = 480
) -> go.Figure:
    sample = df.sample(min(len(df), max_points), random_state=42) if len(df) > max_points else df
    fig = px.scatter(
        sample,
        x=x,
        y=y,
        color="Status",
        color_discrete_map=STATUS_COLORS,
        category_orders={"Status": STATUS_ORDER},
        opacity=0.55,
        hover_data={
            "Geography": True,
            "Gender": True,
            "Age": True,
            "NumOfProducts": True,
            "IsActiveMember": True,
        },
        title=f"{y} vs {x}"
        + (f" (random sample of {max_points:,})" if len(df) > max_points else ""),
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    return _layout(fig, height)


# ------------------------------------------------------------ model charts
FRIENDLY_NAMES = {
    "IsActiveMember": "Active member",
    "HasCrCard": "Has credit card",
    "NumOfProducts": "Number of products",
    "EstimatedSalary": "Estimated salary",
    "CreditScore": "Credit score",
}
RISK_COLORS = {"Low": "#f6c9ad", "Medium": "#f08a55", "High": "#a8401a"}  # one hue, light -> dark
RISK_ORDER = ["Low", "Medium", "High"]


def confusion_matrix_fig(tn: int, fp: int, fn: int, tp: int, height: int = 360) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    total = [[tn + fp] * 2, [fn + tp] * 2]
    text = [
        [f"<b>{v:,}</b><br>{v / t:.0%}" if t else "0" for v, t in zip(zr, tr, strict=False)]
        for zr, tr in zip(z, total, strict=False)
    ]
    names = [["True negatives", "False positives"], ["False negatives", "True positives"]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Retained", "Churned"],
            y=["Retained", "Churned"],
            text=text,
            texttemplate="%{text}",
            customdata=names,
            hovertemplate="%{customdata}: %{z:,}<extra></extra>",
            colorscale=[[0, "#fdf1ea"], [1, CHURNED]],
            showscale=False,
            xgap=3,
            ygap=3,
            textfont=dict(size=15),
        )
    )
    fig.update_yaxes(autorange="reversed", title="Actual")
    fig.update_xaxes(title="Predicted", side="bottom")
    return _layout(fig, height, title="Confusion matrix — test set (% of actual class)")


def roc_pr_figs(y_true, proba, threshold: float, height: int = 380) -> tuple[go.Figure, go.Figure]:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    fpr, tpr, roc_t = roc_curve(y_true, proba)
    roc = go.Figure()
    roc.add_scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        line=dict(color=RETAINED, width=2),
        name=f"Model (AUC {roc_auc_score(y_true, proba):.3f})",
        customdata=roc_t,
        hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<br>threshold %{customdata:.2f}"
        "<extra></extra>",
    )
    roc.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random (AUC 0.500)",
        line=dict(color=NEUTRAL, dash="dash", width=1),
        hoverinfo="skip",
    )
    pred = proba >= threshold
    pos, neg = (y_true == 1), (y_true == 0)
    roc.add_scatter(
        x=[(pred & neg).sum() / neg.sum()],
        y=[(pred & pos).sum() / pos.sum()],
        mode="markers",
        name=f"Threshold {threshold:.2f}",
        marker=dict(color=CHURNED, size=12, line=dict(color="white", width=2)),
    )
    roc.update_xaxes(title="False positive rate", range=[0, 1])
    roc.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    _layout(roc, height, title="ROC curve — test set")
    roc.update_layout(
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(0,0,0,0)",
        )
    )

    prec, rec, pr_t = precision_recall_curve(y_true, proba)
    pr = go.Figure()
    pr.add_scatter(
        x=rec[:-1],
        y=prec[:-1],
        mode="lines",
        line=dict(color=RETAINED, width=2),
        name=f"Model (AP {average_precision_score(y_true, proba):.3f})",
        customdata=pr_t,
        hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<br>threshold "
        "%{customdata:.2f}<extra></extra>",
    )
    base = float(y_true.mean())
    pr.add_scatter(
        x=[0, 1],
        y=[base, base],
        mode="lines",
        name=f"No skill (AP {base:.3f})",
        line=dict(color=NEUTRAL, dash="dash", width=1),
        hoverinfo="skip",
    )
    tp = (pred & pos).sum()
    pr.add_scatter(
        x=[tp / pos.sum()],
        y=[tp / max(pred.sum(), 1)],
        mode="markers",
        name=f"Threshold {threshold:.2f}",
        marker=dict(color=CHURNED, size=12, line=dict(color="white", width=2)),
    )
    pr.update_xaxes(title="Recall", range=[0, 1])
    pr.update_yaxes(title="Precision", range=[0, 1.02])
    _layout(pr, height, title="Precision-Recall curve — test set")
    pr.update_layout(
        legend=dict(
            orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(0,0,0,0)"
        )
    )
    return roc, pr


def importance_bar(
    df: pd.DataFrame, value: str, error: str | None, label: str, title: str, height: int = 400
) -> go.Figure:
    d = df.sort_values(value)
    fig = go.Figure(
        go.Bar(
            x=d[value],
            y=d["feature"],
            orientation="h",
            marker_color=RETAINED,
            error_x=(
                dict(type="data", array=d[error], color=NEUTRAL, thickness=1) if error else None
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title=label)
    return _layout(fig, height, title=title)


def gauge(
    probability: float, threshold: float, low: float, high: float, height: int = 280
) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(suffix="%", valueformat=".1f"),
            gauge=dict(
                axis=dict(range=[0, 100], ticksuffix="%"),
                bar=dict(color="#3b3a37", thickness=0.28),
                steps=[
                    dict(range=[0, low * 100], color=RISK_COLORS["Low"]),
                    dict(range=[low * 100, high * 100], color=RISK_COLORS["Medium"]),
                    dict(range=[high * 100, 100], color=RISK_COLORS["High"]),
                ],
                threshold=dict(
                    line=dict(color=RETAINED, width=4), thickness=0.85, value=threshold * 100
                ),
            ),
            title=dict(text="Churn probability"),
        )
    )
    fig.update_layout(
        height=height, margin=dict(l=30, r=40, t=60, b=10), modebar=dict(orientation="v")
    )
    return fig


def waterfall(
    contributions: pd.Series, base_value: float, max_items: int = 10, height: int = 440
) -> go.Figure:
    """SHAP waterfall (log-odds) of grouped contributions; remainder folded into 'Other'."""
    contributions = contributions.rename(lambda n: FRIENDLY_NAMES.get(n, n))
    top = contributions.head(max_items)
    rest = contributions.iloc[max_items:].sum()
    items = list(top.items()) + (
        [("Other features", rest)] if len(contributions) > max_items else []
    )
    items = items[::-1]  # largest at the top of a horizontal waterfall
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            base=base_value,
            y=[name for name, _ in items],
            x=[v for _, v in items],
            measure=["relative"] * len(items),
            text=[f"{v:+.2f}" for _, v in items],
            textposition="outside",
            increasing=dict(marker=dict(color=CHURNED)),
            decreasing=dict(marker=dict(color=RETAINED)),
            connector=dict(line=dict(color=NEUTRAL, width=1, dash="dot")),
            hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>",
        )
    )
    fig.add_vline(x=base_value, line_color=NEUTRAL, line_dash="dash", line_width=1)
    fig.update_xaxes(title="Contribution to churn log-odds (orange ↑ risk, blue ↓ risk)")
    return _layout(fig, height, title="Why this prediction? (SHAP)", showlegend=False)


def risk_band_bar(bands: pd.Series, height: int = 320) -> go.Figure:
    counts = bands.value_counts().reindex(RISK_ORDER, fill_value=0)
    pct = counts / max(counts.sum(), 1) * 100
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[RISK_COLORS[b] for b in counts.index],
            text=[f"{n:,} ({p:.0f}%)" for n, p in zip(counts.values, pct.values, strict=False)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:,} customers<extra></extra>",
        )
    )
    fig.update_yaxes(title="Customers", range=[0, max(counts.max() * 1.2, 1)])
    return _layout(fig, height, title="Customers per risk band")
