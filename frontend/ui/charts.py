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
DIVERGING = [[0.0, RETAINED], [0.25, "#a9c7ee"], [0.5, "#efeee9"], [0.75, "#f6b48f"],
             [1.0, CHURNED]]


def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      hoverlabel=dict(font_size=13), **kwargs)
    return fig


def churn_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["Exited"]
    out = pd.DataFrame({"customers": g.size(), "churned": g.sum(),
                        "churn_rate": g.mean() * 100}).reset_index()
    out[col] = out[col].astype(str)
    return out


def churn_rate_bar(df: pd.DataFrame, col: str, overall: float | None = None,
                   title: str | None = None, height: int = 340) -> go.Figure:
    """Churn rate (%) per level of ``col``; counts in the hover and bar labels."""
    t = churn_rate_table(df, col)
    fig = px.bar(t, x=col, y="churn_rate", text=t["churn_rate"].map("{:.1f}%".format),
                 custom_data=["customers", "churned"],
                 title=title or f"Churn rate by {col}")
    fig.update_traces(marker_color=CHURNED, textposition="outside", cliponaxis=False,
                      hovertemplate=(f"<b>{col}: %{{x}}</b><br>Churn rate: %{{y:.1f}}%"
                                     "<br>Churned: %{customdata[1]:,} of %{customdata[0]:,}"
                                     "<extra></extra>"))
    if overall is not None:
        fig.add_hline(y=overall, line_dash="dash", line_color=NEUTRAL, line_width=1)
    fig.update_yaxes(title="Churn rate (%)", range=[0, max(t["churn_rate"].max() * 1.2, 5)])
    fig.update_xaxes(title=None, type="category")
    return _layout(fig, height)


def count_by_status(df: pd.DataFrame, col: str, height: int = 340) -> go.Figure:
    """Customer counts per level of ``col``, stacked by churn status."""
    t = (df.groupby([col, "Status"], observed=True).size().rename("customers")
         .reset_index())
    t[col] = t[col].astype(str)
    fig = px.bar(t, x=col, y="customers", color="Status", color_discrete_map=STATUS_COLORS,
                 category_orders={"Status": STATUS_ORDER},
                 title=f"Customers by {col} and churn status")
    fig.update_traces(hovertemplate=f"<b>{col}: %{{x}}</b><br>%{{fullData.name}}: "
                                    "%{y:,}<extra></extra>")
    fig.update_xaxes(title=None, type="category")
    fig.update_yaxes(title="Customers")
    return _layout(fig, height, bargap=0.25)


def histogram_by_status(df: pd.DataFrame, col: str, normalise: bool = True,
                        bins: int = 40, height: int = 380) -> go.Figure:
    """Overlaid histograms of ``col`` for retained vs churned customers."""
    fig = px.histogram(df, x=col, color="Status", color_discrete_map=STATUS_COLORS,
                       category_orders={"Status": STATUS_ORDER}, nbins=bins,
                       barmode="overlay", opacity=0.6,
                       histnorm="percent" if normalise else None,
                       title=f"Distribution of {col} by churn status")
    fig.update_yaxes(title="% of group" if normalise else "Customers")
    fig.update_traces(marker_line_width=0)
    return _layout(fig, height)


def box_by_status(df: pd.DataFrame, col: str, height: int = 380) -> go.Figure:
    fig = px.box(df, x="Status", y=col, color="Status", color_discrete_map=STATUS_COLORS,
                 category_orders={"Status": STATUS_ORDER}, points=False,
                 title=f"{col} spread by churn status")
    fig.update_xaxes(title=None)
    return _layout(fig, height, showlegend=False)


def correlation_heatmap(df: pd.DataFrame, cols: list[str], method: str = "pearson",
                        height: int = 520) -> go.Figure:
    corr = df[cols].corr(method=method)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=DIVERGING, zmin=-1, zmax=1,
                    aspect="auto", title=f"{method.capitalize()} correlation")
    fig.update_traces(hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                      xgap=2, ygap=2)
    return _layout(fig, height, coloraxis_colorbar=dict(title="r"))


def scatter_by_status(df: pd.DataFrame, x: str, y: str, max_points: int = 3000,
                      height: int = 480) -> go.Figure:
    sample = df.sample(min(len(df), max_points), random_state=42) if len(df) > max_points else df
    fig = px.scatter(sample, x=x, y=y, color="Status", color_discrete_map=STATUS_COLORS,
                     category_orders={"Status": STATUS_ORDER}, opacity=0.55,
                     hover_data={"Geography": True, "Gender": True, "Age": True,
                                 "NumOfProducts": True, "IsActiveMember": True},
                     title=f"{y} vs {x}"
                           + (f" (random sample of {max_points:,})" if len(df) > max_points else ""))
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    return _layout(fig, height)
