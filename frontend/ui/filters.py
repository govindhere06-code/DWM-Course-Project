"""Sidebar filters shared by the Overview and EDA Explorer pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

ACTIVE_OPTIONS = {"All": None, "Active only": 1, "Inactive only": 0}


def _defaults(df: pd.DataFrame) -> dict:
    return {
        "f_geo": sorted(df["Geography"].unique()),
        "f_gender": sorted(df["Gender"].unique()),
        "f_age": (int(df["Age"].min()), int(df["Age"].max())),
        "f_active": "All",
    }


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render the filters in the sidebar and return the filtered frame.

    Widget keys live in st.session_state, so selections persist across pages.
    """
    defaults = _defaults(df)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.markdown("### Filters")
        st.multiselect("Geography", defaults["f_geo"], key="f_geo")
        st.multiselect("Gender", defaults["f_gender"], key="f_gender")
        st.slider("Age range", *defaults["f_age"], key="f_age")
        st.radio("Activity status", list(ACTIVE_OPTIONS), key="f_active", horizontal=True)
        if st.button("Reset filters", width="stretch"):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.rerun()

    lo, hi = st.session_state["f_age"]
    mask = (
        df["Geography"].isin(st.session_state["f_geo"])
        & df["Gender"].isin(st.session_state["f_gender"])
        & df["Age"].between(lo, hi)
    )
    active = ACTIVE_OPTIONS[st.session_state["f_active"]]
    if active is not None:
        mask &= df["IsActiveMember"] == active
    filtered = df[mask]

    st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} customers")
    return filtered


def require_rows(df: pd.DataFrame) -> bool:
    """Show a friendly message (instead of empty charts) when filters match nothing."""
    if df.empty:
        st.info(
            "No customers match the current filters. Widen the selection in the sidebar "
            "or press **Reset filters**."
        )
        return False
    return True
