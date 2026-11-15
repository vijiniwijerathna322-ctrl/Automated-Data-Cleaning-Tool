"""Plotly figures built from API profile data."""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def missing_values_chart(column_profiles: list[dict[str, Any]]) -> Figure | None:
    """Bar chart of missing values per column, or ``None`` when nothing is missing."""
    rows = [
        {"Column": c["name"], "Missing Values": c["missing"]}
        for c in column_profiles
        if c["missing"] > 0
    ]

    if not rows:
        return None

    fig = px.bar(
        pd.DataFrame(rows), x="Column", y="Missing Values", title="Missing Values by Column"
    )
    fig.update_layout(height=350)
    return fig


def histogram_chart(column_profile: dict[str, Any]) -> Figure | None:
    """Histogram from pre-binned counts, or ``None`` for a column without numbers."""
    hist = column_profile.get("histogram")

    if not hist:
        return None

    edges, counts = hist["bin_edges"], hist["counts"]
    frame = pd.DataFrame(
        {
            "Value": [(a + b) / 2 for a, b in itertools.pairwise(edges)],
            "Count": counts,
        }
    )
    fig = px.bar(frame, x="Value", y="Count", title=f"{column_profile['name']} Distribution")
    fig.update_layout(height=350, bargap=0.05)
    return fig
