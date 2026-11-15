"""Helpers for showing API table previews."""

from __future__ import annotations

from typing import Any

import pandas as pd


def preview_to_frame(preview: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(preview["rows"], columns=preview["columns"])


def preview_caption(preview: dict[str, Any]) -> str | None:
    if preview["truncated"]:
        return f"Showing the first {len(preview['rows']):,} of {preview['total_rows']:,} rows."

    return None


def report_table(report: dict[str, Any]) -> pd.DataFrame:
    before, after = report["before"], report["after"]
    metrics = [
        ("Original Rows", before["rows"]),
        ("Final Rows", after["rows"]),
        ("Original Columns", before["columns"]),
        ("Final Columns", after["columns"]),
        ("Missing Values Before", before["missing_values"]),
        ("Missing Values After", after["missing_values"]),
        ("Missing Values Fixed", report["missing_fixed"]),
        ("Duplicates Before", before["duplicate_rows"]),
        ("Duplicates Removed", report["duplicates_removed"]),
        ("Potential Outlier Columns", len(report["outliers"])),
        ("Features Created", len(report["created_features"])),
        ("Encoded Columns", len(report["encoded_columns"])),
        ("Scaled Columns", len(report["scaled_columns"])),
        ("Columns Left Out of ML Dataset", len(report["dropped_from_ml"])),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value"])
