"""Dataset profiler — pure pandas, no Streamlit, no LLM calls.

Computes per-column descriptive statistics and dataset-level summaries,
returning a DataProfile that can be consumed by any page or module.

Usage
-----
from core.profiler import profile_dataset, get_column_profile

data_profile = profile_dataset(df, meta)
col_stats    = get_column_profile(data_profile, "price")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.data_loader import DatasetMeta


# ── Public types ───────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """Per-column statistics.

    ``col_type`` mirrors the value in ``DatasetMeta.column_types``
    (one of ``"numeric"``, ``"categorical"``, ``"datetime"``, ``"text"``, ``"id"``).

    ``stats`` is a flat dict whose keys depend on ``col_type``; see the
    individual ``_profile_*`` functions for the full key list.
    """
    col_type: str
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProfile:
    """Full profile for a loaded dataset.

    Attributes
    ----------
    dataset_summary : dict
        Dataset-level metrics (rows, memory, type breakdown, …).
    column_profiles : dict[str, ColumnProfile]
        One entry per column, keyed by the original column name.
    """
    dataset_summary: dict[str, Any]
    column_profiles: dict[str, ColumnProfile]


# ── Internal per-type profilers ────────────────────────────────────────────────

def _profile_numeric(series: pd.Series) -> dict[str, Any]:
    """Stats for a numeric column."""
    non_null = series.dropna()
    n_total  = len(series)
    n_null   = int(series.isnull().sum())
    n        = len(non_null)

    if n == 0:
        return {
            "count": 0, "null_count": n_null,
            "mean": None, "median": None, "std": None,
            "min": None, "max": None, "p25": None, "p75": None,
            "skewness": None, "kurtosis": None,
            "zero_count": 0, "negative_count": 0,
        }

    desc = non_null.describe(percentiles=[0.25, 0.75])

    return {
        "count":          n,
        "null_count":     n_null,
        "mean":           round(float(desc["mean"]),   4),
        "median":         round(float(non_null.median()), 4),
        "std":            round(float(desc["std"]),    4),
        "min":            round(float(desc["min"]),    4),
        "max":            round(float(desc["max"]),    4),
        "p25":            round(float(desc["25%"]),    4),
        "p75":            round(float(desc["75%"]),    4),
        "skewness":       round(float(non_null.skew()),  4),
        "kurtosis":       round(float(non_null.kurt()),  4),
        "zero_count":     int((non_null == 0).sum()),
        "negative_count": int((non_null < 0).sum()),
    }


def _profile_categorical(series: pd.Series) -> dict[str, Any]:
    """Stats for a categorical column."""
    non_null = series.dropna()
    n        = len(non_null)
    n_null   = int(series.isnull().sum())

    value_counts = non_null.value_counts(dropna=True)
    top5 = [
        {
            "value": str(val),
            "count": int(cnt),
            "pct":   round(cnt / n * 100, 2) if n else 0.0,
        }
        for val, cnt in value_counts.head(5).items()
    ]

    mode_val = str(value_counts.index[0]) if not value_counts.empty else None

    return {
        "count":        n,
        "null_count":   n_null,
        "unique_count": int(non_null.nunique()),
        "top_values":   top5,
        "mode":         mode_val,
    }


def _profile_datetime(series: pd.Series) -> dict[str, Any]:
    """Stats for a datetime column.

    Handles both native datetime64 dtype and object/string columns whose
    name contains 'date' or 'time' (as classified by ``_infer_column_types``).
    """
    n_null = int(series.isnull().sum())

    # Coerce to datetime; already-datetime columns pass through unchanged.
    parsed = pd.to_datetime(series, errors="coerce")
    non_null = parsed.dropna()
    n = len(non_null)

    if n == 0:
        return {
            "count": 0, "null_count": n_null,
            "min_date": None, "max_date": None,
            "date_range_days": None,
            "most_common_year": None, "most_common_month": None,
        }

    min_date = non_null.min()
    max_date = non_null.max()
    date_range_days = int((max_date - min_date).days)

    year_counts  = non_null.dt.year.value_counts()
    month_counts = non_null.dt.month.value_counts()

    return {
        "count":             n,
        "null_count":        n_null,
        "min_date":          min_date.isoformat(),
        "max_date":          max_date.isoformat(),
        "date_range_days":   date_range_days,
        "most_common_year":  int(year_counts.index[0]) if not year_counts.empty else None,
        "most_common_month": int(month_counts.index[0]) if not month_counts.empty else None,
    }


def _profile_text(series: pd.Series) -> dict[str, Any]:
    """Stats for a free-text column (avg char length etc.)."""
    non_null = series.dropna().astype(str)
    n        = len(non_null)
    n_null   = int(series.isnull().sum())

    if n == 0:
        return {
            "count": 0, "null_count": n_null, "unique_count": 0,
            "avg_char_length": None, "min_char_length": None, "max_char_length": None,
        }

    char_lengths = non_null.str.len()

    return {
        "count":           n,
        "null_count":      n_null,
        "unique_count":    int(non_null.nunique()),
        "avg_char_length": round(float(char_lengths.mean()), 1),
        "min_char_length": int(char_lengths.min()),
        "max_char_length": int(char_lengths.max()),
    }


def _profile_id(series: pd.Series) -> dict[str, Any]:
    """Stats for an ID column."""
    non_null = series.dropna()
    n        = len(non_null)
    n_null   = int(series.isnull().sum())
    n_unique = int(non_null.nunique())

    return {
        "count":            n,
        "null_count":       n_null,
        "unique_count":     n_unique,
        "uniqueness_ratio": round(n_unique / n, 4) if n else 0.0,
    }


# ── Dispatch table ─────────────────────────────────────────────────────────────

_PROFILERS = {
    "numeric":     _profile_numeric,
    "categorical": _profile_categorical,
    "datetime":    _profile_datetime,
    "text":        _profile_text,
    "id":          _profile_id,
}


# ── Dataset-level summary ──────────────────────────────────────────────────────

def _dataset_summary(df: pd.DataFrame, meta: DatasetMeta) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    total_cells    = n_rows * n_cols
    total_missing  = int(df.isnull().sum().sum())

    memory_bytes   = df.memory_usage(deep=True).sum()
    memory_mb      = round(memory_bytes / 1024 ** 2, 3)

    type_counts: dict[str, int] = {}
    for col_type in ("numeric", "categorical", "datetime", "text", "id"):
        type_counts[col_type] = sum(
            1 for t in meta.column_types.values() if t == col_type
        )

    duplicate_count = int(df.duplicated().sum())

    return {
        "total_rows":         n_rows,
        "total_columns":      n_cols,
        "total_missing_cells": total_missing,
        "missing_pct":        round(total_missing / total_cells * 100, 2) if total_cells else 0.0,
        "memory_mb":          memory_mb,
        "column_type_counts": type_counts,
        "duplicate_row_count": duplicate_count,
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, meta: DatasetMeta) -> DataProfile:
    """Compute a full DataProfile for *df*.

    Parameters
    ----------
    df   : The loaded DataFrame.
    meta : DatasetMeta produced by ``data_loader.render_upload_widget``.

    Returns
    -------
    DataProfile
        Contains ``dataset_summary`` (dict) and ``column_profiles``
        (dict mapping column name → ColumnProfile).
    """
    summary = _dataset_summary(df, meta)

    column_profiles: dict[str, ColumnProfile] = {}
    for col, col_type in meta.column_types.items():
        if col not in df.columns:
            continue
        profiler = _PROFILERS.get(col_type)
        if profiler is None:
            continue
        stats = profiler(df[col])
        column_profiles[col] = ColumnProfile(col_type=col_type, stats=stats)

    return DataProfile(dataset_summary=summary, column_profiles=column_profiles)


# ── Column accessor ────────────────────────────────────────────────────────────

def get_column_profile(profile: DataProfile, col: str) -> ColumnProfile | None:
    """Return the ColumnProfile for *col*, or None if not found.

    Parameters
    ----------
    profile : DataProfile returned by ``profile_dataset``.
    col     : Exact column name (case-sensitive, as it appears in the DataFrame).
    """
    return profile.column_profiles.get(col)
