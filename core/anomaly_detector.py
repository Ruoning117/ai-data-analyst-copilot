"""Row-level anomaly detection using three complementary methods.

Methods
-------
iqr              : Tukey fences on each numeric column individually.
zscore           : Z-score threshold, only on columns that pass a normality test.
isolation_forest : Multivariate detection across all eligible numeric columns.

Each flagged row yields an AnomalyRecord.  A composite score per row rewards
rows that are flagged by more than one method.

No Streamlit code in this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import normaltest
from sklearn.ensemble import IsolationForest

from core.data_loader import DatasetMeta

# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_ROWS: int = 10          # minimum non-null rows required to run any detector
_IF_MAX_MISSING: float = 0.20  # drop a column from IF if > this fraction is missing
_IF_CONTAMINATION: float = 0.05
_IF_RANDOM_STATE: int = 42

# IQR severity thresholds (k = distance from fence in units of IQR)
_IQR_MODERATE_K: float = 2.0
_IQR_HIGH_K: float = 3.0

# Z-score severity thresholds
_ZSCORE_FLAG: float = 3.0
_ZSCORE_HIGH: float = 5.0

# Isolation Forest severity threshold on the 0-1 score
_IF_HIGH_SCORE: float = 0.7

# Multi-method consensus bonus added to the composite score per extra method
_CONSENSUS_BONUS: float = 0.15


# ── Public types ──────────────────────────────────────────────────────────────

@dataclass
class AnomalyRecord:
    row_index: int
    column: str                                          # column name, or "multivariate"
    method_used: Literal["iqr", "zscore", "isolation_forest"]
    anomaly_score: float                                 # 0–1; higher = more anomalous
    severity: Literal["low", "moderate", "high"]
    raw_value: float                                     # flagged value (IF: primary driver's value)
    expected_range: str                                  # human-readable normal range
    primary_driver_feature: str                          # column most responsible for anomaly


@dataclass
class AnomalyDetectionResult:
    records: list[AnomalyRecord]
    composite_scores: dict[int, float]   # row_index → composite score (0–1)
    method_summary: dict[str, int]       # method_used → number of records produced


# ── IQR detection ─────────────────────────────────────────────────────────────

def _iqr_severity(k: float) -> Literal["low", "moderate", "high"]:
    if k >= _IQR_HIGH_K:
        return "high"
    if k >= _IQR_MODERATE_K:
        return "moderate"
    return "low"


def _detect_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> list[AnomalyRecord]:
    """Tukey fence method.  Flag values outside Q1 ± 1.5*IQR or Q3 ± 1.5*IQR.

    Severity is determined by how many IQR units beyond the fence the value sits:
      low      : k ∈ [1.5, 2.0)
      moderate : k ∈ [2.0, 3.0)
      high     : k ≥ 3.0
    """
    records: list[AnomalyRecord] = []

    for col in numeric_cols:
        non_null = df[col].dropna()
        if len(non_null) < _MIN_ROWS:
            continue

        q1 = float(non_null.quantile(0.25))
        q3 = float(non_null.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0.0:
            continue  # all non-null values are identical; no outliers possible

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        range_str = f"[{lower_fence:.4g}, {upper_fence:.4g}]"

        notna = df[col].notna()
        below = notna & (df[col] < lower_fence)
        above = notna & (df[col] > upper_fence)
        flagged = df.index[below | above]

        for idx in flagged:
            value = float(df.at[idx, col])
            k = (q1 - value) / iqr if value < lower_fence else (value - q3) / iqr
            score = float(min(1.0, (k - 1.5) / 3.5))   # 0 at k=1.5, 1 at k=5.0
            records.append(AnomalyRecord(
                row_index=int(idx),
                column=col,
                method_used="iqr",
                anomaly_score=score,
                severity=_iqr_severity(k),
                raw_value=value,
                expected_range=range_str,
                primary_driver_feature=col,
            ))

    return records


# ── Z-score detection ─────────────────────────────────────────────────────────

def _zscore_severity(abs_z: float) -> Literal["low", "moderate", "high"]:
    return "high" if abs_z >= _ZSCORE_HIGH else "moderate"


def _detect_zscore(
    df: pd.DataFrame,
    numeric_cols: list[str],
    zscore_threshold: float = _ZSCORE_FLAG,
) -> list[AnomalyRecord]:
    """Z-score method, applied only to columns consistent with normality.

    scipy.stats.normaltest is used as a gate (p > 0.05 means we fail to reject
    normality, so z-scores are meaningful).  Flags |z| > zscore_threshold.

    Severity:
      moderate : |z| ∈ [threshold, threshold+2)
      high     : |z| ≥ threshold+2
    """
    records: list[AnomalyRecord] = []

    for col in numeric_cols:
        non_null = df[col].dropna()
        if len(non_null) < _MIN_ROWS:
            continue

        try:
            _, p_value = normaltest(non_null.values)
        except Exception:
            continue
        if p_value <= 0.05:
            continue  # distribution is non-normal; z-score inappropriate here

        mean = float(non_null.mean())
        std = float(non_null.std())
        if std == 0.0:
            continue

        lower = mean - zscore_threshold * std
        upper = mean + zscore_threshold * std
        range_str = f"[{lower:.4g}, {upper:.4g}]  (μ ± {zscore_threshold:.1f}σ)"

        # Compute absolute z-score for every non-null row.
        abs_z_series = ((df[col] - mean) / std).abs()
        flagged = df.index[df[col].notna() & (abs_z_series > zscore_threshold)]

        for idx in flagged:
            abs_z = float(abs_z_series.at[idx])
            score = float(min(1.0, (abs_z - zscore_threshold) / 4.0))
            high_threshold = zscore_threshold + 2.0
            severity: Literal["low", "moderate", "high"] = (
                "high" if abs_z >= high_threshold else "moderate"
            )
            records.append(AnomalyRecord(
                row_index=int(idx),
                column=col,
                method_used="zscore",
                anomaly_score=score,
                severity=severity,
                raw_value=float(df.at[idx, col]),
                expected_range=range_str,
                primary_driver_feature=col,
            ))

    return records


# ── Isolation Forest detection ────────────────────────────────────────────────

def _relative_diff(mean_flagged: float, mean_normal: float) -> float:
    """|a − b| / (|b| + ε) — relative mean shift for a feature."""
    return abs(mean_flagged - mean_normal) / (abs(mean_normal) + 1e-9)


def _detect_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: list[str],
    contamination: float = _IF_CONTAMINATION,
) -> list[AnomalyRecord]:
    """Multivariate anomaly detection via sklearn IsolationForest.

    Columns with > _IF_MAX_MISSING fraction of missing values are excluded.
    Remaining NaNs are imputed with the column median before training.

    Primary driver: the eligible feature whose mean shifts most (relative to its
    own scale) between flagged and non-flagged rows.

    Severity:
      moderate : normalised anomaly score < 0.7
      high     : normalised anomaly score ≥ 0.7
    """
    eligible = [
        col for col in numeric_cols
        if df[col].isnull().mean() <= _IF_MAX_MISSING
    ]
    if len(eligible) < 2 or len(df) < _MIN_ROWS:
        return []

    # Build the feature matrix; impute remaining NaNs with column medians.
    X_df = df[eligible].copy()
    X_df = X_df.fillna(X_df.median())
    X = X_df.to_numpy(dtype=float)

    clf = IsolationForest(
        contamination=contamination,
        random_state=_IF_RANDOM_STATE,
    )
    predictions = clf.fit_predict(X)   # 1 = inlier, -1 = outlier

    flagged_mask = predictions == -1
    if not flagged_mask.any():
        return []

    # Normalise score_samples to 0–1 (1 = most anomalous).
    # score_samples returns lower values for more anomalous points.
    raw_scores = clf.score_samples(X)
    score_min, score_max = raw_scores.min(), raw_scores.max()
    scores_01 = (score_max - raw_scores) / (score_max - score_min + 1e-10)

    # Identify the primary driver: the feature with the largest relative mean
    # difference between flagged and non-flagged rows.
    normal_mask = ~flagged_mask
    primary_driver = eligible[0]
    max_rel = 0.0
    for col_pos, col_name in enumerate(eligible):
        rel = _relative_diff(
            X[flagged_mask, col_pos].mean(),
            X[normal_mask, col_pos].mean(),
        )
        if rel > max_rel:
            max_rel = rel
            primary_driver = col_name

    # Expected range expressed as the primary driver's ±3σ interval.
    pd_vals = X_df[primary_driver]
    pd_mean, pd_std = pd_vals.mean(), pd_vals.std()
    range_str = (
        f"[{pd_mean - 3 * pd_std:.4g}, {pd_mean + 3 * pd_std:.4g}]"
        f"  (primary driver: {primary_driver})"
    )

    records: list[AnomalyRecord] = []
    for arr_pos, orig_idx in enumerate(df.index):
        if not flagged_mask[arr_pos]:
            continue

        score_01 = float(scores_01[arr_pos])
        severity: Literal["low", "moderate", "high"] = (
            "high" if score_01 >= _IF_HIGH_SCORE else "moderate"
        )
        records.append(AnomalyRecord(
            row_index=int(orig_idx),
            column="multivariate",
            method_used="isolation_forest",
            anomaly_score=score_01,
            severity=severity,
            raw_value=float(X_df.at[orig_idx, primary_driver]),
            expected_range=range_str,
            primary_driver_feature=primary_driver,
        ))

    return records


# ── Composite scoring ─────────────────────────────────────────────────────────

def _compute_composite_scores(records: list[AnomalyRecord]) -> dict[int, float]:
    """Aggregate per-row anomaly scores across all methods.

    Strategy
    --------
    1. For each (row, method) pair take the *maximum* score across all records
       from that method — prevents double-counting when one method flags the
       same row via multiple columns.
    2. Take the best single-method score as the base.
    3. Add _CONSENSUS_BONUS for each additional method beyond the first.
       A row confirmed by 2 methods will always score higher than the same
       row confirmed by only 1.
    4. Cap at 1.0.
    """
    by_row: dict[int, list[AnomalyRecord]] = defaultdict(list)
    for r in records:
        by_row[r.row_index].append(r)

    composite: dict[int, float] = {}
    for row_idx, row_records in by_row.items():
        method_best: dict[str, float] = {}
        for r in row_records:
            method_best[r.method_used] = max(
                method_best.get(r.method_used, 0.0), r.anomaly_score
            )
        best_score = max(method_best.values())
        n_methods = len(method_best)
        composite[row_idx] = min(1.0, best_score + _CONSENSUS_BONUS * (n_methods - 1))

    return composite


# ── Public entry point ────────────────────────────────────────────────────────

def run_anomaly_detection(
    df: pd.DataFrame,
    meta: DatasetMeta,
    *,
    contamination: float = _IF_CONTAMINATION,
    zscore_threshold: float = _ZSCORE_FLAG,
) -> AnomalyDetectionResult:
    """Run IQR, Z-score, and Isolation Forest detection on all numeric columns.

    Parameters
    ----------
    contamination : float
        Fraction of outliers expected (Isolation Forest ``contamination`` param).
    zscore_threshold : float
        Absolute z-score above which a value is considered anomalous.

    Returns an AnomalyDetectionResult containing:
      records          — one AnomalyRecord per (row, method, column) triple
      composite_scores — per-row score rewarding multi-method agreement
      method_summary   — count of records produced by each method
    """
    numeric_cols = [
        col
        for col, col_type in meta.column_types.items()
        if col_type == "numeric" and col in df.columns
    ]

    if not numeric_cols:
        return AnomalyDetectionResult(
            records=[], composite_scores={}, method_summary={}
        )

    records: list[AnomalyRecord] = []
    records.extend(_detect_iqr(df, numeric_cols))
    records.extend(_detect_zscore(df, numeric_cols, zscore_threshold=zscore_threshold))
    records.extend(_detect_isolation_forest(df, numeric_cols, contamination=contamination))

    composite_scores = _compute_composite_scores(records)

    method_summary: dict[str, int] = defaultdict(int)
    for r in records:
        method_summary[r.method_used] += 1

    return AnomalyDetectionResult(
        records=records,
        composite_scores=composite_scores,
        method_summary=dict(method_summary),
    )
