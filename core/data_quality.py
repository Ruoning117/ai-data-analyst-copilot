from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from core.data_loader import DatasetMeta


# Column name keywords that imply non-negative values
_NON_NEGATIVE_KEYWORDS = {
    "age", "price", "cost", "amount", "quantity", "count",
    "salary", "revenue", "distance", "weight", "height",
    "duration", "rate", "score", "size",
}


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

@dataclass
class MissingDetail:
    count: int
    pct: float
    pattern: Literal["none", "scattered", "clustered"]


@dataclass
class MissingReport:
    total_pct: float
    columns: dict[str, MissingDetail] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

@dataclass
class NearDuplicateGroup:
    row_indices: list[int]
    id_column: str  # the ID column that differs between these rows


@dataclass
class DuplicateReport:
    exact_count: int
    exact_pct: float
    id_column: str | None  # None if no ID column detected
    near_duplicate_groups: list[NearDuplicateGroup] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Type violations
# ---------------------------------------------------------------------------

@dataclass
class TypeViolationDetail:
    violation_count: int
    sample_bad_values: list[str] = field(default_factory=list)  # up to 5 examples


# ---------------------------------------------------------------------------
# Cardinality anomalies
# ---------------------------------------------------------------------------

@dataclass
class CardinalityAnomaly:
    column: str
    kind: Literal["high_cardinality_categorical", "low_cardinality_numeric"]
    unique_count: int
    unique_pct: float


# ---------------------------------------------------------------------------
# Whitespace / casing
# ---------------------------------------------------------------------------

@dataclass
class WhitespaceDetail:
    has_leading_trailing: bool
    has_casing_variants: bool


# ---------------------------------------------------------------------------
# Negative value flags
# ---------------------------------------------------------------------------

@dataclass
class NegativeValueDetail:
    count: int
    pct: float


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    health_score: float                                       # 0–100
    missing: MissingReport
    duplicates: DuplicateReport
    type_violations: dict[str, TypeViolationDetail]          = field(default_factory=dict)
    cardinality_anomalies: list[CardinalityAnomaly]          = field(default_factory=list)
    whitespace_issues: dict[str, WhitespaceDetail]           = field(default_factory=dict)
    negative_value_flags: dict[str, NegativeValueDetail]     = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_missing(df: pd.DataFrame) -> MissingReport:
    n_cells = df.size
    total_missing = int(df.isnull().sum().sum())
    total_pct = round(total_missing / n_cells * 100, 2) if n_cells > 0 else 0.0

    columns: dict[str, MissingDetail] = {}
    for col in df.columns:
        col_missing = df[col].isnull()
        count = int(col_missing.sum())

        if count == 0:
            pattern: Literal["none", "scattered", "clustered"] = "none"
            pct = 0.0
        else:
            pct = round(count / len(df) * 100, 2)
            # Find contiguous runs of missing values in this column.
            # If the longest run exceeds 20% of all missing values → clustered.
            changes = col_missing.ne(col_missing.shift())
            run_ids = changes.cumsum()
            run_lengths = col_missing[col_missing].groupby(run_ids[col_missing]).transform("count")
            longest_run = int(run_lengths.max()) if len(run_lengths) > 0 else 0
            pattern = "clustered" if longest_run > 0.2 * count else "scattered"

        columns[col] = MissingDetail(count=count, pct=pct, pattern=pattern)

    return MissingReport(total_pct=total_pct, columns=columns)


def _check_duplicates(df: pd.DataFrame, meta: DatasetMeta) -> DuplicateReport:
    exact_count = int(df.duplicated().sum())
    exact_pct = round(exact_count / len(df) * 100, 2) if len(df) > 0 else 0.0

    # Use the pre-computed column_types from meta so ID detection is consistent
    # with the rest of the app (avoids the old "id" substring false-positive bug).
    id_column: str | None = next(
        (col for col, t in meta.column_types.items() if t == "id" and col in df.columns),
        None,
    )

    near_duplicate_groups: list[NearDuplicateGroup] = []

    if id_column is not None:
        non_id_cols = [col for col in df.columns if col != id_column]
        if non_id_cols:
            # Cap at 50,000 rows for performance.
            sample = df.head(50_000)
            grouped = sample.groupby(non_id_cols, dropna=False)
            for _, group in grouped:
                if len(group) >= 2:
                    near_duplicate_groups.append(
                        NearDuplicateGroup(
                            row_indices=group.index.tolist(),
                            id_column=id_column,
                        )
                    )

    return DuplicateReport(
        exact_count=exact_count,
        exact_pct=exact_pct,
        id_column=id_column,
        near_duplicate_groups=near_duplicate_groups,
    )


def _check_type_violations(
    df: pd.DataFrame, meta: DatasetMeta
) -> dict[str, TypeViolationDetail]:
    violations: dict[str, TypeViolationDetail] = {}

    for col, col_type in meta.column_types.items():
        if col not in df.columns:
            continue
        if df[col].dtype != object:
            continue  # already a proper numeric/datetime dtype, no violation possible

        if col_type == "numeric":
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad_mask = coerced.isnull() & df[col].notnull()
        elif col_type == "datetime":
            coerced = pd.to_datetime(df[col], errors="coerce")
            bad_mask = coerced.isnull() & df[col].notnull()
        else:
            continue

        count = int(bad_mask.sum())
        if count > 0:
            samples = df.loc[bad_mask, col].dropna().astype(str).unique()[:5].tolist()
            violations[col] = TypeViolationDetail(
                violation_count=count,
                sample_bad_values=samples,
            )

    return violations


def _check_cardinality(
    df: pd.DataFrame, meta: DatasetMeta
) -> list[CardinalityAnomaly]:
    if len(df) <= 50:
        return []

    anomalies: list[CardinalityAnomaly] = []
    n_rows = len(df)

    for col, col_type in meta.column_types.items():
        if col not in df.columns:
            continue

        unique_count = df[col].nunique()
        unique_pct = round(unique_count / n_rows * 100, 2)

        if col_type == "categorical" and unique_pct > 50:
            anomalies.append(CardinalityAnomaly(
                column=col,
                kind="high_cardinality_categorical",
                unique_count=unique_count,
                unique_pct=unique_pct,
            ))
        elif col_type == "numeric" and unique_count < 10:
            anomalies.append(CardinalityAnomaly(
                column=col,
                kind="low_cardinality_numeric",
                unique_count=unique_count,
                unique_pct=unique_pct,
            ))

    return anomalies


def _check_whitespace(df: pd.DataFrame) -> dict[str, WhitespaceDetail]:
    issues: dict[str, WhitespaceDetail] = {}

    for col in df.select_dtypes(include="object").columns:
        values = df[col].dropna().astype(str)
        if values.empty:
            continue

        has_leading_trailing = bool((values != values.str.strip()).any())

        # Casing variant: two distinct values that are equal after lowercasing.
        lowered = values.str.strip().str.lower()
        has_casing_variants = bool(lowered.duplicated(keep=False).any() and values.str.strip().duplicated(keep=False).eq(False).any())

        if has_leading_trailing or has_casing_variants:
            issues[col] = WhitespaceDetail(
                has_leading_trailing=has_leading_trailing,
                has_casing_variants=has_casing_variants,
            )

    return issues


def _check_negative_values(df: pd.DataFrame) -> dict[str, NegativeValueDetail]:
    flags: dict[str, NegativeValueDetail] = {}

    for col in df.select_dtypes(include="number").columns:
        col_lower = col.lower()
        if not any(kw in col_lower for kw in _NON_NEGATIVE_KEYWORDS):
            continue

        neg_mask = df[col] < 0
        count = int(neg_mask.sum())
        if count > 0:
            pct = round(count / df[col].notnull().sum() * 100, 2)
            flags[col] = NegativeValueDetail(count=count, pct=pct)

    return flags


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------

def _compute_health_score(
    missing: MissingReport,
    duplicates: DuplicateReport,
    type_violations: dict[str, TypeViolationDetail],
    cardinality_anomalies: list[CardinalityAnomaly],
    whitespace_issues: dict[str, WhitespaceDetail],
    negative_value_flags: dict[str, NegativeValueDetail],
    n_cols: int,
) -> float:
    # Missing: penalise proportionally to total missing cell percentage.
    missing_score = max(0.0, 100.0 - missing.total_pct * 3)

    # Duplicates: penalise proportionally to duplicate row percentage.
    duplicate_score = max(0.0, 100.0 - duplicates.exact_pct * 5)

    # Type violations: penalise per column with violations.
    if n_cols > 0:
        type_score = 100.0 * (1 - len(type_violations) / n_cols)
    else:
        type_score = 100.0

    # Other: each category with any issues deducts 10 points.
    other_score = 100.0
    if cardinality_anomalies:
        other_score -= 10.0
    if whitespace_issues:
        other_score -= 10.0
    if negative_value_flags:
        other_score -= 10.0
    other_score = max(0.0, other_score)

    health = (
        0.30 * missing_score
        + 0.20 * duplicate_score
        + 0.20 * type_score
        + 0.30 * other_score
    )
    return round(health, 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame, meta: DatasetMeta) -> DataQualityReport:
    missing = _check_missing(df)
    duplicates = _check_duplicates(df, meta)
    type_violations = _check_type_violations(df, meta)
    cardinality_anomalies = _check_cardinality(df, meta)
    whitespace_issues = _check_whitespace(df)
    negative_value_flags = _check_negative_values(df)

    health_score = _compute_health_score(
        missing=missing,
        duplicates=duplicates,
        type_violations=type_violations,
        cardinality_anomalies=cardinality_anomalies,
        whitespace_issues=whitespace_issues,
        negative_value_flags=negative_value_flags,
        n_cols=len(df.columns),
    )

    return DataQualityReport(
        health_score=health_score,
        missing=missing,
        duplicates=duplicates,
        type_violations=type_violations,
        cardinality_anomalies=cardinality_anomalies,
        whitespace_issues=whitespace_issues,
        negative_value_flags=negative_value_flags,
    )
