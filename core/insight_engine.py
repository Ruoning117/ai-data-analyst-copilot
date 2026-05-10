from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from core.data_loader import DatasetMeta
from core.profiler import DataProfile, profile_dataset
from shared.llm_client import LLMClient
from shared.prompts import INSIGHT_ENRICHMENT_SYSTEM, insight_enrichment_user


# ---------------------------------------------------------------------------
# Public output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Insight:
    columns: list[str]
    finding_type: str
    statistic: str
    llm_explanation: str                     # 2 sentences: what + why it matters
    severity: Literal["high", "medium", "low"]
    recommended_action: str                  # 1 sentence


# ---------------------------------------------------------------------------
# Internal: raw finding before LLM enrichment
# ---------------------------------------------------------------------------

@dataclass
class _RawFinding:
    columns: list[str]
    finding_type: str
    statistic: str
    severity: Literal["high", "medium", "low"]
    context: dict                            # structured dict sent to the LLM


# ---------------------------------------------------------------------------
# Statistical detections (pandas only, no LLM)
# ---------------------------------------------------------------------------

def _detect_skew(meta: DatasetMeta, profile: DataProfile) -> list[_RawFinding]:
    findings = []
    numeric_cols = [c for c, t in meta.column_types.items() if t == "numeric"]
    for col in numeric_cols:
        cp = profile.column_profiles.get(col)
        if cp is None or cp.stats["count"] < 10:
            continue
        skew = cp.stats["skewness"]
        if skew is None or abs(skew) <= 2:
            continue
        severity: Literal["high", "medium", "low"] = "high" if abs(skew) > 5 else "medium"
        findings.append(_RawFinding(
            columns=[col],
            finding_type="distribution_skew",
            statistic=f"skewness={skew:.2f}",
            severity=severity,
            context={
                "column": col,
                "skewness": round(skew, 2),
                "direction": (
                    "right-skewed (long tail of high values)"
                    if skew > 0 else
                    "left-skewed (long tail of low values)"
                ),
                "mean": round(cp.stats["mean"], 2),
                "median": round(cp.stats["median"], 2),
            },
        ))
    return findings


def _detect_high_correlations(df: pd.DataFrame) -> list[_RawFinding]:
    findings = []
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return findings

    corr = numeric_df.corr()
    cols = corr.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            r = float(corr.loc[col_a, col_b])
            if abs(r) <= 0.85:
                continue
            severity = "high" if abs(r) > 0.95 else "medium"
            findings.append(_RawFinding(
                columns=[col_a, col_b],
                finding_type="high_correlation",
                statistic=f"r={r:.2f}",
                severity=severity,
                context={
                    "column_a": col_a,
                    "column_b": col_b,
                    "pearson_r": round(r, 2),
                    "direction": "positive" if r > 0 else "negative",
                },
            ))
    return findings


def _detect_outlier_prevalence(df: pd.DataFrame, profile: DataProfile) -> list[_RawFinding]:
    findings = []
    for col, cp in profile.column_profiles.items():
        if cp.col_type != "numeric":
            continue
        mean = cp.stats["mean"]
        std  = cp.stats["std"]
        if cp.stats["count"] < 10 or mean is None or std is None or std == 0:
            continue
        data = df[col].dropna()
        outlier_pct = float(((data - mean).abs() > 3 * std).sum() / len(data) * 100)
        if outlier_pct <= 5:
            continue
        severity = "high" if outlier_pct > 15 else "medium"
        findings.append(_RawFinding(
            columns=[col],
            finding_type="outlier_prevalence",
            statistic=f"{outlier_pct:.1f}% outliers (±3σ)",
            severity=severity,
            context={
                "column": col,
                "outlier_pct": round(outlier_pct, 1),
                "mean": round(mean, 2),
                "std": round(std, 2),
                "bounds": f"{round(mean - 3 * std, 2)} to {round(mean + 3 * std, 2)}",
            },
        ))
    return findings


def _detect_categorical_dominance(
    meta: DatasetMeta, profile: DataProfile
) -> list[_RawFinding]:
    findings = []
    cat_cols = [c for c, t in meta.column_types.items() if t == "categorical"]
    for col in cat_cols:
        cp = profile.column_profiles.get(col)
        if cp is None:
            continue
        top_values = cp.stats["top_values"]
        if not top_values:
            continue
        top_value = top_values[0]["value"]
        top_pct   = top_values[0]["pct"]
        if top_pct <= 70:
            continue
        severity = "high" if top_pct > 90 else "medium"
        findings.append(_RawFinding(
            columns=[col],
            finding_type="categorical_dominance",
            statistic=f"'{top_value}' in {top_pct:.1f}% of rows",
            severity=severity,
            context={
                "column": col,
                "dominant_value": top_value,
                "dominant_pct": round(top_pct, 1),
                "unique_value_count": cp.stats["unique_count"],
            },
        ))
    return findings


def _fmt_timedelta(td: pd.Timedelta) -> str:
    secs = td.total_seconds()
    if secs < 7200:
        return f"{secs:.0f} seconds"
    elif secs < 172_800:
        return f"{secs / 3600:.1f} hours"
    return f"{td.days} days"


def _detect_date_gaps(
    df: pd.DataFrame, meta: DatasetMeta, profile: DataProfile
) -> list[_RawFinding]:
    findings = []
    date_cols = [c for c, t in meta.column_types.items()
                 if t == "datetime" and c in df.columns]
    for col in date_cols:
        cp = profile.column_profiles.get(col)
        if cp is None or cp.stats["count"] < 10:
            continue
        series = pd.to_datetime(df[col], errors="coerce").dropna().sort_values()
        gaps = series.diff().dropna()
        mean_gap = gaps.mean()
        std_gap = gaps.std()
        if mean_gap.total_seconds() == 0:
            continue
        cv = std_gap.total_seconds() / mean_gap.total_seconds()
        if cv <= 0.5:
            continue
        severity = "high" if cv > 2 else "medium"
        findings.append(_RawFinding(
            columns=[col],
            finding_type="date_gaps",
            statistic=f"gap CV={cv:.2f}",
            severity=severity,
            context={
                "column": col,
                "mean_gap": _fmt_timedelta(mean_gap),
                "std_gap": _fmt_timedelta(std_gap),
                "coefficient_of_variation": round(cv, 2),
                "note": "High CV means the time between records is irregular.",
            },
        ))
    return findings


# Pairs where column_a should be ≤ column_b
_CONTRADICTION_PAIRS = [
    ("start", "end"),
    ("begin", "end"),
    ("from", "to"),
    ("min", "max"),
    ("low", "high"),
    ("open", "close"),
]


def _detect_contradictions(df: pd.DataFrame) -> list[_RawFinding]:
    findings = []
    numeric_df = df.select_dtypes(include="number")
    # Map lowercased column name → original name
    cols_lower = {c.lower(): c for c in numeric_df.columns}

    for kw_a, kw_b in _CONTRADICTION_PAIRS:
        col_a = next((cols_lower[c] for c in cols_lower if kw_a in c), None)
        col_b = next((cols_lower[c] for c in cols_lower if kw_b in c), None)
        if col_a is None or col_b is None or col_a == col_b:
            continue

        valid = numeric_df[[col_a, col_b]].dropna()
        violations = int((valid[col_a] > valid[col_b]).sum())
        if violations == 0:
            continue

        violation_pct = round(violations / len(valid) * 100, 1)
        severity = "high" if violation_pct > 10 else "medium"
        findings.append(_RawFinding(
            columns=[col_a, col_b],
            finding_type="cross_column_contradiction",
            statistic=f"{violations} rows where {col_a} > {col_b}",
            severity=severity,
            context={
                "column_a": col_a,
                "column_b": col_b,
                "rule_violated": f"{col_a} should be ≤ {col_b}",
                "violation_count": violations,
                "violation_pct": violation_pct,
            },
        ))
    return findings


# ---------------------------------------------------------------------------
# LLM enrichment — one API call for all findings
# ---------------------------------------------------------------------------

class _ExplanationItem(BaseModel):
    explanation: str        # 2 sentences: what happened + why it matters
    recommended_action: str # 1 sentence


class _LLMResponse(BaseModel):
    items: list[_ExplanationItem]


def _enrich_with_llm(findings: list[_RawFinding]) -> list[Insight]:
    findings_text = "\n\n".join(
        f"Finding {i + 1}:\n{json.dumps(f.context, indent=2)}"
        for i, f in enumerate(findings)
    )

    parsed: _LLMResponse = LLMClient().complete_structured(
        prompt=insight_enrichment_user(findings_text, len(findings)),
        output_format=_LLMResponse,
        system_prompt=INSIGHT_ENRICHMENT_SYSTEM,
        max_tokens=4096,
    )

    llm_items = parsed.items

    # Safety: if the LLM returns fewer items than expected, pad with placeholders.
    while len(llm_items) < len(findings):
        llm_items.append(_ExplanationItem(
            explanation="No explanation available.",
            recommended_action="Review this finding manually.",
        ))

    return [
        Insight(
            columns=f.columns,
            finding_type=f.finding_type,
            statistic=f.statistic,
            llm_explanation=llm_items[i].explanation,
            severity=f.severity,
            recommended_action=llm_items[i].recommended_action,
        )
        for i, f in enumerate(findings)
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_insight_engine(df: pd.DataFrame, meta: DatasetMeta) -> list[Insight]:
    """Run all statistical detections, then enrich findings with Claude."""
    profile = profile_dataset(df, meta)

    raw: list[_RawFinding] = []
    raw.extend(_detect_skew(meta, profile))
    raw.extend(_detect_high_correlations(df))
    raw.extend(_detect_outlier_prevalence(df, profile))
    raw.extend(_detect_categorical_dominance(meta, profile))
    raw.extend(_detect_date_gaps(df, meta, profile))
    raw.extend(_detect_contradictions(df))

    if not raw:
        return []

    return _enrich_with_llm(raw)
