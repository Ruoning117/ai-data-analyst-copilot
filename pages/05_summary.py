"""Executive Summary page for the AI Data Analyst Copilot.

Aggregates outputs from data_quality, insight_engine, and anomaly_detector into
a single LLM prompt and renders a 3-paragraph business-ready narrative, a compact
metric table, and a copy-friendly code block.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.anomaly_detector import AnomalyDetectionResult, run_anomaly_detection
from core.data_loader import DatasetMeta, render_upload_widget
from core.data_quality import DataQualityReport, run_quality_checks
from core.insight_engine import Insight, run_insight_engine
from core.profiler import DataProfile, profile_dataset
from shared.llm_client import LLMClient
from shared.prompts import EXECUTIVE_SUMMARY_SYSTEM, executive_summary_full_user

_SUMMARY_MODEL = "claude-sonnet-4-20250514"

# ── Prompt assembly helpers ────────────────────────────────────────────────────

def _quality_bullets(report: DataQualityReport, meta: DatasetMeta) -> str:
    lines: list[str] = [f"Health score: {report.health_score}/100"]

    if report.missing.total_pct > 0:
        worst_col = max(
            report.missing.columns.items(),
            key=lambda kv: kv[1].pct,
        )
        lines.append(
            f"Missing data: {report.missing.total_pct:.1f}% of all cells; "
            f"worst column is '{worst_col[0]}' ({worst_col[1].pct:.1f}% missing, "
            f"{worst_col[1].pattern} pattern)"
        )
    else:
        lines.append("No missing values detected.")

    if report.duplicates.exact_count > 0:
        lines.append(
            f"Exact duplicates: {report.duplicates.exact_count:,} rows "
            f"({report.duplicates.exact_pct:.1f}% of dataset)"
        )
    else:
        lines.append("No exact duplicate rows found.")

    if report.type_violations:
        cols = ", ".join(f"'{c}'" for c in list(report.type_violations)[:3])
        lines.append(f"Type violations in {len(report.type_violations)} column(s): {cols}")

    if report.whitespace_issues:
        lines.append(
            f"Whitespace / casing inconsistencies in "
            f"{len(report.whitespace_issues)} column(s)"
        )

    if report.negative_value_flags:
        neg_cols = ", ".join(
            f"'{c}' ({v.pct:.1f}%)" for c, v in list(report.negative_value_flags.items())[:3]
        )
        lines.append(f"Unexpected negative values: {neg_cols}")

    return "\n".join(f"- {ln}" for ln in lines)


def _insight_bullets(insights: list[Insight]) -> str:
    if not insights:
        return "- No significant statistical patterns detected."
    lines = []
    for ins in insights[:6]:
        cols = ", ".join(f"'{c}'" for c in ins.columns)
        lines.append(
            f"- [{ins.severity.upper()}] {ins.finding_type.replace('_', ' ').title()} "
            f"in {cols} ({ins.statistic}): {ins.llm_explanation}"
        )
    return "\n".join(lines)


def _anomaly_bullets(result: AnomalyDetectionResult, df: pd.DataFrame) -> str:
    n_flagged = len(result.composite_scores)
    if n_flagged == 0:
        return "- No anomalies detected."

    pct = round(n_flagged / len(df) * 100, 1) if len(df) else 0
    lines = [f"- {n_flagged:,} rows flagged ({pct}% of dataset)"]

    for method, count in result.method_summary.items():
        label = {"iqr": "IQR", "zscore": "Z-Score", "isolation_forest": "Isolation Forest"}.get(method, method)
        lines.append(f"- {label}: {count:,} records")

    multi = sum(
        1 for idx in result.composite_scores
        if len({r.method_used for r in result.records if r.row_index == idx}) >= 2
    )
    if multi:
        lines.append(f"- {multi} rows flagged by 2+ methods (high confidence)")

    # Top anomaly
    if result.composite_scores:
        top_idx = max(result.composite_scores, key=result.composite_scores.__getitem__)
        top_score = result.composite_scores[top_idx]
        top_rec = next(
            (r for r in sorted(result.records, key=lambda r: -r.anomaly_score)
             if r.row_index == top_idx),
            None,
        )
        if top_rec:
            lines.append(
                f"- Highest-scoring anomaly: row {top_idx}, "
                f"driven by '{top_rec.primary_driver_feature}' "
                f"(composite score {top_score:.2f})"
            )

    return "\n".join(lines)


def _column_types_summary(profile: DataProfile) -> str:
    counts: dict[str, int] = profile.dataset_summary["column_type_counts"]
    parts = []
    for t in ("numeric", "categorical", "datetime", "id", "text"):
        if counts.get(t):
            parts.append(f"{counts[t]} {t}")
    return ", ".join(parts) if parts else "unknown"


# ── LLM call ──────────────────────────────────────────────────────────────────

def _generate_summary(
    df: pd.DataFrame,
    meta: DatasetMeta,
    report: DataQualityReport,
    insights: list[Insight],
    anomaly_result: AnomalyDetectionResult,
    profile: DataProfile,
) -> str:
    prompt = executive_summary_full_user(
        filename=meta.filename,
        row_count=meta.row_count,
        col_count=meta.col_count,
        column_types_summary=_column_types_summary(profile),
        memory_mb=profile.dataset_summary["memory_mb"],
        quality_bullets=_quality_bullets(report, meta),
        insight_bullets=_insight_bullets(insights),
        anomaly_bullets=_anomaly_bullets(anomaly_result, df),
    )
    return LLMClient().complete(
        prompt=prompt,
        system_prompt=EXECUTIVE_SUMMARY_SYSTEM,
        max_tokens=800,
        model=_SUMMARY_MODEL,
    )


# ── Compact metric table ───────────────────────────────────────────────────────

def _render_metric_table(
    report: DataQualityReport,
    insights: list[Insight],
    anomaly_result: AnomalyDetectionResult,
) -> None:
    top_insight = "None"
    if insights:
        ins = insights[0]
        cols = ", ".join(f"'{c}'" for c in ins.columns)
        top_insight = f"{ins.finding_type.replace('_', ' ').title()} in {cols} ({ins.statistic})"

    top_anomaly = "None"
    if anomaly_result.composite_scores:
        top_idx = max(anomaly_result.composite_scores, key=anomaly_result.composite_scores.__getitem__)
        top_score = anomaly_result.composite_scores[top_idx]
        top_rec = next(
            (r for r in sorted(anomaly_result.records, key=lambda r: -r.anomaly_score)
             if r.row_index == top_idx),
            None,
        )
        if top_rec:
            top_anomaly = (
                f"Row {top_idx} — '{top_rec.primary_driver_feature}' "
                f"(score {top_score:.2f}, {top_rec.severity} severity)"
            )

    health = report.health_score
    color = "#2e7d32" if health > 80 else "#e65100" if health >= 50 else "#b71c1c"
    label = "Good" if health > 80 else "Fair" if health >= 50 else "Poor"

    rows = [
        ("Health Score",    f"<span style='color:{color}; font-weight:700'>{health}/100 — {label}</span>"),
        ("Insight Count",   str(len(insights))),
        ("Anomalous Rows",  str(len(anomaly_result.composite_scores))),
        ("Top Insight",     top_insight),
        ("Top Anomaly",     top_anomaly),
    ]

    html_rows = "".join(
        f"<tr>"
        f"<td style='padding:6px 12px; font-weight:600; white-space:nowrap; "
        f"background:#f9f9f9; border-bottom:1px solid #eee'>{name}</td>"
        f"<td style='padding:6px 12px; border-bottom:1px solid #eee'>{value}</td>"
        f"</tr>"
        for name, value in rows
    )
    st.markdown(
        f"<table style='width:100%; border-collapse:collapse; font-size:0.88rem'>"
        f"{html_rows}</table>",
        unsafe_allow_html=True,
    )


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("Executive Summary")

df, meta = render_upload_widget()

if df is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

# Per-dataset cache keys — naturally invalidated when the file changes.
base_key      = f"summary_{meta.filename}_{meta.row_count}"
report_key    = f"{base_key}_report"
insights_key  = f"{base_key}_insights"
anomaly_key   = f"{base_key}_anomalies"
summary_key   = f"{base_key}_text"
gen_count_key = f"{base_key}_gen"

# ── Run (or reuse) the four analysis modules ──────────────────────────────────

# DataProfile: reuse what the Overview page already computed when possible.
if "data_profile" not in st.session_state:
    st.session_state["data_profile"] = profile_dataset(df, meta)
profile: DataProfile = st.session_state["data_profile"]

if report_key not in st.session_state:
    with st.spinner("Running data quality checks…"):
        st.session_state[report_key] = run_quality_checks(df, meta)
        # Keep in sync with the sidebar health score shown in app.py
        st.session_state["quality_report"] = st.session_state[report_key]

if insights_key not in st.session_state:
    with st.spinner("Running insight engine…"):
        st.session_state[insights_key] = run_insight_engine(df, meta)

if anomaly_key not in st.session_state:
    with st.spinner("Running anomaly detection…"):
        st.session_state[anomaly_key] = run_anomaly_detection(df, meta)

report: DataQualityReport              = st.session_state[report_key]
insights: list[Insight]                = st.session_state[insights_key]
anomaly_result: AnomalyDetectionResult = st.session_state[anomaly_key]

# ── Compact metrics table ─────────────────────────────────────────────────────

st.subheader("At a Glance")
_render_metric_table(report, insights, anomaly_result)
st.divider()

# ── LLM narrative ─────────────────────────────────────────────────────────────

st.subheader("Business Narrative")

# Regenerate button increments a counter, which busts the summary cache.
gen_count = st.session_state.get(gen_count_key, 0)
col_title, col_btn = st.columns([5, 1])
with col_btn:
    if st.button("Regenerate", key="summary_regen"):
        st.session_state[gen_count_key] = gen_count + 1
        gen_count += 1
        # Clear only the text so metrics don't re-run.
        st.session_state.pop(summary_key, None)

versioned_key = f"{summary_key}_v{gen_count}"

if versioned_key not in st.session_state:
    try:
        with st.spinner("Generating executive summary…"):
            st.session_state[versioned_key] = _generate_summary(
                df, meta, report, insights, anomaly_result, profile
            )
    except EnvironmentError as exc:
        st.error(str(exc))
        st.stop()

narrative: str = st.session_state[versioned_key]

# Render as formatted prose then in a code block for easy copying.
for para in narrative.strip().split("\n\n"):
    st.markdown(para.strip())

st.divider()
st.caption("Copy-ready plain text:")
st.code(narrative, language="text")
