"""Anomaly Detection page for the AI Data Analyst Copilot.

Shows a summary panel, sortable anomaly table, method explainers, a Plotly
scatter with highlighted anomalies, a column-level summary, and a settings
sidebar with sliders that trigger re-detection automatically.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.anomaly_detector import (
    AnomalyDetectionResult,
    run_anomaly_detection,
)
from core.data_loader import DatasetMeta, render_upload_widget

# ── Constants ─────────────────────────────────────────────────────────────────

_SEVERITY_COLORS = {"high": "#b71c1c", "moderate": "#e65100", "low": "#1565c0"}
_SEVERITY_EMOJI  = {"high": "🔴", "moderate": "🟠", "low": "🔵"}

_METHOD_LABELS = {
    "iqr":              "IQR",
    "zscore":           "Z-Score",
    "isolation_forest": "Isolation Forest",
}

_METHOD_EXPLAINERS = {
    "IQR (Interquartile Range)": (
        "**What it detects:** Values that fall far outside the middle 50 % of a "
        "column's distribution — specifically beyond Q1 − 1.5×IQR or "
        "Q3 + 1.5×IQR (Tukey fences).\n\n"
        "**When it's most useful:** Any numeric column, regardless of distribution "
        "shape. It is robust to skewed data and does not assume normality. It works "
        "column-by-column, so it catches univariate outliers only.\n\n"
        "**Severity:** determined by how many IQR units the value sits beyond "
        "the fence — *low* (1.5–2×), *moderate* (2–3×), *high* (3×+)."
    ),
    "Z-Score": (
        "**What it detects:** Values more than *N* standard deviations away from "
        "the column mean, where N is set by the Z-score threshold slider.\n\n"
        "**When it's most useful:** Columns whose values are approximately normally "
        "distributed. This page applies a normality test (scipy `normaltest`) first "
        "and skips non-normal columns — so you may see fewer rows flagged than with "
        "IQR.\n\n"
        "**Severity:** *moderate* for |z| in [threshold, threshold+2), "
        "*high* for |z| ≥ threshold+2."
    ),
    "Isolation Forest": (
        "**What it detects:** Rows that are anomalous across *multiple* columns "
        "simultaneously — multivariate outliers that may not look extreme in any "
        "single column alone.\n\n"
        "**When it's most useful:** Wide datasets with several correlated numeric "
        "features. It is tree-based and handles non-linear relationships.\n\n"
        "**Contamination slider:** tells the algorithm what fraction of the dataset "
        "to treat as anomalies. Increase it if you suspect more outliers; decrease "
        "it if you're getting too many false positives.\n\n"
        "**Severity:** *moderate* below score 0.7, *high* at 0.7+."
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _severity_badge(severity: str) -> str:
    color = _SEVERITY_COLORS.get(severity, "#888")
    emoji = _SEVERITY_EMOJI.get(severity, "")
    return (
        f"<span style='background:{color}; color:white; font-size:0.72rem; "
        f"padding:2px 8px; border-radius:10px; font-weight:600'>"
        f"{emoji} {severity.capitalize()}</span>"
    )


def _score_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#b71c1c" if score >= 0.7 else "#e65100" if score >= 0.4 else "#1565c0"
    return (
        f"<div style='display:flex; align-items:center; gap:6px'>"
        f"<div style='flex:1; background:#eee; border-radius:4px; height:8px'>"
        f"<div style='width:{pct}%; background:{color}; height:8px; border-radius:4px'></div>"
        f"</div>"
        f"<span style='font-size:0.75rem; color:#555; min-width:2.5rem'>{score:.2f}</span>"
        f"</div>"
    )


def _build_anomaly_table(result: AnomalyDetectionResult) -> pd.DataFrame:
    if not result.records:
        return pd.DataFrame()

    rows = []
    for r in result.records:
        rows.append({
            "Row": r.row_index,
            "Column": r.column,
            "Method": _METHOD_LABELS.get(r.method_used, r.method_used),
            "Severity": r.severity,
            "Score": round(result.composite_scores.get(r.row_index, r.anomaly_score), 3),
            "Raw Value": r.raw_value,
            "Expected Range": r.expected_range,
        })

    return pd.DataFrame(rows)


def _numeric_cols_in_result(result: AnomalyDetectionResult) -> list[str]:
    cols = []
    seen: set[str] = set()
    for r in result.records:
        if r.primary_driver_feature not in seen:
            cols.append(r.primary_driver_feature)
            seen.add(r.primary_driver_feature)
    return cols


# ── Summary panel ─────────────────────────────────────────────────────────────

def _render_summary(result: AnomalyDetectionResult, df: pd.DataFrame) -> None:
    total_rows_flagged = len(result.composite_scores)
    multi_method = sum(
        1 for idx in result.composite_scores
        if len({r.method_used for r in result.records if r.row_index == idx}) >= 2
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomaly Records", len(result.records))
    c2.metric("Rows Flagged", total_rows_flagged)
    c3.metric("Multi-method Rows", multi_method,
              help="Rows flagged by 2 or more detection methods — higher confidence")
    pct = round(total_rows_flagged / len(df) * 100, 1) if len(df) else 0
    c4.metric("% of Dataset", f"{pct}%")

    st.markdown("**Records by method**")
    method_cols = st.columns(3)
    for i, (method, label) in enumerate(_METHOD_LABELS.items()):
        count = result.method_summary.get(method, 0)
        method_cols[i].metric(label, count)


# ── Anomaly table ─────────────────────────────────────────────────────────────

def _render_anomaly_table(result: AnomalyDetectionResult) -> None:
    st.subheader("Anomaly Records")

    df_tbl = _build_anomaly_table(result)
    if df_tbl.empty:
        st.info("No anomalies found with the current settings.")
        return

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        method_filter = st.multiselect(
            "Filter by method",
            options=list(_METHOD_LABELS.values()),
            default=list(_METHOD_LABELS.values()),
            key="anom_method_filter",
        )
    with col2:
        severity_filter = st.multiselect(
            "Filter by severity",
            options=["high", "moderate", "low"],
            default=["high", "moderate", "low"],
            key="anom_severity_filter",
        )
    with col3:
        sort_col = st.selectbox(
            "Sort by",
            options=["Score", "Row", "Severity", "Method"],
            key="anom_sort_col",
        )

    filtered = df_tbl[
        df_tbl["Method"].isin(method_filter) &
        df_tbl["Severity"].isin(severity_filter)
    ]

    severity_order = {"high": 0, "moderate": 1, "low": 2}
    if sort_col == "Severity":
        filtered = filtered.assign(
            _sev_ord=filtered["Severity"].map(severity_order)
        ).sort_values("_sev_ord").drop(columns=["_sev_ord"])
    else:
        filtered = filtered.sort_values(sort_col, ascending=(sort_col != "Score"))

    # Render with HTML badges and score bars
    html_rows = []
    for _, row in filtered.head(500).iterrows():
        html_rows.append(
            f"<tr>"
            f"<td style='padding:4px 8px'>{int(row['Row'])}</td>"
            f"<td style='padding:4px 8px'><code>{row['Column']}</code></td>"
            f"<td style='padding:4px 8px'>{row['Method']}</td>"
            f"<td style='padding:4px 8px'>{_severity_badge(row['Severity'])}</td>"
            f"<td style='padding:4px 8px; min-width:140px'>{_score_bar(row['Score'])}</td>"
            f"<td style='padding:4px 8px; font-family:monospace'>{row['Raw Value']:.4g}</td>"
            f"<td style='padding:4px 8px; font-size:0.8rem; color:#555'>{row['Expected Range']}</td>"
            f"</tr>"
        )

    header = (
        "<thead><tr style='background:#f5f5f5; font-size:0.8rem; text-transform:uppercase; "
        "letter-spacing:0.05em'>"
        "<th style='padding:6px 8px; text-align:left'>Row</th>"
        "<th style='padding:6px 8px; text-align:left'>Column</th>"
        "<th style='padding:6px 8px; text-align:left'>Method</th>"
        "<th style='padding:6px 8px; text-align:left'>Severity</th>"
        "<th style='padding:6px 8px; text-align:left'>Score</th>"
        "<th style='padding:6px 8px; text-align:left'>Raw Value</th>"
        "<th style='padding:6px 8px; text-align:left'>Expected Range</th>"
        "</tr></thead>"
    )
    table_html = (
        f"<div style='overflow-x:auto'>"
        f"<table style='width:100%; border-collapse:collapse; font-size:0.85rem'>"
        f"{header}<tbody>{''.join(html_rows)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    if len(filtered) > 500:
        st.caption(f"Showing first 500 of {len(filtered):,} records.")


# ── Method explainers ─────────────────────────────────────────────────────────

def _render_explainers() -> None:
    st.subheader("How Each Method Works")
    for title, body in _METHOD_EXPLAINERS.items():
        with st.expander(title):
            st.markdown(body)


# ── Scatter plot ───────────────────────────────────────────────────────────────

def _render_scatter(
    df: pd.DataFrame,
    result: AnomalyDetectionResult,
    meta: DatasetMeta,
) -> None:
    st.subheader("Scatter Plot with Anomaly Highlights")

    numeric_cols = [
        col for col, t in meta.column_types.items()
        if t == "numeric" and col in df.columns
    ]

    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns to draw a scatter plot.")
        return

    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X axis", numeric_cols, key="scatter_x")
    remaining = [c for c in numeric_cols if c != x_col]
    y_col = c2.selectbox("Y axis", remaining, key="scatter_y")

    flagged_rows = set(result.composite_scores.keys())
    label = df.index.map(lambda i: "Anomaly" if i in flagged_rows else "Normal")

    plot_df = df[[x_col, y_col]].copy()
    plot_df["_label"] = label.values
    plot_df["_score"] = [
        round(result.composite_scores.get(i, 0.0), 3) for i in df.index
    ]

    color_map = {"Normal": "#90caf9", "Anomaly": "#b71c1c"}
    size_map  = {"Normal": 5, "Anomaly": 9}

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="_label",
        color_discrete_map=color_map,
        size=[size_map[lbl] for lbl in plot_df["_label"]],
        size_max=12,
        hover_data={"_score": True, "_label": False},
        labels={"_label": "Status", "_score": "Anomaly Score"},
        title=f"{x_col} vs {y_col}",
    )
    fig.update_layout(
        legend_title_text="",
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Column-level summary ───────────────────────────────────────────────────────

def _render_column_summary(result: AnomalyDetectionResult) -> None:
    st.subheader("Column-Level Anomaly Summary")

    if not result.records:
        st.info("No anomalies detected.")
        return

    from collections import defaultdict
    col_data: dict[str, dict] = defaultdict(lambda: {"count": 0, "methods": set()})
    for r in result.records:
        col_data[r.column]["count"] += 1
        col_data[r.column]["methods"].add(_METHOD_LABELS.get(r.method_used, r.method_used))

    rows = [
        {
            "Column": col,
            "Anomaly Records": info["count"],
            "Methods": ", ".join(sorted(info["methods"])),
        }
        for col, info in sorted(col_data.items(), key=lambda kv: -kv[1]["count"])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("Anomaly Detection")

df, meta = render_upload_widget()

if df is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

# ── Settings sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Detection Settings")
    contamination = st.slider(
        "Contamination (Isolation Forest)",
        min_value=0.01,
        max_value=0.15,
        value=0.05,
        step=0.01,
        help="Expected fraction of anomalies in the dataset.",
        key="anom_contamination",
    )
    zscore_threshold = st.slider(
        "Z-Score Threshold",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        help="Minimum |z| to flag a value as anomalous.",
        key="anom_zscore",
    )
    st.caption(
        "Changing either slider automatically re-runs detection on the current dataset."
    )

# ── Run / cache detection ─────────────────────────────────────────────────────

cache_key = (
    f"anomalies_{meta.filename}_{meta.row_count}"
    f"_{contamination:.2f}_{zscore_threshold:.1f}"
)

if cache_key not in st.session_state:
    with st.spinner("Running anomaly detection…"):
        st.session_state[cache_key] = run_anomaly_detection(
            df, meta,
            contamination=contamination,
            zscore_threshold=zscore_threshold,
        )

result: AnomalyDetectionResult = st.session_state[cache_key]

# ── Render sections ───────────────────────────────────────────────────────────

_render_summary(result, df)
st.divider()
_render_anomaly_table(result)
st.divider()
_render_explainers()
st.divider()
_render_scatter(df, result, meta)
st.divider()
_render_column_summary(result)
