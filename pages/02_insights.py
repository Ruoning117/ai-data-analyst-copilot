from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import DatasetMeta, render_upload_widget
from core.insight_engine import Insight, run_insight_engine
from shared.llm_client import LLMClient
from shared.prompts import executive_summary_user

st.title("Analyst Insights")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = ["high", "medium", "low"]

_BADGE_STYLE: dict[str, str] = {
    "high":   "background:#fdecea; color:#b71c1c; border:1px solid #ef9a9a;",
    "medium": "background:#fff8e1; color:#e65100; border:1px solid #ffcc80;",
    "low":    "background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7;",
}

_FINDING_LABELS: dict[str, str] = {
    "distribution_skew":          "Distribution Skew",
    "high_correlation":           "High Correlation",
    "outlier_prevalence":         "Outlier Prevalence",
    "categorical_dominance":      "Categorical Dominance",
    "date_gaps":                  "Date Gaps",
    "cross_column_contradiction": "Cross-Column Contradiction",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_badge(severity: str) -> str:
    style = _BADGE_STYLE.get(severity, "")
    label = severity.upper()
    return (
        f"<span style='padding:2px 8px; border-radius:4px; font-size:0.75rem; "
        f"font-weight:700; {style}'>{label}</span>"
    )


def _finding_badge(finding_type: str) -> str:
    label = _FINDING_LABELS.get(finding_type, finding_type.replace("_", " ").title())
    return (
        f"<span style='padding:2px 8px; border-radius:4px; font-size:0.75rem; "
        f"font-weight:600; background:#e3f2fd; color:#0d47a1; "
        f"border:1px solid #90caf9;'>{label}</span>"
    )


def _render_insight_card(insight: Insight) -> None:
    badges = (
        _severity_badge(insight.severity)
        + "&nbsp;&nbsp;"
        + _finding_badge(insight.finding_type)
    )
    cols_str = ", ".join(f"`{c}`" for c in insight.columns)

    with st.container(border=True):
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown(f"**Column(s):** {cols_str}")
        st.markdown(insight.llm_explanation)
        st.caption(f"Recommended action: {insight.recommended_action}")


def _generate_executive_summary(
    insights: list[Insight],
    df: pd.DataFrame,
    meta: DatasetMeta,
) -> str:
    """Ask Claude for a one-sentence executive summary of all insights."""
    try:
        client = LLMClient()
    except EnvironmentError:
        return "API key not configured — executive summary unavailable."

    bullets = "\n".join(
        f"- [{i.severity.upper()}] {i.finding_type} on {', '.join(i.columns)}: {i.statistic}"
        for i in insights
    )
    return client.complete(
        prompt=executive_summary_user(meta.filename, meta.row_count, meta.col_count, bullets),
        max_tokens=128,
    ).strip()


def _run_insights_cached(df: pd.DataFrame, meta: DatasetMeta) -> list[Insight]:
    """Run the insight engine, caching the result in session state."""
    cache_key = f"insights_{meta.filename}_{meta.row_count}"
    if cache_key not in st.session_state:
        with st.spinner("Analyzing your data..."):
            st.session_state[cache_key] = run_insight_engine(df, meta)
    return st.session_state[cache_key]


def _get_exec_summary_cached(
    insights: list[Insight],
    meta: DatasetMeta,
    df: pd.DataFrame,
) -> str:
    cache_key = f"exec_summary_{meta.filename}_{meta.row_count}"
    if cache_key not in st.session_state:
        with st.spinner("Analyzing your data..."):
            st.session_state[cache_key] = _generate_executive_summary(
                insights, df, meta
            )
    return st.session_state[cache_key]


def _clear_insight_cache(meta: DatasetMeta) -> None:
    prefix = f"insights_{meta.filename}_{meta.row_count}"
    summary_prefix = f"exec_summary_{meta.filename}_{meta.row_count}"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix) or key.startswith(summary_prefix):
            del st.session_state[key]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

df, meta = render_upload_widget()

if df is None:
    st.info("Upload a CSV file to get started.")
    st.stop()


# ---------------------------------------------------------------------------
# Regenerate button
# ---------------------------------------------------------------------------

if st.button("Regenerate insights", type="secondary"):
    _clear_insight_cache(meta)
    st.rerun()


# ---------------------------------------------------------------------------
# Run insight engine
# ---------------------------------------------------------------------------

insights = _run_insights_cached(df, meta)

n_insights = len(insights)
n_cols = len({c for i in insights for c in i.columns})


# ---------------------------------------------------------------------------
# Headline summary
# ---------------------------------------------------------------------------

if n_insights == 0:
    st.success("No significant insights found in this dataset.")
    st.stop()

exec_sentence = _get_exec_summary_cached(insights, meta, df)

st.markdown(
    f"### {n_insights} insight{'s' if n_insights != 1 else ''} found "
    f"across {n_cols} column{'s' if n_cols != 1 else ''}"
)
st.markdown(
    f"<p style='font-size:1.05rem; color:#555; margin-top:-8px;'>{exec_sentence}</p>",
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------------------------
# Insights grouped by severity
# ---------------------------------------------------------------------------

for severity in _SEVERITY_ORDER:
    group = [i for i in insights if i.severity == severity]
    if not group:
        continue

    label = severity.capitalize()
    count = len(group)
    st.subheader(f"{label} severity  ·  {count} finding{'s' if count != 1 else ''}")

    for insight in group:
        _render_insight_card(insight)

    st.divider()


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    st.subheader("Correlation Heatmap")

    corr = numeric_df.corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=max(350, 50 * len(corr.columns)),
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()


# ---------------------------------------------------------------------------
# Distribution panel — flagged numeric columns
# ---------------------------------------------------------------------------

flagged_numeric_cols: list[str] = []
for insight in insights:
    for col in insight.columns:
        if (
            col in numeric_df.columns
            and col not in flagged_numeric_cols
            and insight.finding_type in (
                "distribution_skew",
                "outlier_prevalence",
                "high_correlation",
            )
        ):
            flagged_numeric_cols.append(col)

if flagged_numeric_cols:
    st.subheader("Distribution Panel  ·  Flagged Columns")
    st.caption(
        "Histograms for numeric columns involved in distribution, outlier, "
        "or correlation findings. Skew value annotated where |skew| > 2."
    )

    cols_per_row = 3
    rows = [
        flagged_numeric_cols[i : i + cols_per_row]
        for i in range(0, len(flagged_numeric_cols), cols_per_row)
    ]

    for row_cols in rows:
        grid = st.columns(len(row_cols))
        for col_name, grid_col in zip(row_cols, grid):
            data = df[col_name].dropna()
            skew_val = float(data.skew())

            fig = px.histogram(
                data.to_frame(),
                x=col_name,
                nbins=40,
                template="simple_white",
            )
            fig.update_traces(marker_color="#5c85d6")

            annotation_text = f"skew = {skew_val:+.2f}" if abs(skew_val) > 2 else ""
            if annotation_text:
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.97,
                    y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=11, color="#b71c1c" if abs(skew_val) > 5 else "#e65100"),
                    align="right",
                    bgcolor="rgba(255,255,255,0.7)",
                )

            fig.update_layout(
                title=dict(text=col_name, font=dict(size=13)),
                margin=dict(l=10, r=10, t=36, b=10),
                height=220,
                showlegend=False,
                xaxis_title=None,
                yaxis_title="count",
            )

            with grid_col:
                st.plotly_chart(fig, use_container_width=True)
