import pandas as pd
import streamlit as st

from core.data_loader import render_upload_widget
from core.data_quality import DataQualityReport, run_quality_checks
from core.profiler import profile_dataset
from core.semantic_roles import ROLE_LABELS, filter_issues, infer_semantic_roles

st.title("Dataset Overview")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _health_color(score: float) -> str:
    if score > 80:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def _health_label(score: float) -> str:
    if score > 80:
        return "Good"
    elif score >= 50:
        return "Fair"
    return "Poor"


def _missing_severity(pct: float) -> str:
    if pct > 30:
        return "High"
    elif pct > 10:
        return "Medium"
    return "Low"


def _build_issues_table(report: DataQualityReport) -> list[dict]:
    rows = []

    # Missing values — one row per affected column
    for col, detail in report.missing.columns.items():
        if detail.count > 0:
            rows.append({
                "Column": col,
                "Issue Type": "Missing Values",
                "Severity": _missing_severity(detail.pct),
                "Detail": f"{detail.count} missing ({detail.pct}%, {detail.pattern})",
            })

    # Exact duplicates
    if report.duplicates.exact_count > 0:
        sev = "High" if report.duplicates.exact_pct > 10 else "Medium"
        rows.append({
            "Column": "(all columns)",
            "Issue Type": "Duplicate Rows",
            "Severity": sev,
            "Detail": f"{report.duplicates.exact_count} exact duplicates ({report.duplicates.exact_pct}%)",
        })

    # Near-duplicates (same data, different ID)
    if report.duplicates.near_duplicate_groups:
        rows.append({
            "Column": report.duplicates.id_column,
            "Issue Type": "Near-Duplicate Rows",
            "Severity": "Low",
            "Detail": (
                f"{len(report.duplicates.near_duplicate_groups)} row groups identical "
                f"except in '{report.duplicates.id_column}'"
            ),
        })

    # Type violations
    for col, detail in report.type_violations.items():
        samples = ", ".join(detail.sample_bad_values)
        rows.append({
            "Column": col,
            "Issue Type": "Type Violation",
            "Severity": "Medium",
            "Detail": f"{detail.violation_count} values can't be cast (e.g. {samples})",
        })

    # Cardinality anomalies
    for anomaly in report.cardinality_anomalies:
        if anomaly.kind == "high_cardinality_categorical":
            detail = (
                f"{anomaly.unique_count} unique values ({anomaly.unique_pct}%) "
                "— may be free text or an ID column"
            )
        else:
            detail = (
                f"Only {anomaly.unique_count} unique values "
                "— may be categorical"
            )
        rows.append({
            "Column": anomaly.column,
            "Issue Type": "Cardinality Anomaly",
            "Severity": "Low",
            "Detail": detail,
        })

    # Whitespace / casing
    for col, detail in report.whitespace_issues.items():
        parts = []
        if detail.has_leading_trailing:
            parts.append("leading/trailing spaces")
        if detail.has_casing_variants:
            parts.append("casing variants (e.g. 'Apple' vs 'apple')")
        rows.append({
            "Column": col,
            "Issue Type": "Whitespace / Casing",
            "Severity": "Low",
            "Detail": ", ".join(parts),
        })

    # Unexpected negatives
    for col, detail in report.negative_value_flags.items():
        rows.append({
            "Column": col,
            "Issue Type": "Unexpected Negatives",
            "Severity": "Medium",
            "Detail": f"{detail.count} negative values ({detail.pct}%)",
        })

    return rows


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

df, meta = render_upload_widget()

if df is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

report = run_quality_checks(df, meta)
st.session_state["quality_report"] = report

profile = profile_dataset(df, meta)
st.session_state["data_profile"] = profile

roles = infer_semantic_roles(df, meta)
st.session_state["semantic_roles"] = roles


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Rows", f"{meta.row_count:,}")

with c2:
    st.metric("Columns", meta.col_count)

with c3:
    color = _health_color(report.health_score)
    label = _health_label(report.health_score)
    st.metric("Health Score", f"{report.health_score} / 100")
    st.markdown(
        f"<span style='color:{color}; font-weight:600'>● {label}</span>",
        unsafe_allow_html=True,
    )

with c4:
    st.metric("Duplicate Rows", report.duplicates.exact_count)

st.divider()


# ---------------------------------------------------------------------------
# Column type breakdown
# ---------------------------------------------------------------------------

st.subheader("Column Types")

type_counts: dict[str, int] = profile.dataset_summary["column_type_counts"]

type_cols = st.columns(len(type_counts))
for i, (type_name, count) in enumerate(sorted(type_counts.items())):
    with type_cols[i]:
        st.metric(type_name.capitalize(), count)

st.divider()


# ---------------------------------------------------------------------------
# Semantic column roles
# ---------------------------------------------------------------------------

st.subheader("Semantic Column Roles")

role_counts: dict[str, int] = {}
for r in roles.values():
    role_counts[r] = role_counts.get(r, 0) + 1

# Show one metric per role that appears in this dataset (skip zeros).
role_order = ["id", "numeric_measure", "datetime", "categorical_low",
              "categorical_high", "free_text", "optional_text"]
active_roles = [r for r in role_order if r in role_counts]
role_cols = st.columns(max(len(active_roles), 1))
for i, role in enumerate(active_roles):
    with role_cols[i]:
        st.metric(ROLE_LABELS[role], role_counts[role])

with st.expander("Column → role mapping"):
    role_table = [
        {"Column": col, "Role": ROLE_LABELS.get(role, role)}
        for col, role in sorted(roles.items())
    ]
    st.dataframe(pd.DataFrame(role_table), use_container_width=True, hide_index=True)

st.divider()


# ---------------------------------------------------------------------------
# Data quality issues table (semantically filtered)
# ---------------------------------------------------------------------------

st.subheader("Data Quality Issues")

raw_issues = _build_issues_table(report)
actionable, suppressed = filter_issues(raw_issues, roles)

if not actionable:
    st.success("No actionable data quality issues found.")
else:
    st.dataframe(
        pd.DataFrame(actionable),
        use_container_width=True,
        hide_index=True,
    )

if suppressed:
    n = len(suppressed)
    label = f"{n} issue{'s' if n > 1 else ''} suppressed — expected for column type"
    with st.expander(label):
        st.caption(
            "These issues are statistically real but not actionable given each "
            "column's inferred role (e.g. high cardinality for an ID column, "
            "missing values for an optional field)."
        )
        st.dataframe(
            pd.DataFrame(suppressed),
            use_container_width=True,
            hide_index=True,
        )

st.divider()


# ---------------------------------------------------------------------------
# Raw data preview
# ---------------------------------------------------------------------------

with st.expander("Raw Data Preview (first 100 rows)"):
    st.dataframe(df.head(100), use_container_width=True)
