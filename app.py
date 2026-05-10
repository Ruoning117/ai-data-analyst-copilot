import streamlit as st

st.set_page_config(
    page_title="AI Data Analyst Copilot",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — branding and live dataset context
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 AI Data Analyst")
    st.caption("Powered by Claude")
    st.divider()

    meta = st.session_state.get("current_dataset")
    if meta is not None:
        st.markdown(f"**Dataset**")
        st.markdown(f"`{meta.filename}`")
        st.markdown(f"{meta.row_count:,} rows · {meta.col_count} columns")

        report = st.session_state.get("quality_report")
        if report is not None:
            score = report.health_score
            if score > 80:
                color, label = "#2e7d32", "Good"
            elif score >= 50:
                color, label = "#e65100", "Fair"
            else:
                color, label = "#b71c1c", "Poor"
            st.markdown(
                f"**Health Score:** "
                f"<span style='color:{color}; font-weight:700'>"
                f"{score}/100 — {label}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No file uploaded yet.")

    st.divider()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

pages = [
    st.Page("pages/01_overview.py",  title="Overview",   icon="🔍"),
    st.Page("pages/02_insights.py",  title="Insights",   icon="💡"),
    st.Page("pages/03_qa.py",        title="Ask",        icon="💬"),
    st.Page("pages/04_anomalies.py", title="Anomalies",  icon="🚨"),
    st.Page("pages/05_summary.py",   title="Summary",    icon="📋"),
]

pg = st.navigation(pages)
pg.run()
