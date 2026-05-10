from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from core.data_loader import DatasetMeta, render_upload_widget
from core.qa_engine import QAResult, _build_schema_context, run_qa
from shared.llm_client import LLMClient
from shared.prompts import QA_SUGGESTIONS_SYSTEM, qa_suggestions_user

_QA_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _exec_badge(ms: float) -> str:
    return (
        "<span style='font-size:0.72rem; color:#888; background:#f5f5f5; "
        "padding:1px 8px; border-radius:10px; font-family:monospace'>"
        f"⏱ {ms:.0f} ms</span>"
    )


def _render_result_value(result) -> None:
    """Display tabular results inline; scalars are covered by the interpretation."""
    if isinstance(result, pd.DataFrame) and not result.empty:
        st.dataframe(result, use_container_width=True)
    elif isinstance(result, pd.Series) and not result.empty:
        st.dataframe(result.rename("value").to_frame(), use_container_width=True)


def _render_answer(qa: QAResult) -> None:
    """Render one QAResult inside an active st.chat_message block."""
    if qa.execution_success:
        st.markdown(qa.interpretation)
        _render_result_value(qa.result)
        st.markdown(_exec_badge(qa.execution_time_ms), unsafe_allow_html=True)
        with st.expander("Show generated code"):
            st.code(qa.generated_code, language="python")
    else:
        st.warning(
            "Could not answer this question after one correction attempt. "
            "Try rephrasing or simplifying it."
        )
        st.markdown(_exec_badge(qa.execution_time_ms), unsafe_allow_html=True)
        with st.expander("Show error and attempted code"):
            st.markdown("**Error**")
            st.code(qa.error_message, language="text")
            st.markdown("**Final attempted code**")
            st.code(qa.generated_code, language="python")


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

def _generate_suggestions(df: pd.DataFrame, meta: DatasetMeta) -> list[str]:
    try:
        client = LLMClient()
    except EnvironmentError:
        return []

    schema = _build_schema_context(df, meta)
    raw = client.complete(
        prompt=qa_suggestions_user(schema),
        system_prompt=QA_SUGGESTIONS_SYSTEM,
        max_tokens=400,
        model=_QA_MODEL,
    )

    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    # Strip leading "1. ", "- ", "* ", "• " prefixes the model sometimes adds.
    cleaned = [re.sub(r"^[\d\-\.\)\*•]+\s*", "", ln).strip() for ln in lines]
    return [q for q in cleaned if q][:5]


def _render_suggestions(df: pd.DataFrame, meta: DatasetMeta, cache_key: str) -> None:
    """Render clickable suggestion buttons, generating them once per dataset."""
    if cache_key not in st.session_state:
        with st.spinner("Generating suggestions..."):
            st.session_state[cache_key] = _generate_suggestions(df, meta)

    suggestions: list[str] = st.session_state[cache_key]

    if not suggestions:
        st.caption("Could not generate suggestions — check your API key.")
        return

    for i, question in enumerate(suggestions):
        if st.button(question, key=f"qa_sugg_{i}", use_container_width=True):
            st.session_state["pending_question"] = question


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Ask Your Data")

df, meta = render_upload_widget()

if df is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

# Per-dataset session state keys — naturally invalidated when the file changes,
# because filename + row_count will differ.
history_key = f"qa_history_{meta.filename}_{meta.row_count}"
suggestions_key = f"qa_suggestions_{meta.filename}_{meta.row_count}"

if history_key not in st.session_state:
    st.session_state[history_key] = []

# ---------------------------------------------------------------------------
# Sidebar: suggested questions
# The sidebar block runs before pending_question is consumed below, so a button
# click sets the key here and pop() picks it up in the same render pass.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Suggested Questions")
    st.caption("Click any question to send it.")
    st.divider()
    _render_suggestions(df, meta, suggestions_key)

# ---------------------------------------------------------------------------
# Consume any pending question from a sidebar button click.
# pop() must run after the sidebar block (where it is set) and before the chat
# input check below, so the question flows into exactly one branch.
# ---------------------------------------------------------------------------

pending: str | None = st.session_state.pop("pending_question", None)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

history: list[QAResult] = st.session_state[history_key]

for qa in history:
    with st.chat_message("user"):
        st.markdown(qa.question)
    with st.chat_message("assistant"):
        _render_answer(qa)

# ---------------------------------------------------------------------------
# New question — typed input or sidebar button
# ---------------------------------------------------------------------------

user_input: str | None = st.chat_input("Ask a question about your data…")
question: str | None = pending or user_input

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            qa_result = run_qa(question, df, meta)
        _render_answer(qa_result)

    st.session_state[history_key].append(qa_result)
