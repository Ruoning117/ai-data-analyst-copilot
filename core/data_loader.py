from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st


@dataclass
class DatasetMeta:
    filename: str
    row_count: int
    col_count: int
    column_types: dict[str, str]  # col → "numeric" | "categorical" | "datetime" | "text" | "id"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_id_column(col_name: str) -> bool:
    """Return True only when 'id' is the first or last word token in the name.

    Splits on whitespace, underscores, and hyphens so that:
      - "id", "user_id", "order id", "ID_CODE"  → True
      - "provider", "MEDICARE ID EFFECTIVE DATE" → False
    """
    tokens = re.split(r"[\s_\-]+", col_name.lower())
    return tokens[0] == "id" or tokens[-1] == "id"


def _infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    types: dict[str, str] = {}
    for col in df.columns:
        col_lower = col.lower()
        if _is_id_column(col):
            types[col] = "id"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            non_null = df[col].dropna()
            if "date" in col_lower or "time" in col_lower:
                types[col] = "datetime"
            elif not non_null.empty and non_null.astype(str).str.len().mean() > 50:
                types[col] = "text"
            else:
                types[col] = "categorical"
    return types


# ---------------------------------------------------------------------------
# Streamlit upload widget
# ---------------------------------------------------------------------------

def render_upload_widget() -> tuple[pd.DataFrame | None, DatasetMeta | None]:
    """Render a CSV file uploader and return (df, meta).

    Caches the parsed result in session state so other pages can access it
    without re-uploading. Returns (None, None) if no file has been uploaded.
    """
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Only re-parse when a new file is selected.
        if st.session_state.get("filename") != uploaded_file.name:
            df = pd.read_csv(uploaded_file)
            meta = DatasetMeta(
                filename=uploaded_file.name,
                row_count=len(df),
                col_count=len(df.columns),
                column_types=_infer_column_types(df),
            )
            st.session_state["df"] = df
            st.session_state["current_dataset"] = meta
            st.session_state["filename"] = uploaded_file.name
            # Invalidate derived caches from the previous file.
            st.session_state.pop("quality_report", None)
        return st.session_state["df"], st.session_state["current_dataset"]

    # No file in the uploader — return whatever is cached (supports page navigation).
    return st.session_state.get("df"), st.session_state.get("current_dataset")
