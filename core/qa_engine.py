"""Natural-language Q&A engine.

Flow
----
1. Build a schema context string from df + DatasetMeta (column names, types,
   sample values, numeric stats).
2. Ask Claude to write pandas code that answers the question and stores the
   result in a variable called `result`.
3. Execute that code in a sandboxed namespace {df, pd, np/numpy}.
4. On failure, send the error back for one correction attempt.
5. On success, ask Claude to interpret the result in plain English.

No Streamlit code in this module.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.data_loader import DatasetMeta
from core.semantic_roles import ROLE_LABELS, infer_semantic_roles
from shared.llm_client import LLMClient
from shared.prompts import (
    QA_CODEGEN_SYSTEM,
    QA_INTERPRETATION_SYSTEM,
    qa_codegen_correction_user,
    qa_codegen_user,
    qa_interpretation_user,
)

_QA_MODEL = "claude-sonnet-4-20250514"

# Maximum rows / values shown to the LLM when formatting a result for interpretation.
_MAX_RESULT_ROWS = 50


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    question: str
    generated_code: str       # final code that was executed (or last attempted)
    result: Any               # the actual pandas/Python value; None if failed
    interpretation: str       # plain-English answer; empty string if failed
    execution_success: bool
    error_message: str        # empty string on success
    execution_time_ms: float


# ---------------------------------------------------------------------------
# Schema context builder
# ---------------------------------------------------------------------------

def _build_schema_context(df: pd.DataFrame, meta: DatasetMeta) -> str:
    """Return a compact, LLM-readable description of the dataset's schema."""
    roles = infer_semantic_roles(df, meta)

    lines: list[str] = [
        f"Dataset: {meta.filename}  ({meta.row_count:,} rows × {meta.col_count} columns)",
        "",
        "Columns:",
    ]

    for col, col_type in meta.column_types.items():
        if col not in df.columns:
            continue

        role = roles.get(col, col_type)
        role_label = ROLE_LABELS.get(role, col_type)

        non_null = df[col].dropna()

        # Sample values — up to 5 non-null entries.
        raw_samples = non_null.iloc[:5].tolist()
        sample_str = ", ".join(repr(v) for v in raw_samples) if raw_samples else "(all null)"

        # Numeric stats appended inline.
        stats_str = ""
        if col_type == "numeric" and len(non_null) > 0:
            try:
                nums = pd.to_numeric(non_null, errors="coerce").dropna()
                if len(nums) > 0:
                    stats_str = (
                        f"  |  min={nums.min():.4g}, "
                        f"max={nums.max():.4g}, "
                        f"mean={nums.mean():.4g}"
                    )
            except Exception:
                pass

        lines.append(f"  {col!r:40s} [{role_label}]  samples: {sample_str}{stats_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code helpers
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    """Strip markdown code fences that the model may include despite instructions."""
    raw = raw.strip()
    if raw.startswith("```"):
        fence_end = raw.find("\n")
        raw = raw[fence_end + 1:] if fence_end != -1 else raw
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return raw.strip()


def _execute(code: str, df: pd.DataFrame) -> tuple[Any, str | None]:
    """Run *code* in a sandboxed namespace.

    Returns (result_value, None) on success or (None, error_string) on failure.
    The namespace is intentionally minimal: df, pd, np, numpy.
    """
    namespace: dict[str, Any] = {
        "df": df,
        "pd": pd,
        "np": np,
        "numpy": np,
    }
    try:
        exec(compile(code, "<qa_engine>", "exec"), namespace)  # noqa: S102
        return namespace.get("result"), None
    except Exception:
        # Return the full traceback so the LLM has enough context to fix it.
        return None, traceback.format_exc(limit=6).strip()


def _format_result(result: Any) -> str:
    """Convert *result* to a readable string for the interpretation prompt."""
    if result is None:
        return "None"
    if isinstance(result, pd.DataFrame):
        preview = result.head(_MAX_RESULT_ROWS)
        suffix = f"\n... ({len(result):,} total rows)" if len(result) > _MAX_RESULT_ROWS else ""
        return preview.to_string() + suffix
    if isinstance(result, pd.Series):
        preview = result.head(_MAX_RESULT_ROWS)
        suffix = f"\n... ({len(result):,} total values)" if len(result) > _MAX_RESULT_ROWS else ""
        return preview.to_string() + suffix
    return str(result)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_qa(question: str, df: pd.DataFrame, meta: DatasetMeta) -> QAResult:
    """Answer *question* about *df* using Claude-generated pandas code.

    Steps:
      1. Build schema context and ask Claude for pandas code.
      2. Execute in a sandboxed namespace; measure wall-clock time.
      3. On failure, send the error back for one correction attempt.
      4. On success, ask Claude for a plain-English interpretation.
    """
    client = LLMClient()
    schema = _build_schema_context(df, meta)

    # ------------------------------------------------------------------ #
    # Step 1 — generate code                                              #
    # ------------------------------------------------------------------ #
    first_response = client.complete(
        prompt=qa_codegen_user(schema, question),
        system_prompt=QA_CODEGEN_SYSTEM,
        max_tokens=1024,
        model=_QA_MODEL,
    )
    code = _extract_code(first_response)

    # ------------------------------------------------------------------ #
    # Step 2 — execute                                                    #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    result, error = _execute(code, df)
    exec_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ #
    # Step 3 — one correction attempt on failure                         #
    # ------------------------------------------------------------------ #
    if error is not None:
        correction_response = client.chat(
            messages=[
                {"role": "user",      "content": qa_codegen_user(schema, question)},
                {"role": "assistant", "content": first_response},
                {"role": "user",      "content": qa_codegen_correction_user(error)},
            ],
            system_prompt=QA_CODEGEN_SYSTEM,
            max_tokens=1024,
            model=_QA_MODEL,
        )
        code = _extract_code(correction_response)

        t0 = time.perf_counter()
        result, error = _execute(code, df)
        exec_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ #
    # Step 4 — interpret on success                                       #
    # ------------------------------------------------------------------ #
    if error is not None:
        return QAResult(
            question=question,
            generated_code=code,
            result=None,
            interpretation="",
            execution_success=False,
            error_message=error,
            execution_time_ms=round(exec_ms, 1),
        )

    interpretation = client.complete(
        prompt=qa_interpretation_user(question, _format_result(result)),
        system_prompt=QA_INTERPRETATION_SYSTEM,
        max_tokens=256,
        model=_QA_MODEL,
    )

    return QAResult(
        question=question,
        generated_code=code,
        result=result,
        interpretation=interpretation,
        execution_success=True,
        error_message="",
        execution_time_ms=round(exec_ms, 1),
    )
