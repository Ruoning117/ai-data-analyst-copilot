"""Semantic role inference and issue filtering.

Sits between raw data-quality checks and the display layer.
Assigns each column one of seven semantic roles, then uses a rule table
to suppress or downgrade issues that are statistically true but not
actionable for a given role.

Extending:
  - Add a new role to SemanticRole and to infer_semantic_roles().
  - Add new rows to _RULES to cover that role.
  - No other files need to change.
"""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from core.data_loader import DatasetMeta


# ---------------------------------------------------------------------------
# Role taxonomy
# ---------------------------------------------------------------------------

SemanticRole = Literal[
    "id",                   # identifier column — high cardinality by design
    "optional_text",        # supplementary field often left blank (address2, fax, …)
    "free_text",            # user-written prose: notes, descriptions, comments
    "categorical_low",      # ≤ 20 distinct values — fits a legend or dropdown
    "categorical_high",     # many distinct values but not free-form prose
    "numeric_measure",      # quantity or measurement
    "datetime",
]

# Human-readable label for each role (used in UI)
ROLE_LABELS: dict[str, str] = {
    "id":               "ID",
    "optional_text":    "Optional",
    "free_text":        "Free Text",
    "categorical_low":  "Categorical",
    "categorical_high": "High-Card. Cat.",
    "numeric_measure":  "Numeric",
    "datetime":         "Datetime",
}


# ---------------------------------------------------------------------------
# Keyword sets for name-based heuristics
# ---------------------------------------------------------------------------

# Whole-token matches (split on whitespace / _ / -)
_OPTIONAL_TOKENS: frozenset[str] = frozenset({
    "suite", "apt", "apartment", "unit",
    "fax",
    "middle",
    "nickname", "alias",
    "optional", "secondary", "alt",
    "extension", "ext",
    "line2",
})

# Substring matches on the collapsed name (e.g. "address2", "addr2")
_OPTIONAL_SUBSTRINGS: tuple[str, ...] = ("address2", "addr2", "line2", "address_2")

# Whole-token matches for free-form prose columns
_FREE_TEXT_TOKENS: frozenset[str] = frozenset({
    "description", "desc",
    "note", "notes",
    "comment", "comments",
    "narrative", "summary",
    "detail", "details",
    "text", "body",
    "feedback", "reason",
    "remarks", "remark",
    "memo", "explanation",
})

_CATEGORICAL_LOW_MAX_UNIQUE = 20


# ---------------------------------------------------------------------------
# Name-matching helpers
# ---------------------------------------------------------------------------

def _tokens(col_name: str) -> set[str]:
    return set(re.split(r"[\s_\-]+", col_name.lower()))


def _is_optional_name(col_name: str) -> bool:
    if _tokens(col_name) & _OPTIONAL_TOKENS:
        return True
    collapsed = re.sub(r"[\s_\-]", "", col_name.lower())
    return any(sub in collapsed for sub in _OPTIONAL_SUBSTRINGS)


def _is_free_text_name(col_name: str) -> bool:
    return bool(_tokens(col_name) & _FREE_TEXT_TOKENS)


# ---------------------------------------------------------------------------
# Role inference
# ---------------------------------------------------------------------------

def infer_semantic_roles(df: pd.DataFrame, meta: DatasetMeta) -> dict[str, SemanticRole]:
    """Return a semantic role for every column in *meta*.

    Decision order within string-based columns:
      1. free_text  — name signals prose, or high-cardinality + long values
      2. optional_text — name signals supplementary field, or mostly-empty + short values
      3. categorical_low / categorical_high — split on unique count
    """
    roles: dict[str, SemanticRole] = {}
    n_rows = len(df)

    for col, col_type in meta.column_types.items():
        if col not in df.columns:
            continue

        # Coarse-type shortcuts — no data inspection needed.
        if col_type == "id":
            roles[col] = "id"
            continue
        if col_type == "datetime":
            roles[col] = "datetime"
            continue
        if col_type == "numeric":
            roles[col] = "numeric_measure"
            continue

        # String-based columns (text | categorical): inspect data shape.
        non_null = df[col].dropna()
        n_non_null = len(non_null)
        missing_pct = (n_rows - n_non_null) / n_rows * 100 if n_rows > 0 else 0.0
        unique_count = df[col].nunique()
        unique_pct = unique_count / n_rows * 100 if n_rows > 0 else 0.0
        avg_len = float(non_null.astype(str).str.len().mean()) if n_non_null > 0 else 0.0

        # free_text: name signals prose, or data shows high-cardinality long values.
        if _is_free_text_name(col) or (unique_pct > 50 and avg_len > 30):
            roles[col] = "free_text"
            continue

        # optional_text: name signals a supplementary field, or the column is
        # mostly empty with short values (characteristic of optional form fields).
        if _is_optional_name(col) or (missing_pct > 40 and avg_len <= 30):
            roles[col] = "optional_text"
            continue

        roles[col] = (
            "categorical_low"
            if unique_count <= _CATEGORICAL_LOW_MAX_UNIQUE
            else "categorical_high"
        )

    return roles


# ---------------------------------------------------------------------------
# Issue filter rules
# ---------------------------------------------------------------------------

# Each rule is (issue_type_substring, role, action).
# issue_type_substring is matched with `in` so "Missing" covers "Missing Values".
# Actions: "suppress" removes the row; "downgrade" sets Severity → Low.
_RULES: list[tuple[str, str, str]] = [
    # Missing values
    ("Missing Values",      "optional_text",    "suppress"),    # by design: optional fields are often blank
    ("Missing Values",      "free_text",         "downgrade"),   # prose fields are often not filled in
    # Cardinality
    ("Cardinality Anomaly", "id",                "suppress"),    # IDs are expected to be unique
    ("Cardinality Anomaly", "free_text",         "suppress"),    # prose naturally has high cardinality
    ("Cardinality Anomaly", "categorical_high",  "suppress"),    # already classified as high-cardinality
    # Whitespace / casing
    ("Whitespace / Casing", "id",                "suppress"),    # case variation in IDs is a system artefact
    ("Whitespace / Casing", "free_text",         "suppress"),    # prose varies naturally
    ("Whitespace / Casing", "optional_text",     "downgrade"),   # less critical for optional fields
]

_RULE_LOOKUP: dict[tuple[str, str], str] = {
    (issue_type, role): action
    for issue_type, role, action in _RULES
}


def filter_issues(
    issues: list[dict],
    roles: dict[str, SemanticRole],
) -> tuple[list[dict], list[dict]]:
    """Split issues into (actionable, suppressed) based on semantic roles.

    Suppressed issues are returned separately so the caller can optionally
    show them in a collapsed expander rather than discarding them.
    Downgraded issues appear in *actionable* with Severity set to "Low".
    """
    actionable: list[dict] = []
    suppressed: list[dict] = []

    for issue in issues:
        col = issue.get("Column", "")
        role = roles.get(col)           # None for multi-column rows like "(all columns)"
        issue_type = issue.get("Issue Type", "")

        action = _RULE_LOOKUP.get((issue_type, role), "keep") if role else "keep"

        if action == "suppress":
            suppressed.append(issue)
        elif action == "downgrade":
            actionable.append({**issue, "Severity": "Low"})
        else:
            actionable.append(issue)

    return actionable, suppressed
