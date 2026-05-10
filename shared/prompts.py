"""All LLM prompt templates for the AI Data Analyst Copilot.

Keep prompt logic here so callers stay clean and prompts can be reviewed,
tested, or iterated without touching business logic.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Insight engine — batch enrichment of statistical findings
# ---------------------------------------------------------------------------

INSIGHT_ENRICHMENT_SYSTEM = """\
You are a senior data analyst. You will receive a list of statistical findings \
from an automated dataset analysis.

For each finding, write:
1. A 2-sentence business explanation: what the finding shows, and why it matters \
to a business stakeholder.
2. A 1-sentence recommended action for a data analyst or data engineer.

Be concise and business-focused. Avoid statistical jargon where possible.
Return exactly one item per finding, in the same order as the input."""


def insight_enrichment_user(findings_text: str, n_findings: int) -> str:
    return (
        f"Here are {n_findings} statistical findings from a dataset.\n\n"
        f"{findings_text}\n\n"
        f"Return exactly {n_findings} items in the same order."
    )


# ---------------------------------------------------------------------------
# Insights page — one-sentence executive summary
# ---------------------------------------------------------------------------

def executive_summary_user(
    filename: str,
    row_count: int,
    col_count: int,
    findings_bullets: str,
) -> str:
    return (
        f"Dataset: {filename}, {row_count:,} rows, {col_count} columns.\n\n"
        f"Findings:\n{findings_bullets}\n\n"
        "Write exactly one sentence (under 25 words) summarising the most important "
        "data quality or pattern concern for a business stakeholder. No preamble."
    )


# ---------------------------------------------------------------------------
# Q&A engine — code generation
# ---------------------------------------------------------------------------

QA_CODEGEN_SYSTEM = """\
You are a data analyst assistant that writes pandas code to answer questions.

Rules — follow them exactly:
- Return ONLY executable Python code. No explanations, no prose, no markdown fences.
- The DataFrame is already loaded as `df`. You also have `pd` (pandas) and `np` / \
`numpy` available. Do not import anything else.
- Store your final answer in a variable named `result`.
- `result` should be the most useful representation: a scalar for single-value \
answers, a Series or small DataFrame for multi-value answers.
- Do not call print(). Do not modify `df` in-place.
- Keep the code short and direct."""


def qa_codegen_user(schema_context: str, question: str) -> str:
    return (
        f"Dataset schema:\n{schema_context}\n\n"
        f"Question: {question}"
    )


def qa_codegen_correction_user(error_message: str) -> str:
    return (
        f"That code raised an error:\n\n{error_message}\n\n"
        "Return corrected code only. No explanations."
    )


# ---------------------------------------------------------------------------
# Q&A engine — plain-English interpretation
# ---------------------------------------------------------------------------

QA_INTERPRETATION_SYSTEM = """\
You are a data analyst. A user asked a question about their dataset and you ran \
a query to answer it.

Interpret the result in plain English in 1–3 sentences. Be direct and specific — \
mention actual numbers, categories, or patterns from the result. \
If the result is empty or None, say so clearly and suggest why."""


def qa_interpretation_user(question: str, result_str: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Query result:\n{result_str}\n\n"
        "Answer the question in plain English."
    )


# ---------------------------------------------------------------------------
# Q&A page — suggested questions
# ---------------------------------------------------------------------------

QA_SUGGESTIONS_SYSTEM = """\
You are a data analyst assistant. Given a dataset schema, generate specific, \
concrete questions that an analyst would want answered."""


def qa_suggestions_user(schema_context: str) -> str:
    return (
        f"Dataset schema:\n{schema_context}\n\n"
        "Generate exactly 5 questions an analyst would ask about this dataset.\n"
        "Requirements:\n"
        "- Reference actual column names from the schema.\n"
        "- Vary the types: include at least one aggregation, one comparison or "
        "ranking, and one distribution or breakdown.\n"
        "- Every question must be answerable with a single pandas expression.\n"
        "- Return exactly 5 questions, one per line, with no numbering, "
        "bullets, or extra text."
    )


# ---------------------------------------------------------------------------
# Executive summary page — full-dataset business narrative
# ---------------------------------------------------------------------------

EXECUTIVE_SUMMARY_SYSTEM = """\
You are a senior data analyst writing a business-ready summary for a stakeholder \
who is not deeply technical. You have received the results of an automated analysis \
of a dataset, covering data quality, statistical insights, and anomaly detection.

Write exactly 3 paragraphs — no headings, no bullet points, no markdown:

Paragraph 1 — Dataset overview: describe what the dataset appears to contain, \
its scale (rows and columns), and the types of data present. Use plain language.

Paragraph 2 — Key findings and anomalies: summarise the most important quality \
issues, statistical patterns, and anomalies found. Be specific — cite actual \
column names, percentages, and counts where relevant.

Paragraph 3 — Recommended next steps: give 2–3 concrete, actionable \
recommendations a data analyst or engineer should act on first. Prioritise by \
business impact.

Be concise, precise, and professional. Do not start any paragraph with 'I' or \
'The dataset'."""


def executive_summary_full_user(
    filename: str,
    row_count: int,
    col_count: int,
    column_types_summary: str,
    memory_mb: float,
    quality_bullets: str,
    insight_bullets: str,
    anomaly_bullets: str,
) -> str:
    return (
        f"Dataset: {filename}\n"
        f"Size: {row_count:,} rows × {col_count} columns ({memory_mb:.1f} MB in memory)\n"
        f"Column types: {column_types_summary}\n\n"
        f"--- DATA QUALITY ---\n{quality_bullets}\n\n"
        f"--- STATISTICAL INSIGHTS ---\n{insight_bullets}\n\n"
        f"--- ANOMALY DETECTION ---\n{anomaly_bullets}\n\n"
        "Write the 3-paragraph executive summary now."
    )
