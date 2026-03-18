# AI Data Analyst Copilot

A Streamlit-based data analysis tool that allows users to upload datasets and automatically generate data quality checks, visualizations, and outlier detection for messy business data.

This project is part of my effort to build practical AI-assisted tools for exploratory data analysis.

---

## Features

- Upload a CSV dataset
- Automatic dataset profiling:
  - number of rows and columns
  - missing values per column
  - duplicate rows
  - column data types

- Data quality warnings:
  - columns with missing values
  - high-missingness columns
  - duplicate row detection
  - constant-value columns

- Missing data visualization:
  - bar chart of missing values per column

- Automatic visualizations:
  - numeric column distributions (histograms)
  - categorical value counts (bar charts)

- Outlier detection:
  - identifies potential outliers using IQR method
  - summarizes outlier counts by column

---

## Tech Stack

- Python
- Streamlit
- Pandas
- uv (Python package manager)

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Ruoning117/ai-data-analyst-copilot.git
cd ai-data-analyst-copilot
```

Install dependencies:

```bash
uv sync
```

Run the app:
```bash
uv run streamlit run main.py
```

Open the app in your browser:

```
http://localhost:8501
```

---

## Future Improvements

- Generate automatic insights and summaries using LLMs
- Allow users to ask questions in natural language
- Support Excel file uploads
- Export a clean analysis report (PDF or HTML)
- Add time-series analysis and trend detection