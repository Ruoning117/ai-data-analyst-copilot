# AI Data Analyst Copilot

A Streamlit-based data analysis tool that allows users to upload datasets and instantly generate basic profiling summaries such as missing values, duplicate rows, and column data types.

This project is part of my effort to build practical AI-assisted tools for exploratory data analysis.

---

## Demo

![App Demo](docs/app_demo.png)

---

## Features

- Upload a CSV dataset
- Automatically generate dataset profiling
- Display:
  - number of rows and columns
  - missing values per column
  - duplicate rows
  - column data types

---

## Tech Stack

- Python
- Streamlit
- Pandas
- uv (Python package manager)

---

## Project Structure

```
ai-data-analyst-copilot/
│
├── main.py                # Streamlit app entry point
├── src/
│   └── data_profile.py    # Data profiling logic
│
├── data/
│   └── sample_data.csv    # Example dataset
│
├── docs/
│   └── app_demo.png       # App screenshot
│
├── pyproject.toml         # Project dependencies
├── uv.lock                # Dependency lockfile
└── README.md
```

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

- Add visualization of distributions
- Generate automatic insights using LLMs
- Support Excel file uploads
- Export profiling report