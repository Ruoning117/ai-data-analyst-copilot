import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_profile import profile_data, quality_warnings, detect_outliers

st.title("AI Data Analyst Copilot")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    summary = profile_data(df)

    st.header("Data Profile Summary")

    st.write(f"Number of rows: {summary['rows']}")
    st.write(f"Number of columns: {summary['columns']}")

    st.write("Missing values per column:")
    st.json(summary["missing_values"])

    st.write(f"Number of duplicate rows: {summary['duplicate_rows']}")

    st.write("Column data types:")
    st.json(summary["column_types"])

    # --- Data Quality Warnings ---
    warnings = quality_warnings(summary, df)

    st.header("Data Quality Warnings")

    if not warnings:
        st.success("No data quality issues found.")
    else:
        for w in warnings:
            st.warning(w)

    # --- Missing Data Overview ---
    st.header("Missing Data Overview")

    # Keep only columns that actually have missing values.
    missing = {col: count for col, count in summary["missing_values"].items() if count > 0}

    if not missing:
        st.info("No missing values found in this dataset.")
    else:
        # Build the bar chart with matplotlib.
        fig, ax = plt.subplots()

        ax.bar(missing.keys(), missing.values(), color="steelblue")

        # Label each axis so the chart is self-explanatory.
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing value count")
        ax.set_title("Missing Values per Column")

        # Force y-axis to show only whole numbers (counts can't be fractions).
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Rotate x-axis labels so long column names don't overlap.
        plt.xticks(rotation=45, ha="right")

        # Tight layout prevents labels from being clipped.
        plt.tight_layout()

        st.pyplot(fig)

        # Always close the figure to free memory.
        plt.close(fig)

    # --- Visualizations ---
    with st.expander("Visualizations"):

        # -- Numeric columns: histograms --
        st.subheader("Numeric Column Distributions")

        # select_dtypes returns only columns whose dtype is a number (int or float).
        # We then skip likely ID columns — they're numeric but carry no meaningful
        # distribution. A column is treated as an ID if its name contains "id"
        # (case-insensitive), e.g. "id", "user_id", "order_ID".
        numeric_cols = [
            col
            for col in df.select_dtypes(include="number").columns
            if "id" not in col.lower()
        ]

        # Cap at 3 so the page doesn't get overwhelmed.
        numeric_cols = numeric_cols[:3]

        if not numeric_cols:
            all_numeric = df.select_dtypes(include="number").columns.tolist()
            if all_numeric:
                # There are numeric columns but all were skipped as IDs.
                st.info(
                    "No numeric columns to plot — all numeric columns appear to be "
                    "ID columns and were skipped."
                )
            else:
                st.info("No numeric columns found.")
        else:
            for col in numeric_cols:
                fig, ax = plt.subplots()

                # Drop missing values so matplotlib doesn't error on NaN.
                col_data = df[col].dropna()
                ax.hist(col_data, bins=20, color="steelblue", edgecolor="white")

                ax.set_title(f"Distribution of '{col}'")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")

                # If all values are whole numbers (e.g. a float column that is
                # actually integers), force the x-axis to display as integers too.
                if (col_data == col_data.astype(int)).all():
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                # Y-axis counts must be whole numbers.
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # -- Categorical columns: bar charts of value counts --
        st.subheader("Categorical Column Distributions")

        # select_dtypes(include="object") picks columns stored as strings.
        # We then keep only those with 20 or fewer unique values — columns with
        # hundreds of unique values (e.g. free-text) produce unreadable charts.
        # We also skip date-like columns: pandas reads dates as strings unless
        # explicitly parsed, so we try converting each column and exclude those
        # that succeed.
        def looks_like_dates(series):
            try:
                pd.to_datetime(series, errors="raise")
                return True
            except Exception:
                return False

        categorical_cols = [
            col
            for col in df.select_dtypes(include=["object", "category"]).columns
            if df[col].nunique() <= 20 and not looks_like_dates(df[col])
        ]

        # Cap at 2 columns.
        categorical_cols = categorical_cols[:2]

        if not categorical_cols:
            st.info("No categorical columns with 20 or fewer unique values found.")
        else:
            for col in categorical_cols:
                # value_counts() tallies how many rows have each unique value.
                counts = df[col].value_counts().head(10)

                fig, ax = plt.subplots()

                ax.bar(counts.index, counts.values, color="coral", edgecolor="white")

                ax.set_title(f"Value Counts for '{col}'")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")

                # Y-axis counts must be whole numbers.
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

                # Rotate labels in case category names are long.
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # --- Potential Outliers ---
    st.header("Potential Outliers")

    # Check whether the dataset has any numeric columns at all.
    has_numeric = not df.select_dtypes(include="number").empty

    if not has_numeric:
        st.info("No numeric columns found — outlier detection was skipped.")
    else:
        outliers = detect_outliers(df)

        if not outliers:
            st.success("No outliers detected in any numeric column.")
        else:
            for col, count in outliers.items():
                st.warning(
                    f"Column '{col}' has {count} potential outlier(s) "
                    f"(values unusually far from the rest of the data)."
                )
