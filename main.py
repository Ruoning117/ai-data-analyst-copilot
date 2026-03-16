import streamlit as st
import pandas as pd
from src.data_profile import profile_data

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