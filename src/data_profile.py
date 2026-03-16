import pandas as pd

def profile_data(df: pd.DataFrame) -> dict:
    summary = {}

    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]
    summary["missing_values"] = df.isnull().sum().to_dict()
    summary["duplicate_rows"] = int(df.duplicated().sum())
    summary["column_types"] = df.dtypes.astype(str).to_dict()

    return summary