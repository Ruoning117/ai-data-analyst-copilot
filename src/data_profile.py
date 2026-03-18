import pandas as pd

def profile_data(df: pd.DataFrame) -> dict:
    summary = {}

    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]
    summary["missing_values"] = df.isnull().sum().to_dict()
    summary["duplicate_rows"] = int(df.duplicated().sum())
    summary["column_types"] = df.dtypes.astype(str).to_dict()

    return summary


def quality_warnings(summary: dict, df: pd.DataFrame) -> list:
    """
    Inspect the dataset and return a list of human-readable warning strings.

    Parameters:
        summary: the dict returned by profile_data()
        df:      the original DataFrame

    Returns:
        A list of warning strings. Empty list means no issues found.
    """
    warnings = []

    # --- Warning 1: duplicate rows ---
    # summary["duplicate_rows"] was already computed by profile_data.
    if summary["duplicate_rows"] > 0:
        warnings.append(
            f"Dataset contains {summary['duplicate_rows']} duplicate row(s)."
        )

    # --- Warnings 2 & 3: missing values ---
    # Iterate over every column's missing-value count.
    total_rows = summary["rows"]

    for col, missing_count in summary["missing_values"].items():
        if missing_count == 0:
            continue  # no problem in this column

        # Calculate what percentage of rows are missing.
        missing_pct = missing_count / total_rows * 100

        if missing_pct > 30:
            # Severe: flag with a stronger message.
            warnings.append(
                f"Column '{col}' is missing {missing_count} value(s) "
                f"({missing_pct:.1f}% of rows) — consider dropping or imputing."
            )
        else:
            # Mild: just note the count.
            warnings.append(
                f"Column '{col}' has {missing_count} missing value(s) "
                f"({missing_pct:.1f}% of rows)."
            )

    # --- Warning 4: columns with only one unique value ---
    # A column with a single unique value carries no information.
    for col in df.columns:
        if df[col].nunique() == 1:
            warnings.append(
                f"Column '{col}' has only one unique value — it may not be useful."
            )

    return warnings


def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Detect outliers in every numeric column using the IQR method.

    How IQR works:
        1. Sort the values and find Q1 (25th percentile) and Q3 (75th percentile).
        2. IQR = Q3 - Q1  (the spread of the middle 50% of values).
        3. Any value below  Q1 - 1.5 * IQR  or above  Q3 + 1.5 * IQR
           is considered an outlier. The multiplier 1.5 is a widely used
           convention introduced by John Tukey (the same rule used in box plots).

    Parameters:
        df: the original DataFrame

    Returns:
        A dict mapping column name to the number of outliers found.
        Only columns with at least one outlier are included.
        Columns with no numeric data are skipped entirely.
    """
    outliers = {}

    # Work only on numeric columns — IQR is meaningless for strings.
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        # Drop missing values before computing percentiles; NaN would skew results.
        col_data = df[col].dropna()

        # Need at least a few values to compute a meaningful IQR.
        if len(col_data) < 4:
            continue

        q1 = col_data.quantile(0.25)  # 25th percentile
        q3 = col_data.quantile(0.75)  # 75th percentile
        iqr = q3 - q1                 # interquartile range

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Count values that fall outside the bounds.
        outlier_count = int(((col_data < lower_bound) | (col_data > upper_bound)).sum())

        if outlier_count > 0:
            outliers[col] = outlier_count

    return outliers