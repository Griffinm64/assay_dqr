import pandas as pd
import numpy as np


def count_total_nulls(df: pd.DataFrame) -> int:
    """Return the total number of null values in the dataframe."""
    return df.isnull().sum().sum()


def nulls_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return null count and percentage of nulls per column."""
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    return pd.DataFrame({
        "null_count": null_counts,
        "null_percent": null_percent.round(2)
    })


def dataset_shape(df: pd.DataFrame) -> dict:
    """Return number of rows and columns."""
    return {"rows": df.shape[0], "columns": df.shape[1]}


def column_types(df: pd.DataFrame) -> pd.Series:
    """Return the data types of each column."""
    return df.dtypes


def unique_counts(df: pd.DataFrame) -> pd.Series:
    """Return the number of unique values per column."""
    return df.nunique()


def duplicate_rows(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows in the dataframe."""
    return df.duplicated().sum()


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.describe().T


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Return top categories for each categorical column."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    summary = {}
    for col in cat_cols:
        summary[col] = df[col].value_counts().head(top_n).to_dict()
    return summary


def outlier_counts(df: pd.DataFrame) -> dict:
    """Detect outliers per numeric column using IQR method."""
    outlier_dict = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
        outlier_dict[col] = int(outliers)
    return outlier_dict


def date_summary(df: pd.DataFrame) -> dict:
    """Return min and max for datetime columns."""
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    summary = {}
    for col in date_cols:
        summary[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "nulls": df[col].isnull().sum()
        }
    return summary