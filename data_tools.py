import pandas as pd
import numpy as np


# ============================================================
# COLUMN HELPERS
# ============================================================

def get_numeric_columns(df):

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def get_categorical_columns(df):

    return df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


# ============================================================
# BASIC CALCULATIONS
# ============================================================

def calculate_sum(df, column):

    return {
        "operation": "sum",
        "column": column,
        "result": float(
            df[column].sum()
        )
    }


def calculate_average(df, column):

    return {
        "operation": "average",
        "column": column,
        "result": float(
            df[column].mean()
        )
    }


def calculate_min(df, column):

    return {
        "operation": "minimum",
        "column": column,
        "result": float(
            df[column].min()
        )
    }


def calculate_max(df, column):

    return {
        "operation": "maximum",
        "column": column,
        "result": float(
            df[column].max()
        )
    }


def calculate_count(df):

    return {
        "operation": "row_count",
        "result": int(
            len(df)
        )
    }


# ============================================================
# GROUP BY
# ============================================================

def group_by_sum(
    df,
    category_column,
    value_column
):

    result = (
        df.groupby(category_column)[value_column]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    return {
        "operation": "group_by_sum",
        "group_by": category_column,
        "value": value_column,
        "result": result.head(20).to_dict()
    }


def group_by_average(
    df,
    category_column,
    value_column
):

    result = (
        df.groupby(category_column)[value_column]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    return {
        "operation": "group_by_average",
        "group_by": category_column,
        "value": value_column,
        "result": result.head(20).to_dict()
    }


# ============================================================
# TOP N
# ============================================================

def top_n(
    df,
    column,
    n=5
):

    result = (
        df[column]
        .value_counts()
        .head(n)
    )

    return {
        "operation": "top_n",
        "column": column,
        "result": result.to_dict()
    }


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlation(
    df,
    column1,
    column2
):

    correlation = df[column1].corr(
        df[column2]
    )

    return {
        "operation": "correlation",
        "column1": column1,
        "column2": column2,
        "result": float(
            correlation
        )
    }


# ============================================================
# DATASET LOOKUP
# ============================================================

def lookup_data(
    df,
    filters,
    select_columns=None
):

    filtered_df = df.copy()

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    for filter_item in filters:

        column = filter_item["column"]
        value = filter_item["value"]

        if column not in filtered_df.columns:

            raise ValueError(
                f"Column '{column}' does not exist."
            )

        series = filtered_df[column]

        # -----------------------------------------------
        # Numeric matching
        # -----------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            try:

                numeric_value = float(value)

                filtered_df = filtered_df[
                    series == numeric_value
                ]

            except (ValueError, TypeError):

                filtered_df = filtered_df[
                    series.astype(str).str.lower()
                    == str(value).lower()
                ]

        # -----------------------------------------------
        # Text matching
        # -----------------------------------------------

        else:

            value_string = str(value).strip().lower()

            filtered_df = filtered_df[
                series.astype(str)
                .str.strip()
                .str.lower()
                == value_string
            ]

    # --------------------------------------------------------
    # SELECT COLUMNS
    # --------------------------------------------------------

    if select_columns:

        valid_columns = [
            column
            for column in select_columns
            if column in filtered_df.columns
        ]

        if valid_columns:

            filtered_df = filtered_df[
                valid_columns
            ]

    # --------------------------------------------------------
    # CONVERT TO SIMPLE RECORDS
    # --------------------------------------------------------

    records = filtered_df.head(20).copy()

    records = records.where(
        pd.notna(records),
        None
    )

    return {
        "operation": "lookup",
        "row_count": len(filtered_df),
        "result": records.to_dict(
            orient="records"
        )
    }