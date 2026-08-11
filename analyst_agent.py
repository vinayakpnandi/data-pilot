import re
import ollama
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "qwen3:1.7b"


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower()
    ).strip()


def find_column(question, columns):

    question_normalized = normalize_text(
        question
    )

    # --------------------------------------------------------
    # Exact column name match
    # --------------------------------------------------------

    for column in columns:

        column_normalized = normalize_text(
            column
        )

        if column_normalized in question_normalized:

            return column

    # --------------------------------------------------------
    # Word matching
    # --------------------------------------------------------

    question_words = set(
        question_normalized.split()
    )

    best_column = None
    best_score = 0

    for column in columns:

        column_words = set(
            normalize_text(column).split()
        )

        if not column_words:
            continue

        overlap = len(
            question_words & column_words
        )

        score = overlap / len(
            column_words
        )

        if score > best_score:

            best_score = score
            best_column = column

    if best_score >= 0.5:

        return best_column

    return None


# ============================================================
# FIND NUMERIC COLUMN
# ============================================================

def find_numeric_column(
    question,
    df
):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        return None

    # --------------------------------------------------------
    # Direct column matching
    # --------------------------------------------------------

    column = find_column(
        question,
        numeric_columns
    )

    if column:

        return column

    # --------------------------------------------------------
    # Common metric synonyms
    # --------------------------------------------------------

    question_lower = question.lower()

    synonyms = {

        "revenue": [
            "revenue",
            "sales",
            "income",
            "earnings"
        ],

        "sales": [
            "sales",
            "revenue"
        ],

        "profit": [
            "profit",
            "earnings",
            "margin"
        ],

        "price": [
            "price",
            "cost"
        ],

        "quantity": [
            "quantity",
            "units",
            "volume"
        ],

        "score": [
            "score",
            "marks",
            "rating"
        ],

        "rating": [
            "rating",
            "score"
        ]
    }

    for key, words in synonyms.items():

        if key in question_lower:

            for word in words:

                for actual_column in numeric_columns:

                    if word in normalize_text(
                        actual_column
                    ):

                        return actual_column

    return None


# ============================================================
# FIND CATEGORY COLUMN
# ============================================================

def find_category_column(
    question,
    df
):

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns.tolist()

    if not categorical_columns:

        return None

    # --------------------------------------------------------
    # Direct matching
    # --------------------------------------------------------

    column = find_column(
        question,
        categorical_columns
    )

    if column:

        return column

    # --------------------------------------------------------
    # Common category names
    # --------------------------------------------------------

    question_lower = question.lower()

    common_words = {

        "category": [
            "category",
            "categories"
        ],

        "state": [
            "state",
            "states"
        ],

        "region": [
            "region",
            "regions"
        ],

        "product": [
            "product",
            "products"
        ],

        "customer": [
            "customer",
            "customers"
        ],

        "class": [
            "class",
            "classes"
        ],

        "title": [
            "title",
            "titles"
        ]
    }

    for key, words in common_words.items():

        if key in question_lower:

            for word in words:

                for actual_column in categorical_columns:

                    if word in normalize_text(
                        actual_column
                    ):

                        return actual_column

    return None


# ============================================================
# BASIC CALCULATIONS
# ============================================================

def perform_calculation(
    question,
    df
):

    question_lower = question.lower()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        return None

    column = find_numeric_column(
        question,
        df
    )

    if not column:

        return None

    # --------------------------------------------------------
    # AVERAGE
    # --------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "average",
            "mean",
            "avg"
        ]
    ):

        return {

            "type": "calculation",

            "operation": "average",

            "column": column,

            "value": df[column].mean()
        }

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "total",
            "sum",
            "overall"
        ]
    ):

        return {

            "type": "calculation",

            "operation": "sum",

            "column": column,

            "value": df[column].sum()
        }

    # --------------------------------------------------------
    # MAXIMUM
    # --------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "maximum",
            "max",
            "highest",
            "largest"
        ]
    ):

        return {

            "type": "calculation",

            "operation": "maximum",

            "column": column,

            "value": df[column].max()
        }

    # --------------------------------------------------------
    # MINIMUM
    # --------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "minimum",
            "min",
            "lowest",
            "smallest"
        ]
    ):

        return {

            "type": "calculation",

            "operation": "minimum",

            "column": column,

            "value": df[column].min()
        }

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    if any(
        phrase in question_lower
        for phrase in [
            "how many records",
            "how many rows",
            "number of records",
            "number of rows",
            "row count",
            "how many entries"
        ]
    ):

        return {

            "type": "calculation",

            "operation": "count",

            "value": len(df)
        }

    return None


# ============================================================
# TOP RECORD / HIGHEST / LOWEST
# ============================================================

def perform_top_record(
    question,
    df
):

    question_lower = question.lower()

    ranking_words = [

        "highest",
        "lowest",
        "largest",
        "smallest",
        "maximum",
        "minimum",
        "most",
        "least"
    ]

    if not any(
        word in question_lower
        for word in ranking_words
    ):

        return None

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        return None

    # --------------------------------------------------------
    # Find metric
    # --------------------------------------------------------

    metric_column = find_numeric_column(
        question,
        df
    )

    if metric_column is None:

        return None

    # --------------------------------------------------------
    # Find requested target column
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns.tolist()

    if not categorical_columns:

        return None

    target_column = None

    question_words = set(
        normalize_text(
            question
        ).split()
    )

    best_score = 0

    # --------------------------------------------------------
    # Try matching column name
    # --------------------------------------------------------

    for column in categorical_columns:

        column_words = set(
            normalize_text(
                column
            ).split()
        )

        if not column_words:

            continue

        overlap = len(
            question_words &
            column_words
        )

        score = overlap / len(
            column_words
        )

        if score > best_score:

            best_score = score

            target_column = column

    # --------------------------------------------------------
    # Common descriptive columns
    # --------------------------------------------------------

    if target_column is None:

        common_targets = [

            "title",
            "name",
            "product",
            "item",
            "category",
            "class",
            "customer",
            "state",
            "region"
        ]

        for column in categorical_columns:

            column_name = normalize_text(
                column
            )

            if any(
                word in column_name
                for word in common_targets
            ):

                target_column = column

                break

    if target_column is None:

        return None

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    valid_df = df.dropna(
        subset=[
            metric_column
        ]
    )

    if valid_df.empty:

        return None

    # --------------------------------------------------------
    # Find highest / lowest row
    # --------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "highest",
            "largest",
            "maximum",
            "most"
        ]
    ):

        index = valid_df[
            metric_column
        ].idxmax()

    else:

        index = valid_df[
            metric_column
        ].idxmin()

    row = valid_df.loc[
        index
    ]

    return {

        "type": "top_record",

        "target_column": target_column,

        "target_value": row[
            target_column
        ],

        "metric_column": metric_column,

        "metric_value": row[
            metric_column
        ]
    }


# ============================================================
# GROUP / COMPARISON ANALYSIS
# ============================================================

def perform_group_analysis(
    question,
    df
):

    question_lower = question.lower()

    ranking_words = [

        "best",
        "highest",
        "top",
        "most",
        "largest",
        "strongest",
        "performs best",
        "perform best",
        "better"
    ]

    comparison_words = [

        "compare",
        "comparison",
        "versus",
        "vs",
        "difference"
    ]

    is_ranking = any(
        word in question_lower
        for word in ranking_words
    )

    is_comparison = any(
        word in question_lower
        for word in comparison_words
    )

    if not is_ranking and not is_comparison:

        return None

    category_column = find_category_column(
        question,
        df
    )

    if category_column is None:

        return None

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        return None

    metric_column = find_numeric_column(
        question,
        df
    )

    if metric_column is None:

        metric_column = numeric_columns[0]

    # --------------------------------------------------------
    # Group
    # --------------------------------------------------------

    grouped = (
        df.groupby(
            category_column
        )[metric_column]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if grouped.empty:

        return None

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    if is_ranking:

        best_category = grouped.index[0]

        best_value = grouped.iloc[0]

        top_values = grouped.head(
            5
        )

        return {

            "type": "group_ranking",

            "category_column":
                category_column,

            "metric_column":
                metric_column,

            "best_category":
                best_category,

            "best_value":
                best_value,

            "top_values":
                top_values.to_dict()
        }

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    return {

        "type": "group_comparison",

        "category_column":
            category_column,

        "metric_column":
            metric_column,

        "values":
            grouped.head(
                10
            ).to_dict()
    }


# ============================================================
# FIND VALUE IN DATASET
# ============================================================

def find_value_in_dataset(
    question,
    df
):

    question_normalized = normalize_text(
        question
    )

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns

    # --------------------------------------------------------
    # Search text values
    # --------------------------------------------------------

    for column in categorical_columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
        )

        for value in values:

            normalized_value = normalize_text(
                value
            )

            if (
                normalized_value
                and
                normalized_value
                in question_normalized
            ):

                return column, value

    # --------------------------------------------------------
    # Search numeric values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    numbers = re.findall(
        r"\b\d+(?:\.\d+)?\b",
        question
    )

    for number in numbers:

        try:

            number_value = float(
                number
            )

        except ValueError:

            continue

        for column in numeric_columns:

            numeric_series = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if (
                numeric_series
                == number_value
            ).any():

                return column, number_value

    return None, None


# ============================================================
# LOOKUP
# ============================================================

def perform_lookup(
    question,
    df
):

    filter_column, filter_value = (
        find_value_in_dataset(
            question,
            df
        )
    )

    if filter_column is None:

        return None

    target_column = find_column(
        question,
        df.columns
    )

    # --------------------------------------------------------
    # If target column isn't obvious
    # --------------------------------------------------------

    if (
        target_column == filter_column
        or target_column is None
    ):

        remaining_columns = [

            column

            for column in df.columns

            if column != filter_column
        ]

        question_words = normalize_text(
            question
        ).split()

        for column in remaining_columns:

            column_words = normalize_text(
                column
            ).split()

            if any(
                word in question_words
                for word in column_words
            ):

                target_column = column

                break

    # --------------------------------------------------------
    # Filter dataset
    # --------------------------------------------------------

    filtered = df[
        df[filter_column]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        str(filter_value)
        .strip()
        .lower()
    ]

    # --------------------------------------------------------
    # Numeric fallback
    # --------------------------------------------------------

    if filtered.empty:

        try:

            filtered = df[
                pd.to_numeric(
                    df[filter_column],
                    errors="coerce"
                )
                ==
                float(
                    filter_value
                )
            ]

        except Exception:

            pass

    if filtered.empty:

        return {

            "type": "not_found",

            "filter_column":
                filter_column,

            "filter_value":
                filter_value
        }

    # --------------------------------------------------------
    # Return requested column
    # --------------------------------------------------------

    if (
        target_column
        and
        target_column != filter_column
    ):

        values = (
            filtered[
                target_column
            ]
            .dropna()
            .tolist()
        )

        return {

            "type": "lookup",

            "filter_column":
                filter_column,

            "filter_value":
                filter_value,

            "target_column":
                target_column,

            "values":
                values
        }

    # --------------------------------------------------------
    # Return rows
    # --------------------------------------------------------

    return {

        "type": "rows",

        "rows":
            filtered
            .head(10)
            .to_dict(
                orient="records"
            )
    }


# ============================================================
# FORMAT ANSWERS
# ============================================================

def format_answer(
    result
):

    if result is None:

        return None

    result_type = result.get(
        "type"
    )

    # ========================================================
    # BASIC CALCULATION
    # ========================================================

    if result_type == "calculation":

        operation = result[
            "operation"
        ]

        if operation == "count":

            return (
                "The dataset contains "
                f"**{result['value']:,} records**."
            )

        value = result[
            "value"
        ]

        column = result[
            "column"
        ]

        if pd.isna(value):

            return None

        if operation == "average":

            return (
                f"The average **{column}** is "
                f"**{value:,.2f}**."
            )

        if operation == "sum":

            return (
                f"The total **{column}** is "
                f"**{value:,.2f}**."
            )

        if operation == "maximum":

            return (
                f"The highest **{column}** is "
                f"**{value:,.2f}**."
            )

        if operation == "minimum":

            return (
                f"The lowest **{column}** is "
                f"**{value:,.2f}**."
            )

    # ========================================================
    # TOP RECORD
    # ========================================================

    if result_type == "top_record":

        target_value = result[
            "target_value"
        ]

        metric_column = result[
            "metric_column"
        ]

        metric_value = result[
            "metric_value"
        ]

        return (
            f"**{target_value}** has the highest "
            f"**{metric_column}**, at "
            f"**{metric_value:,.2f}**."
        )

    # ========================================================
    # GROUP RANKING
    # ========================================================

    if result_type == "group_ranking":

        category = result[
            "best_category"
        ]

        value = result[
            "best_value"
        ]

        metric = result[
            "metric_column"
        ]

        top_values = result[
            "top_values"
        ]

        response = (
            f"**{category}** performs best based on "
            f"total **{metric}**, with a value of "
            f"**{value:,.2f}**."
        )

        # Only show top performers if useful
        if len(top_values) > 1:

            response += (
                "\n\n**Top performers:**\n"
            )

            for name, amount in (
                top_values.items()
            ):

                response += (
                    f"- {name}: "
                    f"{amount:,.2f}\n"
                )

        return response

    # ========================================================
    # GROUP COMPARISON
    # ========================================================

    if result_type == "group_comparison":

        values = result[
            "values"
        ]

        metric = result[
            "metric_column"
        ]

        category = result[
            "category_column"
        ]

        response = (
            f"Here's the comparison of "
            f"**{metric}** by **{category}**:"
            "\n\n"
        )

        for name, value in values.items():

            response += (
                f"- **{name}:** "
                f"{value:,.2f}\n"
            )

        return response

    # ========================================================
    # LOOKUP
    # ========================================================

    if result_type == "lookup":

        values = result[
            "values"
        ]

        if len(values) == 1:

            return (
                f"The **{result['target_column']}** "
                f"for **{result['filter_column']} "
                f"{result['filter_value']}** is "
                f"**{values[0]}**."
            )

        formatted_values = ", ".join(
            map(
                str,
                values
            )
        )

        return (
            f"The values are: "
            f"**{formatted_values}**."
        )

    # ========================================================
    # NOT FOUND
    # ========================================================

    if result_type == "not_found":

        return (
            "I couldn't find any records matching "
            f"**{result['filter_column']} = "
            f"{result['filter_value']}**."
        )

    # ========================================================
    # ROWS
    # ========================================================

    if result_type == "rows":

        rows = result[
            "rows"
        ]

        if len(rows) == 1:

            row = rows[0]

            return "\n".join(

                f"**{key}:** {value}"

                for key, value
                in row.items()
            )

        return (
            f"I found **{len(rows)} "
            f"matching records**."
        )

    return None


# ============================================================
# QWEN FALLBACK
# ============================================================

def ask_qwen(
    question,
    df
):

    columns = df.columns.tolist()

    # Keep prompt extremely small
    prompt = f"""
You are InsightAI, a concise AI Data Analyst.

User question:
{question}

Available dataset columns:
{columns}

Answer the question naturally.

Rules:
- Be concise.
- Answer only what was asked.
- Do not mention Python.
- Do not mention Pandas.
- Do not mention internal tools.
- Do not invent numbers.
- If the dataset does not contain enough information,
  say that clearly.
- Keep the answer under 100 words.

/no_think
"""

    response = ollama.chat(

        model=MODEL,

        messages=[
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ],

        options={
            "num_predict": 128
        }
    )

    answer = (
        response[
            "message"
        ][
            "content"
        ]
        .strip()
    )

    # --------------------------------------------------------
    # Never return a blank answer
    # --------------------------------------------------------

    if not answer:

        return (
            "I couldn't generate an answer for that "
            "question. Try asking about a specific "
            "column, category, or metric."
        )

    return answer


# ============================================================
# MAIN ANALYST
# ============================================================

def analyze_question(
    question,
    df
):

    question = question.strip()

    if not question:

        return {
            "answer":
                "Please enter a question."
        }

    # ========================================================
    # 1. TOP RECORD
    # ========================================================

    top_result = perform_top_record(
        question,
        df
    )

    if top_result is not None:

        answer = format_answer(
            top_result
        )

        if answer:

            return {
                "answer":
                    answer
            }

    # ========================================================
    # 2. GROUP / COMPARISON
    # ========================================================

    group_result = perform_group_analysis(
        question,
        df
    )

    if group_result is not None:

        answer = format_answer(
            group_result
        )

        if answer:

            return {
                "answer":
                    answer
            }

    # ========================================================
    # 3. BASIC CALCULATION
    # ========================================================

    calculation_result = perform_calculation(
        question,
        df
    )

    if calculation_result is not None:

        answer = format_answer(
            calculation_result
        )

        if answer:

            return {
                "answer":
                    answer
            }

    # ========================================================
    # 4. LOOKUP
    # ========================================================

    lookup_result = perform_lookup(
        question,
        df
    )

    if lookup_result is not None:

        answer = format_answer(
            lookup_result
        )

        if answer:

            return {
                "answer":
                    answer
            }

    # ========================================================
    # 5. QWEN FALLBACK
    # ========================================================

    answer = ask_qwen(
        question,
        df
    )

    return {
        "answer":
            answer
    }