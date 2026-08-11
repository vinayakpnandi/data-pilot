import streamlit as st
import pandas as pd
import numpy as np

from analyst_agent import analyze_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightAI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 20px;
    }

    .stMetric {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def format_number(value):

    if pd.isna(value):
        return "N/A"

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.2f}"


# ============================================================
# AUTOMATIC DATASET INSIGHTS
# ============================================================

def generate_insights(df):

    insights = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # --------------------------------------------------------
    # DATASET SIZE
    # --------------------------------------------------------

    insights.append(
        f"The dataset contains **{df.shape[0]:,} rows** "
        f"and **{df.shape[1]} columns**."
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:

        worst_column = missing.idxmax()
        worst_count = missing.max()

        percentage = (
            worst_count / len(df)
        ) * 100

        insights.append(
            f"⚠️ **{worst_column}** has the highest number "
            f"of missing values: **{worst_count:,} "
            f"({percentage:.1f}%)**."
        )

    else:

        insights.append(
            "✅ No missing values were found."
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicates = df.duplicated().sum()

    if duplicates > 0:

        insights.append(
            f"⚠️ The dataset contains **{duplicates:,} "
            f"duplicate rows**."
        )

    else:

        insights.append(
            "✅ No duplicate rows were detected."
        )

    # --------------------------------------------------------
    # NUMERICAL INSIGHTS
    # --------------------------------------------------------

    for column in numeric_columns[:5]:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        average = series.mean()
        maximum = series.max()
        minimum = series.min()

        insights.append(
            f"📈 **{column}** has an average of "
            f"**{format_number(average)}**, with values "
            f"ranging from **{format_number(minimum)}** "
            f"to **{format_number(maximum)}**."
        )

    # --------------------------------------------------------
    # HIGHEST VALUES
    # --------------------------------------------------------

    for column in numeric_columns[:5]:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        max_index = series.idxmax()

        if categorical_columns:

            category_column = categorical_columns[0]

            category_value = df.loc[
                max_index,
                category_column
            ]

            insights.append(
                f"🏆 The highest **{column}** value is "
                f"**{format_number(series.max())}**, "
                f"associated with **{category_column}: "
                f"{category_value}**."
            )

    # --------------------------------------------------------
    # CATEGORICAL INSIGHTS
    # --------------------------------------------------------

    for column in categorical_columns[:5]:

        value_counts = df[column].value_counts()

        if len(value_counts) == 0:
            continue

        top_value = value_counts.index[0]
        top_count = value_counts.iloc[0]

        percentage = (
            top_count / len(df)
        ) * 100

        insights.append(
            f"🔎 In **{column}**, the most common value "
            f"is **{top_value}**, appearing in "
            f"**{percentage:.1f}%** of records."
        )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    if len(numeric_columns) >= 2:

        correlation_matrix = df[
            numeric_columns
        ].corr()

        highest_pair = None
        highest_corr = 0

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                corr = correlation_matrix.iloc[i, j]

                if pd.isna(corr):
                    continue

                if abs(corr) > abs(highest_corr):

                    highest_corr = corr

                    highest_pair = (
                        numeric_columns[i],
                        numeric_columns[j]
                    )

        if highest_pair:

            insights.append(
                f"🔗 **{highest_pair[0]}** and "
                f"**{highest_pair[1]}** have a correlation "
                f"of **{highest_corr:.2f}**."
            )

    return insights


# ============================================================
# CACHED DATA CONTEXT
# ============================================================

@st.cache_data
def create_data_context(df):

    context = []

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    context.append(
        f"Dataset size: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    context.append(
        f"Columns: {', '.join(df.columns.tolist())}"
    )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    context.append(
        "\nDATA TYPES:"
    )

    for column in df.columns:

        context.append(
            f"- {column}: {df[column].dtype}"
        )

    # --------------------------------------------------------
    # NUMERICAL STATISTICS
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if not numeric_df.empty:

        context.append(
            "\nNUMERICAL STATISTICS:"
        )

        stats = numeric_df.describe().round(2)

        context.append(
            stats.to_string()
        )

    # --------------------------------------------------------
    # CATEGORICAL INFORMATION
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:

        context.append(
            "\nCATEGORICAL INFORMATION:"
        )

        for column in categorical_columns[:10]:

            values = (
                df[column]
                .value_counts()
                .head(10)
            )

            context.append(
                f"\n{column}:\n{values.to_string()}"
            )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    context.append(
        "\nMISSING VALUES:"
    )

    context.append(
        df.isna().sum().to_string()
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    context.append(
        f"\nDUPLICATE ROWS: "
        f"{df.duplicated().sum():,}"
    )

    return "\n".join(context)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 InsightAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your AI-powered Data Analyst Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a dataset and chat with your data "
    "using natural language."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 InsightAI")

    st.write(
        "AI Data Analyst Assistant"
    )

    st.divider()

    st.markdown(
        """
        **Technology**

        🐼 Pandas  
        🤖 Qwen3 4B  
        🦙 Ollama  
        ⚡ Streamlit  
        📈 Data Analysis
        """
    )

    st.divider()

    st.caption(
        "InsightAI uses Python for data calculations "
        "and Qwen for reasoning and explanations."
    )

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload your dataset",
    type=["csv", "xlsx"]
)


# ============================================================
# NO FILE UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a CSV or Excel file to start."
    )

    st.markdown(
        """
        ### What can InsightAI do?

        **📊 Understand your dataset**
        - Dataset size
        - Column types
        - Missing values
        - Duplicate records

        **💡 Find insights**
        - Averages
        - Totals
        - Maximum/minimum values
        - Category patterns
        - Correlations

        **💬 Chat with your data**
        - Ask questions naturally
        - Get business explanations
        - Ask follow-up questions
        - Explore your dataset conversationally
        """
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    if uploaded_file.name.lower().endswith(
        ".csv"
    ):

        df = pd.read_csv(
            uploaded_file
        )

    else:

        df = pd.read_excel(
            uploaded_file
        )

except Exception as e:

    st.error(
        "Unable to read the uploaded file."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"Successfully loaded **{uploaded_file.name}**"
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader(
    "📊 Dataset Overview"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:

    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )

with col4:

    st.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}"
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "👀 Preview Dataset",
    expanded=False
):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander(
    "🔎 Column Information",
    expanded=False
):

    column_info = pd.DataFrame({

        "Column": df.columns,

        "Data Type":
            df.dtypes.astype(str),

        "Missing Values":
            df.isna().sum().values,

        "Unique Values": [
            df[col].nunique()
            for col in df.columns
        ]

    })

    st.dataframe(
        column_info,
        use_container_width=True
    )


# ============================================================
# AUTOMATIC INSIGHTS
# ============================================================

with st.expander(
    "💡 Automatic Insights",
    expanded=False
):

    insights = generate_insights(
        df
    )

    for insight in insights:

        st.markdown(
            f"- {insight}"
        )


# ============================================================
# NUMERICAL ANALYSIS
# ============================================================

with st.expander(
    "📈 Numerical Analysis",
    expanded=False
):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if not numeric_df.empty:

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns detected."
        )


# ============================================================
# CORRELATION
# ============================================================

with st.expander(
    "🔗 Correlation Analysis",
    expanded=False
):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if len(numeric_df.columns) >= 2:

        correlation = numeric_df.corr()

        st.dataframe(
            correlation.round(2),
            use_container_width=True
        )

    else:

        st.info(
            "At least two numerical columns "
            "are required for correlation analysis."
        )


# ============================================================
# CREATE CACHED CONTEXT
# ============================================================

data_context = create_data_context(
    df
)


# ============================================================
# CHAT SECTION
# ============================================================

st.divider()

st.subheader(
    "💬 Chat with Your Data"
)

st.caption(
    "Ask questions naturally. InsightAI will perform "
    "the required analysis and show only the relevant answer."
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask InsightAI something about your data..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            try:

                result = analyze_question(
                    question,
                    df
                )

                answer = result.get(
                    "answer",
                    "I couldn't generate an answer."
                )

                st.markdown(
                    answer
                )

                # Save assistant response

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception:

                answer = (
                    "I couldn't answer that using "
                    "the available data. Try asking "
                    "the question in a different way."
                )

                st.markdown(
                    answer
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )