import pandas as pd

from analyst_agent import analyze_question


# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------

df = pd.DataFrame({

    "Region": [
        "North",
        "South",
        "North",
        "West",
        "South"
    ],

    "Sales": [
        1000,
        800,
        1200,
        700,
        900
    ],

    "Profit": [
        200,
        150,
        300,
        100,
        180
    ]
})


# ------------------------------------------------------------
# TEST QUESTION
# ------------------------------------------------------------

question = "Which region has the highest profit?"


# ------------------------------------------------------------
# RUN AGENT
# ------------------------------------------------------------

result = analyze_question(
    question,
    df
)


print("\n===================================")
print("OPERATION")
print("===================================")

print(result["operation"])


print("\n===================================")
print("PYTHON RESULT")
print("===================================")

print(result["result"])


print("\n===================================")
print("AI ANSWER")
print("===================================")

print(result["answer"])