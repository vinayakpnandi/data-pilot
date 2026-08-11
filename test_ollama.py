import ollama

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Say hello to InsightAI in one sentence."
        }
    ]
)

print(response["message"]["content"])