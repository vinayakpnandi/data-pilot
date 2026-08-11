import time
import ollama


print("Starting Qwen 1.7B test...")

start = time.time()

response = ollama.chat(
    model="qwen3:1.7b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Hello"
        }
    ],
    options={
        "num_predict": 20
    }
)

end = time.time()

print("\nResponse:")
print(response["message"]["content"])

print("\nTime taken:")
print(round(end - start, 2), "seconds")