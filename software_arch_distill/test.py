import os
import requests

key = os.getenv("OPENROUTER_API_KEY")
print(f"Key starts with: {key[:10] if key else 'None'}")
print(f"Key length: {len(key) if key else 0}")

resp = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text}")
