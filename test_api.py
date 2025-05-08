import os, requests

key = os.getenv("GROQ_API_KEY")
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "llama3-70b-8192",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 10
}
response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
print(response.status_code)
print(response.text)