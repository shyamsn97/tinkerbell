import requests

BASE_URL = "https://jesterlabs--sglang-model.modal.run"
# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Generate text
response = requests.post(
    f"{BASE_URL}/batch_generate",
    json=[{
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0.7
    }]
)
print(response.status_code)
print(response.json())