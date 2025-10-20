import httpx
from pydantic import BaseModel
import json

BASE_URL = "https://jesterlabs--sglang-model.modal.run"
# Health check
response = httpx.get(f"{BASE_URL}/health")
print(response.json())

class Character(BaseModel):
    name: str
    description: str
    personality: str
    backstory: str
    appearance: str
    dialogue: str

schema = Character.model_json_schema()
print(schema)

# Generate text
client = httpx.Client(timeout=60.0)

response = client.post(
    f"{BASE_URL}/batch_generate",
    json={
        "prompts": ["Please generate a fantasy character."],
        "sampling_params": {
            "max_new_tokens": 1024,
            "temperature": 1.0,
            "json_schema": json.dumps(schema),
        },
        "engine_kwargs": {
        }
    }
)

print(response.status_code)
print(response.json())