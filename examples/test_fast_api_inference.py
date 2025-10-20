import requests
from pydantic import BaseModel
import json

class Character(BaseModel):
    name: str
    description: str
    personality: str
    backstory: str
    appearance: str
    dialogue: str

BASE_URL = "https://jesterlabs--example-fastapi-main.modal.run"

schema = Character.model_json_schema()

health_response = requests.get(f"{BASE_URL}/health")
print(health_response)
print(health_response.json())

response = requests.post(f"{BASE_URL}/batch_generate", json={
    "prompts": ["Please generate a character in json format"],
    "sampling_params": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "json_schema": json.dumps(schema),
    },
    "engine_kwargs": {}
})

print(response.json())