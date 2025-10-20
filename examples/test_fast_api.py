from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import modal
from modal import runner
import json

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "uvicorn")

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    sampling_params: Dict[str, Any] = Field(
        default_factory=dict
    )
    engine_kwargs: Dict[str, Any] = Field(default_factory=dict)

app = modal.App("example-fastapi")

@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def main():
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/batch_generate")
    async def batch_generate(requests: BatchGenerateRequest):
        # print(requests)
        print(requests.json())
        loaded_json = json.loads(requests.json())
        print("SAMPLING PARAMS TYPE", type(loaded_json.get("sampling_params")))
        return requests.json()

    return app

with modal.enable_output():
    runner.deploy_app(app)