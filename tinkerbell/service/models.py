from pydantic import BaseModel, Field
from typing import Any

class CreateTrainingServerRequest(BaseModel):
    model_name: str
    model_config: dict[str, Any] = Field(default_factory=dict)
    lora_config: dict[str, Any] = Field(default_factory=dict)
    parallelize_plan: dict[str, str] = Field(default_factory=dict)

class CreateTrainingServerResponse(BaseModel):
    success: bool
    message: str
    training_server_url: str

class HealthResponse(BaseModel):
    status: str
    server_url: str
