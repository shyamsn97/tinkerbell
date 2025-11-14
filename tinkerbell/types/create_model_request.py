from typing import Any, Optional

from typing_extensions import Literal

from ._models import StrictBase
from .lora_config import LoraConfig

__all__ = ["CreateModelRequest"]


class CreateModelRequest(StrictBase):
    """Request to create/initialize a model."""

    session_id: str
    """Associated session ID"""

    model_seq_id: int
    """Sequence ID for this model in the session"""

    base_model: str
    """Base model name or path"""

    user_metadata: Optional[dict[str, Any]] = None
    """Optional metadata about this model/training run, set by the end-user"""

    lora_config: Optional[LoraConfig] = None
    """Optional LoRA configuration"""

    type: Literal["create_model"] = "create_model"
