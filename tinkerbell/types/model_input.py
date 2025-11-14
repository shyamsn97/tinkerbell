from typing import Dict, List

from ._models import StrictBase
from .data import TensorData

__all__ = ["ModelInput"]


class ModelInput(StrictBase):
    """Model input representing token sequences and associated data.

    Simplified from tinker's chunk-based approach for tinkerbell compatibility.
    """

    tokens: List[int]
    """List of input token IDs"""

    attention_mask: List[int] | None = None
    """Optional attention mask"""

    additional_inputs: Dict[str, TensorData] | None = None
    """Optional additional inputs as tensors"""

    @classmethod
    def from_tokens(cls, tokens: List[int]) -> "ModelInput":
        """Create a ModelInput from a list of token IDs."""
        return cls(tokens=tokens)

    @property
    def length(self) -> int:
        """Return the total context length."""
        return len(self.tokens)
