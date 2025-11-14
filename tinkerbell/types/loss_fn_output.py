from typing import Dict

from typing_extensions import TypeAlias

from .data import TensorData

__all__ = ["LossFnOutput"]

LossFnOutput: TypeAlias = Dict[str, TensorData]
