"""Tensor data types and padding utilities.

Re-exports TensorData from tinker SDK. Keeps internal helpers like PaddingStrategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from tinker.types.tensor_data import TensorData
from tinker.types.tensor_dtype import TensorDtype

from .base import StrictBase

if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any

__all__ = ["TensorData", "PaddingStrategy"]


def tensor_data_from_list(data: List[Any]) -> TensorData:
    """Create TensorData from a plain Python list (convenience helper).

    Tinker's TensorData doesn't have from_list(), so we provide this.
    """
    arr = np.array(data)
    dtype: TensorDtype = "int64" if arr.dtype.kind == "i" else "float32"
    return TensorData(data=data, dtype=dtype, shape=[len(data)])


# Monkey-patch from_list onto TensorData for backwards compatibility
if not hasattr(TensorData, "from_list"):
    TensorData.from_list = classmethod(lambda cls, data: tensor_data_from_list(data))  # type: ignore[attr-defined]


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, Dict]
MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]
MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],
    List[MultimodalDataInputItem],
    MultimodalDataInputItem,
]


class PaddingStrategy(StrictBase):
    padding_side: str = "left"
    padding_value: int | float = 0

    def pad_sequence(
        self,
        data: list[TensorData] | list[torch.Tensor],
    ) -> torch.Tensor:
        """Pad a list of 1D tensors to the same length."""
        from tinkerbell.utils import pad_sequence

        if isinstance(data[0], TensorData):
            tensors = [d.to_torch() for d in data]
        else:
            tensors = data
        return pad_sequence(
            tensors, padding_side=self.padding_side, pad_value=self.padding_value
        )
