# from https://github.com/thinking-machines-lab/tinker/blob/main/src/tinker/types/tensor_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from ._models import StrictBase
from .tensor_dtype import TensorDtype

if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any


def _convert_tensor_dtype_to_numpy(dtype: TensorDtype) -> npt.DTypeLike:
    """Convert TensorDtype to numpy dtype-like."""
    if dtype == "float32":
        return np.float32
    elif dtype == "int64":
        return np.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_tensor_dtype_to_torch(dtype: TensorDtype) -> torch.dtype:
    """Convert TensorDtype to torch dtype."""
    if dtype == "float32":
        return torch.float32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_numpy_dtype_to_tensor(dtype: np.dtype[Any]) -> TensorDtype:
    """Convert numpy dtype to TensorDtype."""
    if dtype.kind == "f":
        return "float32"
    elif dtype.kind == "i":
        return "int64"
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")


def _convert_torch_dtype_to_tensor(dtype: torch.dtype) -> TensorDtype:
    """Convert torch dtype to TensorDtype."""
    # torch.dtype objects have .is_floating_point
    if getattr(dtype, "is_floating_point", False):
        return "float32"
    else:
        return "int64"


class TensorData(StrictBase):
    data: List[int] | List[float]
    """Flattened tensor data as array of numbers."""

    dtype: TensorDtype

    shape: List[int]
    """The shape of the tensor (see PyTorch tensor.shape)."""

    @classmethod
    def from_numpy(cls, array: npt.NDArray[Any]) -> TensorData:
        return cls(
            data=array.flatten().tolist(),
            dtype=_convert_numpy_dtype_to_tensor(array.dtype),
            shape=list(array.shape),
        )

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> TensorData:
        return cls(
            data=tensor.flatten().tolist(),
            dtype=_convert_torch_dtype_to_tensor(tensor.dtype),
            shape=list(tensor.shape),
        )

    def to_numpy(self) -> npt.NDArray[Any]:
        """Convert TensorData to numpy array."""
        numpy_dtype = _convert_tensor_dtype_to_numpy(self.dtype)
        arr = np.array(self.data, dtype=numpy_dtype)
        if self.shape is not None:
            arr = arr.reshape(self.shape)
        return arr

    def to_torch(self, device: Any = None) -> torch.Tensor:
        """Convert TensorData to torch tensor."""
        torch_dtype = _convert_tensor_dtype_to_torch(self.dtype)
        tensor = torch.tensor(self.data, dtype=torch_dtype)
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def tolist(self) -> List[Any]:
        return self.to_numpy().tolist()

    def slice(self, index: int) -> TensorData:
        torch_tensor = self.to_torch()
        torch_tensor = torch_tensor[index]
        return TensorData.from_torch(torch_tensor)


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


# Type definitions for multimodal input data
# Individual data item types for each modality
ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, Dict]
# Union type for any multimodal data item
MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]
# Format types supporting single items, lists, or nested lists for batch processing
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
