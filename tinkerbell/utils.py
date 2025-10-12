"""Utility functions for efficient tensor serialization/deserialization."""

import io
from typing import Any

import torch


def serialize_tensor(obj: Any) -> bytes:
    """Serialize a tensor or nested structure of tensors to bytes.
    Args:
        obj: A torch tensor, list, dict, or nested structure containing tensors
    Returns:
        bytes: Serialized representation
    """
    buffer = io.BytesIO()
    torch.save(obj, buffer, _use_new_zipfile_serialization=False)
    buffer.seek(0)
    return buffer.read()


def deserialize_tensor(data: bytes) -> Any:
    """Deserialize bytes back to tensor or nested structure.

    Args:
        data: Serialized bytes from serialize_tensor
    Returns:
        The deserialized tensor or nested structure
    """
    buffer = io.BytesIO(data)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu")


def serialize_payload(data: list[Any], loss_fn: Any = None, **kwargs) -> bytes:
    """Serialize a complete payload including data and additional parameters.

    Args:
        data: List of data points (can contain tensors)
        loss_fn: Optional loss function or serializable representation
        **kwargs: Additional parameters to serialize

    Returns:
        bytes: Serialized payload
    """
    payload = {"data": data, "loss_fn": loss_fn, **kwargs}
    return serialize_tensor(payload)


def deserialize_payload(data: bytes) -> dict:
    """Deserialize a complete payload.

    Args:
        data: Serialized bytes

    Returns:
        dict: Deserialized payload with 'data', 'loss_fn', and other fields
    """
    return deserialize_tensor(data)
