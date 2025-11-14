"""Utility functions for efficient tensor serialization/deserialization."""

import fnmatch
import pickle
import re
import socket
from typing import Any

import dill


def get_free_port() -> int:
    """
    Get a free port on the local machine.

    Returns:
        int: An available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_submodules_with_wildcard(model, pattern):
    """Get all submodules matching a wildcard pattern."""
    regex_pattern = fnmatch.translate(pattern)
    regex = re.compile(regex_pattern)

    matching_modules = []
    for name, module in model.named_modules():
        if regex.match(name):
            matching_modules.append(name)

    return matching_modules


def serialize_tensor(obj: Any) -> bytes:
    """Serialize a tensor or nested structure of tensors to bytes.
    Args:
        obj: A torch tensor, list, dict, or nested structure containing tensors
    Returns:
        bytes: Serialized representation
    """
    return dill.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_tensor(data: bytes) -> Any:
    """Deserialize bytes back to tensor or nested structure.

    Args:
        data: Serialized bytes from serialize_tensor
    Returns:
        The deserialized tensor or nested structure
    """
    return dill.loads(data)


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


def serialize_class(cls: type) -> bytes:
    """Serialize a class definition using dill."""
    return dill.dumps(cls, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_class(data: bytes) -> type:
    """Deserialize a class definition using dill."""
    return dill.loads(data)


def get_host_and_port(server_url: str) -> tuple[str, int | None]:
    """
    Parse server URL to extract host and port.

    Args:
        server_url: URL in format "http://host:port", "host:port", "http://host", or "host"

    Returns:
        Tuple of (host, port) where port is None if not specified
    """
    from urllib.parse import urlparse

    # Add scheme if not present to help urlparse
    if not server_url.startswith(("http://", "https://")):
        server_url = "https://" + server_url

    parsed = urlparse(server_url)
    host = parsed.hostname or parsed.netloc.split(":")[0]
    port = parsed.port

    return host, port
