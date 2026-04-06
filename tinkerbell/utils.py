"""Utility functions for efficient tensor serialization/deserialization."""

import fnmatch
import json
import os
import pickle
import re
import signal
import socket
import sys
import threading
from typing import Any

import dill
import numpy as np
import psutil
import torch


def save_dict_to_json(data: dict[str, Any], filepath: str) -> None:
    """Save a dictionary to a JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to the JSON file
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_dict_from_json(filepath: str) -> dict[str, Any]:
    """Load a dictionary from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary loaded from the file
    """
    with open(filepath, "r") as f:
        return json.load(f)


def clean_model_name(name: str) -> str:
    """Convert a model path to a clean actor name.

    e.g., "Qwen/Qwen3-0.6B" -> "qwen_qwen3-0.6b"
    """
    return name.replace("/", "_").replace(":", "_").lower()


def convert_to_tensor_data(key: str, value: Any) -> Any:
    """Convert torch.Tensor, numpy array, dict, or 1-D list to TensorData if needed."""
    from tinker.types import TensorData

    from tinkerbell.types._models import _key_to_type

    if isinstance(value, TensorData):
        return value
    elif isinstance(value, torch.Tensor):
        return TensorData.from_torch(value)
    elif isinstance(value, np.ndarray):
        return TensorData.from_numpy(value)
    elif (
        isinstance(value, dict)
        and "data" in value
        and "dtype" in value
    ):
        return TensorData(**value)
    elif isinstance(value, list):
        return TensorData(
            data=value, dtype=_key_to_type.get(key, "float32"), shape=[len(value)]
        )
    else:
        return value


def process_dict_values(data: dict[str, Any], converter_fn) -> dict[str, Any]:
    """Apply converter function to all dictionary values."""
    return {key: converter_fn(key, value) for key, value in data.items()}


def pad_sequence(
    tensors: list[torch.Tensor],
    padding_side: str = "right",
    pad_value: int = 0,
) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length.

    Args:
        tensors: List of 1D tensors to pad
        padding_side: 'left' or 'right' padding
        pad_value: Value to use for padding

    Returns:
        Stacked tensor of shape (batch_size, max_length)
    """
    max_len = max(len(t) for t in tensors)
    batch_size = len(tensors)
    device = tensors[0].device
    dtype = tensors[0].dtype

    padded = torch.full((batch_size, max_len), pad_value, dtype=dtype, device=device)

    for i, tensor in enumerate(tensors):
        length = len(tensor)
        if padding_side == "left":
            padded[i, max_len - length :] = tensor
        else:
            padded[i, :length] = tensor

    return padded


def stack_or_cat_tensors(
    tensors: list[torch.Tensor],
    padding_side: str = "left",
    pad_value: int = 0,
) -> torch.Tensor:
    """Stack or concatenate tensors, handling already-batched and variable-length cases."""
    if len(tensors) == 1:
        return tensors[0] if tensors[0].ndim >= 2 else tensors[0].unsqueeze(0)

    lengths = [t.shape[-1] if t.ndim >= 2 else len(t) for t in tensors]
    if len(set(lengths)) > 1:
        return pad_sequence(tensors, padding_side=padding_side, pad_value=pad_value)
    return torch.cat(tensors) if tensors[0].ndim >= 2 else torch.stack(tensors)


def get_actor_names_by_prefix(prefix: str, actors: list[dict[str, Any]]) -> list[str]:
    return [
        actor["name"]
        for actor in actors
        if actor["name"].startswith(prefix)
        if actor["namespace"] == "tinkerbell"
    ]


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


def serialize(obj: Any) -> bytes:
    """Serialize any object (tensor, class, nested structure) to bytes using dill."""
    return dill.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(data: bytes) -> Any:
    """Deserialize bytes back to the original object."""
    return dill.loads(data)


# Aliases for backwards compatibility
serialize_tensor = serialize
deserialize_tensor = deserialize
serialize_class = serialize
deserialize_class = deserialize


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


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def get_nested(data: dict, path: str, default: Any = None, separator: str = ".") -> Any:
    """Get a value from a nested dictionary using dot notation.

    Args:
        data: The dictionary to traverse
        path: Dot-separated path (e.g., "outer.inner.key")
        default: Default value if path not found
        separator: Path separator (default: ".")

    Returns:
        The value at the nested path, or default if not found

    Examples:
        >>> data = {"a": {"b": {"c": 42}}}
        >>> get_nested(data, "a.b.c")
        42
        >>> get_nested(data, "a.b.x", default=0)
        0
    """
    keys = path.split(separator)
    result = data

    try:
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default


def set_nested(data: dict, path: str, value: Any, separator: str = ".") -> None:
    """Set a value in a nested dictionary using dot notation.

    Args:
        data: The dictionary to modify (modified in-place)
        path: Dot-separated path (e.g., "outer.inner.key")
        value: Value to set
        separator: Path separator (default: ".")

    Examples:
        >>> data = {}
        >>> set_nested(data, "a.b.c", 42)
        >>> data
        {"a": {"b": {"c": 42}}}
    """
    keys = path.split(separator)
    current = data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def model_to_dict(obj, exclude: list[str] = [], exclude_none: bool = False):
    """Convert a Pydantic model instance to a dictionary.

    Args:
        obj: Pydantic model instance
        exclude: List of keys to exclude from the output
        exclude_none: If True, exclude fields with None values
    """
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        out = obj.model_dump(exclude_none=exclude_none)  # For Pydantic v2
        for key in exclude:
            out.pop(key, None)
        return out
    else:
        raise ValueError(f"Object {obj} is not a Pydantic model instance")
