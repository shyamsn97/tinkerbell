"""Utility functions for efficient tensor serialization/deserialization."""

import fnmatch
import os
import pickle
import re
import signal
import socket
import sys
import threading
from typing import Any

import dill
import psutil


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
