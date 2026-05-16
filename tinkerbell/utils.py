"""Utility functions for tinkerbell."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import signal
import socket
import sys
import threading
from typing import Any

import psutil
import torch

logger = logging.getLogger(__name__)


def clean_model_name(name: str) -> str:
    """Convert a model path to a clean actor name.

    e.g., "Qwen/Qwen3-0.6B" -> "qwen_qwen3-0.6b"
    """
    return name.replace("/", "_").replace(":", "_").lower()


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
