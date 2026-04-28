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
import time
from typing import TYPE_CHECKING, Any

import psutil
import torch

if TYPE_CHECKING:
    from tinkerbell.training.engine import TrainingEngine
    from tinkerbell.types.route import RouteKey

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


# ---------------------------------------------------------------------------
# Ray named-actor discovery (used by the API on boot to back-fill Registry).
# ---------------------------------------------------------------------------


def list_tinkerbell_actors() -> list[dict]:
    """Return all named Ray actors in the "tinkerbell" namespace.

    Normalizes the shape across Ray versions: some return `list[str]`, some
    return `list[dict]`. Always returns `list[dict]` with `name` + `namespace`.
    """
    import ray

    try:
        actors = ray.util.list_named_actors(all_namespaces=True)
    except TypeError:
        actors = ray.util.list_named_actors()
    normalized = []
    for a in actors:
        if isinstance(a, str):
            normalized.append({"name": a, "namespace": "tinkerbell"})
        elif isinstance(a, dict):
            normalized.append(a)
    return [a for a in normalized if a.get("namespace") == "tinkerbell"]


def rehydrate_registry(
    registry, base_model_hint: dict[RouteKey, str] | None = None
) -> int:
    """Inspect Ray for training/sampling actors and register any missing ones.

    Called once on API boot. Walks the "tinkerbell" Ray namespace for actors
    matching known naming conventions and re-registers them if the Registry
    doesn't know about them (e.g. after a full cluster restart where detached
    actors survived but the Registry did not).

    Args:
        registry: a Registry Ray actor handle.
        base_model_hint: optional {route: base_model} to fill in base_model
            on rehydrated records (otherwise we use model_name as a stand-in).

    Returns the number of records added.
    """
    import ray

    from tinkerbell.state.registry import EngineRecord
    from tinkerbell.types.route import RouteKey as _RouteKey

    base_model_hint = base_model_hint or {}
    existing: dict[str, EngineRecord] = {
        str(r.route): r for r in ray.get(registry.list.remote())
    }
    added = 0

    # Training actors: naming is training_actor_{model_clean}_{rank}
    training_by_model: dict[str, list[str]] = {}
    sampling_by_model: dict[str, str] = {}
    for a in list_tinkerbell_actors():
        name = a.get("name", "")
        if name.startswith("training_actor_"):
            # Strip the trailing _{rank}
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                model_clean = parts[0][len("training_actor_") :]
                training_by_model.setdefault(model_clean, []).append(name)
        elif name.startswith("sampling_actor_"):
            model_clean = name[len("sampling_actor_") :]
            sampling_by_model[model_clean] = name

    now = time.time()
    for model_clean, worker_names in training_by_model.items():
        worker_names.sort()
        route = _RouteKey(model=model_clean, adapter=None)
        if str(route) in existing:
            continue
        record = EngineRecord(
            kind="training",
            route=route,
            base_model=base_model_hint.get(route, model_clean),
            worker_names=worker_names,
            created_at=now,
        )
        ray.get(registry.register.remote(record))
        added += 1
        logger.info(f"Rehydrated training engine {route} ({len(worker_names)} workers)")

    for model_clean, name in sampling_by_model.items():
        route = _RouteKey(model=model_clean, adapter=None)
        if str(route) in existing:
            continue
        record = EngineRecord(
            kind="sampling",
            route=route,
            base_model=base_model_hint.get(route, model_clean),
            worker_names=[name],
            created_at=now,
        )
        ray.get(registry.register.remote(record))
        added += 1
        logger.info(f"Rehydrated sampling engine {route}")

    return added


# ---------------------------------------------------------------------------
# Background job wrappers: run an engine method, write the result to JobStore.
# Kicked off by the API as fire-and-forget asyncio tasks so one stuck upload
# can't stall other work on the same engine.
# ---------------------------------------------------------------------------


async def run_save_checkpoint_job(
    engine: TrainingEngine,
    registry: Any,
    job_store: Any,
    job_id: str,
    checkpoint_path: str,
    adapter_name: str | None = None,
) -> None:
    from tinkerbell.types.jobs import ErrorRecord

    try:
        await engine.save_checkpoint(checkpoint_path, adapter_name=adapter_name)
        if adapter_name:
            await registry.update_adapter.remote(
                engine.route, adapter_name, checkpoint_path
            )
        await job_store.set_success.remote(
            job_id,
            {
                "model_name": engine.route.model,
                "success": True,
                "path": checkpoint_path,
            },
        )
    except Exception as e:
        logger.error(f"save_checkpoint failed: {e}", exc_info=True)
        await job_store.set_error.remote(job_id, ErrorRecord.from_exception(e))


async def run_push_to_hub_job(
    engine: TrainingEngine,
    job_store: Any,
    job_id: str,
    **push_kwargs: Any,
) -> None:
    from tinkerbell.types.jobs import ErrorRecord

    try:
        await engine.push_to_hub(**push_kwargs)
        await job_store.set_success.remote(
            job_id,
            {
                "model_name": engine.route.model,
                "success": True,
                "repo_id": push_kwargs.get("repo_id"),
            },
        )
    except Exception as e:
        logger.error(f"push_to_hub failed: {e}", exc_info=True)
        await job_store.set_error.remote(job_id, ErrorRecord.from_exception(e))
