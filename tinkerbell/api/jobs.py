"""Tiny shared job records for HTTP submit/poll."""

from __future__ import annotations

import json
from typing import Any

JOB_KV_NAMESPACE = "tinkerbell"
JOB_KV_PREFIX = "tinkerbell:job:"


def _internal_kv():
    from ray.experimental import internal_kv

    return internal_kv


def _key(job_id: str) -> bytes:
    return f"{JOB_KV_PREFIX}{job_id}".encode("utf-8")


def _encode(record: dict[str, Any]) -> bytes:
    return json.dumps(record).encode("utf-8")


def _decode(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def create_job(job_id: str) -> None:
    _internal_kv()._internal_kv_put(
        _key(job_id),
        _encode({"status": "pending"}),
        overwrite=True,
        namespace=JOB_KV_NAMESPACE,
    )


def set_job(job_id: str, record: dict[str, Any]) -> None:
    _internal_kv()._internal_kv_put(
        _key(job_id),
        _encode(record),
        overwrite=True,
        namespace=JOB_KV_NAMESPACE,
    )


def poll_job(job_id: str, pop: bool = True) -> dict[str, Any]:
    raw = _internal_kv()._internal_kv_get(_key(job_id), namespace=JOB_KV_NAMESPACE)
    if raw is None:
        return {"status": "error", "error": f"unknown job_id {job_id}"}
    record = _decode(raw)
    if pop and record.get("status") != "pending":
        _internal_kv()._internal_kv_del(_key(job_id), namespace=JOB_KV_NAMESPACE)
    return record


__all__ = ["create_job", "poll_job", "set_job"]
