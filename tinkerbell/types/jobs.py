"""Job-system types.

Every async op the API exposes goes through the same pipeline:
  submit → JobHandle(job_id) → client polls /poll → JobRecord

This file defines the models exchanged over the wire for that pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from tinkerbell.types.base import BaseModel


class JobKind(str, Enum):
    """Every op kind that can produce a JobRecord.

    Kept coarse: the exact op-specific payload lives in JobRecord.result.
    """

    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    ZERO_GRAD = "zero_grad"
    OPTIM_STEP = "optim_step"
    SAVE_CHECKPOINT = "save_checkpoint"
    PUSH_TO_HUB = "push_to_hub"
    LOAD_CHECKPOINT = "load_checkpoint"
    SAMPLE = "sample"
    SHUTDOWN_SAMPLING = "shutdown_sampling"
    SAMPLING_ACTOR_STATUS = "sampling_actor_status"


JobStatus = Literal["pending", "success", "error"]


class ErrorRecord(BaseModel):
    """Structured error payload. Replaces str(e) + the `{success: False}` dicts."""

    type: str
    message: str
    traceback: str | None = None

    @classmethod
    def from_exception(cls, exc: BaseException) -> "ErrorRecord":
        import traceback as _tb

        msg = str(exc).strip() or repr(exc)
        tb_str: str | None
        try:
            tb_str = "".join(
                _tb.format_exception(type(exc), exc, exc.__traceback__)
            ).strip()
        except Exception:
            tb_str = None
        return cls(type=type(exc).__name__, message=msg, traceback=tb_str)


class JobRecord(BaseModel):
    """One entry in the JobStore. Returned by /poll."""

    job_id: str
    kind: JobKind
    status: JobStatus
    result: dict[str, Any] | None = None
    error: ErrorRecord | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


class JobHandle(BaseModel):
    """Returned by the gateway after submitting any async op."""

    job_id: str
    kind: JobKind | None = None


class PollRequest(BaseModel):
    job_id: str
    pop: bool = True  # delete on read (default: yes, one-shot semantics)


class SubmitResponse(BaseModel):
    """Generic gateway response for any submit-style endpoint."""

    job_id: str
    kind: JobKind
    # Optional eager metadata the caller may need before polling.
    extras: dict[str, Any] = Field(default_factory=dict)
