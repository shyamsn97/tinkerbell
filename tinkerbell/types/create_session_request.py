from __future__ import annotations

from typing import Any

from typing_extensions import Literal

from ._models import StrictBase

__all__ = ["CreateSessionRequest"]


class CreateSessionRequest(StrictBase):
    """Request to create a new training session."""

    tags: list[str]
    """Tags to categorize this session"""

    user_metadata: dict[str, Any] | None = None
    """Optional user-defined metadata"""

    sdk_version: str
    """Version of the SDK being used"""

    type: Literal["create_session"] = "create_session"
