"""Tinkerbell HTTP API.

Stateless FastAPI surface + Ray Serve deployment. Translates HTTP requests
into `Op`s and puts them on `WorkQueue`; polls results from `JobStore`.
"""

from tinkerbell.api.deploy import deploy_on_modal, deploy_service

__all__ = ["deploy_service", "deploy_on_modal"]
