from tinkerbell.client.common import BaseClient, JobHandle
from tinkerbell.client.sampling import SamplingClient
from tinkerbell.client.service import ServiceClient
from tinkerbell.client.training import TrainingClient
from tinkerbell.client.transport import AsyncTransport

# Backwards-compat alias: v1 scripts imported TinkerbellFuture.
TinkerbellFuture = JobHandle

__all__ = [
    "AsyncTransport",
    "BaseClient",
    "JobHandle",
    "SamplingClient",
    "ServiceClient",
    "TinkerbellFuture",
    "TrainingClient",
]
