"""System state: JobStore, WorkQueue, Registry.

Three small, detached Ray actors that hold the live state of the cluster.
Together they replace the old monolithic `GlobalStore`. See
docs/internal/design-v2.md §4.
"""

from tinkerbell.state.job_store import JobStore
from tinkerbell.state.registry import GroupHandle, Registry
from tinkerbell.state.work_queue import WorkQueue

__all__ = ["JobStore", "WorkQueue", "Registry", "GroupHandle"]
