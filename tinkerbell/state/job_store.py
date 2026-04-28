"""JobStore: typed async job-result store.

Replaces `GlobalStore.results`. One concern: map job_id → JobRecord.
TTL purge runs from inside the actor on a timer; no external supervisor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import ray

from tinkerbell.types.jobs import ErrorRecord, JobKind, JobRecord

logger = logging.getLogger(__name__)


JOB_STORE_ACTOR_NAME = "tinkerbell:job_store"
DEFAULT_JOB_TTL_S = 3600.0  # 1 hour
PURGE_INTERVAL_S = 300.0  # 5 minutes


@ray.remote(num_cpus=0)
class JobStore:
    def __init__(self, ttl_s: float = DEFAULT_JOB_TTL_S):
        self.jobs: dict[str, JobRecord] = {}
        self.ttl_s = ttl_s
        # Started lazily on first method call (Ray's event loop isn't
        # guaranteed to be running inside __init__).
        self._purge_task: asyncio.Task | None = None

    def _ensure_purge_task(self) -> None:
        if self._purge_task is None or self._purge_task.done():
            self._purge_task = asyncio.create_task(self._purge_loop())

    async def _purge_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(PURGE_INTERVAL_S)
                expired = await self.purge_expired()
                if expired:
                    logger.info(f"JobStore: purged {expired} expired job records")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"JobStore purge tick failed: {e}")

    async def create(self, job_id: str, kind: JobKind) -> None:
        self._ensure_purge_task()
        now = time.time()
        self.jobs[job_id] = JobRecord(
            job_id=job_id,
            kind=kind,
            status="pending",
            created_at=now,
            updated_at=now,
        )

    async def set_success(self, job_id: str, result: dict[str, Any]) -> None:
        existing = self.jobs.get(job_id)
        kind = existing.kind if existing else JobKind.FORWARD
        self.jobs[job_id] = JobRecord(
            job_id=job_id,
            kind=kind,
            status="success",
            result=result,
            created_at=existing.created_at if existing else time.time(),
            updated_at=time.time(),
        )

    async def set_error(self, job_id: str, error: ErrorRecord) -> None:
        existing = self.jobs.get(job_id)
        kind = existing.kind if existing else JobKind.FORWARD
        self.jobs[job_id] = JobRecord(
            job_id=job_id,
            kind=kind,
            status="error",
            error=error,
            created_at=existing.created_at if existing else time.time(),
            updated_at=time.time(),
        )
        logger.error(f"Job {job_id} ({kind}) errored: {error.type}: {error.message}")

    async def poll(self, job_id: str, pop: bool = True) -> JobRecord | None:
        """Return job record; if completed (success/error) and `pop`, delete after read."""
        record = self.jobs.get(job_id)
        if record is None:
            return None
        if pop and record.status != "pending":
            del self.jobs[job_id]
        return record

    async def purge_expired(self) -> int:
        """Drop jobs older than ttl_s; return count removed."""
        cutoff = time.time() - self.ttl_s
        expired = [jid for jid, r in self.jobs.items() if r.updated_at < cutoff]
        for jid in expired:
            del self.jobs[jid]
        return len(expired)

    async def depth(self) -> int:
        return len(self.jobs)


def get_or_create_job_store(ttl_s: float = DEFAULT_JOB_TTL_S) -> Any:
    """Create or attach to the detached JobStore actor."""
    return JobStore.options(
        num_cpus=0,
        get_if_exists=True,
        lifetime="detached",
        name=JOB_STORE_ACTOR_NAME,
        namespace="tinkerbell",
    ).remote(ttl_s=ttl_s)
