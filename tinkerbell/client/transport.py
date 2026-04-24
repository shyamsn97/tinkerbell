"""HTTP transport + uniform JobHandle.

Consolidates the v1 `BaseClient` + `BaseFuture` + `TinkerbellFuture` +
`AsyncTinkerbellFuture` + `retry_on_transient_error` into a single async-
first transport with a thin sync wrapper.

Every async op on the gateway returns `{"job_id": "...", "kind": "..."}`.
Clients wrap that in a `JobHandle` that supports both `handle.result()`
(sync poll loop) and `await handle` (async poll loop). The parser is
attached per-request so typed responses come out of `.result()`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Generic, Optional, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = (408, 429, 502, 503, 504)


class AsyncTransport:
    """Owns one httpx.Client + one httpx.AsyncClient, plus server_url."""

    def __init__(
        self,
        server_url: str,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._sync: httpx.Client | None = None
        self._async: httpx.AsyncClient | None = None

    # ---- transport config ----

    def _config(self) -> tuple[httpx.Timeout, httpx.Limits]:
        t = httpx.Timeout(connect=30.0, read=self.timeout, write=30.0, pool=30.0)
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        return t, limits

    @property
    def sync(self) -> httpx.Client:
        if self._sync is None:
            t, limits = self._config()
            self._sync = httpx.Client(
                base_url=self.server_url,
                timeout=t,
                http2=True,
                transport=httpx.HTTPTransport(retries=3, limits=limits),
                follow_redirects=True,
            )
        return self._sync

    @property
    def aio(self) -> httpx.AsyncClient:
        if self._async is None:
            t, limits = self._config()
            self._async = httpx.AsyncClient(
                base_url=self.server_url,
                timeout=t,
                http2=True,
                transport=httpx.AsyncHTTPTransport(retries=3, limits=limits),
                follow_redirects=True,
            )
        return self._async

    # ---- submit / poll ----

    def submit(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST and return parsed JSON (typically a JobHandle-shaped dict)."""
        last_err: Exception | None = None
        delay = 1.0
        for attempt in range(5):
            try:
                r = self.sync.post(endpoint, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in RETRYABLE_STATUS_CODES:
                    last_err = e
                    if attempt < 4:
                        logger.warning(
                            f"{endpoint}: {e.response.status_code}, retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise
        if last_err:
            raise last_err
        return {}

    def poll(self, job_id: str) -> dict[str, Any]:
        r = self.sync.post("/poll", json={"request_id": job_id})
        r.raise_for_status()
        return r.json()

    async def poll_async(self, job_id: str) -> dict[str, Any]:
        r = await self.aio.post("/poll", json={"request_id": job_id})
        r.raise_for_status()
        return r.json()

    # ---- cleanup ----

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None
        if self._async is not None:
            await self._async.aclose()
            self._async = None


# ---------------------------------------------------------------------------
# JobHandle: sync .result() AND `await handle`
# ---------------------------------------------------------------------------


class JobHandle(Generic[T]):
    """Polls `/poll` until a job_id resolves. Typed by `parse`.

    Use .result() for sync callers, `await handle` for async callers. The
    same object supports both.
    """

    def __init__(
        self,
        transport: AsyncTransport,
        job_id: str,
        parse: Callable[[dict[str, Any]], T] = (lambda r: r),  # type: ignore[assignment]
        timeout: Optional[float] = None,
    ):
        self.transport = transport
        self.job_id = job_id
        self.parse = parse
        self.timeout = timeout
        self._resolved: bool = False
        self._value: T | None = None

    @property
    def done(self) -> bool:
        return self._resolved

    # ---- sync ----

    def result(self) -> T:
        if self._resolved:
            return self._value  # type: ignore[return-value]
        start = time.time()
        while True:
            if self.timeout and time.time() - start > self.timeout:
                raise TimeoutError(
                    f"Timeout waiting for result after {self.timeout}s (job_id={self.job_id})"
                )
            data = self.transport.poll(self.job_id)
            resolved = self._handle_poll(data)
            if resolved is not None:
                return resolved
            time.sleep(self.transport.poll_interval)

    # ---- async ----

    def __await__(self):
        return self._poll_async().__await__()

    async def _poll_async(self) -> T:
        if self._resolved:
            return self._value  # type: ignore[return-value]
        start = time.time()
        while True:
            if self.timeout and time.time() - start > self.timeout:
                raise TimeoutError(
                    f"Timeout waiting for result after {self.timeout}s (job_id={self.job_id})"
                )
            data = await self.transport.poll_async(self.job_id)
            resolved = self._handle_poll(data)
            if resolved is not None:
                return resolved
            await asyncio.sleep(self.transport.poll_interval)

    def _handle_poll(self, data: dict[str, Any]) -> T | None:
        status = data.get("status", "pending")
        if status == "completed":
            self._value = self.parse(data.get("result") or {})
            self._resolved = True
            return self._value
        if status == "error":
            err = data.get("error")
            if not err:
                # Server indicated failure but didn't populate `error`.
                # Dump the full payload so the failure is at least debuggable.
                err = f"error with empty detail; full response={data!r}"
            raise RuntimeError(f"Job {self.job_id} failed: {err}")
        return None  # pending
