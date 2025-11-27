import asyncio
import time
from abc import ABC
from typing import Any, Callable, Generic, TypeVar

import httpx

from tinkerbell.types.responses import RemoteFuture

T = TypeVar("T")


class BaseFuture(ABC, Generic[T]):
    """Base class for Tinkerbell futures."""

    def __init__(
        self,
        remote_future: RemoteFuture,
        server_url: str,
        result_parser: Callable[[dict[str, Any]], T],
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ):
        self._remote_future = remote_future
        self._server_url = server_url
        self._result_parser = result_parser
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._result = None
        self._resolved = False

    @property
    def done(self) -> bool:
        return self._resolved

    @property
    def request_id(self) -> str:
        return self._remote_future.request_id

    def _get_client_config(self) -> tuple[httpx.Timeout, httpx.Limits]:
        """Get HTTP client configuration."""
        timeout_config = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
        return timeout_config, limits

    def _handle_poll_response(self, status: str, data: dict[str, Any]) -> T | None:
        """Handle poll response. Returns result if completed, None if pending, raises on error."""
        if status == "completed":
            self._result = self._result_parser(data.get("result"))
            self._resolved = True
            return self._result
        elif status == "error":
            raise RuntimeError(f"Request failed: {data.get('error', 'Unknown error')}")
        elif status == "pending":
            return None
        else:
            raise Exception(f"Unknown status: {status}")

    def _check_timeout(self, start_time: float) -> None:
        """Check if timeout has been exceeded."""
        if self._timeout and (time.time() - start_time) > self._timeout:
            raise TimeoutError(f"Timeout waiting for result after {self._timeout}s")

    def _make_poll_payload(self) -> dict[str, Any]:
        """Create poll request payload."""
        return self._remote_future.model_dump(exclude_none=True)


class TinkerbellFuture(BaseFuture[T]):
    """Sync future for Tinkerbell operations."""

    def result(self) -> T:
        if self._resolved:
            return self._result

        start_time = time.time()
        time.sleep(0.1)

        timeout_config, limits = self._get_client_config()

        with httpx.Client(
            base_url=self._server_url, timeout=timeout_config, limits=limits
        ) as client:
            while True:
                if self._resolved:
                    return self._result

                self._check_timeout(start_time)

                response = client.post("/poll_result", json=self._make_poll_payload())
                response.raise_for_status()
                data = response.json()

                result = self._handle_poll_response(data.get("status", "pending"), data)
                if result is not None:
                    return result

                time.sleep(self._poll_interval)


class AsyncTinkerbellFuture(BaseFuture[T]):
    """Async future for Tinkerbell operations. Usage: future = await client.op_async(); result = await future"""

    async def _poll_for_result(self) -> T:
        if self._resolved:
            return self._result

        start_time = time.time()
        await asyncio.sleep(0.1)

        timeout_config, limits = self._get_client_config()

        async with httpx.AsyncClient(
            base_url=self._server_url, timeout=timeout_config, limits=limits
        ) as client:
            while True:
                if self._resolved:
                    return self._result

                self._check_timeout(start_time)

                response = await client.post(
                    "/poll_result", json=self._make_poll_payload()
                )
                response.raise_for_status()
                data = response.json()

                result = self._handle_poll_response(data.get("status", "pending"), data)
                if result is not None:
                    return result

                await asyncio.sleep(self._poll_interval)

    def __await__(self):
        return self._poll_for_result().__await__()


class BaseClient:
    def __init__(self, server_url: str | None = None, timeout: float = 600.0):
        self.server_url = server_url
        self.timeout = timeout
        self._client = None
        self._async_client = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @client.setter
    def client(self, client: httpx.Client):
        self._client = client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = self._create_async_client()
        return self._async_client

    def _create_client(self) -> httpx.Client:
        timeout_config = httpx.Timeout(
            connect=30.0, read=self.timeout, write=30.0, pool=30.0
        )
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        transport = httpx.HTTPTransport(retries=3, limits=limits)
        return httpx.Client(
            base_url=self.server_url,
            timeout=timeout_config,
            transport=transport,
            follow_redirects=True,
        )

    def _create_async_client(self) -> httpx.AsyncClient:
        timeout_config = httpx.Timeout(
            connect=30.0, read=self.timeout, write=30.0, pool=30.0
        )
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        transport = httpx.AsyncHTTPTransport(retries=3, limits=limits)
        return httpx.AsyncClient(
            base_url=self.server_url,
            timeout=timeout_config,
            transport=transport,
            follow_redirects=True,
        )

    def create_future(
        self,
        request: Any,
        endpoint: str,
        parse_result_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> TinkerbellFuture[Any]:
        if parse_result_fn is None:
            parse_result_fn = lambda x: x

        response = self.client.post(
            endpoint, json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        remote_future = RemoteFuture(**response.json())
        return self.create_future_from_request_id(
            remote_future=remote_future, parse_result_fn=parse_result_fn
        )

    def create_future_from_request_id(
        self,
        remote_future: RemoteFuture,
        parse_result_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> TinkerbellFuture[Any]:
        if parse_result_fn is None:
            parse_result_fn = lambda x: x

        return TinkerbellFuture(
            remote_future=remote_future,
            server_url=self.server_url,
            result_parser=parse_result_fn,
            poll_interval=1.0,
            timeout=self.timeout,
        )
