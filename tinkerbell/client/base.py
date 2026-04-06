import asyncio
import functools
import logging
import time
from abc import ABC
from typing import Any, Callable, Generic, TypeVar

import httpx

from tinker.types import Datum

from tinkerbell.renderer import MASK_TOKEN_ID, Renderer, RenderMode, TrainOnWhat
from tinkerbell.types.responses import RemoteFuture

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Transient HTTP status codes that should be retried
RETRYABLE_STATUS_CODES = (408, 429, 502, 503, 504)


def retry_on_transient_error(
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_multiplier: float = 2.0,
    retryable_codes: tuple[int, ...] = RETRYABLE_STATUS_CODES,
):
    """
    Decorator that retries a function on transient HTTP errors.

    Works with functions that return an httpx.Response or call response.raise_for_status().

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        retryable_codes: HTTP status codes that trigger a retry
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in retryable_codes:
                        last_exception = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"{func.__name__} got {e.response.status_code}, "
                                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(delay)
                            delay *= backoff_multiplier
                            continue
                    raise

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class BaseFuture(ABC, Generic[T]):
    """Base class for Tinkerbell futures."""

    def __init__(
        self,
        remote_future: RemoteFuture,
        server_url: str,
        result_parser: Callable[[dict[str, Any]], T],
        poll_interval: float = 0.5,  # 500ms default poll interval
        timeout: float | None = None,
        client: httpx.Client | None = None,  # Reuse existing client
    ):
        self._remote_future = remote_future
        self._server_url = server_url
        self._result_parser = result_parser
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._result = None
        self._resolved = False
        self._shared_client = client  # Reuse connection

    @property
    def done(self) -> bool:
        return self._resolved

    @property
    def request_id(self) -> str:
        return self._remote_future.request_id

    def _get_client_config(self) -> tuple[httpx.Timeout, httpx.Limits]:
        """Get HTTP client configuration."""
        timeout_config = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
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

    def _poll_once(self, client: httpx.Client) -> T | None:
        """Single poll attempt. Returns result if ready, None if pending."""
        response = client.post("/poll_result", json=self._make_poll_payload())
        response.raise_for_status()
        data = response.json()
        return self._handle_poll_response(data.get("status", "pending"), data)

    def result(self) -> T:
        if self._resolved:
            return self._result

        start_time = time.time()

        # Reuse shared client if available (faster - keeps connection alive)
        if self._shared_client is not None:
            while True:
                self._check_timeout(start_time)
                result = self._poll_once(self._shared_client)
                if result is not None:
                    return result
                time.sleep(self._poll_interval)

        # Fallback: create new client (slower but works without shared client)
        timeout_config, limits = self._get_client_config()
        with httpx.Client(
            base_url=self._server_url, timeout=timeout_config, limits=limits
        ) as client:
            while True:
                self._check_timeout(start_time)
                result = self._poll_once(client)
                if result is not None:
                    return result
                time.sleep(self._poll_interval)


class AsyncTinkerbellFuture(BaseFuture[T]):
    """Async future for Tinkerbell operations. Usage: future = await client.op_async(); result = await future"""

    def __init__(self, *args, async_client: httpx.AsyncClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_async_client = async_client

    async def _poll_once(self, client: httpx.AsyncClient) -> T | None:
        """Single poll attempt. Returns result if ready, None if pending."""
        response = await client.post("/poll_result", json=self._make_poll_payload())
        response.raise_for_status()
        data = response.json()
        return self._handle_poll_response(data.get("status", "pending"), data)

    async def _poll_for_result(self) -> T:
        if self._resolved:
            return self._result

        start_time = time.time()

        # Reuse shared client if available
        if self._shared_async_client is not None:
            while True:
                self._check_timeout(start_time)
                result = await self._poll_once(self._shared_async_client)
                if result is not None:
                    return result
                await asyncio.sleep(self._poll_interval)

        # Fallback: create new client
        timeout_config, limits = self._get_client_config()
        async with httpx.AsyncClient(
            base_url=self._server_url, timeout=timeout_config, limits=limits
        ) as client:
            while True:
                self._check_timeout(start_time)
                result = await self._poll_once(client)
                if result is not None:
                    return result
                await asyncio.sleep(self._poll_interval)

    def __await__(self):
        return self._poll_for_result().__await__()


class BaseClient:
    def __init__(
        self,
        server_url: str | None = None,
        timeout: float = 600.0,
        base_model: str | None = None,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.base_model = base_model
        self._client = None
        self._async_client = None
        self._tokenizer = None
        self._renderer = None

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

    def _get_client_config(self):
        timeout = httpx.Timeout(connect=30.0, read=self.timeout, write=30.0, pool=30.0)
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        return timeout, limits

    def _create_client(self) -> httpx.Client:
        timeout, limits = self._get_client_config()
        return httpx.Client(
            base_url=self.server_url,
            timeout=timeout,
            http2=True,
            transport=httpx.HTTPTransport(retries=3, limits=limits),
            follow_redirects=True,
        )

    def _create_async_client(self) -> httpx.AsyncClient:
        timeout, limits = self._get_client_config()
        return httpx.AsyncClient(
            base_url=self.server_url,
            timeout=timeout,
            http2=True,
            transport=httpx.AsyncHTTPTransport(retries=3, limits=limits),
            follow_redirects=True,
        )

    def _make_future(self, remote_future: RemoteFuture, parse_fn, is_async=False):
        if is_async:
            return AsyncTinkerbellFuture(
                remote_future=remote_future,
                server_url=self.server_url,
                result_parser=parse_fn or (lambda x: x),
                poll_interval=0.5,  # 500ms polling
                timeout=self.timeout,
                async_client=self._async_client,  # Reuse connection
            )
        else:
            return TinkerbellFuture(
                remote_future=remote_future,
                server_url=self.server_url,
                result_parser=parse_fn or (lambda x: x),
                poll_interval=0.5,  # 500ms polling
                timeout=self.timeout,
                client=self._client,  # Reuse connection
            )

    def create_future(
        self,
        request: Any,
        endpoint: str,
        parse_result_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> TinkerbellFuture[Any]:
        response = self.client.post(
            endpoint, json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return self._make_future(RemoteFuture(**response.json()), parse_result_fn)

    def create_future_from_remote(
        self,
        remote_future: RemoteFuture,
        parse_result_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> TinkerbellFuture[Any]:
        return self._make_future(remote_future, parse_result_fn)

    async def create_async_future(
        self,
        request: Any,
        endpoint: str,
        parse_result_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> AsyncTinkerbellFuture[Any]:
        response = await self.async_client.post(
            endpoint, json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return self._make_future(
            RemoteFuture(**response.json()), parse_result_fn, is_async=True
        )

    def create_async_future_from_request_id(
        self,
        remote_future: RemoteFuture,
        parse_result_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> AsyncTinkerbellFuture[Any]:
        return self._make_future(remote_future, parse_result_fn, is_async=True)

    def get_tokenizer(self):
        """Get or create a tokenizer for the base model."""
        if self.base_model is None:
            raise ValueError("base_model is required to get tokenizer")
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or 0
        return self._tokenizer

    def get_renderer(self) -> Renderer:
        """Get or create a renderer for the base model."""
        if self.base_model is None:
            raise ValueError("base_model is required to get renderer")
        if self._renderer is None:
            self._renderer = Renderer(self.get_tokenizer())
        return self._renderer

    def render(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        mode: RenderMode = RenderMode.TRAINING,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        continue_final_message: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> list[Datum]:
        """Render chat messages into Datum objects.

        Args:
            messages: Single conversation or batch of conversations.
            mode: TRAINING (with labels) or INFERENCE (for sampling).
            train_on_what: Which messages to train on (only for TRAINING mode).
            mask_value: Value for masked tokens (only for TRAINING mode).
            continue_final_message: If True in INFERENCE mode, continue from a
                                    partial assistant message.]
            add_generation_prompt: If True, add the generation prompt to the messages.
            **kwargs: Additional args passed to apply_chat_template.

        Returns:
            List of Datum objects ready for training or inference.
        """
        renderer = self.get_renderer()
        return renderer.render(
            messages=messages,
            mode=mode,
            train_on_what=train_on_what,
            mask_value=mask_value,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def build_chat_samples(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        **kwargs,
    ) -> list[Datum]:
        """Backwards-compatible alias for render(mode=TRAINING)."""
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render(messages, mode, train_on_what, mask_value, **kwargs)

    def build_message_samples(
        self,
        messages: list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        **kwargs,
    ) -> Datum:
        """Backwards-compatible alias for render(mode=TRAINING) with single message."""
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render([messages], mode, train_on_what, mask_value, **kwargs)[0]

    def close(self):
        """Close the HTTP client."""
        self.client.close()
        if self._async_client is not None:
            import warnings

            warnings.warn(
                "Async client not closed. Use 'async with' or call await client.aclose()",
                ResourceWarning,
            )

    async def aclose(self):
        """Close both sync and async HTTP clients."""
        self.client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
