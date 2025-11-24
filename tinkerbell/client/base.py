from typing import Any, Callable, Generic, TypeVar

import httpx

from tinkerbell.types.responses import RemoteFuture

T = TypeVar("T")


class TinkerbellFuture(Generic[T]):
    """
    A future-like object that wraps async operations in Tinkerbell.

    This allows for non-blocking API calls where the actual result
    can be retrieved later using the .result() method.

    The future polls the server using a request_id to check if the
    operation is complete.
    """

    def __init__(
        self,
        request_id: str,
        server_url: str,
        poll_endpoint: str,
        result_parser: Callable[[dict[str, Any]], T],
        poll_interval: float = 0.1,
        timeout: float | None = None,
        client_timeout: float = 30.0,
        model_id: str | None = None,
    ):
        """
        Initialize a TinkerbellFuture.

        Args:
            request_id: The request ID string to poll for
            server_url: Base URL of the server
            poll_endpoint: Endpoint to poll for results (e.g., "/get_result")
            result_parser: Function to parse the response into the result type
            poll_interval: Interval between polls in seconds
            timeout: Maximum time to wait for result
            client_timeout: Timeout for individual HTTP requests
            model_id: Optional model ID
        """
        self._request_id = RemoteFuture(request_id=request_id, model_id=model_id)
        self._server_url = server_url
        self._poll_endpoint = poll_endpoint
        self._result_parser = result_parser
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._client_timeout = client_timeout
        self._result = None
        self._resolved = False

    def _get_client(self) -> httpx.Client:
        """Get or create a light httpx client for polling."""
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=self._client_timeout,
            write=10.0,
            pool=10.0,
        )
        # Force new connection each time by limiting connection pool
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
        return httpx.Client(
            base_url=self._server_url,
            timeout=timeout_config,
            limits=limits,
        )

    @property
    def done(self) -> bool:
        """Check if the future is resolved."""
        return self._resolved

    @property
    def request_id(self) -> str:
        """Get the request ID for this future."""
        return self._request_id

    def result(self) -> Any:
        if self._resolved:
            return self._result

        import time

        start_time = time.time()
        # Small delay before first poll to let server process the request
        time.sleep(0.1)
        # Create fresh client each time to avoid connection issues
        client = self._get_client()
        with client:
            while True:
                # Check if already resolved (race condition protection)
                if self._resolved:
                    return self._result

                # Check timeout
                if self._timeout and (time.time() - start_time) > self._timeout:
                    raise TimeoutError(
                        f"Timeout waiting for result after {self._timeout}s"
                    )
                # Make the poll request
                try:
                    payload = self._request_id.model_dump(exclude_none=True)
                    response = client.post(self._poll_endpoint, json=payload)
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError:
                    raise
                status = data.get("status", "pending")

                if status == "completed":
                    result = self._result_parser(data.get("result"))
                    self._result = result
                    self._resolved = True
                    return result
                elif status == "error":
                    error_msg = data.get("error", "Unknown error")
                    raise RuntimeError(f"Request failed: {error_msg}")
                elif status == "pending":
                    # Sleep and retry
                    time.sleep(self._poll_interval)
                else:
                    raise Exception(f"Unknown status: {status}")


class BaseClient:
    def __init__(
        self,
        server_url: str | None = None,
        timeout: float = 600.0,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self._client = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @client.setter
    def client(self, client: httpx.Client):
        self._client = client

    def _create_client(self) -> httpx.Client:
        """Create an httpx client with robust timeout and transport settings."""
        timeout_config = httpx.Timeout(
            connect=30.0,  # Connection timeout
            read=self.timeout,  # Read timeout
            write=30.0,  # Write timeout
            pool=30.0,  # Pool timeout
        )

        limits = httpx.Limits(
            max_connections=200,  # Total connection pool size
            max_keepalive_connections=100,  # Keep-alive connections
        )

        transport = httpx.HTTPTransport(
            retries=3,  # Retry failed connections
            limits=limits,
        )

        return httpx.Client(
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
            endpoint,
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        remote_future = response.json()
        return self.create_future_from_request_id(
            request_id=remote_future,
            parse_result_fn=parse_result_fn,
        )

    def create_future_from_request_id(
        self,
        request_id: str | dict[str, Any] | RemoteFuture,
        parse_result_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> TinkerbellFuture[Any]:
        """
        Create a TinkerbellFuture from an existing request_id.

        Args:
            request_id: The request ID to poll for (string, dict, or RemoteFuture)
            parse_result_fn: Optional function to parse the result

        Returns:
            TinkerbellFuture that polls for the given request_id
        """
        if parse_result_fn is None:
            parse_result_fn = lambda x: x

        # Extract request_id string and model_id from various formats
        if isinstance(request_id, RemoteFuture):
            rid = request_id.request_id
            mid = request_id.model_id
        elif isinstance(request_id, dict):
            rid = request_id.get("request_id")
            mid = request_id.get("model_id")
        else:
            rid = request_id
            mid = None

        return TinkerbellFuture(
            request_id=rid,
            server_url=self.server_url,
            poll_endpoint="/poll_result",
            result_parser=parse_result_fn,
            poll_interval=1.0,
            timeout=self.timeout,
            model_id=mid,
        )
