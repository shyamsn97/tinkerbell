import time
from typing import Any, Callable, Generic, TypeVar

import httpx

T = TypeVar('T')


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
    ):
        """
        Initialize a TinkerbellFuture.
        
        Args:
            request_id: The request ID to poll for
            server_url: Base URL of the server
            poll_endpoint: Endpoint to poll for results (e.g., "/get_result")
            result_parser: Function to parse the response into the result type
            poll_interval: Interval between polls in seconds
            timeout: Maximum time to wait for result
            client_timeout: Timeout for individual HTTP requests
        """
        self._request_id = request_id
        self._server_url = server_url
        self._poll_endpoint = poll_endpoint
        self._result_parser = result_parser
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._client_timeout = client_timeout
        self._result = None
        self._resolved = False
        self._exception = None
    
    def _get_client(self) -> httpx.Client:
        """Get or create a light httpx client for polling."""
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=self._client_timeout,
            write=10.0,
            pool=10.0,
        )
        return httpx.Client(
            base_url=self._server_url,
            timeout=timeout_config,
        )
    
    def result(self, timeout: float | None = None) -> T:
        """
        Block and wait for the result by polling the server.
        
        Args:
            timeout: Maximum time to wait (overrides instance timeout)
            
        Returns:
            The result of the operation
            
        Raises:
            TimeoutError: If timeout is exceeded
            Exception: Any exception raised during execution
        """
        if self._resolved:
            if self._exception:
                raise self._exception
            return self._result
        
        timeout = timeout or self._timeout
        start_time = time.time()
        
        try:
            with self._get_client() as client:
                while True:
                    # Check timeout
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed > timeout:
                            raise TimeoutError(
                                f"Future timed out after {elapsed:.1f}s (timeout={timeout}s)"
                            )

                    # Poll the server
                    response = client.post(
                        self._poll_endpoint,
                        json={"request_id": self._request_id},
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Check if result is ready
                    if data.get("status") == "completed":
                        # Parse the result
                        self._result = self._result_parser(data.get("result"))
                        self._resolved = True
                        return self._result
                    elif data.get("status") == "error":
                        error_msg = data.get("error", "Unknown error")
                        raise RuntimeError(f"Operation failed: {error_msg}")
                    elif data.get("status") == "pending":
                        # Still processing, wait and retry
                        time.sleep(self._poll_interval)
                    else:
                        # Unknown status
                        raise RuntimeError(
                            f"Unknown status: {data.get('status')}"
                        )
        except Exception as e:
            self._exception = e
            self._resolved = True
            raise

    def done(self) -> bool:
        """Check if the future is resolved."""
        return self._resolved
    
    def exception(self) -> Exception | None:
        """Get the exception if one occurred."""
        return self._exception
    
    @property
    def request_id(self) -> str:
        """Get the request ID for this future."""
        return self._request_id


class BaseClient:
    def __init__(
        self,
        server_url: str,
        timeout: float = 600.0,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.client = self._create_client()

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
        return TinkerbellFuture(
            request_id=remote_future["request_id"],
            server_url=self.server_url,
            poll_endpoint="/poll_result",
            result_parser=parse_result_fn,
            poll_interval=0.1,
            timeout=self.timeout,
        )
