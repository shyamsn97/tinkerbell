import logging
import multiprocessing
import socket
import time
from typing import Any, Dict, Optional

import httpx
import ray

from tinkerbell.utils import kill_process_tree

logger = logging.getLogger(__name__)


def launch_server_process(server_args, launch_server_fn) -> multiprocessing.Process:
    """Launch SGLang server in a separate process.

    Note: SGLang server output goes to Ray actor logs.
    Use `ray logs <actor_name>` to view them.
    """
    p = multiprocessing.Process(target=launch_server_fn, args=(server_args,))
    p.start()
    return p


@ray.remote
class SGLangInferenceActor:
    def __init__(self, model_id: str, tp_size: int, engine_kwargs: dict = {}):
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs

        print("=" * 80)
        print("🚀 Initializing SGLang Inference Actor")
        print(f"   Model: {model_id}")
        print(f"   TP Size: {tp_size}")
        print(f"   Engine kwargs: {engine_kwargs}")
        print("=" * 80)

        self.client = None
        self.server_process = None
        # Find available port
        self.port = self._find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        print(f"📡 Allocated port: {self.port}")

        engine_kwargs["model_path"] = model_id
        engine_kwargs["tp_size"] = tp_size
        engine_kwargs["port"] = self.port
        engine_kwargs["host"] = "127.0.0.1"

        print("🔧 Starting SGLang server process...")
        server_args = ServerArgs(**engine_kwargs)
        self.server_process = launch_server_process(server_args, launch_server)
        print(f"Process PID: {self.server_process.pid}")

        # Wait for server to be ready
        try:
            self._wait_for_server()
        except Exception as e:
            self.shutdown()
            raise e

        # Create HTTP client for making requests
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(600.0),
        )

        print(f"✓ SGLang server running on {self.base_url}")

    def _find_free_port(self) -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _wait_for_server(self, timeout: int = 600):
        """Wait for SGLang server to be ready."""
        start_time = time.time()
        last_log_time = start_time

        print(f"Waiting for SGLang server to start on {self.base_url}...")
        print("This may take several minutes while the model loads...")

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time

            # Check if process is still alive (for multiprocessing.Process, use is_alive())
            if not self.server_process.is_alive():
                # Process died
                exitcode = self.server_process.exitcode
                raise RuntimeError(
                    f"SGLang server process died with exit code {exitcode}"
                )

            try:
                response = httpx.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    print(f"✓ SGLang server ready after {elapsed:.1f}s")
                    return
            except Exception:
                # Log every 10 seconds with elapsed time
                if time.time() - last_log_time >= 10:
                    print(
                        f"[{elapsed:.1f}s] Still waiting for SGLang server... (process is alive)"
                    )
                    last_log_time = time.time()
                pass
            time.sleep(2)
        raise RuntimeError(f"SGLang server failed to start within {timeout}s")

    def is_ready(self) -> bool:
        """Simple method to check if actor is initialized"""
        return True

    def update_weights_from_disk(
        self, checkpoint_path: str, load_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint from disk."""
        import os

        print("=" * 80)
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"Checkpoint path exists: {os.path.exists(checkpoint_path)}")

        if os.path.exists(checkpoint_path):
            files_in_checkpoint = os.listdir(checkpoint_path)
            print(f"Files in checkpoint directory: {files_in_checkpoint}")
        else:
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )

        print("=" * 80)

        try:
            # Use SGLang server's update weights endpoint
            response = self.client.post(
                "/update_weights_from_disk",
                json={
                    "model_path": checkpoint_path,
                    "load_format": load_format,
                },
                timeout=600.0,
            )
            response.raise_for_status()
            result = response.json()
            print(f"✓ Checkpoint loaded successfully from {checkpoint_path}")
            return result
        except Exception as e:
            print(f"✗ Failed to load checkpoint from {checkpoint_path}")
            print(f"Error: {type(e).__name__}: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            import traceback

            traceback.print_exc()
            raise

    def generate(
        self,
        generate_request: dict[str, Any],
    ):
        """
        Generate text from the request.

        This uses SGLang's HTTP server which properly batches concurrent requests
        through its continuous batching scheduler.

        Args:
            generate_request: Dictionary of the GenerateRequest
        """
        # Use SGLang's native /generate endpoint which supports both text and input_ids
        response = self.client.post("/generate", json=generate_request)
        response.raise_for_status()
        result = response.json()

        # Handle different SGLang response formats
        # SGLang can return:
        # 1. A dictionary with "text" field: {"text": "..."}
        # 2. A list of strings: ["text1", "text2"]
        # 3. A list of dicts: [{"text": "..."}, {"text": "..."}]

        if isinstance(result, dict):
            # Single response as dictionary
            if "text" in result:
                return [result["text"]]
            else:
                logger.warning(f"Unexpected dict format from SGLang: {result}")
                return [str(result)]
        elif isinstance(result, list):
            # List of responses
            if len(result) > 0 and isinstance(result[0], dict):
                # List of dicts with "text" field
                return [output.get("text", str(output)) for output in result]
            else:
                # List of strings
                return result
        else:
            # Fallback for unexpected format
            logger.warning(
                f"Unexpected response format from SGLang: {type(result)}, {result}"
            )
            return [str(result)]

    def shutdown(self):
        """Shutdown the SGLang server."""
        try:
            if self.client is not None:
                self.client.close()
            if self.server_process is not None:
                kill_process_tree(self.server_process.pid)
        except Exception as e:
            raise e
        return True
