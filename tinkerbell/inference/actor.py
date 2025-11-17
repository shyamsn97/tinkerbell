import logging
import socket
import subprocess
import time
from typing import Any, Dict, Optional

import httpx
import ray

logger = logging.getLogger(__name__)


@ray.remote
class SGLangInferenceActor:
    def __init__(self, model_id: str, tp_size: int, engine_kwargs: dict = {}):
        import signal
        import threading

        # Monkey patch signal.signal to ignore if not in main thread
        _original_signal = signal.signal

        def patched_signal(signalnum, handler):
            if threading.current_thread() is threading.main_thread():
                return _original_signal(signalnum, handler)
            return None

        signal.signal = patched_signal

        print(f"Setting up SGLang server for model {model_id} with {tp_size} GPUs")

        # Find available port
        self.port = self._find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        # Build server command
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_id,
            "--tp-size",
            str(tp_size),
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
        ]

        # Add engine kwargs as CLI args
        for key, value in engine_kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Start SGLang server as subprocess
        self.server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for server to be ready
        self._wait_for_server()

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

    def _wait_for_server(self, timeout: int = 300):
        """Wait for SGLang server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if subprocess is still alive
            if self.server_process.poll() is not None:
                # Process died, get stderr
                stderr = self.server_process.stderr.read()
                stdout = self.server_process.stdout.read()
                raise RuntimeError(
                    f"SGLang server process died with exit code {self.server_process.returncode}\n"
                    f"STDOUT: {stdout}\n"
                    f"STDERR: {stderr}"
                )

            try:
                response = httpx.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    return
            except Exception as e:
                # Only log occasionally to avoid spam
                if (time.time() - start_time) % 10 < 2:
                    logger.info(f"Waiting for SGLang server to be ready... ({e})")
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
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Use SGLang server's update weights endpoint
        response = self.client.post(
            "/update_weights_from_disk",
            json={
                "model_path": checkpoint_path,
                "load_format": load_format,
            },
        )
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        prompts: list[str] = None,
        input_ids: list[list[int]] = None,
        sampling_params: dict = {},
    ):
        """
        Generate text from prompts or input_ids.

        This uses SGLang's HTTP server which properly batches concurrent requests
        through its continuous batching scheduler.

        Args:
            prompts: List of text prompts (mutually exclusive with input_ids)
            input_ids: List of token id sequences (mutually exclusive with prompts)
            sampling_params: Sampling parameters dict
        """
        # Use SGLang's native /generate endpoint which supports both text and input_ids
        json_data = {"sampling_params": sampling_params}

        if prompts is not None:
            json_data["text"] = prompts
        elif input_ids is not None:
            json_data["input_ids"] = input_ids
        else:
            raise ValueError("Either prompts or input_ids must be provided")

        response = self.client.post("/generate", json=json_data)
        response.raise_for_status()
        result = response.json()

        # Extract text from response
        return [output["text"] for output in result]

    def shutdown(self):
        """Shutdown the SGLang server."""
        try:
            self.client.close()
            self.server_process.terminate()
            self.server_process.wait(timeout=10)
        except Exception as e:
            logger.error(f"Error shutting down SGLang server: {e}")
            self.server_process.kill()
        return True
