import logging
import multiprocessing
import socket
import time
from typing import Any, Dict, Optional

import httpx
import ray

from tinkerbell.utils import kill_process_tree

logger = logging.getLogger(__name__)


SUPPORTED_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "gate_up_proj",
]


def launch_server_process(server_args, launch_server_fn) -> multiprocessing.Process:
    """Launch SGLang server in a separate process.

    Note: SGLang server output goes to Ray actor logs.
    Use `ray logs <actor_name>` to view them.
    """
    p = multiprocessing.Process(target=launch_server_fn, args=(server_args,))
    p.start()
    return p


@ray.remote
class SGLangSamplingActor:
    def __init__(self, model_id: str, tp_size: int, engine_kwargs: dict = {}):
        import sys

        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs

        # Flush all logs immediately for debugging
        init_msg = (
            f"\n{'=' * 80}\n"
            f"🚀 Initializing SGLang Sampling Actor\n"
            f"   Model: {model_id}\n"
            f"   TP Size: {tp_size}\n"
            f"   Engine kwargs: {engine_kwargs}\n"
            f"{'=' * 80}\n"
        )
        print(init_msg, flush=True)
        logger.info(init_msg)
        sys.stdout.flush()
        sys.stderr.flush()

        self.client = None
        self.server_process = None
        # Find available port
        self.port = self._find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        port_msg = f"📡 Allocated port: {self.port}"
        print(port_msg, flush=True)
        logger.info(port_msg)
        sys.stdout.flush()

        engine_kwargs["model_path"] = model_id
        engine_kwargs["tp_size"] = tp_size
        engine_kwargs["port"] = self.port
        engine_kwargs["host"] = "127.0.0.1"
        engine_kwargs["enable_lora"] = True
        if "max_loras_per_batch" not in engine_kwargs:
            engine_kwargs["max_loras_per_batch"] = 2
        if "max_lora_rank" not in engine_kwargs:
            engine_kwargs["max_lora_rank"] = 256
        engine_kwargs["lora_target_modules"] = SUPPORTED_LORA_TARGET_MODULES

        start_msg = "🔧 Starting SGLang server process..."
        logger.info(start_msg)

        server_args = ServerArgs(**engine_kwargs)
        self.server_process = launch_server_process(server_args, launch_server)

        pid_msg = f"Process PID: {self.server_process.pid}"
        logger.info(pid_msg)

        # Wait for server to be ready
        wait_msg = f"⏳ Waiting for SGLang server to become ready at {self.base_url}..."
        logger.info(wait_msg)

        try:
            self._wait_for_server()
        except Exception as e:
            error_msg = f"✗ Server failed to start: {e}"
            logger.error(error_msg)
            self.shutdown()
            raise e

        # Create HTTP client for making requests
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(600.0),
        )

        ready_msg = f"✓ SGLang server running on {self.base_url}"
        print(ready_msg, flush=True)
        logger.info(ready_msg)
        sys.stdout.flush()

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

        logger.info(f"Waiting for SGLang server to start on {self.base_url}...")
        logger.info("This may take several minutes while the model loads...")

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time

            # Check if process is still alive (for multiprocessing.Process, use is_alive())
            if not self.server_process.is_alive():
                # Process died
                exitcode = self.server_process.exitcode
                logger.error("=" * 80)
                logger.error("ERROR: SGLang server process died during startup!")
                logger.error(f"Exit code: {exitcode}")
                logger.error("=" * 80)
                logger.error("Common causes:")
                logger.error("  1. GPU out of memory (OOM)")
                logger.error("  2. Model files not found or corrupted")
                logger.error("  3. CUDA/GPU driver issues")
                logger.error("  4. Insufficient system RAM")
                logger.error("=" * 80)
                raise RuntimeError(
                    f"SGLang server process died with exit code {exitcode}. "
                    f"Check Ray actor logs for more details."
                )

            try:
                response = httpx.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info(f"✓ SGLang server ready after {elapsed:.1f}s")
                    return
            except Exception:
                # Log every 10 seconds with elapsed time
                if time.time() - last_log_time >= 10:
                    logger.info(
                        f"[{elapsed:.1f}s] Still waiting for SGLang server... (process is alive)"
                    )
                    last_log_time = time.time()
                pass
            time.sleep(2)

        # Timeout reached - check if process is still alive
        if not self.server_process.is_alive():
            exitcode = self.server_process.exitcode
            raise RuntimeError(
                f"SGLang server process died (exit code {exitcode}) before becoming ready"
            )
        else:
            raise RuntimeError(
                f"SGLang server failed to start within {timeout}s (process still alive but not responding)"
            )

    def is_ready(self) -> bool:
        """Simple method to check if actor is initialized"""
        return True

    def is_server_alive(self) -> bool:
        """Check if the underlying SGLang server process is still alive"""
        if self.server_process is None:
            return False
        return self.server_process.is_alive()

    def _is_lora_adapter_path(self, path: str) -> bool:
        """Check if the given path contains a LoRA adapter.

        LoRA adapters are identified by the presence of adapter-specific files
        like adapter_config.json, adapter_model.bin, or adapter_model.safetensors.
        """
        import os

        if not os.path.exists(path):
            return False

        files = os.listdir(path)
        # Check for common LoRA adapter files
        lora_indicators = [
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
        ]
        return any(indicator in files for indicator in lora_indicators)

    def update_weights_from_disk(
        self,
        checkpoint_path: str,
        load_format: Optional[str] = None,
        pin_lora: bool = False,
    ) -> Dict[str, Any]:
        """Load model checkpoint or LoRA adapter from disk.

        If the checkpoint_path contains a LoRA adapter (detected by the presence
        of adapter_config.json), it will use the /load_lora_adapter endpoint.
        Otherwise, it will use the /update_weights_from_disk endpoint.
        """
        import os

        logger.info("=" * 80)
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        logger.info(f"Checkpoint path exists: {os.path.exists(checkpoint_path)}")

        if os.path.exists(checkpoint_path):
            files_in_checkpoint = os.listdir(checkpoint_path)
            logger.info(f"Files in checkpoint directory: {files_in_checkpoint}")
        else:
            error_msg = f"Checkpoint path does not exist: {checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Check if server is still alive before attempting to load
        if not self.is_server_alive():
            raise RuntimeError(
                "SGLang server process is not alive. Cannot load checkpoint."
            )

        # Detect if this is a LoRA adapter
        is_lora = self._is_lora_adapter_path(checkpoint_path)

        if is_lora:
            logger.info(f"Detected LoRA adapter at {checkpoint_path}")
            endpoint = "/load_lora_adapter"
            # Extract adapter name from path (use last directory name)
            lora_name = os.path.basename(os.path.normpath(checkpoint_path))
            request_data = {
                "lora_name": lora_name,  # REQUIRED
                "lora_path": checkpoint_path,  # REQUIRED
                "pinned": pin_lora,  # Optional: whether to pin adapter in memory
            }
        else:
            logger.info(f"Loading full model checkpoint from {checkpoint_path}")
            endpoint = "/update_weights_from_disk"
            request_data = {
                "model_path": checkpoint_path,
                "load_format": load_format,
            }

        logger.info("=" * 80)

        try:
            # Use appropriate SGLang endpoint
            response = self.client.post(
                endpoint,
                json=request_data,
                timeout=600.0,
            )

            # Check if server died during the request
            if not self.is_server_alive():
                raise RuntimeError(
                    "SGLang server process died during checkpoint loading. "
                    "This is likely due to OOM or incompatible checkpoint."
                )

            response.raise_for_status()
            result = response.json()

            if is_lora:
                msg = f"✓ LoRA adapter loaded successfully from {checkpoint_path}"
                print(msg, flush=True)
                logger.info(msg)
            else:
                msg = f"✓ Checkpoint loaded successfully from {checkpoint_path}"
                print(msg, flush=True)
                logger.info(msg)
            return result
        except httpx.HTTPError as e:
            # Check if server died
            if not self.is_server_alive():
                exitcode = self.server_process.exitcode
                error_msg = (
                    f"SGLang server process died (exit code {exitcode}) during checkpoint loading. "
                    f"This is likely due to GPU OOM or incompatible checkpoint format."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            msg = f"✗ Failed to load checkpoint from {checkpoint_path}"
            print(msg, flush=True)
            logger.error(msg)
            logger.error(f"HTTP Error: {type(e).__name__}: {str(e)}")
            print(f"HTTP Error: {type(e).__name__}: {str(e)}", flush=True)
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status: {e.response.status_code}", flush=True)
                print(f"Response text: {e.response.text}", flush=True)
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            import traceback

            traceback.print_exc()
            raise
        except Exception as e:
            # Check if server died
            if not self.is_server_alive():
                exitcode = self.server_process.exitcode
                error_msg = (
                    f"SGLang server process died (exit code {exitcode}) during checkpoint loading. "
                    f"This is likely due to GPU OOM or incompatible checkpoint format."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            msg = f"✗ Failed to load checkpoint from {checkpoint_path}"
            print(msg, flush=True)
            logger.error(msg)
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            print(f"Error: {type(e).__name__}: {str(e)}", flush=True)
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status: {e.response.status_code}", flush=True)
                print(f"Response text: {e.response.text}", flush=True)
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            import traceback

            traceback.print_exc()
            raise

    def sample(
        self,
        sample_request: dict[str, Any],
    ):
        """
        Sample text from the request.

        This uses SGLang's HTTP server which properly batches concurrent requests
        through its continuous batching scheduler.

        Args:
            sample_request: Dictionary of the SampleRequest

        Returns:
            Dictionary with outputs, logprobs (if available), and other metadata
        """
        # Use SGLang's native /generate endpoint which supports both text and input_ids
        response = self.client.post("/generate", json=sample_request)
        response.raise_for_status()
        result = response.json()

        # Initialize response structure with logprobs support
        response_data = {
            "outputs": [],
            "logprobs": None,
            "top_logprobs": None,
            "output_token_ids": None,
            "finish_reasons": None,
            "meta_info": {},
        }

        # Handle different SGLang response formats
        # SGLang can return:
        # 1. A dictionary with "text" field: {"text": "...", "meta_info": {...}, "logprobs": [...]}
        # 2. A list of strings: ["text1", "text2"]
        # 3. A list of dicts: [{"text": "...", "logprobs": [...]}, {"text": "...", "logprobs": [...]}]

        if isinstance(result, dict):
            # Single response as dictionary
            if "text" in result:
                response_data["outputs"] = [result["text"]]
                # Extract logprobs if available
                if "meta_info" in result:
                    meta_info = result["meta_info"]
                    if "logprobs" in meta_info:
                        response_data["logprobs"] = [meta_info["logprobs"]]
                    if "top_logprobs" in meta_info:
                        response_data["top_logprobs"] = [meta_info["top_logprobs"]]
                    if "output_token_ids" in meta_info:
                        response_data["output_token_ids"] = [
                            meta_info["output_token_ids"]
                        ]
                    if "finish_reason" in meta_info:
                        finish_reason = meta_info["finish_reason"]
                        # Convert dict to string if needed
                        if isinstance(finish_reason, dict):
                            finish_reason = finish_reason.get(
                                "type", str(finish_reason)
                            )
                        response_data["finish_reasons"] = [finish_reason]
                    response_data["meta_info"] = meta_info
                # Also check top-level for logprobs (alternative SGLang format)
                elif "logprobs" in result:
                    response_data["logprobs"] = [result["logprobs"]]
            else:
                logger.warning(f"Unexpected dict format from SGLang: {result}")
                response_data["outputs"] = [str(result)]
        elif isinstance(result, list):
            # List of responses
            if len(result) > 0 and isinstance(result[0], dict):
                # List of dicts with "text" field
                response_data["outputs"] = [
                    output.get("text", str(output)) for output in result
                ]
                # Try to extract logprobs from each output
                all_logprobs = []
                all_top_logprobs = []
                all_token_ids = []
                all_finish_reasons = []
                for output in result:
                    if "meta_info" in output:
                        meta = output["meta_info"]
                        all_logprobs.append(meta.get("logprobs"))
                        all_top_logprobs.append(meta.get("top_logprobs"))
                        all_token_ids.append(meta.get("output_token_ids"))
                        # Convert finish_reason dict to string if needed
                        finish_reason = meta.get("finish_reason")
                        if isinstance(finish_reason, dict):
                            finish_reason = finish_reason.get(
                                "type", str(finish_reason)
                            )
                        all_finish_reasons.append(finish_reason)
                    elif "logprobs" in output:
                        all_logprobs.append(output.get("logprobs"))

                if any(x is not None for x in all_logprobs):
                    response_data["logprobs"] = all_logprobs
                if any(x is not None for x in all_top_logprobs):
                    response_data["top_logprobs"] = all_top_logprobs
                if any(x is not None for x in all_token_ids):
                    response_data["output_token_ids"] = all_token_ids
                if any(x is not None for x in all_finish_reasons):
                    response_data["finish_reasons"] = all_finish_reasons
            else:
                # List of strings
                response_data["outputs"] = result
        else:
            # Fallback for unexpected format
            logger.warning(
                f"Unexpected response format from SGLang: {type(result)}, {result}"
            )
            response_data["outputs"] = [str(result)]

        return response_data

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
