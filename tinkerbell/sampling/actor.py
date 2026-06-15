import logging
import multiprocessing
import os
import socket
import time
from typing import Any, Dict, Optional

import httpx
import ray

from tinkerbell.types.lora_config import SUPPORTED_LORA_TARGET_MODULES
from tinkerbell.utils import kill_process_tree

logger = logging.getLogger(__name__)


def launch_server_process(server_args, launch_server_fn) -> multiprocessing.Process:
    p = multiprocessing.Process(target=launch_server_fn, args=(server_args,))
    p.start()
    return p


@ray.remote
class SGLangSamplingActor:
    def __init__(self, base_model: str, tp_size: int, engine_kwargs: dict = {}):
        try:
            self._setup(base_model, tp_size, engine_kwargs)
        except Exception as e:
            logger.error(f"FATAL: SGLangSamplingActor init failed: {e}")
            raise

    def _setup(self, base_model: str, tp_size: int, engine_kwargs: dict):
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs

        logger.info(f"Initializing SGLang: model={base_model}, tp={tp_size}")

        self.client = None
        # AsyncClient is created lazily on first sample() call, so the asyncio
        # loop (created by Ray for async actors) is guaranteed to exist.
        self.async_client: httpx.AsyncClient | None = None
        self.server_process = None
        self._loaded_adapters = {}
        self._inflight_samples = 0
        self._loading_weights = False
        self._barrier = None
        self.port = self._find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        engine_kwargs["model_path"] = base_model
        engine_kwargs["tp_size"] = tp_size
        engine_kwargs["port"] = self.port
        engine_kwargs["host"] = "127.0.0.1"

        if engine_kwargs.get("enable_lora", None) is not False:
            engine_kwargs["enable_lora"] = True
            engine_kwargs.setdefault("max_loras_per_batch", 8)
            engine_kwargs.setdefault("max_lora_rank", 256)
            engine_kwargs["lora_target_modules"] = list(SUPPORTED_LORA_TARGET_MODULES)
            logger.info(
                f"LoRA enabled: max_rank={engine_kwargs['max_lora_rank']}, "
                f"target_modules={engine_kwargs['lora_target_modules']}, "
                f"lora_paths={engine_kwargs.get('lora_paths', [])}"
            )

        engine_kwargs.setdefault("enable_deterministic_inference", False)

        server_args = ServerArgs(**engine_kwargs)
        logger.info(f"ServerArgs: {server_args}")
        self.server_process = launch_server_process(server_args, launch_server)

        try:
            self._wait_for_server()
        except Exception as e:
            logger.error(f"Server failed to start: {e}")
            self.shutdown()
            raise

        self.client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(600.0))
        logger.info(f"SGLang server running on {self.base_url}")

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            return s.getsockname()[1]

    def _wait_for_server(self, timeout: int = 600):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.server_process.is_alive():
                raise RuntimeError(
                    f"SGLang process died (exit={self.server_process.exitcode})"
                )
            try:
                if httpx.get(f"{self.base_url}/health", timeout=5.0).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError(f"SGLang server timeout after {timeout}s")

    def is_ready(self) -> bool:
        return True

    def is_server_alive(self) -> bool:
        return self.server_process is not None and self.server_process.is_alive()

    def _is_lora_adapter_path(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        indicators = [
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
        ]
        files = os.listdir(path)
        if any(f in files for f in indicators):
            return True
        for f in files:
            subdir = os.path.join(path, f)
            if os.path.isdir(subdir) and any(
                ind in os.listdir(subdir) for ind in indicators
            ):
                return True
        return False

    async def _ensure_async_client(self) -> httpx.AsyncClient:
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url, timeout=httpx.Timeout(600.0)
            )
        return self.async_client

    def _ensure_barrier(self):
        import asyncio

        if self._barrier is None:
            self._barrier = asyncio.Condition()
        return self._barrier

    async def _begin_sample(self) -> None:
        barrier = self._ensure_barrier()
        async with barrier:
            while self._loading_weights:
                await barrier.wait()
            self._inflight_samples += 1

    async def _end_sample(self) -> None:
        barrier = self._ensure_barrier()
        async with barrier:
            self._inflight_samples -= 1
            if self._inflight_samples == 0:
                barrier.notify_all()

    async def _begin_weight_mutation(self) -> None:
        barrier = self._ensure_barrier()
        async with barrier:
            while self._loading_weights or self._inflight_samples:
                await barrier.wait()
            self._loading_weights = True

    async def _end_weight_mutation(self) -> None:
        barrier = self._ensure_barrier()
        async with barrier:
            self._loading_weights = False
            barrier.notify_all()

    async def update_weights_from_disk(
        self,
        checkpoint_path: str,
        load_format: Optional[str] = None,
        pin_lora: bool = False,
    ) -> Dict[str, Any]:
        """Async so weight loads (which can take 30-60s on a big LoRA) don't
        block the actor's asyncio loop. While this is awaiting on SGLang HTTP
        responses, in-flight `sample()` tasks continue making progress.
        """
        await self._begin_weight_mutation()
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            if not self.is_server_alive():
                raise RuntimeError("SGLang server not alive")

            client = await self._ensure_async_client()
            is_lora = self._is_lora_adapter_path(checkpoint_path)
            lora_name = None
            lora_path = checkpoint_path
            t0 = time.time()

            if is_lora:
                if "adapter_config.json" not in os.listdir(checkpoint_path):
                    for f in os.listdir(checkpoint_path):
                        subdir = os.path.join(checkpoint_path, f)
                        if os.path.isdir(
                            subdir
                        ) and "adapter_config.json" in os.listdir(subdir):
                            lora_path = subdir
                            break

                lora_name = os.path.normpath(lora_path)
                logger.info(f"Loading LoRA '{lora_name}' from path: {lora_path}")

                # Keep SGLang GPU memory bounded across repeated train/sample syncs.
                try:
                    info_resp = await client.get("/get_server_info", timeout=10.0)
                    if info_resp.status_code == 200:
                        lora_paths_info = info_resp.json().get("lora_paths", []) or []
                        base_path = (
                            lora_path.rsplit("-step", 1)[0]
                            if "-step" in lora_path
                            else lora_path
                        )
                        for lp in lora_paths_info:
                            existing_name = lp.get("lora_name", "")
                            existing_base = (
                                existing_name.rsplit("-step", 1)[0]
                                if "-step" in existing_name
                                else existing_name
                            )
                            if existing_base == base_path or existing_name == lora_name:
                                logger.info(f"Unloading existing LoRA: {existing_name}")
                                await client.post(
                                    "/unload_lora_adapter",
                                    json={"lora_name": existing_name},
                                    timeout=60.0,
                                )
                except Exception as e:
                    logger.warning(f"Error checking/unloading existing LoRAs: {e}")

                response = await client.post(
                    "/load_lora_adapter",
                    json={"lora_name": lora_name, "lora_path": lora_path},
                    timeout=600.0,
                )
                if response.status_code == 200:
                    try:
                        self._loaded_adapters.update(
                            response.json().get("loaded_adapters", {})
                        )
                    except Exception as e:
                        logger.warning(f"Error parsing load response: {e}")
            else:
                response = await client.post(
                    "/update_weights_from_disk",
                    json={"model_path": checkpoint_path, "load_format": load_format},
                    timeout=600.0,
                )

            if not self.is_server_alive():
                raise RuntimeError("SGLang server died during checkpoint loading")
            if response.status_code != 200:
                logger.error(f"SGLang response {response.status_code}: {response.text}")
            response.raise_for_status()
            logger.info(
                f"{'LoRA' if is_lora else 'Checkpoint'} loaded: {checkpoint_path} "
                f"in {time.time() - t0:.1f}s"
            )
            return {"is_lora": is_lora, "lora_name": lora_name}
        finally:
            await self._end_weight_mutation()

    def get_lora_info(self) -> dict[str, Any]:
        """Get info about loaded LoRAs from SGLang."""
        try:
            response = self.client.get("/get_server_info", timeout=10.0)
            if response.status_code == 200:
                server_info = response.json()
                # Get lora_paths (startup adapters)
                lora_paths = server_info.get("lora_paths", [])
                startup_loras = (
                    [lp.get("lora_name") for lp in lora_paths]
                    if isinstance(lora_paths, list)
                    else []
                )

                # Note: dynamically loaded adapters are returned in /load_lora_adapter response
                # but not in /get_server_info. We track them via self._loaded_adapters if available.
                loaded_adapters = getattr(self, "_loaded_adapters", {})

                return {
                    "startup_loras": startup_loras,
                    "dynamically_loaded": list(loaded_adapters.keys()),
                }
        except Exception as e:
            logger.warning(f"Failed to get LoRA info: {e}")
        return {"error": "Failed to get server info"}

    async def sample(self, sample_request: dict[str, Any]):
        """Async so Ray fans out many concurrent /generate calls instead of
        serializing them. Without this, SGLang's continuous batcher only ever
        sees one request at a time."""
        import asyncio

        try:
            await self._begin_sample()
            # Keep direct actor calls compatible with the client defaults:
            # GRPO relies on output token logprobs from SGLang.
            sample_request.setdefault("return_logprob", True)
            sample_request.setdefault("top_logprobs_num", 1)
            lora_path = sample_request.get("lora_path")
            if lora_path and "lora_name" not in sample_request:
                sample_request["lora_name"] = lora_path

            # Retry transport-level errors. SGLang's HTTP server occasionally
            # drops a connection mid-request (especially right after a LoRA
            # hot-swap or under load); a stale keep-alive in the pool surfaces
            # as ReadError/RemoteProtocolError. Reset the client and try again.
            last_exc: Exception | None = None
            for attempt in range(3):
                client = await self._ensure_async_client()
                t0 = time.time()
                try:
                    response = await client.post("/generate", json=sample_request)
                    elapsed = time.time() - t0
                    if elapsed > 30.0:
                        logger.warning(f"sample /generate took {elapsed:.1f}s (slow)")
                    response.raise_for_status()
                    return self._parse_sglang_response(response.json())
                except httpx.ReadTimeout:
                    logger.error(
                        f"sample /generate timed out after {time.time() - t0:.0f}s "
                        f"(req max_new_tokens={sample_request.get('sampling_params', {}).get('max_new_tokens')})"
                    )
                    raise
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    last_exc = e
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    transient = isinstance(e, httpx.TransportError) or status in {
                        408,
                        425,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    if not transient or attempt == 2:
                        raise
                    logger.warning(
                        f"sample /generate transient error ({type(e).__name__}: {e}); "
                        f"resetting client and retrying (attempt {attempt + 1}/3)"
                    )
                    await self._reset_async_client()
                    await asyncio.sleep(0.5 * (2**attempt))
            assert last_exc is not None
            raise last_exc
        finally:
            await self._end_sample()

    async def _reset_async_client(self) -> None:
        client = self.async_client
        self.async_client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass

    def _parse_sglang_response(self, result) -> dict:
        response_data = {
            "outputs": [],
            "logprobs": None,
            "top_logprobs": None,
            "output_token_ids": None,
            "finish_reasons": None,
            "meta_info": {},
        }

        def _wrap(val):
            return [val] if val is not None else None

        if isinstance(result, dict) and "text" in result:
            response_data["outputs"] = [result["text"]]
            if meta := result.get("meta_info"):
                response_data["meta_info"] = meta
                response_data["logprobs"] = _wrap(
                    meta.get("output_top_logprobs")
                    or meta.get("output_token_logprobs")
                    or meta.get("logprobs")
                )
                response_data["top_logprobs"] = _wrap(
                    meta.get("output_top_logprobs") or meta.get("top_logprobs")
                )
                response_data["output_token_ids"] = _wrap(meta.get("output_token_ids"))
                if fr := meta.get("finish_reason"):
                    response_data["finish_reasons"] = [
                        fr.get("type", str(fr)) if isinstance(fr, dict) else fr
                    ]
        elif isinstance(result, list):
            if result and isinstance(result[0], dict):
                response_data["outputs"] = [o.get("text", str(o)) for o in result]
                metas = [o.get("meta_info", {}) for o in result]
                logprobs = [
                    m.get("output_top_logprobs")
                    or m.get("output_token_logprobs")
                    or m.get("logprobs")
                    for m in metas
                ]
                if any(logprobs):
                    response_data["logprobs"] = logprobs
                if any(m.get("output_token_ids") for m in metas):
                    response_data["output_token_ids"] = [
                        m.get("output_token_ids") for m in metas
                    ]
            else:
                response_data["outputs"] = result
        else:
            response_data["outputs"] = [str(result)]
        return response_data

    async def shutdown(self):
        await self._begin_weight_mutation()
        try:
            if self.client:
                self.client.close()
            if self.async_client is not None:
                await self.async_client.aclose()
                self.async_client = None
            if self.server_process:
                kill_process_tree(self.server_process.pid)
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        finally:
            await self._end_weight_mutation()
        return True
