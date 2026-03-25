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

        print(f"🚀 Initializing SGLang: model={base_model}, tp={tp_size}", flush=True)

        self.client = None
        self.server_process = None
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
        print(f"✓ SGLang server running on {self.base_url}", flush=True)

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

    def update_weights_from_disk(
        self,
        checkpoint_path: str,
        load_format: Optional[str] = None,
        pin_lora: bool = False,
    ) -> Dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not self.is_server_alive():
            raise RuntimeError("SGLang server not alive")

        is_lora = self._is_lora_adapter_path(checkpoint_path)
        lora_name = None
        lora_path = checkpoint_path

        if is_lora:
            # Find actual adapter path (might be in subdirectory)
            if "adapter_config.json" not in os.listdir(checkpoint_path):
                for f in os.listdir(checkpoint_path):
                    subdir = os.path.join(checkpoint_path, f)
                    if os.path.isdir(subdir) and "adapter_config.json" in os.listdir(
                        subdir
                    ):
                        lora_path = subdir
                        break

            lora_name = os.path.normpath(lora_path)

            # Log adapter config for debugging
            adapter_config_path = os.path.join(lora_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                import json

                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                logger.info(f"Loading LoRA '{lora_name}' from {lora_path}")
                logger.info(
                    f"  Adapter target_modules: {adapter_config.get('target_modules')}"
                )
                logger.info(f"  Adapter rank (r): {adapter_config.get('r')}")
                logger.info(
                    f"  SGLang lora_target_modules: {list(SUPPORTED_LORA_TARGET_MODULES)}"
                )
            # Try to unload any existing LoRAs with similar paths to force fresh load
            try:
                # Get list of currently loaded LoRAs
                info_resp = self.client.get("/get_server_info", timeout=10.0)
                if info_resp.status_code == 200:
                    server_info = info_resp.json()
                    lora_paths_info = server_info.get("lora_paths", [])
                    print(
                        f"Currently loaded LoRAs: {[lp.get('lora_name') for lp in lora_paths_info]}",
                        flush=True,
                    )

                    # Unload any LoRA with matching base path
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
                            print(
                                f"Unloading existing LoRA: {existing_name}", flush=True
                            )
                            unload_resp = self.client.post(
                                "/unload_lora_adapter",
                                json={"lora_name": existing_name},
                                timeout=60.0,
                            )
                            print(
                                f"Unload response: {unload_resp.status_code} - {unload_resp.text}",
                                flush=True,
                            )
            except Exception as e:
                print(f"Error checking/unloading existing LoRAs: {e}", flush=True)

            print(f"Loading LoRA '{lora_name}' from path: {lora_path}", flush=True)
            response = self.client.post(
                "/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                },
                timeout=600.0,
            )
            print(
                f"Load LoRA response: {response.status_code} - {response.text}",
                flush=True,
            )

            # Track dynamically loaded adapters from the response
            if response.status_code == 200:
                try:
                    load_result = response.json()
                    loaded_adapters = load_result.get("loaded_adapters", {})
                    if not hasattr(self, "_loaded_adapters"):
                        self._loaded_adapters = {}
                    self._loaded_adapters.update(loaded_adapters)
                    print(
                        f"Tracked loaded adapters: {list(self._loaded_adapters.keys())}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"Error parsing load response: {e}", flush=True)

            # Verify the LoRA was actually loaded (check our tracked adapters, not server_info)
            if hasattr(self, "_loaded_adapters") and lora_name in self._loaded_adapters:
                print(
                    f"✓ Verified: LoRA '{lora_name}' is in tracked loaded adapters",
                    flush=True,
                )
            else:
                print(
                    f"⚠️  WARNING: LoRA '{lora_name}' not in tracked loaded adapters!",
                    flush=True,
                )
        else:
            response = self.client.post(
                "/update_weights_from_disk",
                json={"model_path": checkpoint_path, "load_format": load_format},
                timeout=600.0,
            )

        if not self.is_server_alive():
            raise RuntimeError("SGLang server died during checkpoint loading")

        if response.status_code != 200:
            logger.error(f"SGLang response {response.status_code}: {response.text}")
        response.raise_for_status()
        print(
            f"✓ {'LoRA' if is_lora else 'Checkpoint'} loaded: {checkpoint_path}",
            flush=True,
        )
        return {"is_lora": is_lora, "lora_name": lora_name}

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

    def sample(self, sample_request: dict[str, Any]):
        # Debug: log the lora_path being used
        lora_path = sample_request.get("lora_path")
        # Also set lora_name in case SGLang expects that instead
        if lora_path and "lora_name" not in sample_request:
            sample_request["lora_name"] = lora_path
        print(
            f"[SAMPLE] lora_path={lora_path}, lora_name={sample_request.get('lora_name')}",
            flush=True,
        )
        response = self.client.post("/generate", json=sample_request)
        response.raise_for_status()
        result = response.json()
        return self._parse_sglang_response(result)

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
                response_data["logprobs"] = _wrap(meta.get("logprobs"))
                response_data["top_logprobs"] = _wrap(meta.get("top_logprobs"))
                response_data["output_token_ids"] = _wrap(meta.get("output_token_ids"))
                if fr := meta.get("finish_reason"):
                    response_data["finish_reasons"] = [
                        fr.get("type", str(fr)) if isinstance(fr, dict) else fr
                    ]
        elif isinstance(result, list):
            if result and isinstance(result[0], dict):
                response_data["outputs"] = [o.get("text", str(o)) for o in result]
                metas = [o.get("meta_info", {}) for o in result]
                if any(m.get("logprobs") for m in metas):
                    response_data["logprobs"] = [m.get("logprobs") for m in metas]
                if any(m.get("output_token_ids") for m in metas):
                    response_data["output_token_ids"] = [
                        m.get("output_token_ids") for m in metas
                    ]
            else:
                response_data["outputs"] = result
        else:
            response_data["outputs"] = [str(result)]
        return response_data

    def shutdown(self):
        try:
            if self.client:
                self.client.close()
            if self.server_process:
                kill_process_tree(self.server_process.pid)
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        return True
