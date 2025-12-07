import logging
import multiprocessing
import os
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
    p = multiprocessing.Process(target=launch_server_fn, args=(server_args,))
    p.start()
    return p


@ray.remote
class SGLangSamplingActor:
    def __init__(self, model_id: str, tp_size: int, engine_kwargs: dict = {}):
        try:
            self._setup(model_id, tp_size, engine_kwargs)
        except Exception as e:
            logger.error(f"FATAL: SGLangSamplingActor init failed: {e}")
            raise

    def _setup(self, model_id: str, tp_size: int, engine_kwargs: dict):
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs

        print(f"🚀 Initializing SGLang: model={model_id}, tp={tp_size}", flush=True)

        self.client = None
        self.server_process = None
        self.port = self._find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        engine_kwargs["model_path"] = model_id
        engine_kwargs["tp_size"] = tp_size
        engine_kwargs["port"] = self.port
        engine_kwargs["host"] = "127.0.0.1"

        if engine_kwargs.get("enable_lora", None) is not False:
            engine_kwargs["enable_lora"] = True
            engine_kwargs.setdefault("max_loras_per_batch", 256)
            engine_kwargs.setdefault("max_lora_rank", 256)
            engine_kwargs["lora_target_modules"] = SUPPORTED_LORA_TARGET_MODULES

        engine_kwargs.setdefault("enable_deterministic_inference", True)

        server_args = ServerArgs(**engine_kwargs)
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

            lora_name = os.path.basename(os.path.normpath(lora_path))
            try:
                self.client.post(
                    "/unload_lora_adapter", json={"lora_name": lora_name}, timeout=60.0
                )
            except Exception:
                pass
            response = self.client.post(
                "/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                    "pinned": pin_lora,
                },
                timeout=600.0,
            )
        else:
            response = self.client.post(
                "/update_weights_from_disk",
                json={"model_path": checkpoint_path, "load_format": load_format},
                timeout=600.0,
            )

        if not self.is_server_alive():
            raise RuntimeError("SGLang server died during checkpoint loading")

        response.raise_for_status()
        print(
            f"✓ {'LoRA' if is_lora else 'Checkpoint'} loaded: {checkpoint_path}",
            flush=True,
        )
        return {"is_lora": is_lora, "lora_name": lora_name}

    def sample(self, sample_request: dict[str, Any]):
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

        if isinstance(result, dict) and "text" in result:
            response_data["outputs"] = [result["text"]]
            meta = result.get("meta_info")
            if meta:
                response_data["meta_info"] = meta
                response_data["logprobs"] = (
                    [meta["logprobs"]] if "logprobs" in meta else None
                )
                response_data["top_logprobs"] = (
                    [meta["top_logprobs"]] if "top_logprobs" in meta else None
                )
                response_data["output_token_ids"] = (
                    [meta["output_token_ids"]] if "output_token_ids" in meta else None
                )
                fr = meta.get("finish_reason")
                if fr:
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
