"""Small HTTP/Ray Serve gateway over the Ray-native core."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import ray
from fastapi import FastAPI, HTTPException
from ray import serve
from tinker.types import Datum

from tinkerbell.api.jobs import create_job, poll_job, set_job
from tinkerbell.runtime.session import Session
from tinkerbell.utils import clean_model_name

APP = FastAPI()


class TinkerbellServer:
    """Thin server: HTTP in, direct Ray actor calls out."""

    def __init__(self, namespace: str = "tinkerbell"):
        self.session = Session(namespace=namespace)
        self.trainers = {}
        self.samplers = {}
        self.trainer_ready = {}
        self.sampler_ready = {}

    def _model_name(self, payload: dict[str, Any]) -> str:
        base_model = payload.get("base_model")
        model_name = payload.get("model_name")
        if model_name:
            return model_name
        if base_model:
            return clean_model_name(base_model)
        raise HTTPException(
            status_code=400, detail="model_name or base_model is required"
        )

    def _trainer(self, model_name: str):
        trainer = self.trainers.get(model_name)
        if trainer is None:
            raise HTTPException(
                status_code=404, detail=f"trainer {model_name} not found"
            )
        return trainer

    def _sampler(self, model_name: str):
        sampler = self.samplers.get(model_name)
        if sampler is None:
            raise HTTPException(
                status_code=404, detail=f"sampler {model_name} not found"
            )
        return sampler

    def _datums(self, payload: dict[str, Any]) -> list[Datum]:
        return [
            item if isinstance(item, Datum) else Datum(**item)
            for item in (payload.get("data") or [])
        ]

    async def _result(self, future):
        return await asyncio.to_thread(future.result)

    async def _ensure_ready(self, futures: dict[str, Any], model_name: str) -> None:
        future = futures.get(model_name)
        if future is not None:
            await self._result(future)

    def _status(self, futures: dict[str, Any], model_name: str) -> dict[str, str]:
        future = futures.get(model_name)
        if future is None:
            return {"status": "ready", "message": f"{model_name}: ready"}
        if not future.done:
            return {"status": "pending", "message": f"{model_name}: pending"}
        try:
            future.result()
        except Exception as e:
            return {"status": "not_present", "message": f"{type(e).__name__}: {e}"}
        return {"status": "ready", "message": f"{model_name}: ready"}

    async def _submit(self, coro) -> dict[str, str]:
        job_id = str(uuid.uuid4())
        create_job(job_id)
        asyncio.create_task(self._complete_job(job_id, coro), name=f"job:{job_id}")
        return {"job_id": job_id}

    async def _complete_job(self, job_id: str, coro) -> None:
        record = await coro
        set_job(job_id, record)

    async def _ok(self, value) -> dict[str, Any]:
        result = await value
        result = _dump_result(result)
        return {"status": "completed", "result": result}

    async def _run(self, coro) -> dict[str, Any]:
        try:
            return await self._ok(coro)
        except Exception as e:
            return {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }

    @APP.get("/health")
    async def health(self) -> dict[str, str]:
        return {
            "status": "healthy",
            "name": "TinkerbellServer",
            "job_protocol": "ray_internal_kv",
            "container_model": "single",
        }

    @APP.get("/store_keys")
    async def store_keys(self) -> dict[str, list[str]]:
        """Compatibility diagnostic: list live in-memory server handles."""
        return {
            "trainers": sorted(self.trainers),
            "samplers": sorted(self.samplers),
        }

    @APP.post("/poll")
    async def poll(self, payload: dict[str, Any]) -> dict[str, Any]:
        return poll_job(payload["job_id"])

    @APP.post("/trainers")
    async def create_trainer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._create_trainer(payload)))

    async def _create_trainer(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = self._model_name(payload)
        trainer = self.session.trainer(
            base_model=payload["base_model"],
            world_size=payload.get("world_size", payload.get("tp_size", 1)),
            model_name=model_name,
            adapter_name=payload.get("adapter_name"),
            model_kwargs=payload.get("model_kwargs") or {},
            parallelize_plan=payload.get("parallelize_plan") or {},
            lora_config=payload.get("lora_config"),
            initialize_base_model=payload.get("initialize_base_model", False),
        )
        self.trainers[model_name] = trainer
        self.trainer_ready[model_name] = trainer.ready()
        self.trainer_ready[model_name].done
        return {
            "success": True,
            "model_name": model_name,
            "message": f"trainer {model_name} created",
        }

    @APP.post("/trainer_status")
    async def trainer_status(self, payload: dict[str, Any]) -> dict[str, str]:
        return self._status(self.trainer_ready, payload["model_name"])

    @APP.post("/samplers")
    async def create_sampler(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._create_sampler(payload)))

    async def _create_sampler(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = self._model_name(payload)
        sampler = self.session.sampler(
            base_model=payload["base_model"],
            tp_size=payload.get("tp_size", 1),
            model_name=model_name,
            adapter_name=payload.get("adapter_name"),
            engine_kwargs=payload.get("engine_kwargs") or {},
            sample_concurrency=payload.get("sample_concurrency", 64),
        )
        self.samplers[model_name] = sampler
        self.sampler_ready[model_name] = sampler.ready()
        self.sampler_ready[model_name].done
        return {"success": True, "message": f"sampler {model_name} created"}

    @APP.post("/sampler_status")
    async def sampler_status(self, payload: dict[str, Any]) -> dict[str, str]:
        return self._status(self.sampler_ready, payload["model_name"])

    @APP.post("/forward")
    async def forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._forward(payload)))

    async def _forward(self, payload: dict[str, Any]):
        await self._ensure_ready(self.trainer_ready, payload["model_name"])
        result = await self._result(
            self._trainer(payload["model_name"]).forward(
                data=self._datums(payload),
                forward_kwargs=payload.get("forward_kwargs") or {},
            )
        )
        return result

    @APP.post("/forward_backward")
    async def forward_backward(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._forward_backward(payload)))

    async def _forward_backward(self, payload: dict[str, Any]):
        await self._ensure_ready(self.trainer_ready, payload["model_name"])
        result = await self._result(
            self._trainer(payload["model_name"]).forward_backward(
                data=self._datums(payload),
                forward_kwargs=payload.get("forward_kwargs") or {},
                return_logprobs=payload.get("return_logprobs", False),
                zero_grad=payload.get("zero_grad", False),
                optimizer_params=payload.get("optimizer_params"),
                loss_fn=payload.get("loss_fn", "cross_entropy"),
                loss_fn_config=payload.get("loss_fn_config"),
            )
        )
        return result

    @APP.post("/zero_grad")
    async def zero_grad(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._zero_grad(payload)))

    async def _zero_grad(self, payload: dict[str, Any]):
        await self._ensure_ready(self.trainer_ready, payload["model_name"])
        result = await self._result(self._trainer(payload["model_name"]).zero_grad())
        return result

    @APP.post("/optim_step")
    async def optim_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._optim_step(payload)))

    async def _optim_step(self, payload: dict[str, Any]):
        await self._ensure_ready(self.trainer_ready, payload["model_name"])
        result = await self._result(
            self._trainer(payload["model_name"]).optim_step(
                optimizer_params=payload.get("optimizer_params") or {}
            )
        )
        return result

    @APP.post("/save_checkpoint")
    async def save_checkpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._save_checkpoint(payload)))

    async def _save_checkpoint(self, payload: dict[str, Any]):
        await self._ensure_ready(self.trainer_ready, payload["model_name"])
        result = await self._result(
            self._trainer(payload["model_name"]).save_checkpoint(
                payload["checkpoint_path"]
            )
        )
        return result

    @APP.post("/push_to_hub")
    async def push_to_hub(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._push_to_hub(payload)))

    async def _push_to_hub(self, payload: dict[str, Any]):
        await self._ensure_ready(self.trainer_ready, payload["model_name"])
        result = await self._result(
            self._trainer(payload["model_name"]).push_to_hub(
                repo_id=payload["repo_id"],
                token=payload.get("token"),
                private=payload.get("private", False),
                commit_message=payload.get("commit_message"),
                push_kwargs=payload.get("push_kwargs") or {},
            )
        )
        return result

    @APP.post("/sample")
    async def sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._sample(payload)))

    async def _sample(self, payload: dict[str, Any]):
        model_name = payload.pop("model_name")
        await self._ensure_ready(self.sampler_ready, model_name)
        result = await self._result(self._sampler(model_name).sample(**payload))
        return result

    @APP.post("/sample_batch")
    async def sample_batch(self, payload: dict[str, Any]) -> dict[str, str]:
        return await self._submit(self._run(self._sample_batch(payload)))

    async def _sample_batch(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        await self._ensure_ready(self.sampler_ready, payload["model_name"])
        sampler = self._sampler(payload["model_name"])
        futures = sampler.sample_batch(payload.get("batch_kwargs") or [])
        results = await asyncio.gather(*[asyncio.to_thread(f.result) for f in futures])
        return [_dump_result(r) for r in results]

    @APP.post("/load_checkpoint")
    async def load_checkpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._load_checkpoint(payload)))

    async def _load_checkpoint(self, payload: dict[str, Any]):
        await self._ensure_ready(self.sampler_ready, payload["model_name"])
        result = await self._result(
            self._sampler(payload["model_name"]).load_checkpoint(
                checkpoint_path=payload["checkpoint_path"],
                pin_lora=payload.get("pin_lora", False),
            )
        )
        return result

    @APP.post("/get_lora_info")
    async def get_lora_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._get_lora_info(payload)))

    async def _get_lora_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_ready(self.sampler_ready, payload["model_name"])
        return await asyncio.to_thread(
            self._sampler(payload["model_name"]).get_lora_info
        )

    @APP.post("/shutdown_sampler")
    async def shutdown_sampler(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._submit(self._run(self._shutdown_sampler(payload)))

    async def _shutdown_sampler(self, payload: dict[str, Any]):
        result = await self._result(self._sampler(payload["model_name"]).shutdown())
        self.samplers.pop(payload["model_name"], None)
        return result


def deploy_service(
    server_url: str = "http://127.0.0.1:8000",
    namespace: str = "tinkerbell",
    **deployment_kwargs: Any,
) -> str:
    if not ray.is_initialized():
        ray.init(namespace=namespace)
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    deployment_kwargs.setdefault("ray_actor_options", {"num_gpus": 0})
    deployment = serve.deployment(**deployment_kwargs)(
        serve.ingress(APP)(TinkerbellServer)
    )
    serve.run(deployment.bind(namespace=namespace))
    return server_url


__all__ = ["APP", "TinkerbellServer", "deploy_service"]


def _dump_result(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_dump_result(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump_result(v) for k, v in value.items() if v is not None}
    return value
