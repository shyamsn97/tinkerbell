"""Tiny HTTP client for the Tinkerbell server."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import time
from typing import Any, Callable, Generic, TypeVar

import httpx
from tinker.types import Datum

from tinkerbell.renderer import MASK_TOKEN_ID, Renderer, RenderMode, TrainOnWhat
from tinkerbell.types.responses import (
    ActorStatusResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    HealthResponse,
    LoadCheckpointResponse,
    OptimStepResponse,
    PushToHubResponse,
    SampleResponse,
    SaveCheckpointResponse,
    ShutdownSamplingActorResponse,
    ZeroGradResponse,
)
from tinkerbell.utils import clean_model_name

T = TypeVar("T")


def _tensordata_to_wire(td: Any) -> dict[str, Any]:
    """Serialize a tinker `TensorData` into the stable `{data, dtype, shape}`
    wire format accepted by `TensorData.__init__` on both old (pydantic) and
    new (dataclass) tinker releases."""
    arr = td.to_numpy()
    return {
        "data": arr.tolist(),
        "dtype": td.dtype,
        "shape": list(td.shape) if td.shape is not None else list(arr.shape),
    }


def _jsonable(value):
    from tinker.types import TensorData

    if isinstance(value, TensorData):
        return _tensordata_to_wire(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _drop_none(value):
    """Strip None fields from server JSON before pydantic rehydration.

    Some `TensorData` dumps include sparse_* fields set to None, while older
    tinker models reject those as extra inputs.
    """
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


def _looks_like_tensordata(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "data" in value
        and "dtype" in value
        and "shape" in value
    )


def _to_tensordata(value: Any) -> Any:
    """Rehydrate `TensorData` dicts into TensorData instances.

    The response models intentionally type tensor fields as `Any` so Pydantic
    does not recurse into the (non-pydantic) `TensorData` dataclass. We restore
    the typed object here so callers can keep using `.to_torch()` etc.
    """
    if _looks_like_tensordata(value):
        from tinker.types import TensorData

        return TensorData(**value)
    if isinstance(value, dict):
        return {k: _to_tensordata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_tensordata(v) for v in value]
    return value


class HTTPFuture(Generic[T]):
    def __init__(self, future: concurrent.futures.Future, parse: Callable[[Any], T]):
        self._future = future
        self._parse = parse

    @property
    def done(self) -> bool:
        return self._future.done()

    def result(self) -> T:
        return self._parse(_drop_none(self._future.result()))


class ServiceClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        timeout: float = 1200.0,
        poll_interval: float = 1.0,
        max_workers: int = 128,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.http = httpx.Client(
            base_url=self.server_url,
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=30.0),
            limits=httpx.Limits(max_connections=max_workers),
            follow_redirects=True,
        )
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    @classmethod
    def deploy(
        cls,
        server_url: str = "http://127.0.0.1:8000",
        wait_for_ready: bool = True,
        timeout: float = 1200.0,
        **deploy_kwargs,
    ) -> "ServiceClient":
        from tinkerbell.api.server import deploy_service

        url = deploy_service(server_url=server_url, **deploy_kwargs)
        client = cls(server_url=url, timeout=timeout)
        if wait_for_ready:
            client.wait_until_ready()
        return client

    @classmethod
    def deploy_or_connect(
        cls,
        deploy_config=None,
        redeploy: bool = False,
        wait_for_ready: bool = True,
        timeout: float = 1200.0,
        **deploy_kwargs,
    ) -> "ServiceClient":
        from tinkerbell.api.server import deploy_service

        if deploy_config is not None:
            if redeploy:
                server_url = deploy_config.deploy()
            else:
                try:
                    server_url = deploy_config.connect()
                except Exception:
                    server_url = deploy_config.deploy()
        else:
            server_url = deploy_service(**deploy_kwargs)
        client = cls(server_url=server_url, timeout=timeout)
        if wait_for_ready:
            client.wait_until_ready()
        return client

    _RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def _post(self, endpoint: str, payload: dict[str, Any]) -> Any:
        body = _jsonable(payload)
        backoff = 0.5
        for attempt in range(8):
            try:
                response = self.http.post(endpoint, json=body)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                status = (
                    e.response.status_code
                    if isinstance(e, httpx.HTTPStatusError)
                    else None
                )
                transient = status in self._RETRY_STATUS or status is None
                if not transient or attempt == 7:
                    raise
                time.sleep(min(backoff, 8.0))
                backoff *= 2

    def _submit(self, endpoint: str, payload: dict[str, Any]) -> str:
        data = self._post(endpoint, payload)
        job_id = data.get("job_id")
        if not job_id:
            raise RuntimeError(f"{endpoint} did not return a job_id: {data}")
        return job_id

    def _poll(self, job_id: str) -> dict[str, Any]:
        try:
            return self._post("/poll", {"job_id": job_id})
        except httpx.HTTPStatusError:
            return {"status": "pending"}

    def _wait(self, job_id: str) -> Any:
        deadline = time.time() + self.timeout
        while True:
            data = self._poll(job_id)
            status = data.get("status")
            if status == "completed":
                return data.get("result")
            if status == "error":
                raise RuntimeError(data.get("error") or f"job {job_id} failed")
            if time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for job {job_id} after {self.timeout}s"
                )
            time.sleep(self.poll_interval)

    def _submit_and_wait(self, endpoint: str, payload: dict[str, Any]) -> Any:
        return self._wait(self._submit(endpoint, payload))

    def _wait_status(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float,
        poll_interval: float,
    ) -> None:
        deadline = time.time() + timeout
        last_status = None
        while True:
            data = self._post(endpoint, payload)
            status = data.get("status")
            if status == "ready":
                return
            if status == "not_present":
                raise RuntimeError(data.get("message") or f"{endpoint} failed")
            if status != last_status:
                print(data.get("message") or f"{endpoint}: {status}")
                last_status = status
            if time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for {endpoint} after {timeout}s")
            time.sleep(poll_interval)

    def _future(
        self,
        endpoint: str,
        payload: dict[str, Any],
        parse: Callable[[Any], T],
    ) -> HTTPFuture[T]:
        return HTTPFuture(
            self.pool.submit(self._submit_and_wait, endpoint, payload), parse
        )

    def get_health(self) -> HealthResponse:
        response = self.http.get("/health")
        response.raise_for_status()
        return HealthResponse(**response.json())

    def get_store_keys(self) -> dict[str, list[str]]:
        response = self.http.get("/store_keys")
        response.raise_for_status()
        return response.json()

    def wait_until_ready(
        self, timeout: float = 120.0, poll_interval: float = 1.0
    ) -> None:
        start = time.time()
        while True:
            try:
                self.get_health()
                return
            except Exception:
                if time.time() - start > timeout:
                    raise
                time.sleep(poll_interval)

    def create_training_client(
        self,
        base_model: str,
        tp_size: int = 1,
        world_size: int | None = None,
        model_name: str | None = None,
        adapter_name: str | None = None,
        parallelize_plan: dict[str, str] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        lora_config: dict[str, Any] | None = None,
        wait_until_ready: bool = False,
        initialize_base_model: bool = False,
        **_,
    ) -> "TrainingClient":
        del wait_until_ready
        model_name = model_name or clean_model_name(base_model)
        self._submit_and_wait(
            "/trainers",
            {
                "base_model": base_model,
                "model_name": model_name,
                "adapter_name": adapter_name,
                "world_size": world_size or tp_size,
                "parallelize_plan": parallelize_plan or {},
                "model_kwargs": model_kwargs or {},
                "lora_config": lora_config,
                "initialize_base_model": initialize_base_model,
            },
        )
        return TrainingClient(
            service=self,
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
        )

    def create_sampling_client(
        self,
        base_model: str,
        tp_size: int = 1,
        model_name: str | None = None,
        adapter_name: str | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        wait_until_ready: bool = False,
        **_,
    ) -> "SamplingClient":
        del wait_until_ready
        model_name = model_name or clean_model_name(base_model)
        self._submit_and_wait(
            "/samplers",
            {
                "base_model": base_model,
                "model_name": model_name,
                "adapter_name": adapter_name,
                "tp_size": tp_size,
                "engine_kwargs": engine_kwargs or {},
            },
        )
        return SamplingClient(
            service=self,
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
        )


class TrainingClient:
    def __init__(
        self,
        service: ServiceClient,
        base_model: str,
        model_name: str,
        adapter_name: str | None = None,
    ):
        self.service = service
        self.base_model = base_model
        self.model_name = model_name
        self.adapter_name = adapter_name
        self._tokenizer = None
        self._renderer = None

    def wait_until_ready(
        self, timeout: float = 1200.0, poll_interval: float = 2.0
    ) -> None:
        self.service._wait_status(
            "/trainer_status",
            {"model_name": self.model_name},
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def forward(
        self, data: list[Datum], forward_kwargs: dict[str, Any] | None = None
    ) -> HTTPFuture[ForwardResponse]:
        return self.service._future(
            "/forward",
            {
                "model_name": self.model_name,
                "data": data,
                "forward_kwargs": forward_kwargs or {},
            },
            lambda r: ForwardResponse(**_to_tensordata(r)),
        )

    def forward_backward(
        self,
        data: list[Datum],
        forward_kwargs: dict[str, Any] | None = None,
        return_logprobs: bool = False,
        zero_grad: bool = True,
        optimizer_params: dict[str, Any] | None = None,
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, float] | None = None,
    ) -> HTTPFuture[ForwardBackwardResponse]:
        return self.service._future(
            "/forward_backward",
            {
                "model_name": self.model_name,
                "data": data,
                "forward_kwargs": forward_kwargs or {},
                "return_logprobs": return_logprobs,
                "zero_grad": zero_grad,
                "optimizer_params": optimizer_params,
                "loss_fn": loss_fn,
                "loss_fn_config": loss_fn_config,
            },
            lambda r: ForwardBackwardResponse(**_to_tensordata(r)),
        )

    def zero_grad(self) -> HTTPFuture[ZeroGradResponse]:
        return self.service._future(
            "/zero_grad",
            {"model_name": self.model_name},
            lambda r: ZeroGradResponse(**r),
        )

    def optim_step(
        self, optimizer_params: dict[str, Any] | None = None
    ) -> HTTPFuture[OptimStepResponse]:
        return self.service._future(
            "/optim_step",
            {"model_name": self.model_name, "optimizer_params": optimizer_params or {}},
            lambda r: OptimStepResponse(**r),
        )

    def save_checkpoint(
        self, checkpoint_path: str
    ) -> HTTPFuture[SaveCheckpointResponse]:
        return self.service._future(
            "/save_checkpoint",
            {"model_name": self.model_name, "checkpoint_path": checkpoint_path},
            lambda r: SaveCheckpointResponse(**r),
        )

    def push_to_hub(self, repo_id: str, **kwargs) -> HTTPFuture[PushToHubResponse]:
        payload = {"model_name": self.model_name, "repo_id": repo_id, **kwargs}
        return self.service._future(
            "/push_to_hub", payload, lambda r: PushToHubResponse(**r)
        )

    def save_weights_and_get_sampling_client(
        self,
        checkpoint_path: str,
        tp_size: int | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        wait_until_ready: bool = False,
    ) -> "SamplingClient":
        self.save_checkpoint(checkpoint_path).result()
        final_kwargs = dict(engine_kwargs or {})
        if self.adapter_name:
            final_kwargs["lora_paths"] = [checkpoint_path]
            actor_base_model = self.base_model
        else:
            final_kwargs["enable_lora"] = False
            actor_base_model = checkpoint_path
        sampler = self.service.create_sampling_client(
            base_model=actor_base_model,
            tp_size=tp_size or 1,
            model_name=self.model_name,
            adapter_name=self.adapter_name,
            engine_kwargs=final_kwargs,
            wait_until_ready=wait_until_ready,
        )
        if self.adapter_name:
            sampler.load_checkpoint(checkpoint_path).result()
        return sampler

    def get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or 0
        return self._tokenizer

    def get_renderer(self) -> Renderer:
        if self._renderer is None:
            self._renderer = Renderer(self.get_tokenizer())
        return self._renderer

    def render(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        mode: RenderMode = RenderMode.TRAINING,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        **kwargs,
    ):
        return self.get_renderer().render(
            messages=messages,
            mode=mode,
            train_on_what=train_on_what,
            mask_value=mask_value,
            **kwargs,
        )

    def build_chat_samples(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        **kwargs,
    ):
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render(messages, mode, train_on_what, mask_value, **kwargs)


class SamplingClient:
    def __init__(
        self,
        service: ServiceClient,
        base_model: str,
        model_name: str,
        adapter_name: str | None = None,
    ):
        self.service = service
        self.base_model = base_model
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.lora_path = None

    def wait_until_ready(
        self, timeout: float = 1200.0, poll_interval: float = 2.0, *_, **__
    ) -> None:
        self.service._wait_status(
            "/sampler_status",
            {"model_name": self.model_name},
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def sample(self, **kwargs) -> HTTPFuture[SampleResponse]:
        return self.service._future(
            "/sample",
            {"model_name": self.model_name, **kwargs},
            lambda r: SampleResponse(**_to_tensordata(r)),
        )

    def sample_batch(
        self, batch_kwargs: list[dict[str, Any]]
    ) -> list[HTTPFuture[SampleResponse]]:
        return [self.sample(**kw) for kw in batch_kwargs]

    def sample_many(
        self, batch_kwargs: list[dict[str, Any]]
    ) -> HTTPFuture[list[SampleResponse]]:
        return self.service._future(
            "/sample_batch",
            {"model_name": self.model_name, "batch_kwargs": batch_kwargs},
            lambda rows: [SampleResponse(**_to_tensordata(row)) for row in rows],
        )

    def load_checkpoint(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> HTTPFuture[LoadCheckpointResponse]:
        self.lora_path = checkpoint_path
        return self.service._future(
            "/load_checkpoint",
            {
                "model_name": self.model_name,
                "checkpoint_path": checkpoint_path,
                "pin_lora": pin_lora,
            },
            lambda r: LoadCheckpointResponse(**r),
        )

    def get_status(self) -> HTTPFuture[ActorStatusResponse]:
        return HTTPFuture(
            self.service.pool.submit(lambda: {"status": "ready", "message": None}),
            lambda r: ActorStatusResponse(**r),
        )

    def get_lora_info(self) -> dict[str, Any]:
        return self.service._submit_and_wait(
            "/get_lora_info", {"model_name": self.model_name}
        )

    def shutdown(self) -> HTTPFuture[ShutdownSamplingActorResponse]:
        return self.service._future(
            "/shutdown_sampler",
            {"model_name": self.model_name},
            lambda r: ShutdownSamplingActorResponse(**r),
        )


__all__ = ["HTTPFuture", "SamplingClient", "ServiceClient", "TrainingClient"]
