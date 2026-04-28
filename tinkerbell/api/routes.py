"""FastAPI routes + Ray Serve deployment class.

All business logic lives in `TrainingEngine` / `TrainingExecutor` /
`SamplingEngine`. The API is the thinnest possible layer: translate
HTTP → Op (or direct engine call for side-effect ops), return a
`JobHandle`; clients poll `/poll` for the outcome.

Engine handles are cached in-process. Misses go to the `Registry`, which
is the single source of truth; if the Registry doesn't know about a
route, it's a 404.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException
from tinker.types import TensorData

from tinkerbell.sampling.engine import (
    SamplingEngine,
    attach_sampling_engine,
    spawn_sampling_engine,
)
from tinkerbell.state.job_store import get_or_create_job_store
from tinkerbell.state.ops import (
    ForwardBackwardOp,
    ForwardOp,
    Op,
    OptimStepOp,
    ZeroGradOp,
)
from tinkerbell.state.registry import (
    EngineRecord,
    get_or_create_registry,
    resolve_handle,
)
from tinkerbell.state.work_queue import get_or_create_work_queue
from tinkerbell.training.engine import (
    TrainingEngine,
    attach_training_engine,
    spawn_training_engine,
)
from tinkerbell.training.executor import TrainingExecutor
from tinkerbell.types import (
    ActorStatusRequest,
    ActorStatusResponse,
    CreateSamplingActorRequest,
    CreateSamplingActorResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    ErrorRecord,
    ForwardBackwardRequest,
    ForwardRequest,
    HealthResponse,
    JobHandle,
    JobKind,
    JobRecord,
    LoadCheckpointRequest,
    OptimStepRequest,
    PollResultRequest,
    PollResultResponse,
    PushToHubRequest,
    RouteKey,
    SampleRequest,
    SaveCheckpointRequest,
    ShutdownSamplingActorRequest,
    ZeroGradRequest,
)
from tinkerbell.utils import (
    clean_model_name,
    model_to_dict,
    rehydrate_registry,
    run_push_to_hub_job,
    run_save_checkpoint_job,
)

logger = logging.getLogger(__name__)

APP = FastAPI()


def _new_job_id() -> str:
    return str(uuid.uuid4())


def _engine_route(model_name: str) -> RouteKey:
    """Engine cache / registry / queue key.

    Engines are physically per-model: one TP group per training engine,
    with many LoRA adapters hot-swapped internally. The adapter travels
    per-op via `Op.route.adapter`, not via the engine key.
    """
    return RouteKey(model=clean_model_name(model_name), adapter=None)


class TinkerbellAPI:
    """Ray Serve deployment. Stateless modulo in-process engine cache."""

    def __init__(
        self,
        server_url: str,
        max_wait_time: float = 600.0,
        clock_cycle: float = 2.0,
    ):
        self.server_url = server_url
        self.max_wait_time = max_wait_time
        self.clock_cycle = clock_cycle

        self.job_store = get_or_create_job_store()
        self.work_queue = get_or_create_work_queue()
        self.registry = get_or_create_registry()

        self.training_engines: dict[RouteKey, TrainingEngine] = {}
        self.training_executors: dict[RouteKey, TrainingExecutor] = {}
        self.sampling_engines: dict[RouteKey, SamplingEngine] = {}

        # Boot: re-attach to surviving Ray actors and start their executors.
        # One-shot — JobStore self-purges and Ray Serve restarts us on crash.
        asyncio.create_task(self._boot(), name="api_boot")

    async def _boot(self) -> None:
        try:
            added = rehydrate_registry(self.registry)
            if added:
                logger.info(f"Registry rehydrated: {added} records added")
        except Exception as e:
            logger.warning(f"Registry rehydration failed: {e}")

        try:
            records = await self.registry.list.remote("training")
        except Exception as e:
            logger.warning(f"Registry list failed during boot: {e}")
            return

        for rec in records:
            if rec.route in self.training_executors:
                continue
            handle = resolve_handle(rec)
            if handle is None:
                await self.registry.remove.remote(rec.route)
                continue
            engine = attach_training_engine(handle, max_wait_time=self.max_wait_time)
            self._install_training_engine(engine)

    # ------------------------------------------------------------------
    # Engine lookup helpers
    # ------------------------------------------------------------------

    async def _get_training_engine(self, route: RouteKey) -> TrainingEngine | None:
        engine = self.training_engines.get(route)
        if engine is not None:
            return engine
        record: EngineRecord | None = await self.registry.lookup.remote(route)
        if record is None or record.kind != "training":
            return None
        handle = resolve_handle(record)
        if handle is None:
            return None
        engine = attach_training_engine(handle, max_wait_time=self.max_wait_time)
        self._install_training_engine(engine)
        return engine

    def _install_training_engine(self, engine: TrainingEngine) -> None:
        self.training_engines[engine.route] = engine
        executor = TrainingExecutor(
            engine=engine,
            work_queue=self.work_queue,
            job_store=self.job_store,
            clock_cycle=self.clock_cycle,
        )
        self.training_executors[engine.route] = executor
        executor.start()

    async def _get_sampling_engine(self, route: RouteKey) -> SamplingEngine | None:
        engine = self.sampling_engines.get(route)
        if engine is not None:
            return engine
        record: EngineRecord | None = await self.registry.lookup.remote(route)
        if record is None or record.kind != "sampling":
            return None
        handle = resolve_handle(record)
        if handle is None:
            return None
        attached = attach_sampling_engine(handle)
        if attached is None:
            return None
        self.sampling_engines[route] = attached
        return attached

    async def _require_training_engine(self, model_name: str) -> TrainingEngine:
        engine = await self._get_training_engine(_engine_route(model_name))
        if engine is None:
            raise HTTPException(
                status_code=404,
                detail=f"No training engine registered for '{model_name}'",
            )
        return engine

    # ------------------------------------------------------------------
    # Submit helpers — collapse the boilerplate from every endpoint.
    # ------------------------------------------------------------------

    async def _submit_op(self, route: RouteKey, kind: JobKind, op: Op) -> JobHandle:
        """Create a JobRecord and enqueue an Op for the route's executor.

        The Op carries the canonical `job_id`; the JobStore record is keyed
        by that same id so the executor's writes land where /poll reads.
        """
        await self.job_store.create.remote(op.job_id, kind)
        await self.work_queue.submit.remote(route, op)
        return JobHandle(job_id=op.job_id, kind=kind)

    async def _spawn_job(
        self,
        kind: JobKind,
        coro_factory: Callable[[str], Awaitable[None]],
    ) -> JobHandle:
        """Register a JobRecord, run `coro_factory(job_id)` as a background task."""
        job_id = _new_job_id()
        await self.job_store.create.remote(job_id, kind)
        asyncio.create_task(coro_factory(job_id), name=f"{kind}:{job_id}")
        return JobHandle(job_id=job_id, kind=kind)

    # ------------------------------------------------------------------
    # Health / diagnostics
    # ------------------------------------------------------------------

    @APP.get("/health")
    async def health(self) -> HealthResponse:
        return HealthResponse(status="healthy", name="TinkerbellAPI")

    @APP.get("/queue_depth")
    async def queue_depth(self) -> dict[str, int]:
        return await self.work_queue.depth.remote()

    # ------------------------------------------------------------------
    # Polling (single endpoint for every async op)
    # ------------------------------------------------------------------

    @APP.post("/poll")
    async def poll(self, request: PollResultRequest) -> PollResultResponse:
        """Poll by job_id. Pops on completion, leaves pending entries in place."""
        record: JobRecord | None = await self.job_store.poll.remote(
            request.request_id, pop=True
        )
        if record is None or record.status == "pending":
            return PollResultResponse(
                status="pending", request_id=request.request_id, result=None
            )
        if record.status == "success":
            return PollResultResponse(
                status="completed",
                request_id=request.request_id,
                result=record.result,
            )
        # record.status == "error"
        err = record.error
        if err is not None:
            detail = f"{err.type}: {err.message}" if err.message else err.type
            if err.traceback:
                # Include the server-side traceback tail so the client log is
                # actionable without SSH'ing to Ray.
                detail = f"{detail}\n--- server traceback ---\n{err.traceback}"
        else:
            # An error-status record with no ErrorRecord attached is a bug.
            logger.error(
                f"Job {record.job_id} has status=error but error=None; "
                f"record={record!r}"
            )
            detail = f"error with empty ErrorRecord on job {record.job_id}"
        return PollResultResponse(
            status="error",
            request_id=request.request_id,
            result=(
                {
                    "success": False,
                    "error": detail,
                    "error_type": err.type if err else "Error",
                }
                if err
                else None
            ),
            error=detail,
        )

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    @APP.post("/create_training_actors")
    async def create_training_actors(
        self, request: CreateTrainingActorsRequest
    ) -> CreateTrainingActorsResponse:
        model_name = request.model_name or clean_model_name(request.base_model)
        # Engines are keyed by model; adapters live inside the engine.
        engine_route = _engine_route(model_name)

        engine = self.training_engines.get(engine_route)
        if engine is None:
            record = await self.registry.lookup.remote(engine_route)
            if record is not None:
                handle = resolve_handle(record)
                if handle is not None:
                    engine = attach_training_engine(
                        handle, max_wait_time=self.max_wait_time
                    )

        lora_dict = (
            request.lora_config
            if isinstance(request.lora_config, dict)
            else (
                request.lora_config.model_dump()
                if request.lora_config is not None
                else None
            )
        )

        if engine is None:
            engine = spawn_training_engine(
                world_size=request.world_size,
                base_model=request.base_model,
                model_name=model_name,
                adapter_name=request.adapter_name,
                model_kwargs=request.model_kwargs,
                parallelize_plan=request.parallelize_plan,
                lora_config=lora_dict,
                ray_worker_options=request.ray_worker_options,
                initialize_base_model=request.initialize_base_model,
                max_wait_time=self.max_wait_time,
            )
            await self.registry.register.remote(engine.handle.record)
        elif request.adapter_name and lora_dict is not None:
            # Engine already exists (cached or rehydrated); ensure this
            # adapter is installed before ops start landing on it.
            current = await engine.broadcast("get_adapters")
            existing: set[str] = set()
            for ranks_adapters in current:
                if isinstance(ranks_adapters, list):
                    existing.update(ranks_adapters)
            if request.adapter_name not in existing:
                await engine.add_adapter(request.adapter_name, lora_dict)

        if engine_route not in self.training_executors:
            self._install_training_engine(engine)
        else:
            self.training_engines[engine_route] = engine

        return CreateTrainingActorsResponse(
            success=True,
            model_name=model_name,
            message=f"Training actors for {model_name} ready",
        )

    @APP.post("/get_actor_status")
    async def get_actor_status(
        self, request: ActorStatusRequest
    ) -> ActorStatusResponse:
        engine = await self._get_training_engine(_engine_route(request.model_name))
        if engine is None:
            return ActorStatusResponse(
                status="not_present",
                message=f"No training engine for {request.model_name}",
            )
        ready = engine._ready or await engine.wait_until_ready()
        status = "ready" if ready else "pending"
        return ActorStatusResponse(
            status=status, message=f"{request.model_name}: {status}"
        )

    # ------------------------------------------------------------------
    # Training ops (queued)
    # ------------------------------------------------------------------

    @APP.post("/forward")
    async def forward(self, request: ForwardRequest) -> JobHandle:
        engine = await self._require_training_engine(request.model_name)
        return await self._submit_op(
            engine.route,
            JobKind.FORWARD,
            ForwardOp(
                job_id=_new_job_id(),
                route=engine.route,  # forward doesn't use an adapter
                data=request.data,
                forward_kwargs=request.forward_kwargs,
            ),
        )

    @APP.post("/forward_backward")
    async def forward_backward(self, request: ForwardBackwardRequest) -> JobHandle:
        engine = await self._require_training_engine(request.model_name)
        route = RouteKey(model=engine.route.model, adapter=request.adapter_name)

        if request.zero_grad:
            await self._submit_op(
                engine.route,
                JobKind.ZERO_GRAD,
                ZeroGradOp(job_id=_new_job_id(), route=route),
            )

        fb_handle = await self._submit_op(
            engine.route,
            JobKind.FORWARD_BACKWARD,
            ForwardBackwardOp(
                job_id=_new_job_id(),
                route=route,
                data=request.data,
                forward_kwargs=request.forward_kwargs,
                return_logprobs=request.return_logprobs,
                loss_fn=request.loss_fn,
                loss_fn_config=request.loss_fn_config,
            ),
        )

        if request.optimizer_params is not None:
            await self._submit_op(
                engine.route,
                JobKind.OPTIM_STEP,
                OptimStepOp(
                    job_id=_new_job_id(),
                    route=route,
                    optimizer_params=request.optimizer_params,
                ),
            )

        return fb_handle

    @APP.post("/zero_grad")
    async def zero_grad(self, request: ZeroGradRequest) -> JobHandle:
        engine = await self._require_training_engine(request.model_name)
        route = RouteKey(model=engine.route.model, adapter=request.adapter_name)
        return await self._submit_op(
            engine.route,
            JobKind.ZERO_GRAD,
            ZeroGradOp(job_id=_new_job_id(), route=route),
        )

    @APP.post("/optim_step")
    async def optim_step(self, request: OptimStepRequest) -> JobHandle:
        engine = await self._require_training_engine(request.model_name)
        route = RouteKey(model=engine.route.model, adapter=request.adapter_name)
        return await self._submit_op(
            engine.route,
            JobKind.OPTIM_STEP,
            OptimStepOp(
                job_id=_new_job_id(),
                route=route,
                optimizer_params=request.optimizer_params,
            ),
        )

    # ------------------------------------------------------------------
    # Training side-effect ops (save/push) — run out of band
    # ------------------------------------------------------------------

    @APP.post("/save_checkpoint")
    async def save_checkpoint(self, request: SaveCheckpointRequest) -> JobHandle:
        engine = await self._require_training_engine(request.model_name)
        return await self._spawn_job(
            JobKind.SAVE_CHECKPOINT,
            lambda job_id: run_save_checkpoint_job(
                engine=engine,
                registry=self.registry,
                job_store=self.job_store,
                job_id=job_id,
                checkpoint_path=request.checkpoint_path,
                adapter_name=request.adapter_name,
            ),
        )

    @APP.post("/push_to_hub")
    async def push_to_hub(self, request: PushToHubRequest) -> JobHandle:
        engine = await self._require_training_engine(request.model_name)
        return await self._spawn_job(
            JobKind.PUSH_TO_HUB,
            lambda job_id: run_push_to_hub_job(
                engine=engine,
                job_store=self.job_store,
                job_id=job_id,
                repo_id=request.repo_id,
                adapter_name=request.adapter_name,
                token=request.token,
                private=request.private,
                commit_message=request.commit_message,
                push_kwargs=request.push_kwargs,
            ),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @APP.post("/create_sampling_actor")
    async def create_sampling_actor(
        self, request: CreateSamplingActorRequest
    ) -> CreateSamplingActorResponse:
        route = _engine_route(request.model_name)

        existing = self.sampling_engines.get(route)
        if existing is None:
            record = await self.registry.lookup.remote(route)
            if record is not None:
                handle = resolve_handle(record)
                if handle is not None:
                    existing = attach_sampling_engine(handle)

        if existing is not None:
            self.sampling_engines[route] = existing
            return CreateSamplingActorResponse(
                success=True, message=f"Reusing sampling actor for {request.model_name}"
            )

        engine = spawn_sampling_engine(
            base_model=request.base_model,
            route=route,
            tp_size=request.tp_size,
            engine_kwargs=request.engine_kwargs,
        )
        self.sampling_engines[route] = engine
        await self.registry.register.remote(engine.handle.record)
        return CreateSamplingActorResponse(
            success=True, message=f"Sampling actor for {request.model_name} created"
        )

    @APP.post("/get_sampling_actor_status")
    async def get_sampling_actor_status(self, request: ActorStatusRequest) -> JobHandle:
        """Wrapped as a JobHandle for client parity (always completes immediately)."""
        route = _engine_route(request.model_name)
        engine = await self._get_sampling_engine(route)
        if engine is None:
            job_id = _new_job_id()
            await self.job_store.create.remote(job_id, JobKind.SAMPLING_ACTOR_STATUS)
            await self.job_store.set_success.remote(
                job_id,
                {
                    "status": "not_present",
                    "message": f"No sampling actor for {request.model_name}",
                },
            )
            return JobHandle(job_id=job_id, kind=JobKind.SAMPLING_ACTOR_STATUS)

        async def _report(job_id: str) -> None:
            try:
                status = await engine.get_status()
                await self.job_store.set_success.remote(
                    job_id, {"status": status, "message": f"sampling actor is {status}"}
                )
            except Exception as e:
                await self.job_store.set_error.remote(
                    job_id, ErrorRecord.from_exception(e)
                )

        return await self._spawn_job(JobKind.SAMPLING_ACTOR_STATUS, _report)

    @APP.post("/sample")
    async def sample(self, request: SampleRequest) -> JobHandle:
        route = _engine_route(request.model_name)
        engine = await self._get_sampling_engine(route)
        if engine is None:
            raise HTTPException(
                status_code=404,
                detail=f"Sampling actor for '{request.model_name}' not found",
            )

        async def _run(job_id: str) -> None:
            try:
                request_dict = model_to_dict(
                    request, exclude=["model_name"], exclude_none=True
                )
                if isinstance(request.input_ids, TensorData):
                    request_dict["input_ids"] = request.input_ids.tolist()
                if isinstance(request.input_embeds, TensorData):
                    request_dict["input_embeds"] = request.input_embeds.tolist()
                sample_result = await engine.actor.sample.remote(request_dict)
                await self.job_store.set_success.remote(
                    job_id,
                    {
                        "outputs": sample_result.get("outputs", []),
                        "logprobs": sample_result.get("logprobs"),
                        "top_logprobs": sample_result.get("top_logprobs"),
                        "output_token_ids": sample_result.get("output_token_ids"),
                        "finish_reasons": sample_result.get("finish_reasons"),
                        "meta_info": sample_result.get("meta_info"),
                    },
                )
            except Exception as e:
                logger.error(f"sample failed: {e}", exc_info=True)
                await self.job_store.set_error.remote(
                    job_id, ErrorRecord.from_exception(e)
                )

        return await self._spawn_job(JobKind.SAMPLE, _run)

    @APP.post("/load_checkpoint")
    async def load_checkpoint(self, request: LoadCheckpointRequest) -> JobHandle:
        route = _engine_route(request.model_name)
        engine = await self._get_sampling_engine(route)
        if engine is None:
            raise HTTPException(
                status_code=404,
                detail=f"Sampling actor for '{request.model_name}' not found",
            )
        job_id = _new_job_id()
        await self.job_store.create.remote(job_id, JobKind.LOAD_CHECKPOINT)

        try:
            engine.mark_pending(
                engine.actor.update_weights_from_disk.remote(
                    checkpoint_path=request.checkpoint_path,
                    load_format=None,
                    pin_lora=request.pin_lora,
                )
            )
            await self.job_store.set_success.remote(
                job_id,
                {
                    "model_name": request.model_name,
                    "success": True,
                    "message": f"Checkpoint loading started from {request.checkpoint_path}",
                },
            )
        except Exception as e:
            logger.error(f"load_checkpoint failed: {e}", exc_info=True)
            await self.job_store.set_error.remote(job_id, ErrorRecord.from_exception(e))
        return JobHandle(job_id=job_id, kind=JobKind.LOAD_CHECKPOINT)

    @APP.post("/get_lora_info")
    async def get_lora_info(self, request: ActorStatusRequest) -> dict[str, Any]:
        route = _engine_route(request.model_name)
        engine = await self._get_sampling_engine(route)
        if engine is None:
            return {"error": f"Sampling actor for '{request.model_name}' not found"}
        try:
            return await engine.actor.get_lora_info.remote()
        except Exception as e:
            return {"error": str(e)}

    @APP.post("/shutdown_sampling_actor")
    async def shutdown_sampling_actor(
        self, request: ShutdownSamplingActorRequest
    ) -> JobHandle:
        route = _engine_route(request.model_name)
        engine = await self._get_sampling_engine(route)
        if engine is None:
            job_id = _new_job_id()
            await self.job_store.create.remote(job_id, JobKind.SHUTDOWN_SAMPLING)
            await self.job_store.set_success.remote(
                job_id,
                {
                    "model_name": request.model_name,
                    "success": False,
                    "message": f"No sampling actor for {request.model_name}",
                },
            )
            return JobHandle(job_id=job_id, kind=JobKind.SHUTDOWN_SAMPLING)

        async def _shutdown(job_id: str) -> None:
            try:
                await engine.actor.shutdown.remote()
                self.sampling_engines.pop(route, None)
                await self.registry.remove.remote(route)
                await self.job_store.set_success.remote(
                    job_id,
                    {
                        "model_name": request.model_name,
                        "success": True,
                        "message": f"Sampling actor '{request.model_name}' shut down",
                    },
                )
            except Exception as e:
                logger.error(f"shutdown_sampling_actor failed: {e}", exc_info=True)
                await self.job_store.set_error.remote(
                    job_id, ErrorRecord.from_exception(e)
                )

        return await self._spawn_job(JobKind.SHUTDOWN_SAMPLING, _shutdown)
