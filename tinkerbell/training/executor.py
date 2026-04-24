"""TrainingExecutor: per-engine drain-and-dispatch loop.

One executor runs in the gateway process per live `TrainingEngine`. It is
started when an engine is spawned/attached and stopped on shutdown.

Design:
  1. Wait (with timeout = clock cycle) for any ops on our RouteKey.
  2. Drain the queue for this route.
  3. Walk the drained list in order:
     - Coalesce consecutive Forward/ForwardBackward ops into one TP call.
     - ZeroGrad and OptimStep act as barriers: flush any pending batch
       first, then execute.
  4. Per op, write a JobRecord to JobStore (success or error).
  5. One op failing only sets that op's error record; the loop keeps going.

This is the v1 `_process_batch` logic extracted and scoped to one engine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tinkerbell.state.ops import (
    ForwardBackwardOp,
    ForwardOp,
    Op,
    OptimStepOp,
    ZeroGradOp,
)
from tinkerbell.training.engine import TrainingEngine
from tinkerbell.types.jobs import ErrorRecord
from tinkerbell.types.route import RouteKey

logger = logging.getLogger(__name__)


class TrainingExecutor:
    def __init__(
        self,
        engine: TrainingEngine,
        work_queue: Any,
        job_store: Any,
        clock_cycle: float = 2.0,
    ):
        self.engine = engine
        self.work_queue = work_queue
        self.job_store = job_store
        self.clock_cycle = clock_cycle
        self.task: asyncio.Task | None = None
        self.running = False

    @property
    def route(self) -> RouteKey:
        return self.engine.route

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self.run_loop(), name=f"executor:{self.route}")

    async def stop(self) -> None:
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def run_loop(self) -> None:
        """Main loop. MUST NOT DIE — exceptions are logged and swallowed."""
        while self.running:
            try:
                await self.work_queue.wait_for_work.remote(self.route, self.clock_cycle)
                ops: list[Op] = await self.work_queue.drain.remote(self.route)
                if not ops:
                    continue
                await self.dispatch(ops)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[executor:{self.route}] loop error (continuing): {e}",
                    exc_info=True,
                )
                await asyncio.sleep(1.0)

    async def dispatch(self, ops: list[Op]) -> None:
        """Walk ops in order. Coalesce fwd/fb; run zero/optim as barriers.

        Each op gets exactly one JobStore write. We track processed job_ids so
        that a failure mid-batch can fail only the remaining ops.

        Adapter boundaries also flush: fwd/fb ops from different adapters
        cannot be batched (the engine can only have one active adapter at
        a time), so a change in `op.route.adapter` forces a flush.
        """
        pending_fb: list[ForwardBackwardOp] = []
        pending_fwd: list[ForwardOp] = []
        processed: set[str] = set()

        async def flush() -> None:
            if pending_fwd:
                try:
                    await self.run_forward(pending_fwd)
                except Exception as e:
                    logger.error(f"forward batch failed: {e}", exc_info=True)
                    await self.fail_ops(pending_fwd, e)
                processed.update(op.job_id for op in pending_fwd)
                pending_fwd.clear()
            if pending_fb:
                try:
                    await self.run_forward_backward(pending_fb)
                except Exception as e:
                    logger.error(f"forward_backward batch failed: {e}", exc_info=True)
                    await self.fail_ops(pending_fb, e)
                processed.update(op.job_id for op in pending_fb)
                pending_fb.clear()

        logger.debug(f"[executor:{self.route}] dispatching {len(ops)} ops")
        try:
            for op in ops:
                if isinstance(op, ZeroGradOp):
                    await flush()
                    await self.run_zero_grad(op)
                    processed.add(op.job_id)
                elif isinstance(op, OptimStepOp):
                    await flush()
                    await self.run_optim_step(op)
                    processed.add(op.job_id)
                elif isinstance(op, ForwardBackwardOp):
                    if pending_fb and pending_fb[-1].route.adapter != op.route.adapter:
                        await flush()
                    pending_fb.append(op)
                elif isinstance(op, ForwardOp):
                    pending_fwd.append(op)
                else:
                    logger.warning(f"Unknown op type: {type(op).__name__}")
            await flush()
        except Exception as e:
            logger.error(
                f"[executor:{self.route}] dispatch top-level error: {e}", exc_info=True
            )
            unprocessed = [op for op in ops if op.job_id not in processed]
            await self.fail_ops(unprocessed, e)

    # ---- per-op runners ------------------------------------------------

    async def run_forward(self, ops: list[ForwardOp]) -> None:
        batch, sizes = [], []
        for op in ops:
            sizes.append(len(op.data))
            batch.extend(op.data)
        outputs = await self.engine.forward(
            data=batch, forward_kwargs=ops[0].forward_kwargs
        )
        idx = 0
        for op, size in zip(ops, sizes):
            per_op = outputs[idx : idx + size]
            idx += size
            await self.job_store.set_success.remote(op.job_id, {"logprobs": per_op})

    async def run_forward_backward(self, ops: list[ForwardBackwardOp]) -> None:
        batch, sizes, loss_fns = [], [], []
        for op in ops:
            sizes.append(len(op.data))
            batch.extend(op.data)
            loss_fns.extend([op.loss_fn] * len(op.data))
        # `dispatch` guarantees all coalesced ops share the same adapter.
        outputs = await self.engine.forward_backward(
            data=batch,
            loss_fns=loss_fns,
            adapter_name=ops[0].route.adapter,
            forward_kwargs=ops[0].forward_kwargs,
            loss_fn_config=ops[0].loss_fn_config,
        )
        idx = 0
        for op, size in zip(ops, sizes):
            per_op = outputs[idx : idx + size]
            idx += size
            await self.job_store.set_success.remote(
                op.job_id, self.combine_fb_outputs(per_op)
            )

    @staticmethod
    def combine_fb_outputs(per_op_outputs: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge per-example rank-0 dicts into the shape the client expects."""
        if not per_op_outputs:
            return {}
        first = per_op_outputs[0]
        if not isinstance(first, dict) or not first:
            return {}
        combined: dict[str, Any] = {}
        for key in first.keys():
            if key == "sum_gradient":
                # Gradient sums are the same across examples in a batch
                # (computed once across all params); take the first.
                combined[key] = first[key]
            else:
                combined[key] = [
                    o[key] for o in per_op_outputs if isinstance(o, dict) and key in o
                ]
        return combined

    async def run_zero_grad(self, op: ZeroGradOp) -> None:
        try:
            await self.engine.zero_grad()
            await self.job_store.set_success.remote(
                op.job_id,
                {"model_name": op.route.model, "message": "gradients zeroed"},
            )
        except Exception as e:
            logger.error(f"zero_grad failed for {op.route}: {e}", exc_info=True)
            await self.fail_op(op.job_id, e)

    async def run_optim_step(self, op: OptimStepOp) -> None:
        try:
            await self.engine.optim_step(
                adapter_name=op.route.adapter, optimizer_params=op.optimizer_params
            )
            await self.job_store.set_success.remote(
                op.job_id,
                {"model_name": op.route.model, "message": "optimizer stepped"},
            )
        except Exception as e:
            logger.error(f"optim_step failed for {op.route}: {e}", exc_info=True)
            await self.fail_op(op.job_id, e)

    # ---- failure helpers ---------------------------------------------

    async def fail_op(self, job_id: str, exc: BaseException) -> None:
        try:
            await self.job_store.set_error.remote(
                job_id, ErrorRecord.from_exception(exc)
            )
        except Exception as write_err:
            logger.error(f"Failed to write error for job {job_id}: {write_err}")

    async def fail_ops(self, ops: list[Op], exc: BaseException) -> None:
        for op in ops:
            await self.fail_op(op.job_id, exc)
