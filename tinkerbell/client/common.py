"""Shared base for Tinkerbell clients.

Wraps an `AsyncTransport` and adds tokenizer/renderer helpers so that
training and sampling clients share a single HTTP pool.

The old v1 `BaseClient` + `TinkerbellFuture` + `AsyncTinkerbellFuture` are
gone — polling is unified under `JobHandle` in `client.transport`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional

from tinker.types import Datum

from tinkerbell.client.transport import AsyncTransport, JobHandle
from tinkerbell.renderer import MASK_TOKEN_ID, Renderer, RenderMode, TrainOnWhat

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


__all__ = ["BaseClient", "JobHandle"]


class BaseClient:
    def __init__(
        self,
        server_url: str | None = None,
        timeout: float = 600.0,
        base_model: str | None = None,
    ):
        if server_url is None:
            raise ValueError("server_url is required")
        self.server_url = server_url
        self.timeout = timeout
        self.base_model = base_model
        self.transport = AsyncTransport(server_url=server_url, timeout=timeout)
        self._tokenizer: Optional["PreTrainedTokenizerBase"] = None
        self._renderer: Optional[Renderer] = None

    # ---- shared submit helper -------------------------------------------
    # Every queued op POSTs a payload, gets back `{"job_id": ...}`, and
    # wraps that in a JobHandle. Sync caller: `.result()`. Async caller:
    # `await handle`. Same object both ways.

    def _submit(
        self,
        endpoint: str,
        payload: dict[str, Any],
        parse: Callable[[dict[str, Any]], Any] = (lambda r: r),
    ) -> JobHandle[Any]:
        data = self.transport.submit(endpoint, payload)
        job_id = data.get("job_id")
        if not job_id:
            raise RuntimeError(f"{endpoint} response missing job_id: {data}")
        return JobHandle(
            transport=self.transport, job_id=job_id, parse=parse, timeout=self.timeout
        )

    # ---- tokenizer / renderer helpers ------------------------------------

    def get_tokenizer(self):
        if self.base_model is None:
            raise ValueError("base_model is required to get tokenizer")
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or 0
        return self._tokenizer

    def get_renderer(self) -> Renderer:
        if self.base_model is None:
            raise ValueError("base_model is required to get renderer")
        if self._renderer is None:
            self._renderer = Renderer(self.get_tokenizer())
        return self._renderer

    def render(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        mode: RenderMode = RenderMode.TRAINING,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        continue_final_message: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> list[Datum]:
        return self.get_renderer().render(
            messages=messages,
            mode=mode,
            train_on_what=train_on_what,
            mask_value=mask_value,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def build_chat_samples(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        **kwargs,
    ) -> list[Datum]:
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render(messages, mode, train_on_what, mask_value, **kwargs)

    def build_message_samples(
        self,
        messages: list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        **kwargs,
    ) -> Datum:
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render([messages], mode, train_on_what, mask_value, **kwargs)[0]

    # ---- lifecycle ----

    def close(self) -> None:
        self.transport.close()
        if self.transport._async is not None:
            warnings.warn(
                "Async HTTP client not closed. Use `async with` or `await client.aclose()`.",
                ResourceWarning,
            )

    async def aclose(self) -> None:
        await self.transport.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
