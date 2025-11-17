import asyncio
import logging
from typing import Any, Dict, Optional

import ray

logger = logging.getLogger(__name__)


@ray.remote
class SGLangInferenceActor:
    def __init__(self, model_id: str, tp_size: int, engine_kwargs: dict = {}):
        # REALLY ANNOYING BUT NECESSARY FIX for https://github.com/sgl-project/sglang/issues/2536
        import signal
        import threading

        # Monkey patch signal.signal to ignore if not in main thread
        _original_signal = signal.signal

        def patched_signal(signalnum, handler):
            if threading.current_thread() is threading.main_thread():
                return _original_signal(signalnum, handler)
            return None

        signal.signal = patched_signal

        import sglang as sgl

        # Create event loop for this actor
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        print(f"Setting up SGLang engine for model {model_id} with {tp_size} GPUs")

        # Initialize engine directly in __init__
        self.engine = sgl.Engine(
            model_path=model_id,
            tp_size=tp_size,
            **engine_kwargs,
        )
        print(f"✓ SGLang engine initialized with {tp_size} GPUs")

    def is_ready(self) -> bool:
        """Simple method to check if actor is initialized"""
        return True

    async def update_weights_from_disk(
        self, checkpoint_path: str, load_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint from disk (async version)."""
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        obj = UpdateWeightFromDiskReqInput(
            model_path=checkpoint_path,
            load_format=load_format,
        )

        return await self.engine.tokenizer_manager.update_weights_from_disk(obj, None)

    async def generate(self, prompts: list[str], sampling_params: dict = {}):
        """Generate text from a prompt."""

        # Use async_generate with the event loop
        outputs = await self.engine.async_generate(prompts, sampling_params)

        return [output["text"] for output in outputs]

    async def shutdown(self):
        del self.engine
        return True
