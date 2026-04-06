# adapted from https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers.py

from enum import StrEnum

from transformers import AutoTokenizer

from tinker.types import Datum, ModelInput, TensorData

MASK_TOKEN_ID = -100


class RenderMode(StrEnum):
    TRAINING = "training"
    INFERENCE = "inference"


class TrainOnWhat(StrEnum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    ALL_USER_AND_SYSTEM_MESSAGES = "all_user_and_system_messages"


class Renderer:
    """Render chat messages into Datum objects for training or inference."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def _get_message_token_ranges(
        self, messages: list[dict[str, str]], **kwargs
    ) -> list[tuple[int, int]]:
        """Get token ranges (start, end) for each message."""
        ranges = []
        for i in range(len(messages)):
            text_so_far = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False, **kwargs
            )
            tokens_so_far = self.tokenizer.encode(text_so_far, add_special_tokens=True)
            if i > 0:
                text_before = self.tokenizer.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=True, **kwargs
                )
                tokens_before = self.tokenizer.encode(
                    text_before, add_special_tokens=True
                )
                ranges.append((len(tokens_before), len(tokens_so_far)))
            else:
                ranges.append((0, len(tokens_so_far)))
        return ranges

    def _create_labels(
        self,
        messages: list[dict[str, str]],
        full_ids: list[int],
        train_roles: set[str] | None,
        mask_value: int,
        **kwargs,
    ) -> list[int]:
        """Create labels by masking tokens not in train_roles."""
        if train_roles is None:
            return full_ids.copy()
        labels = [mask_value] * len(full_ids)
        for msg, (start, end) in zip(
            messages, self._get_message_token_ranges(messages, **kwargs)
        ):
            if msg["role"] in train_roles:
                for j in range(start, min(end, len(labels))):
                    labels[j] = full_ids[j]
        return labels

    def _build_single(
        self,
        messages: list[dict[str, str]],
        mode: RenderMode = RenderMode.TRAINING,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Datum:
        """Build a single Datum from chat messages."""
        # Inference mode: prepare for generation
        if mode == RenderMode.INFERENCE:
            if continue_final_message:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    **kwargs,
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs,
                )
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            return Datum(
                model_input=ModelInput.from_ints(input_ids),
                loss_fn_inputs={},
            )

        # Training mode: include labels with masking
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        attention_mask = [1] * len(full_ids)

        train_roles = None
        if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            labels = [mask_value] * len(full_ids)
            if messages and messages[-1]["role"] == "assistant":
                start, end = self._get_message_token_ranges(messages, **kwargs)[-1]
                for j in range(start, min(end, len(labels))):
                    labels[j] = full_ids[j]
            else:
                labels = full_ids.copy()
        elif train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            train_roles = {"assistant"}
        elif train_on_what == TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
            train_roles = {"user", "system"}
        elif train_on_what not in (TrainOnWhat.ALL_TOKENS, TrainOnWhat.ALL_MESSAGES):
            raise ValueError(f"Unknown train_on_what mode: {train_on_what}")

        if train_on_what != TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            labels = self._create_labels(
                messages, full_ids, train_roles, mask_value, **kwargs
            )

        labels = labels[1:] + [mask_value]

        return Datum(
            model_input=ModelInput.from_ints(full_ids),
            loss_fn_inputs={
                "labels": TensorData(data=labels, dtype="int64", shape=[len(labels)])
            },
        )

    def render(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        mode: RenderMode = RenderMode.TRAINING,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> list[Datum]:
        """Render chat messages into Datum objects.

        Args:
            messages: Single conversation or batch of conversations.
            mode: TRAINING (with labels) or INFERENCE (for sampling).
            train_on_what: Which messages to train on (only used in TRAINING mode).
            mask_value: Value for masked tokens in labels (only used in TRAINING mode).
            continue_final_message: If True in INFERENCE mode, continue from a partial
                                    assistant message instead of starting a new response.
            add_generation_prompt: If True, add the generation prompt to the messages.
            **kwargs: Additional args passed to apply_chat_template.

        Returns:
            List of Datum objects ready for training or inference.
        """
        if isinstance(messages[0], dict):
            messages = [messages]
        return [
            self._build_single(
                m,
                mode,
                train_on_what,
                mask_value,
                continue_final_message=continue_final_message,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            for m in messages
        ]

    # Backwards compatibility aliases
    def build_chat_samples(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> list[Datum]:
        """Backwards-compatible alias for render(mode=TRAINING)."""
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render(
            messages,
            mode,
            train_on_what,
            mask_value,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def build_message_samples(
        self,
        messages: list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Datum:
        """Backwards-compatible alias for render(mode=TRAINING) with single message."""
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render(
            [messages],
            mode,
            train_on_what,
            mask_value,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )[0]
