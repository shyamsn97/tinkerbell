# adapted from https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers.py

from enum import StrEnum

from transformers import AutoTokenizer

from tinkerbell.types.data import TensorData
from tinkerbell.types.datum import Datum
from tinkerbell.types.model_input import ModelInput

MASK_TOKEN_ID = -100


class TrainOnWhat(StrEnum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    ALL_USER_AND_SYSTEM_MESSAGES = "all_user_and_system_messages"


class Renderer:
    """Render chat messages into training examples with proper masking."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def _get_message_token_ranges(
        self, messages: list[dict[str, str]]
    ) -> list[tuple[int, int]]:
        """Get token ranges (start, end) for each message."""
        ranges = []
        for i in range(len(messages)):
            text_so_far = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            tokens_so_far = self.tokenizer.encode(text_so_far, add_special_tokens=True)
            if i > 0:
                text_before = self.tokenizer.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=True
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
    ) -> list[int]:
        """Create labels by masking tokens not in train_roles."""
        if train_roles is None:
            return full_ids.copy()
        labels = [mask_value] * len(full_ids)
        for msg, (start, end) in zip(
            messages, self._get_message_token_ranges(messages)
        ):
            if msg["role"] in train_roles:
                for j in range(start, min(end, len(labels))):
                    labels[j] = full_ids[j]
        return labels

    def build_message_samples(
        self,
        messages: list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
    ) -> Datum:
        """Build a supervised training example from chat messages."""
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        train_roles = None
        if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            labels = [mask_value] * len(full_ids)
            if messages and messages[-1]["role"] == "assistant":
                start, end = self._get_message_token_ranges(messages)[-1]
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
            labels = self._create_labels(messages, full_ids, train_roles, mask_value)

        labels = labels[1:] + [mask_value]
        attention_mask = [1] * len(full_ids)

        return Datum(
            model_input=ModelInput(
                input_ids=TensorData(
                    data=full_ids, dtype="int64", shape=[len(full_ids)]
                ),
                attention_mask=TensorData(
                    data=attention_mask, dtype="int64", shape=[len(attention_mask)]
                ),
            ),
            loss_fn_inputs={
                "labels": TensorData(data=labels, dtype="int64", shape=[len(labels)])
            },
        )

    def build_chat_samples(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
    ) -> list[Datum]:
        """Build training examples from messages (single or batch)."""
        if isinstance(messages[0], dict):
            messages = [messages]
        return [
            self.build_message_samples(m, train_on_what, mask_value) for m in messages
        ]
