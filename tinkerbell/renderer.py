from enum import StrEnum

from transformers import AutoTokenizer

from tinkerbell.types.data import TensorData
from tinkerbell.types.datum import Datum
from tinkerbell.types.model_input import ModelInput


class TrainOnWhat(StrEnum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    ALL_USER_AND_SYSTEM_MESSAGES = "all_user_and_system_messages"


class Renderer:
    """Helper class to render chat messages into training examples with proper masking."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def _get_message_token_ranges(
        self, messages: list[dict[str, str]]
    ) -> list[tuple[int, int]]:
        """Get token ranges (start, end) for each message.

        Returns list of (start_idx, end_idx) tuples where message i spans tokens [start:end].
        """
        ranges = []
        for i in range(len(messages)):
            # Tokenize up to current message
            text_so_far = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            tokens_so_far = self.tokenizer.encode(text_so_far, add_special_tokens=True)

            # Tokenize up to previous message
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
        """Create labels by masking tokens not in train_roles.

        Args:
            messages: List of messages
            full_ids: Full token IDs
            train_roles: Set of roles to train on, or None to train on all
            mask_value: Value to use for masked tokens
        """
        if train_roles is None:
            return full_ids.copy()

        labels = [mask_value] * len(full_ids)
        ranges = self._get_message_token_ranges(messages)

        for msg, (start, end) in zip(messages, ranges):
            if msg["role"] in train_roles:
                for j in range(start, min(end, len(labels))):
                    labels[j] = full_ids[j]

        return labels

    def build_message_examples(
        self,
        messages: list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = -100,
    ) -> Datum:
        """Build a single supervised training example from chat messages.

        Args:
            messages: List of chat messages with 'role' and 'content' keys
            train_on_what: Which parts of the conversation to train on
            mask_value: Value to use for masked tokens (default: -100)

        Returns:
            Datum object with properly masked and shifted labels for next-token prediction

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "What is 2+2?"},
            ...     {"role": "assistant", "content": "The answer is 4."},
            ... ]
            >>> datum = renderer.build_message_examples(
            ...     messages,
            ...     train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
            ... )
        """
        # Tokenize full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Determine which roles to train on
        if train_on_what == TrainOnWhat.ALL_TOKENS:
            train_roles = None  # Train on everything
        elif train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            # Special case: only train on last assistant message
            labels = [mask_value] * len(full_ids)
            if messages and messages[-1]["role"] == "assistant":
                ranges = self._get_message_token_ranges(messages)
                start, end = ranges[-1]
                for j in range(start, min(end, len(labels))):
                    labels[j] = full_ids[j]
            else:
                labels = full_ids.copy()
        elif train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            train_roles = {"assistant"}
        elif train_on_what == TrainOnWhat.ALL_MESSAGES:
            train_roles = None  # Train on everything
        elif train_on_what == TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
            train_roles = {"user", "system"}
        else:
            raise ValueError(f"Unknown train_on_what mode: {train_on_what}")

        # Create labels (except for LAST_ASSISTANT_MESSAGE which is handled above)
        if train_on_what != TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            labels = self._create_labels(messages, full_ids, train_roles, mask_value)

        # Shift labels for next-token prediction: labels[i] = full_ids[i+1]
        labels = labels[1:] + [mask_value]

        # Create attention mask (all ones since all tokens are valid)
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

    def build_chat_examples(
        self,
        conversations: list[list[dict[str, str]]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = -100,
    ) -> list[Datum]:
        """Build multiple training examples from a list of conversations.

        Args:
            conversations: List of conversations, where each conversation is a list of messages
            train_on_what: Which parts of each conversation to train on
            mask_value: Value to use for masked tokens (default: -100)

        Returns:
            List of Datum objects, one per conversation.

        Example:
            >>> conversations = [
            ...     [
            ...         {"role": "user", "content": "What is 2+2?"},
            ...         {"role": "assistant", "content": "4"},
            ...     ],
            ...     [
            ...         {"role": "user", "content": "What is 3+3?"},
            ...         {"role": "assistant", "content": "6"},
            ...     ],
            ... ]
            >>> data = renderer.build_chat_examples(
            ...     conversations,
            ...     train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
            ... )
        """
        return [
            self.build_message_examples(messages, train_on_what, mask_value)
            for messages in conversations
        ]
