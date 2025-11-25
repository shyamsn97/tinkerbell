from typing import Callable

import torch
import torch.nn as nn

from tinkerbell.types.data import PaddingStrategy
from tinkerbell.types.datum import Datum
from tinkerbell.utils import get_nested, set_nested


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=-100, reduction="none"
    )
    return loss


def ForCausalLMLoss(
    logits,
    labels,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]
    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    labels = labels.to(logits.device)
    loss = fixed_cross_entropy(logits, labels)
    loss = loss.view(batch_size, -1).mean(dim=-1)
    return loss


class LossFn:
    def __init__(
        self,
        loss_fn: Callable,
        padding_strategies: dict[str, PaddingStrategy],
    ):
        self.loss_fn = loss_fn
        self.padding_strategies = padding_strategies

    def pad(
        self,
        data: list[Datum],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Pad a batch of Datum objects and return dictionary of padded tensors.

        Uses padding strategies defined by keys like "model_input.tokens" or "loss_fn_inputs.labels".
        """
        torch_data = [d.to_torch(device=device) for d in data]
        result = {}

        # Iterate through all padding strategy keys and apply them
        for path, padding_strategy in self.padding_strategies.items():
            # Extract values from each datum using the nested path
            values = [get_nested(d, path) for d in torch_data]

            # Pad the values
            padded = padding_strategy.pad_sequence(values)

            # Set the padded result back using the nested path
            set_nested(result, path, padded)

        return result

    def __call__(
        self,
        padded_logits: torch.Tensor,
        padded_loss_fn_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss given logits and loss function inputs."""
        return self.loss_fn(logits=padded_logits, **padded_loss_fn_inputs)


CROSS_ENTROPY_LOSS_FN = LossFn(
    loss_fn=fixed_cross_entropy,
    padding_strategies={
        "model_input.tokens": PaddingStrategy(padding_side="left", padding_value=0),
        "model_input.attention_mask": PaddingStrategy(
            padding_side="left", padding_value=0
        ),
        "loss_fn_inputs.labels": PaddingStrategy(padding_side="left", padding_value=0),
    },
)
