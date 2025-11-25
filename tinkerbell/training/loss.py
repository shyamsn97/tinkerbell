import torch
import torch.nn as nn
from typing import Callable
from tinkerbell.types.data import PaddingStrategy, TensorData
from tinkerbell.types.datum import Datum


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

        Returns:
            Dictionary with keys:
                - 'tokens': padded tokens
                - 'attention_mask': padded attention masks (if present)
                - loss_fn_inputs keys: padded loss function inputs
        """
        default_padding = PaddingStrategy(padding_side="left", padding_value=0)
        result = {"additional_inputs": {}, "loss_fn_inputs": {}}

        torch_data = [d.to_torch(device=device) for d in data]

        # Pad tokens (always present)
        tokens = [d['model_input']['tokens'] for d in torch_data]
        result['tokens'] = self.padding_strategies.get('tokens', default_padding).pad_sequence(tokens)

        # Pad attention_mask if present
        attention_masks = [d['model_input']['attention_mask'] for d in torch_data]
        result['attention_mask'] = self.padding_strategies.get('attention_mask', default_padding).pad_sequence(attention_masks)

        # Pad additional inputs if present
        additional_inputs = [d['model_input']['additional_inputs'] for d in torch_data]
        for key in additional_inputs[0].keys():
            values = [d['model_input']['additional_inputs'][key] for d in torch_data]
            result['additional_inputs'][key] = self.padding_strategies.get('additional_inputs', default_padding).pad_sequence(values)

        # Pad loss_fn_inputs (collect all keys across all data points)
        loss_fn_inputs = [d['loss_fn_inputs'] for d in torch_data]
        for key in loss_fn_inputs[0].keys():
            values = [d['loss_fn_inputs'][key] for d in torch_data]
            result['loss_fn_inputs'][key] = self.padding_strategies.get('loss_fn_inputs', default_padding).pad_sequence(values)

        return result

    def __call__(
        self,
        padded_logits: torch.Tensor,
        padded_loss_fn_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss given logits and loss function inputs."""
        return self.loss_fn(
            logits=padded_logits,
            **padded_loss_fn_inputs
        )

CROSS_ENTROPY_LOSS_FN = LossFn(
    loss_fn=fixed_cross_entropy,
    padding_strategies={
        'tokens': PaddingStrategy(padding_side="left", padding_value=0),
        'attention_mask': PaddingStrategy(padding_side="left", padding_value=0),
        'additional_inputs': PaddingStrategy(padding_side="left", padding_value=0),
        'loss_fn_inputs': PaddingStrategy(padding_side="left", padding_value=0),
    }
)
