import torch
import torch.nn as nn


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss for causal language modeling.

    Args:
        logits: Model logits of shape [batch_size, seq_len, vocab_size]
        labels: Target labels of shape [batch_size, seq_len]

    Returns:
        Per-example loss of shape [batch_size]
    """
    # Upcast to float to avoid precision issues
    logits = logits.float()
    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    labels = labels.to(logits.device)

    # Compute cross-entropy with ignore_index for masked tokens
    loss = nn.functional.cross_entropy(
        logits, labels, ignore_index=-100, reduction="none"
    )

    # Reshape and average over sequence length
    loss = loss.view(batch_size, -1).mean(dim=-1)
    return loss


LOSSES = {
    "cross_entropy": cross_entropy_loss,
}
