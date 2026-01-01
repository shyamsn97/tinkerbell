import torch
import torch.nn as nn

MASK_TOKEN_ID = -100


def cross_entropy_loss(
    logprobs: torch.Tensor,
    labels: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
) -> torch.Tensor:
    """Compute cross-entropy loss for causal language modeling.

    Args:
        logprobs: Model log probabilities of shape [batch_size, seq_len, vocab_size]
        labels: Target labels of shape [batch_size, seq_len]

    Returns:
        Per-example loss of shape [batch_size]
    """
    batch_size = logprobs.shape[0]
    vocab_size = logprobs.shape[-1]

    # Flatten the tokens
    logprobs_flat = logprobs.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    labels_flat = labels_flat.to(logprobs.device)

    # Use logprobs.float() only when needed to avoid unnecessary memory allocation
    if logprobs.dtype != torch.float32:
        logprobs_flat = logprobs_flat.float()

    # Use negative log likelihood loss (since we already have logprobs)
    loss = nn.functional.nll_loss(
        logprobs_flat, labels_flat, ignore_index=mask_token_id, reduction="none"
    )

    # Reshape and average over sequence length
    loss = loss.view(batch_size, -1).mean(dim=-1)
    return loss


def importance_sampling_loss(
    logprobs: torch.Tensor,
    labels: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
) -> torch.Tensor:
    """Compute importance sampling loss for causal language modeling.

    Args:
        logprobs: Model log probabilities of size (batch_size, seq_len, vocab_size) or (batch_size, seq_len)]
        labels: Target labels of shape [batch_size, seq_len]
        sampling_logprobs: Log probabilities from the sampling/behavior policy for each token label of shape [batch_size, seq_len]
        advantages: Advantage values of shape [batch_size, seq_len]

    Returns:
        Per-example loss of shape [batch_size]
    """
    batch_size = labels.shape[0]
    seq_len = labels.shape[1]

    # Ensure logprobs are float32
    logprobs = logprobs.float()
    sampling_logprobs = sampling_logprobs.float()
    advantages = advantages.float()

    if len(logprobs.shape) == 3:
        logprobs = torch.gather(
            logprobs, dim=-1, index=labels.view(batch_size, seq_len, 1)
        )
    else:
        logprobs = logprobs.view(batch_size, seq_len)

    target_logprobs = logprobs.view(batch_size, seq_len)
    sampling_logprobs = sampling_logprobs.view(batch_size, seq_len)
    advantages = advantages.view(batch_size, seq_len)
    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)

    # Mask out ignored tokens (where labels == mask_token_id)
    mask = (labels != mask_token_id).float()
    prob_ratio = prob_ratio * mask
    advantages_masked = advantages * mask

    loss = -(prob_ratio * advantages_masked).sum(dim=-1)
    return loss


LOSSES = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
}
