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
    ratio_clip: float = 10.0,
    debug: bool = True,
) -> torch.Tensor:
    """Compute truncated importance sampling loss per the off-policy RL fix.

    Implements: min(π_learner/π_sampler, C) · R(a) · ∇_θ log π_learner

    This handles the mismatch between vLLM/SGLang (sampler) and HF (learner)
    by using truncated importance sampling.

    Args:
        logprobs: Model log probabilities of size (batch_size, seq_len, vocab_size) or (batch_size, seq_len)]
        labels: Target labels of shape [batch_size, seq_len]
        sampling_logprobs: Log probabilities from the sampler for each token
        advantages: Advantage values of shape [batch_size, seq_len]
        ratio_clip: Maximum value for importance ratio (C in the paper)

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
        # Replace -100 (mask token) with 0 for gathering - these positions will be masked later
        gather_indices = labels.clone()
        gather_indices[gather_indices == mask_token_id] = 0
        logprobs = torch.gather(
            logprobs, dim=-1, index=gather_indices.view(batch_size, seq_len, 1)
        )

    target_logprobs = logprobs.view(batch_size, seq_len)
    sampling_logprobs = sampling_logprobs.view(batch_size, seq_len)
    advantages = advantages.view(batch_size, seq_len)

    # Mask for valid tokens
    mask = (labels != mask_token_id).float()

    # Compute importance ratio: π_learner / π_sampler
    # Use clamp on log-space difference to prevent numerical issues
    log_ratio = target_logprobs - sampling_logprobs
    log_ratio_clamped = torch.clamp(
        log_ratio, min=-20.0, max=20.0
    )  # Prevent exp overflow/underflow
    prob_ratio = torch.exp(log_ratio_clamped)

    # Truncated importance sampling: min(ratio, C)
    # Also add a lower bound to prevent completely dead gradients
    truncated_ratio = torch.clamp(prob_ratio, min=1e-4, max=ratio_clip)

    if debug:
        print(
            f"  [LOSS DEBUG] target_logprobs mean: {(target_logprobs * mask).sum() / mask.sum():.4f}"
        )
        print(
            f"  [LOSS DEBUG] sampling_logprobs mean: {(sampling_logprobs * mask).sum() / mask.sum():.4f}"
        )
        print(
            f"  [LOSS DEBUG] log_ratio mean: {(log_ratio * mask).sum() / mask.sum():.4f}"
        )
        print(
            f"  [LOSS DEBUG] truncated_ratio mean: {(truncated_ratio * mask).sum() / mask.sum():.4f}"
        )
        print(
            f"  [LOSS DEBUG] advantages mean: {(advantages * mask).sum() / mask.sum():.4f}"
        )

    # Loss = -truncated_ratio * advantage * log_π_learner
    # Gradient: truncated_ratio * advantage * ∇log_π_learner (REINFORCE with IS correction)
    # Note: truncated_ratio is detached to only use it as a weight, not for its own gradient
    loss_per_token = -truncated_ratio.detach() * advantages * target_logprobs * mask
    loss = loss_per_token.sum(dim=-1)

    if debug:
        print(f"  [LOSS DEBUG] per-sample loss: {loss.tolist()[:5]}")

    return loss


LOSSES = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
}
