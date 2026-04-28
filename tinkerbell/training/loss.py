import torch
import torch.nn as nn

MASK_TOKEN_ID = -100


def cross_entropy_loss(
    logprobs: torch.Tensor,
    labels: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
    **kwargs,
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
    **kwargs,
) -> torch.Tensor:
    """Compute truncated importance sampling loss for off-policy GRPO.

    Implements: loss = -min(π_learner/π_sampler, C) · advantage · log π_learner

    The truncated ratio handles the log-probability mismatch between the
    sampling engine (SGLang/vLLM) and the training model (HuggingFace) by
    capping the importance weight, preventing exploding/vanishing gradients.

    Args:
        logprobs: [batch, seq, vocab] or [batch, seq] model log-probs.
        labels: [batch, seq] target token ids (-100 = masked).
        sampling_logprobs: [batch, seq] log-probs from the sampler.
        advantages: [batch, seq] per-token advantage values.
        ratio_clip: Upper bound C for the importance ratio.

    Returns:
        Per-example loss of shape [batch].
    """
    batch_size, seq_len = labels.shape

    logprobs = logprobs.float()
    sampling_logprobs = sampling_logprobs.float()
    advantages = advantages.float()

    if logprobs.ndim == 3:
        gather_idx = labels.clone()
        gather_idx[gather_idx == mask_token_id] = 0
        logprobs = torch.gather(logprobs, dim=-1, index=gather_idx.unsqueeze(-1))

    target_lp = logprobs.view(batch_size, seq_len)
    sampling_lp = sampling_logprobs.view(batch_size, seq_len)
    advantages = advantages.view(batch_size, seq_len)
    mask = (labels != mask_token_id).float()

    log_ratio = torch.clamp(target_lp - sampling_lp, min=-20.0, max=20.0)
    truncated_ratio = torch.clamp(torch.exp(log_ratio), min=1e-4, max=ratio_clip)

    loss = -(truncated_ratio.detach() * advantages * target_lp * mask).sum(dim=-1)
    return loss


LOSSES = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
}
