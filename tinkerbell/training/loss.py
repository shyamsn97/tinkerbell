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
    **kwargs,
) -> torch.Tensor:
    """Importance-sampling policy-gradient loss, identical to Tinker's
    ``importance_sampling`` loss as documented at
    https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/:

        prob_ratio = exp(target_logprobs - sampling_logprobs)
        loss = -(prob_ratio * advantages).sum()

    Args:
        logprobs: [batch, seq, vocab] or [batch, seq] model log-probs.
        labels: [batch, seq] target token ids (-100 = masked / non-action).
        sampling_logprobs: [batch, seq] log-probs from the sampler.
        advantages: [batch, seq] per-token advantage values (zero on prompt).
    """
    batch_size, seq_len = labels.shape

    sampling_logprobs = sampling_logprobs.float()
    advantages = advantages.float()

    if logprobs.ndim == 3:
        # Gather before casting. A full [batch, seq, vocab] fp32 copy can be
        # tens of GiB for GRPO batches; we only need target-token log-probs.
        gather_idx = labels.clone()
        gather_idx[gather_idx == mask_token_id] = 0
        logprobs = torch.gather(logprobs, dim=-1, index=gather_idx.unsqueeze(-1))

    target_lp = logprobs.view(batch_size, seq_len).float()
    if sampling_logprobs.numel() != batch_size * seq_len:
        raise ValueError(
            "sampling_logprobs shape does not match labels: "
            f"sampling_logprobs={tuple(sampling_logprobs.shape)}, "
            f"labels={tuple(labels.shape)}"
        )
    if advantages.numel() != batch_size * seq_len:
        raise ValueError(
            "advantages shape does not match labels: "
            f"advantages={tuple(advantages.shape)}, labels={tuple(labels.shape)}"
        )
    sampling_lp = sampling_logprobs.view(batch_size, seq_len)
    advantages = advantages.view(batch_size, seq_len)

    prob_ratio = torch.exp(target_lp - sampling_lp)
    return -(prob_ratio * advantages).sum(dim=-1)


LOSSES = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
}
