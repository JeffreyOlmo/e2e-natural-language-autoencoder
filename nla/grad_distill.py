"""Per-position gradient-distillation loss for AV training.

Construction:
  - For each rollout token, compute g_t = ∂MSE/∂e_t at AR's input embeddings.
  - Per-position L2-normalize: ĝ_t = g_t / ||g_t||  (keeps τ in stable units).
  - Per-vocab scores: s_t(v) = -<ĝ_t, e_v> / τ
  - Teacher: q_t = softmax(log π_AV_detached + s_t)  (π_ref = current AV)
  - Loss: Σ_t mask_t · KL(q_t || π_AV)  / sum(mask_t)

τ = β identity: with π_ref = current AV, this is the closed-form
KL-regularized one-step improvement under reward r(v) = -<ĝ_t, e_v>.
The temperature τ IS the regularization strength; do not multiply by an
additional β factor on top.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GradDistillOutputs:
    loss: torch.Tensor
    teacher_kl: torch.Tensor    # mean per-position KL(q || π_AV) — same as loss
    teacher_ent: torch.Tensor   # mean per-position H(q)
    av_ent: torch.Tensor        # mean per-position H(π_AV)
    top1_agree: torch.Tensor    # frac of positions where argmax(q) == argmax(π_AV)
    grad_norm_mean: torch.Tensor  # mean ||g_t|| (raw)


def build_ar_inputs_from_rollout(
    tokenizer,
    rollout_ids: torch.Tensor,
    rollout_mask: torch.Tensor,
    prefix_str: str = "<text>",
    suffix_str: str = "</text> <summary>",
) -> dict:
    """Build per-row AR input: `prefix + valid_rollout_tokens + suffix` (variable length).
    Pad to batch max. Returns offsets so caller can extract g for rollout tokens only."""
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False).input_ids
    pad_id = tokenizer.pad_token_id

    rows, masks, offsets, lengths = [], [], [], []
    B = rollout_ids.shape[0]
    for i in range(B):
        n = int(rollout_mask[i].sum().item())
        if n == 0:
            # No valid rollout tokens — still build a row so batch shapes line up
            seq = prefix_ids + suffix_ids
            offsets.append(len(prefix_ids))
            lengths.append(0)
        else:
            seq = prefix_ids + rollout_ids[i, :n].tolist() + suffix_ids
            offsets.append(len(prefix_ids))
            lengths.append(n)
        rows.append(seq)
        masks.append([1] * len(seq))

    max_len = max(len(r) for r in rows)
    ar_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    ar_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        ar_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        ar_mask[i, :len(m)] = torch.tensor(m, dtype=torch.long)
    return {
        "ar_ids": ar_ids,
        "ar_mask": ar_mask,
        "rollout_offsets": torch.tensor(offsets, dtype=torch.long),
        "rollout_lengths": torch.tensor(lengths, dtype=torch.long),
    }


def compute_g_at_rollout(
    ar_model,
    ar_ids: torch.Tensor,
    ar_mask: torch.Tensor,
    rollout_offsets: torch.Tensor,
    rollout_lengths: torch.Tensor,
    h_gold: torch.Tensor,
    mse_norm: float = 1.0,
) -> torch.Tensor:
    """Forward AR → MSE → grad at AR input embeddings → extract rollout positions.

    Returns g_at_resp [B, T_resp, d] (zero-padded), where T_resp = max(rollout_lengths).
    Detached from autograd graph.
    """
    device = next(ar_model.parameters()).device
    ar_ids, ar_mask = ar_ids.to(device), ar_mask.to(device)
    h_gold = h_gold.to(device)

    embed_layer = ar_model.get_input_embeddings()
    ar_embeds = embed_layer(ar_ids).detach().clone().requires_grad_(True)
    pred = ar_model(inputs_embeds=ar_embeds, attention_mask=ar_mask)  # [B, d]
    p_norm = F.normalize(pred, dim=-1) * mse_norm
    g_norm_target = F.normalize(h_gold, dim=-1) * mse_norm
    mse = ((p_norm - g_norm_target) ** 2).mean()
    g_at_input = torch.autograd.grad(mse, ar_embeds)[0].detach()  # [B, T_ar, d]

    B, _, d = g_at_input.shape
    T_resp = int(rollout_lengths.max().item()) if rollout_lengths.numel() > 0 else 0
    if T_resp == 0:
        return torch.zeros(B, 0, d, device=device)
    g_at_resp = torch.zeros(B, T_resp, d, device=device)
    for i in range(B):
        off = int(rollout_offsets[i].item())
        n = int(rollout_lengths[i].item())
        if n > 0:
            g_at_resp[i, :n] = g_at_input[i, off : off + n]
    return g_at_resp


def grad_distill_loss(
    av_logits: torch.Tensor,        # [B, T_resp, V]  — π_AV at each rollout position (with grad)
    rollout_mask: torch.Tensor,     # [B, T_resp]
    g_at_resp: torch.Tensor,        # [B, T_resp, d]
    e_v: torch.Tensor,              # [V, d]  — AV's input embedding matrix (no grad on it)
    tau: float,
    pi_ref_mode: str = "current_av",
) -> GradDistillOutputs:
    """Build per-position softmax teacher and return KL(q || π_AV) summed/normalized."""
    B, T, V = av_logits.shape
    device = av_logits.device
    mask_f = rollout_mask.to(device).float()
    n_valid = mask_f.sum().clamp_min(1.0)

    # Per-position normalize g (units-stable τ).
    g_raw_norm = g_at_resp.norm(dim=-1, keepdim=True)
    g_hat = g_at_resp / g_raw_norm.clamp_min(1e-12)  # [B, T, d]

    # Scores: s_t(v) = -<ĝ_t, e_v> / τ   →   [B, T, V]
    scores = -torch.einsum("btd,vd->btv", g_hat, e_v) / tau

    # Teacher q_t
    if pi_ref_mode == "uniform":
        log_q = F.log_softmax(scores, dim=-1)
    elif pi_ref_mode == "current_av":
        log_pi_av_detached = F.log_softmax(av_logits.detach(), dim=-1)
        log_q = F.log_softmax(log_pi_av_detached + scores, dim=-1)
    else:
        raise ValueError(f"unknown pi_ref_mode: {pi_ref_mode}")
    q = log_q.exp()

    # AV log-policy (with grad on av_logits).
    log_pi = F.log_softmax(av_logits, dim=-1)

    # Per-position KL(q || π_AV) = Σ_v q(v) (log q(v) − log π(v))
    kl_per_pos = (q * (log_q - log_pi)).sum(dim=-1)  # [B, T]
    loss = (kl_per_pos * mask_f).sum() / n_valid

    with torch.no_grad():
        teacher_ent = (-(q * log_q).sum(dim=-1) * mask_f).sum() / n_valid
        pi = log_pi.exp()
        av_ent = (-(pi * log_pi).sum(dim=-1) * mask_f).sum() / n_valid
        agree = ((q.argmax(-1) == pi.argmax(-1)).float() * mask_f).sum() / n_valid
        grad_norm_mean = (g_raw_norm.squeeze(-1) * mask_f).sum() / n_valid

    return GradDistillOutputs(
        loss=loss,
        teacher_kl=loss.detach(),
        teacher_ent=teacher_ent,
        av_ent=av_ent,
        top1_agree=agree,
        grad_norm_mean=grad_norm_mean,
    )
