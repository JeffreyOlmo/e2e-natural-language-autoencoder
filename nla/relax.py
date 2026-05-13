"""RELAX-style estimator for AV training.

The RELAX gradient estimator (Tucker et al. 2017, Grathwohl et al. 2018) gives
an unbiased estimator for ∇φ E_{z~π_φ}[L(z)] via:

    ĝ = [L(z) - L(z̃|z)] · ∇φ log π_φ(z)
        + ∇φ L(z̃)
        - ∇φ L(z̃|z)

where:
    z   = hard sample, z = argmax_v(g_v + log π_v)  with g_v ~ Gumbel(0,1)
    z̃   = Gumbel-Softmax relaxation, softmax((logits + g)/τ_g) — same g as z
    z̃|z = relaxation with fresh Gumbels constrained so argmax = z
    L(·) = scalar loss on the (hard or soft) tokens

In NLA-land L is the AR reconstruction loss. L(z̃) feeds soft-token embeddings
(z̃ @ E) to AR; differentiable in φ. L(z̃|z) is the same construction but with
conditional Gumbels — its expectation equals E[L(z̃)] but, conditioned on z,
it correlates strongly with L(z), making it a low-variance baseline.

Key implementation note: the SAME logits-from-AV are used for the Gumbel-Softmax
construction (z̃) and for log π_φ(z). So one teacher-force forward suffices for
all the AV-side gradients.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_gumbel(shape, device, dtype, eps: float = 1e-20):
    """g ~ Gumbel(0, 1) via inverse-CDF on Uniform."""
    u = torch.rand(shape, device=device, dtype=torch.float32).clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(u)).to(dtype)


def gumbel_argmax(logits: torch.Tensor, gumbels: torch.Tensor | None = None):
    """Hard sample via Gumbel-max.

    Returns (z, gumbels). If gumbels is None, samples fresh.
    z has shape logits.shape[:-1]; gumbels has the same shape as logits.
    """
    if gumbels is None:
        gumbels = sample_gumbel(logits.shape, logits.device, logits.dtype)
    y = logits + gumbels
    z = y.argmax(dim=-1)
    return z, gumbels


def gumbel_softmax_from(logits: torch.Tensor, gumbels: torch.Tensor, tau: float):
    """Soft Gumbel-Softmax sample using *given* Gumbel noise (for shared-noise pairing)."""
    return F.softmax((logits + gumbels) / tau, dim=-1)


def conditional_gumbels(logits: torch.Tensor, z: torch.Tensor, eps: float = 1e-20):
    """Re-sample Gumbels given that argmax(logits + g) == z.

    Closed form (Maddison et al. 2017 "Concrete distribution"; Tucker 2017 REBAR):
      Let π_v = softmax(logits)_v. For independent u_v ~ Uniform(0, 1):
        g'_z = -log(-log u_z)                            [unconstrained Gumbel]
        g'_v = -log(-log u_v / π_v - log u_z)  for v ≠ z [truncated Gumbel]

    The truncation enforces g'_v + log π_v <= g'_z + log π_z so the argmax is z.

    Args:
        logits: [..., V] AV logits at each rollout position.
        z:      [...]    hard sample indices.

    Returns:
        gumbels': [..., V] same shape as logits.
    """
    log_pi = F.log_softmax(logits.float(), dim=-1)  # [..., V]
    pi = log_pi.exp().clamp(min=eps)

    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32).clamp(min=eps, max=1 - eps)
    # u_z = u at the argmax position; build by gather
    u_z = u.gather(-1, z.unsqueeze(-1))  # [..., 1]

    # g_z = -log(-log u_z): unconstrained Gumbel for the argmax index
    g_z = -torch.log(-torch.log(u_z))  # [..., 1]

    # g_v for v != z: truncated Gumbel
    # g_v = -log(-log u_v / π_v - log u_z)
    # The "-log u_z" offset is the threshold.
    inner = (-torch.log(u) / pi) + (-torch.log(u_z))  # [..., V]
    g_v = -torch.log(inner.clamp(min=eps))

    # At position z, replace with g_z
    g_cond = g_v.scatter(-1, z.unsqueeze(-1), g_z)
    return g_cond.to(logits.dtype)


def soft_token_embeds(z_tilde: torch.Tensor, embed_weight: torch.Tensor):
    """Compute weighted-sum embeddings for a soft distribution over vocab.

    Args:
        z_tilde: [B, T, V] simplex (rows sum to 1).
        embed_weight: [V, d] embed_tokens.weight.

    Returns:
        [B, T, d] dense embeddings.
    """
    return z_tilde @ embed_weight


def gather_log_prob(log_probs: torch.Tensor, z: torch.Tensor):
    """log π_φ(z_t) for each rollout position. Returns [..., ] (one less dim than log_probs)."""
    return log_probs.gather(-1, z.unsqueeze(-1)).squeeze(-1)
