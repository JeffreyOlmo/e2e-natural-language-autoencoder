"""Calibrate τ for grad-distill: find the value that gives a target mean
per-position KL(q || π_AV) on a representative batch.

Theory: τ in `q_t ∝ π_AV · exp(-<ĝ, e_v>/τ)` IS the per-step KL-regularization
strength. The "right" τ depends on gradient and embedding scales — which depend
on the model and the stage of training.

Procedure:
  1. Sample a batch of activations.
  2. AV rollout → recover g_t at AR's input embeddings via autograd.
  3. Per-position L2-normalize g_t → ĝ_t.
  4. For τ ∈ grid: compute q_t with current_av anchoring, measure mean KL(q || π).
  5. Recommend τ that hits the target KL (default 0.05 nats).

Run on SFT-init checkpoints to choose τ for the START of joint RL. Re-run on
mid-training checkpoints to see how τ should evolve.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import NLAConfig
from nla.data import NLADataset
from nla.grad_distill import build_ar_inputs_from_rollout
from nla.injection import build_av_inputs_embeds, build_av_prompt_ids
from nla.model import ARModel
from nla.rollout import av_rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", default="checkpoints/av_sft/av.pt")
    ap.add_argument("--ar-ckpt", default="checkpoints/ar_sft/ar.pt")
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-kl", type=float, default=0.05,
                    help="target per-position KL(q || π) in nats")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="rl")
    args = ap.parse_args()

    cfg = NLAConfig()
    torch.manual_seed(args.seed)

    print(f"Loading models on {args.device}")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(args.device)
    av.load_state_dict(torch.load(args.av_ckpt, map_location="cpu", weights_only=False)["state_dict"])
    av.eval()
    ar = ARModel(cfg, dtype=torch.float32).to(args.device)
    ar.load_state_dict(torch.load(args.ar_ckpt, map_location="cpu", weights_only=False)["state_dict"])
    ar.eval()

    ds = NLADataset(
        f"{cfg.data_dir}/activations_L{cfg.layer}.parquet",
        f"{cfg.data_dir}/summaries_L{cfg.layer}.parquet",
        split=args.split,
    )
    print(f"  split={args.split}, {len(ds)} records; collecting from {args.n_batches} batches of {args.batch_size}")

    rng = np.random.default_rng(args.seed)
    e_v = av.get_input_embeddings().weight  # [V, d]

    g_hat_list, log_pi_list = [], []
    for _ in range(args.n_batches):
        idxs = rng.choice(len(ds), args.batch_size, replace=False)
        h = torch.stack([ds.records[int(i)].h for i in idxs]).to(args.device)
        roll = av_rollout(
            av, tok, h, marker_token=cfg.marker_token, alpha=cfg.alpha,
            max_new_tokens=args.max_new_tokens, temperature=1.0, top_p=0.95,
        )
        rollout_ids = roll["rollout_ids"]
        rollout_mask = roll["rollout_mask"]
        rollout_logits = roll["rollout_logits"]  # [B, T, V]

        ar_inputs = build_ar_inputs_from_rollout(tok, rollout_ids, rollout_mask)
        ar_ids = ar_inputs["ar_ids"].to(args.device)
        ar_mask = ar_inputs["ar_mask"].to(args.device)
        offsets = ar_inputs["rollout_offsets"]
        lengths = ar_inputs["rollout_lengths"]

        ar_embeds = ar.get_input_embeddings()(ar_ids).detach().clone().requires_grad_(True)
        pred = ar(inputs_embeds=ar_embeds, attention_mask=ar_mask)
        p_norm = F.normalize(pred, dim=-1) * cfg.mse_norm
        g_norm = F.normalize(h, dim=-1) * cfg.mse_norm
        mse = ((p_norm - g_norm) ** 2).mean()
        g_at_input = torch.autograd.grad(mse, ar_embeds)[0].detach()

        B, _, d = g_at_input.shape
        for i in range(B):
            n = int(lengths[i].item())
            if n == 0:
                continue
            off = int(offsets[i].item())
            g_resp_i = g_at_input[i, off : off + n]
            g_norm_i = g_resp_i.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            g_hat_i = g_resp_i / g_norm_i
            log_pi_i = F.log_softmax(rollout_logits[i, :n].detach(), dim=-1)
            g_hat_list.append(g_hat_i.cpu())
            log_pi_list.append(log_pi_i.cpu())

    g_hat_all = torch.cat(g_hat_list, dim=0)  # [N, d]
    log_pi_all = torch.cat(log_pi_list, dim=0)  # [N, V]
    N = g_hat_all.shape[0]
    print(f"  N = {N} valid rollout positions")

    # KL sweep
    g_hat_all = g_hat_all.to(args.device)
    log_pi_all = log_pi_all.to(args.device)
    raw_scores = -(g_hat_all @ e_v.T.to(g_hat_all.dtype))  # [N, V]; divide by τ inside loop
    pi_ent_mean = (-(log_pi_all.exp() * log_pi_all).sum(dim=-1)).mean().item()

    taus = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    print()
    print(f"  AV policy entropy (mean per pos) = {pi_ent_mean:.4f} nats")
    print()
    print(f"  {'τ':>8s} {'KL(q||π)':>12s} {'KL(π||q)':>12s} {'H(q)':>10s} {'top-1 same?':>12s}")
    results = []
    pi_argmax = log_pi_all.argmax(dim=-1)  # [N]
    for tau in taus:
        with torch.no_grad():
            scores = raw_scores / tau
            log_q = F.log_softmax(log_pi_all + scores, dim=-1)
            q = log_q.exp()
            kl_q_pi = (q * (log_q - log_pi_all)).sum(dim=-1).mean().item()
            pi = log_pi_all.exp()
            kl_pi_q = (pi * (log_pi_all - log_q)).sum(dim=-1).mean().item()
            q_ent = (-(q * log_q).sum(dim=-1)).mean().item()
            agree = (q.argmax(dim=-1) == pi_argmax).float().mean().item()
        results.append({"tau": tau, "kl_q_pi": kl_q_pi, "kl_pi_q": kl_pi_q,
                        "q_ent": q_ent, "agree": agree})
        print(f"  {tau:>8.3f} {kl_q_pi:>12.5f} {kl_pi_q:>12.5f} {q_ent:>10.3f} {agree:>11.2%}")

    target = args.target_kl
    print()
    print(f"  Target KL = {target} nats")

    sorted_res = sorted(results, key=lambda r: r["tau"])
    for i in range(len(sorted_res) - 1):
        kl1, kl2 = sorted_res[i]["kl_q_pi"], sorted_res[i + 1]["kl_q_pi"]
        if (kl1 - target) * (kl2 - target) <= 0:
            tau1, tau2 = sorted_res[i]["tau"], sorted_res[i + 1]["tau"]
            log_tau1, log_tau2 = np.log(tau1), np.log(tau2)
            log_kl1, log_kl2 = np.log(max(kl1, 1e-9)), np.log(max(kl2, 1e-9))
            t = (np.log(target) - log_kl1) / (log_kl2 - log_kl1) if log_kl2 != log_kl1 else 0.5
            tau_interp = float(np.exp(log_tau1 + t * (log_tau2 - log_tau1)))
            print(f"  Interpolated τ (log-log) = {tau_interp:.5f}")
            break
    else:
        closest = min(results, key=lambda r: abs(r["kl_q_pi"] - target))
        print(f"  Closest grid τ = {closest['tau']} (KL = {closest['kl_q_pi']:.5f})")
        print(f"  (target KL not bracketed by grid; consider extending the τ range)")


if __name__ == "__main__":
    main()
