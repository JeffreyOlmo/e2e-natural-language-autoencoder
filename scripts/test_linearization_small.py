"""Test the linearization assumption on the 0.5B SFT'd NLA setup.

For each rollout token position t we have:
  - π_t(v|<t): AV's current next-token distribution (from teacher-force).
  - g_t = ∂L/∂e_t: gradient of one-reconstruction MSE at e_t.
  - scores_t(v) = -<ĝ_t, e_v>/τ: linearized one-step reward.
  - q_t = softmax(log π_t + scores_t): closed-form KL-regularized teacher.

Hypothesis (grad-distill claim): substituting tokens at every position with
draws from q_t (rather than π_t) should *decrease* L on average — by the linear
prediction E[<g_t, e_v>] = something negative under q's tilt.

This script measures the ACTUAL ΔFVE across τ values, comparing:
  - z_q: rollout where every position's token is resampled from q_t
  - z_π: rollout where every position's token is resampled from π_t (control)

For each τ, reports:
  - ΔFVE(q_t - π_t) — the linearization-attributable improvement
  - ΔFVE(q_t - original) — total change vs baseline
  - Top-1 agreement between q and π (low = aggressive teacher)

Single-GPU, no DDP. Default settings = 0.5B SFT init + cfg.alpha.
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import NLAConfig
from nla.model import ARModel
from nla.prompts import build_av_messages


AR_PREFIX = "Summary of the following text: <text>"
AR_SUFFIX = "</text> <summary>"


def load_sd(path, device):
    x = torch.load(path, map_location=device, weights_only=False)
    return x["state_dict"] if isinstance(x, dict) and "state_dict" in x else x


def build_ar_inputs_from_rollout(tok, rollout_ids, rollout_mask, device):
    pre_ids = tok(AR_PREFIX, add_special_tokens=False).input_ids
    suf_ids = tok(AR_SUFFIX, add_special_tokens=False).input_ids
    pad_id = tok.pad_token_id
    rows, masks, offsets, lengths = [], [], [], []
    B = rollout_ids.shape[0]
    for i in range(B):
        n = int(rollout_mask[i].sum().item())
        seq = pre_ids + rollout_ids[i, :n].tolist() + suf_ids
        rows.append(seq); masks.append([1] * len(seq))
        offsets.append(len(pre_ids)); lengths.append(n)
    max_len = max(len(r) for r in rows)
    ar_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    ar_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, (r, m) in enumerate(zip(rows, masks)):
        ar_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        ar_mask[i, :len(m)] = torch.tensor(m, dtype=torch.long, device=device)
    return ar_ids, ar_mask, torch.tensor(offsets), torch.tensor(lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-checkpoint", default="checkpoints/av_sft_kitft/av.pt")
    ap.add_argument("--ar-checkpoint", default="checkpoints/ar_sft_kitft/ar.pt")
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--n-rollouts", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--taus", type=str, default="0.001,0.005,0.01,0.05,0.1,0.5,1.0",
                    help="Comma-separated list of τ values to sweep.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-eval-set", action="store_true")
    args = ap.parse_args()

    cfg = NLAConfig()
    device, dtype = args.device, torch.float32
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    taus = [float(t) for t in args.taus.split(",")]

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    print(f"Loading AV: {cfg.base_model} + {args.av_checkpoint}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).to(device)
    av.load_state_dict(load_sd(args.av_checkpoint, device))
    av.eval()

    print(f"Loading AR: identity-init + {args.ar_checkpoint}")
    ar = ARModel(cfg, dtype=dtype).to(device).eval()
    ar.load_state_dict(load_sd(args.ar_checkpoint, device))

    print(f"Loading activations: {args.activations}")
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    acts = torch.from_numpy(flat.reshape(n, d).copy())
    n_eval = max(1, n // 20)
    pool = list(range(n - n_eval, n)) if args.use_eval_set else list(range(n))
    idxs = rng.choice(pool, size=args.n_rollouts, replace=False).tolist()
    h_batch = acts[idxs].to(device)
    mu = acts.mean(dim=0).to(device).to(dtype)
    sc = cfg.mse_norm

    # Baseline MSE (predict-mean on this batch, for FVE denominator)
    p_b = F.normalize(mu.expand_as(h_batch).float(), dim=-1) * sc
    g_b = F.normalize(h_batch.float(), dim=-1) * sc
    base_per = ((p_b - g_b) ** 2).mean(dim=-1)  # [B]

    # ---- 1. Sample rollouts from AV ----
    msgs = build_av_messages(cfg.marker_token)
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    B = h_batch.shape[0]
    prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
    h_unit = F.normalize(h_batch.float(), dim=-1)
    inj = (cfg.alpha * h_unit).to(dtype)
    pos_marker = (prompt_ids_b == marker_id).float().argmax(dim=1)
    embeds = av.get_input_embeddings()(prompt_ids_b).clone()
    embeds[torch.arange(B, device=device), pos_marker] = inj
    attn_mask = torch.ones_like(prompt_ids_b)
    print(f"\nSampling {B} rollouts (max_new_tokens={args.max_new_tokens})...")
    with torch.no_grad():
        gen = av.generate(
            inputs_embeds=embeds, attention_mask=attn_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    rollout_ids = gen  # [B, T_resp]
    eos = tok.eos_token_id
    rollout_mask = ((rollout_ids == eos).cumsum(dim=1) <= 1).long()

    def fve_for_rollout(r_ids, r_mask):
        """One-recon AR forward → per-rollout FVE."""
        ar_ids, ar_mask, _, _ = build_ar_inputs_from_rollout(tok, r_ids, r_mask, device)
        with torch.no_grad():
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask)
        p_norm = F.normalize(pred.float(), dim=-1) * sc
        g_norm = F.normalize(h_batch.float(), dim=-1) * sc
        per_mse = ((p_norm - g_norm) ** 2).mean(dim=-1)
        return 1.0 - per_mse / base_per  # per-rollout FVE

    fve_orig = fve_for_rollout(rollout_ids, rollout_mask)
    print(f"  baseline FVE on these rollouts: mean={fve_orig.mean().item():+.4f}")

    # ---- 2. Compute g_at_resp via AR backward on the inputs_embeds ----
    ar_ids, ar_mask, offsets, lengths = build_ar_inputs_from_rollout(tok, rollout_ids, rollout_mask, device)
    embed_layer_ar = ar.get_input_embeddings()
    ar_embeds_in = embed_layer_ar(ar_ids).detach().clone().requires_grad_(True)
    pred = ar(inputs_embeds=ar_embeds_in, attention_mask=ar_mask)
    p_norm = F.normalize(pred, dim=-1) * sc
    g_norm_target = F.normalize(h_batch, dim=-1) * sc
    mse = ((p_norm - g_norm_target) ** 2).mean()
    g_at_input = torch.autograd.grad(mse, ar_embeds_in)[0].detach()
    T_resp = rollout_ids.shape[1]
    g_at_resp = torch.zeros(B, T_resp, d, device=device, dtype=dtype)
    for i in range(B):
        off = int(offsets[i].item()); n_v = int(lengths[i].item())
        if n_v > 0:
            g_at_resp[i, :n_v] = g_at_input[i, off : off + n_v]
    g_raw_norm = g_at_resp.norm(dim=-1, keepdim=True)
    g_hat = g_at_resp / g_raw_norm.clamp_min(1e-12)

    # ---- 3. AV per-position logits at the rollout (teacher-force) ----
    def teacher_force_logits(model, prompt_embeds, prompt_mask, response_ids, response_mask):
        embed_layer = model.get_input_embeddings()
        response_embeds = embed_layer(response_ids)
        full_embeds = torch.cat([prompt_embeds, response_embeds], dim=1)
        full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        with torch.no_grad():
            out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
        T_pre = prompt_embeds.shape[1]
        return out.logits[:, T_pre - 1 : T_pre - 1 + response_ids.shape[1], :].contiguous()

    av_logits = teacher_force_logits(av, embeds, attn_mask, rollout_ids, rollout_mask)
    # [B, T_resp, V]
    E_av = av.get_input_embeddings().weight.detach()  # [V, d]

    # ---- 4. τ sweep ----
    print(f"\nτ sweep over {taus}; B={B} rollouts, T_resp={T_resp}, V={E_av.shape[0]}")
    results = []
    for tau in taus:
        scores = -torch.einsum("btd,vd->btv", g_hat, E_av) / tau  # [B, T, V]
        log_pi = F.log_softmax(av_logits.float(), dim=-1)
        log_q = F.log_softmax(log_pi + scores, dim=-1)
        # Sample one replacement token per position from q
        q = log_q.exp()
        pi = log_pi.exp()
        # Top-1 agreement
        agree = (q.argmax(-1) == pi.argmax(-1)).float()
        agree_mask = agree * rollout_mask.float()
        n_valid = rollout_mask.float().sum().clamp_min(1.0)
        agree_mean = (agree_mask.sum() / n_valid).item()

        # Sample from q at every position
        q_flat = q.reshape(-1, q.shape[-1])
        v_q_flat = torch.multinomial(q_flat, num_samples=1).squeeze(-1)
        v_q = v_q_flat.view(B, T_resp)
        # Sample from π at every position (control)
        pi_flat = pi.reshape(-1, pi.shape[-1])
        v_pi_flat = torch.multinomial(pi_flat, num_samples=1).squeeze(-1)
        v_pi = v_pi_flat.view(B, T_resp)

        # Replace tokens at valid positions only
        new_q_rollout = torch.where(rollout_mask.bool(), v_q, rollout_ids)
        new_pi_rollout = torch.where(rollout_mask.bool(), v_pi, rollout_ids)

        fve_q = fve_for_rollout(new_q_rollout, rollout_mask)
        fve_pi = fve_for_rollout(new_pi_rollout, rollout_mask)

        d_q = (fve_q - fve_orig).mean().item()
        d_pi = (fve_pi - fve_orig).mean().item()
        d_qp = (fve_q - fve_pi).mean().item()
        results.append({
            "tau": tau,
            "fve_orig_mean": fve_orig.mean().item(),
            "fve_q_mean": fve_q.mean().item(),
            "fve_pi_mean": fve_pi.mean().item(),
            "delta_q": d_q,
            "delta_pi": d_pi,
            "delta_q_minus_pi": d_qp,
            "top1_agree": agree_mean,
        })
        print(f"  τ={tau:.4f}  agree={agree_mean:.3f}  "
              f"FVE(orig)={fve_orig.mean().item():+.4f}  "
              f"FVE(q)={fve_q.mean().item():+.4f}  Δ={d_q:+.4f}  "
              f"FVE(π)={fve_pi.mean().item():+.4f}  Δ={d_pi:+.4f}  "
              f"Δ(q-π)={d_qp:+.4f}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'τ':>8} {'agree':>7} {'ΔFVE(q-orig)':>14} {'ΔFVE(π-orig)':>14} {'ΔFVE(q-π)':>12}")
    for r in results:
        print(f"{r['tau']:>8.4f} {r['top1_agree']:>7.3f} {r['delta_q']:>+14.4f} "
              f"{r['delta_pi']:>+14.4f} {r['delta_q_minus_pi']:>+12.4f}")
    print("\nIf grad-distill is informative: Δ(q-π) should be POSITIVE for some τ (q better than π).")


if __name__ == "__main__":
    main()
