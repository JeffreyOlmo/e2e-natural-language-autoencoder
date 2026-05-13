"""Diagnose the downstream-KL signal in the e2e setup.

Loads vanilla-GRPO AR + frozen LM, picks a few records, computes:
  - orig_logits (unhooked LM forward)
  - patched_logits with ĥ (from AR) replacing h at layer 16 position p
  - patched_logits with PURE NOISE (random direction, same norm as h)
  - patched_logits with the EXACT h (control: KL should be ~0)
and reports per-position KL profile, plus angular error of ĥ vs h.

If patching the exact h does NOT give KL ≈ 0, the hook is broken.
If patching ĥ gives small KL averaged but large KL at p+1, dilution is the story.
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

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl_e2e import (
    make_patch_state, make_patch_hook, tokenize_contexts,
)
from train_small_rl import build_ar_inputs, AR_PREFIX, AR_SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-init", default="checkpoints/rl_small_grpo_cont/ar_step_500.pt")
    ap.add_argument("--av-init", default="checkpoints/rl_small_grpo_cont/av_step_500.pt")
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--summaries", default="data/summaries_kitft.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-records", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--lm-dtype", default="bfloat16")
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load AR
    print(f"Loading AR: {args.ar_init}")
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    sd = torch.load(args.ar_init, map_location=device, weights_only=False)
    sd = sd.get("state_dict", sd)
    ar_state = ar.state_dict()
    ar.load_state_dict({k: v for k, v in sd.items() if k in ar_state}, strict=False)
    ar.eval()

    # Load LM (frozen)
    lm_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[args.lm_dtype]
    print(f"Loading LM (frozen, {args.lm_dtype})")
    lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=lm_dtype).to(device)
    lm.eval()

    patch_state = make_patch_state()
    handle = lm.model.layers[cfg.layer - 1].register_forward_hook(make_patch_hook(patch_state))

    # Load records: activations + summaries to feed AR
    print("Loading records")
    acts_t = pq.read_table(args.activations)
    sums_t = pq.read_table(args.summaries)
    n = len(acts_t)
    flat = np.asarray(acts_t["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = acts_t["text"].to_pylist()
    positions = np.asarray(acts_t["position"].to_pylist(), dtype=np.int64)
    doc_ids_a = np.asarray(acts_t["doc_id"].to_pylist(), dtype=np.int64)
    positions_a = positions
    sum_map = {
        (int(d), int(p)): s for d, p, s in
        zip(sums_t["doc_id"].to_pylist(), sums_t["position"].to_pylist(), sums_t["summary"].to_pylist())
    }
    # Pick n records with p+8 < max_ctx_tokens and a summary present
    chosen = []
    for i in range(n):
        if positions[i] + 8 >= args.max_ctx_tokens:
            continue
        key = (int(doc_ids_a[i]), int(positions_a[i]))
        if key not in sum_map or not sum_map[key]:
            continue
        chosen.append(i)
        if len(chosen) == args.n_records:
            break
    print(f"  picked {len(chosen)} records (positions={positions[chosen].tolist()})")

    summaries = [sum_map[(int(doc_ids_a[i]), int(positions_a[i]))] for i in chosen]
    h_batch = activations[chosen].to(device)  # [B, d]
    text_batch = [texts[i] for i in chosen]
    pos_batch = positions[chosen]

    # Build AR inputs from kitft summaries (using same prefix/suffix as training)
    sum_tok = []
    for s in summaries:
        ids = tok(s, add_special_tokens=False).input_ids
        sum_tok.append(ids)
    max_len = max(len(s) for s in sum_tok)
    pad = tok.pad_token_id
    rollout_ids = torch.full((len(chosen), max_len), pad, dtype=torch.long, device=device)
    rollout_mask = torch.zeros((len(chosen), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(sum_tok):
        rollout_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        rollout_mask[i, :len(s)] = 1
    ar_ids, ar_mask, _, _ = build_ar_inputs(tok, rollout_ids, rollout_mask, device, AR_PREFIX, AR_SUFFIX)

    with torch.no_grad():
        h_hat = ar(input_ids=ar_ids, attention_mask=ar_mask).float()  # [B, d]
    # Magnitudes + cos
    h_norm = h_batch.norm(dim=-1)
    hh_norm = h_hat.norm(dim=-1)
    cos = F.cosine_similarity(h_hat, h_batch, dim=-1)
    print(f"  ||h||: mean={h_norm.mean():.2f} std={h_norm.std():.2f}")
    print(f"  ||ĥ||: mean={hh_norm.mean():.2f} std={hh_norm.std():.2f}")
    print(f"  cos(ĥ, h): mean={cos.mean():.4f}, per-sample={cos.tolist()}")

    # Tokenize context
    ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
        tok, text_batch, pos_batch, args.max_ctx_tokens, device,
    )
    T_ctx = ctx_ids.shape[1]
    print(f"  T_ctx={T_ctx}, positions in ctx={pos_in_ctx.tolist()}")

    # Forward variants
    def forward(h_patch, label):
        patch_state["h_hat"] = h_patch
        patch_state["positions"] = pos_in_ctx if h_patch is not None else None
        with torch.no_grad():
            out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
        patch_state["h_hat"] = None
        return F.log_softmax(out.logits.float(), dim=-1), label

    orig_lp, _ = forward(None, "orig")
    exact_lp, _ = forward(h_batch.to(lm_dtype), "patch_h_exact")
    arpred_lp, _ = forward(h_hat.to(lm_dtype), "patch_h_hat")
    # Noise patch: random direction with same norm
    noise = torch.randn_like(h_batch)
    noise = noise / noise.norm(dim=-1, keepdim=True) * h_norm.unsqueeze(-1)
    noise_lp, _ = forward(noise.to(lm_dtype), "patch_random")

    def kl_per_pos(p_lp, q_lp):
        """KL(p || q) per position. Both are log_probs [B, T, V]."""
        return (p_lp.exp() * (p_lp - q_lp)).sum(-1)

    def report(lab, lp):
        kl = kl_per_pos(orig_lp, lp)  # [B, T]
        # Mask: positions > p AND within ctx_mask
        B, T = kl.shape
        pos_idx = torch.arange(T, device=device).unsqueeze(0)
        future = (pos_idx > pos_in_ctx.unsqueeze(1)) & ctx_mask.bool()
        # Per-sample profile around p
        first_kl = []
        avg_future = []
        peak = []
        for b in range(B):
            p = int(pos_in_ctx[b].item())
            valid = future[b]
            kl_b = kl[b]
            if valid.any():
                avg_future.append(kl_b[valid].mean().item())
                peak.append(kl_b[valid].max().item())
            else:
                avg_future.append(float("nan"))
                peak.append(float("nan"))
            # KL at exactly p+1 (next-token), if available
            if p + 1 < T and ctx_mask[b, p + 1] == 1:
                first_kl.append(kl_b[p + 1].item())
            else:
                first_kl.append(float("nan"))
        print(f"  [{lab}]")
        print(f"    KL@p+1 (next token): mean={np.nanmean(first_kl):.4f}, vals={[f'{x:.3f}' for x in first_kl]}")
        print(f"    KL future avg:       mean={np.nanmean(avg_future):.4f}, vals={[f'{x:.3f}' for x in avg_future]}")
        print(f"    KL future peak:      mean={np.nanmean(peak):.4f}, vals={[f'{x:.3f}' for x in peak]}")
        return kl

    print("\n--- KL diagnostics ---")
    kl_exact = report("exact h (sanity ≈ 0)", exact_lp)
    kl_arpred = report("ĥ (AR pred)", arpred_lp)
    kl_noise = report("random vector @ ||h||", noise_lp)

    # Profile of KL near p+1 for ĥ
    print("\n--- ĥ KL decay profile (avg over batch, near position p) ---")
    profile = []
    for delta in range(0, 32):
        kls = []
        for b in range(orig_lp.shape[0]):
            p = int(pos_in_ctx[b].item())
            if p + delta < T_ctx and ctx_mask[b, p + delta] == 1:
                kls.append(kl_arpred[b, p + delta].item())
        if kls:
            profile.append((delta, sum(kls) / len(kls)))
    for delta, k in profile[:20]:
        bar = "#" * max(1, int(k * 100))
        print(f"  Δ={delta:2d}: KL={k:.4f}  {bar}")

    handle.remove()


if __name__ == "__main__":
    main()
