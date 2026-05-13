"""Evaluate a (AV, AR) checkpoint on downstream-KL — the real e2e metric.

For each held-out record:
  1. AV(h) → text (greedy or sampled, default temp=1.0/top-p=0.95 like training)
  2. AR(text) → ĥ
  3. Run frozen LM on the ORIGINAL source text (the one h was collected from)
     - Once unpatched → orig logits
     - Once with ĥ patched at layer 16 position p → patched logits
  4. KL(orig || patched) over future positions, plus KL@p (next-token), plus
     standard MSE/FVE.

Lets us compare a vanilla-GRPO ckpt vs an e2e ckpt on behavioral fidelity.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig
from nla.model import ARModel

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl_e2e import make_patch_state, make_patch_hook, tokenize_contexts
from train_small_rl import (
    AR_PREFIX, AR_SUFFIX, build_ar_inputs, build_av_prompt_embeds, ddp_generate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-init", required=True)
    ap.add_argument("--av-init", required=True)
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-records", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm-dtype", default="bfloat16")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    # Load AV
    print(f"Loading AV: {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    sd = torch.load(args.av_init, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()

    # Load AR
    print(f"Loading AR: {args.ar_init}")
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    sd_ar = torch.load(args.ar_init, map_location=device, weights_only=False)
    sd_ar = sd_ar.get("state_dict", sd_ar)
    ar_state = ar.state_dict()
    ar.load_state_dict({k: v for k, v in sd_ar.items() if k in ar_state}, strict=False)
    ar.eval()

    # Load LM (frozen, hooked)
    lm_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[args.lm_dtype]
    print(f"Loading LM (frozen, {args.lm_dtype})")
    lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=lm_dtype).to(device)
    lm.eval()
    patch_state = make_patch_state()
    handle = lm.model.layers[cfg.layer - 1].register_forward_hook(make_patch_hook(patch_state))

    # Records (use the same eval slice as training — last 5%)
    print(f"Loading records")
    t = pq.read_table(args.activations)
    n = len(t)
    flat = np.asarray(t["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = t["text"].to_pylist()
    positions = np.asarray(t["position"].to_pylist(), dtype=np.int64)
    n_eval = max(1, n // 20)
    eval_idx_all = list(range(n - n_eval, n))
    # Filter for KL signal
    eval_idx = [i for i in eval_idx_all if positions[i] + 8 < args.max_ctx_tokens]
    eval_idx = eval_idx[: args.n_records]
    print(f"  using {len(eval_idx)} eval records")

    # Baseline MSE
    mu = activations[: n - n_eval].mean(dim=0).to(device)
    sc = cfg.mse_norm

    totals = {"mse": 0.0, "kl_avg": 0.0, "kl_at_p": 0.0, "kl_p_baseline": 0.0, "n": 0}
    pbar = tqdm(range(0, len(eval_idx), args.batch_size))
    for b0 in pbar:
        b1 = min(b0 + args.batch_size, len(eval_idx))
        ids = eval_idx[b0:b1]
        h_batch = activations[ids].to(device)
        text_batch = [texts[i] for i in ids]
        pos_batch = positions[ids]

        # AV → text
        prompt_embeds, prompt_mask = build_av_prompt_embeds(
            av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32, no_prompt=False,
        )
        with torch.no_grad():
            gen = ddp_generate(
                av, prompt_embeds, prompt_mask, max_new_tokens=args.max_new_tokens,
                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            )
            mask = (gen == tok.eos_token_id).cumsum(1) <= 1
            ar_ids, ar_mask, _, _ = build_ar_inputs(tok, gen, mask.long(), device, AR_PREFIX, AR_SUFFIX)
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask)  # [B, d]

            # MSE
            p_n = F.normalize(pred.float(), dim=-1) * sc
            g_n = F.normalize(h_batch.float(), dim=-1) * sc
            mse = ((p_n - g_n) ** 2).mean(dim=-1)  # [B]

            # Downstream KL
            ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
                tok, text_batch, pos_batch, args.max_ctx_tokens, device,
            )
            T_ctx = ctx_ids.shape[1]
            # orig
            patch_state["h_hat"] = None
            orig_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            orig_lp = F.log_softmax(orig_out.logits.float(), dim=-1)
            # patched ĥ
            patch_state["h_hat"] = pred.to(lm_dtype)
            patch_state["positions"] = pos_in_ctx
            pat_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            patch_state["h_hat"] = None
            pat_lp = F.log_softmax(pat_out.logits.float(), dim=-1)

            # Baseline: predict-mean reconstruction
            mu_b = F.normalize(mu, dim=-1) * h_batch.norm(dim=-1, keepdim=True).mean()  # approximate magnitude
            patch_state["h_hat"] = mu_b.unsqueeze(0).expand_as(pred).to(lm_dtype)
            patch_state["positions"] = pos_in_ctx
            mb_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            patch_state["h_hat"] = None
            mb_lp = F.log_softmax(mb_out.logits.float(), dim=-1)

            pos_idx = torch.arange(T_ctx, device=device).unsqueeze(0)
            future = (pos_idx >= pos_in_ctx.unsqueeze(1)) & ctx_mask.bool()
            B = orig_lp.shape[0]
            kl_pp = (orig_lp.exp() * (orig_lp - pat_lp)).sum(-1)
            kl_avg = (kl_pp * future.float()).sum(1) / future.float().sum(1).clamp_min(1.0)
            kl_at_p = kl_pp.gather(1, pos_in_ctx.unsqueeze(1)).squeeze(1)
            kl_pp_mu = (orig_lp.exp() * (orig_lp - mb_lp)).sum(-1)
            kl_at_p_mu = kl_pp_mu.gather(1, pos_in_ctx.unsqueeze(1)).squeeze(1)

        totals["mse"] += mse.sum().item()
        totals["kl_avg"] += kl_avg.sum().item()
        totals["kl_at_p"] += kl_at_p.sum().item()
        totals["kl_p_baseline"] += kl_at_p_mu.sum().item()
        totals["n"] += B
        pbar.set_postfix(mse=f"{totals['mse']/totals['n']:.5f}",
                         kl_p=f"{totals['kl_at_p']/totals['n']:.3f}",
                         kl_avg=f"{totals['kl_avg']/totals['n']:.4f}")

    n = totals["n"]
    res = {
        "mse": totals["mse"] / n,
        "kl_future_avg": totals["kl_avg"] / n,
        "kl_at_p_recon": totals["kl_at_p"] / n,
        "kl_at_p_predict_mean": totals["kl_p_baseline"] / n,
        "n": n,
        "ar_init": args.ar_init,
        "av_init": args.av_init,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"saved → {args.out}")
    handle.remove()


if __name__ == "__main__":
    main()
