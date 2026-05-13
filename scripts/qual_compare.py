"""Side-by-side qualitative comparison: vanilla GRPO vs e2e NLA.

For the same set of records:
  - Sample AV explanation from each (same seed → same h, same temperature noise)
  - Run AR to get ĥ from each
  - Compute KL@p for each
  - Compute cos(ĥ_grpo, ĥ_e2e) — how much did the encoding move
  - Print N most-interesting records sorted by various criteria

Outputs a JSON dump (so we can re-sort without re-running) plus a printed report.
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


def load_av(path, base_model, device):
    av = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32).to(device)
    sd = torch.load(path, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()
    return av


def load_ar(path, cfg, device):
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    sd = torch.load(path, map_location=device, weights_only=False)
    sd = sd.get("state_dict", sd)
    ar_state = ar.state_dict()
    ar.load_state_dict({k: v for k, v in sd.items() if k in ar_state}, strict=False)
    ar.eval()
    return ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo-ar", default="checkpoints/rl_small_grpo_cont/ar_step_500.pt")
    ap.add_argument("--grpo-av", default="checkpoints/rl_small_grpo_cont/av_step_500.pt")
    ap.add_argument("--e2e-ar", default="checkpoints/rl_small_grpo_e2e/ar_step_300.pt")
    ap.add_argument("--e2e-av", default="checkpoints/rl_small_grpo_e2e/av_step_300.pt")
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-records", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm-dtype", default="bfloat16")
    ap.add_argument("--out", default="/tmp/qual_compare.json")
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    print("Loading 4 models (2 AVs, 2 ARs) + frozen LM")
    av_g = load_av(args.grpo_av, cfg.base_model, device)
    av_e = load_av(args.e2e_av, cfg.base_model, device)
    ar_g = load_ar(args.grpo_ar, cfg, device)
    ar_e = load_ar(args.e2e_ar, cfg, device)

    lm_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[args.lm_dtype]
    lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=lm_dtype).to(device)
    lm.eval()
    patch_state = make_patch_state()
    handle = lm.model.layers[cfg.layer - 1].register_forward_hook(make_patch_hook(patch_state))

    # Records
    t = pq.read_table(args.activations)
    n = len(t)
    flat = np.asarray(t["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = t["text"].to_pylist()
    positions = np.asarray(t["position"].to_pylist(), dtype=np.int64)
    n_eval = max(1, n // 20)
    eval_idx_all = list(range(n - n_eval, n))
    eval_idx = [i for i in eval_idx_all if positions[i] + 8 < args.max_ctx_tokens][: args.n_records]
    print(f"  {len(eval_idx)} records")

    records = []
    pbar = tqdm(range(0, len(eval_idx), args.batch_size))
    for b0 in pbar:
        b1 = min(b0 + args.batch_size, len(eval_idx))
        ids = eval_idx[b0:b1]
        h_batch = activations[ids].to(device)
        text_batch = [texts[i] for i in ids]
        pos_batch = positions[ids]
        B = h_batch.shape[0]

        # Sample AV explanations from each model with the SAME seed so the only
        # difference is AV weights, not RNG.
        rollouts = {}
        for name, av in (("grpo", av_g), ("e2e", av_e)):
            emb, mask = build_av_prompt_embeds(av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32)
            torch.manual_seed(args.seed)
            with torch.no_grad():
                gen = ddp_generate(av, emb, mask, max_new_tokens=args.max_new_tokens,
                                   eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                                   temperature=1.0, top_p=0.95)
            gen_mask = (gen == tok.eos_token_id).cumsum(1) <= 1
            rollouts[name] = (gen, gen_mask)

        # AR forward for each
        preds = {}
        for name, ar in (("grpo", ar_g), ("e2e", ar_e)):
            gen, gen_mask = rollouts[name]
            ar_ids, ar_mask, _, _ = build_ar_inputs(tok, gen, gen_mask.long(), device, AR_PREFIX, AR_SUFFIX)
            with torch.no_grad():
                preds[name] = ar(input_ids=ar_ids, attention_mask=ar_mask)  # [B, d]

        # KL@p for each
        ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
            tok, text_batch, pos_batch, args.max_ctx_tokens, device,
        )
        with torch.no_grad():
            patch_state["h_hat"] = None
            orig_lp = F.log_softmax(lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False).logits.float(), dim=-1)
            o_at_p = orig_lp.gather(1, pos_in_ctx.view(-1, 1, 1).expand(-1, 1, orig_lp.shape[-1])).squeeze(1)
            kls = {}
            for name in ("grpo", "e2e"):
                patch_state["h_hat"] = preds[name].to(lm_dtype)
                patch_state["positions"] = pos_in_ctx
                lp = F.log_softmax(lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False).logits.float(), dim=-1)
                p_at_p = lp.gather(1, pos_in_ctx.view(-1, 1, 1).expand(-1, 1, lp.shape[-1])).squeeze(1)
                kls[name] = (o_at_p.exp() * (o_at_p - p_at_p)).sum(-1)
            patch_state["h_hat"] = None

        # Encoding drift between GRPO and e2e
        cos_ge = F.cosine_similarity(preds["grpo"], preds["e2e"], dim=-1)
        drift = (preds["e2e"] - preds["grpo"]).norm(dim=-1) / preds["grpo"].norm(dim=-1).clamp_min(1e-6)

        # MSE for each vs h (the geometric reconstruction quality)
        for name in ("grpo", "e2e"):
            p_n = F.normalize(preds[name].float(), dim=-1)
            g_n = F.normalize(h_batch.float(), dim=-1)
            mse = ((p_n - g_n) ** 2).mean(dim=-1)
            preds[name + "_mse"] = mse

        # cos(ĥ, h) for each (more interpretable than MSE)
        for name in ("grpo", "e2e"):
            preds[name + "_cos_h"] = F.cosine_similarity(preds[name], h_batch, dim=-1)

        for bi in range(B):
            g_gen, g_mask = rollouts["grpo"][0][bi], rollouts["grpo"][1][bi]
            e_gen, e_mask = rollouts["e2e"][0][bi], rollouts["e2e"][1][bi]
            g_text = tok.decode(g_gen[: int(g_mask.sum().item())], skip_special_tokens=True)
            e_text = tok.decode(e_gen[: int(e_mask.sum().item())], skip_special_tokens=True)
            # Source context: take ~80 chars around position p for display
            src = text_batch[bi]
            # Render context around the position p, in tokens
            ctx_tok = tok.encode(src, add_special_tokens=False)[: args.max_ctx_tokens]
            p = int(pos_batch[bi])
            start = max(0, p - 30); end = min(len(ctx_tok), p + 8)
            ctx_window = tok.decode(ctx_tok[start:end], skip_special_tokens=True)
            # Token at position p+1 (the "true" next token in source)
            next_tok = tok.decode([ctx_tok[p + 1]]) if p + 1 < len(ctx_tok) else "<eos>"
            records.append({
                "record_id": int(ids[bi]),
                "position": int(pos_batch[bi]),
                "src_window": ctx_window,
                "true_next_token": next_tok,
                "grpo_explanation": g_text,
                "e2e_explanation": e_text,
                "grpo_kl_at_p": float(kls["grpo"][bi].item()),
                "e2e_kl_at_p": float(kls["e2e"][bi].item()),
                "kl_delta": float(kls["e2e"][bi].item() - kls["grpo"][bi].item()),
                "cos_ge": float(cos_ge[bi].item()),
                "drift_pct": float(drift[bi].item() * 100),
                "grpo_cos_h": float(preds["grpo_cos_h"][bi].item()),
                "e2e_cos_h": float(preds["e2e_cos_h"][bi].item()),
                "grpo_explanation_len": int(g_mask.sum().item()),
                "e2e_explanation_len": int(e_mask.sum().item()),
            })

    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"saved {len(records)} records → {args.out}")
    handle.remove()


if __name__ == "__main__":
    main()
