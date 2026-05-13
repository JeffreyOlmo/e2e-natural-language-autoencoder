"""Evaluate a 7B (AV, AR) checkpoint on downstream-KL.

Mirrors eval_downstream_kl.py but uses the 7B config (alpha=150, layer=20,
marker='㈎') and KitftAR. Runs single-GPU sequentially: AV→generate, free AV;
load AR→reconstruct ĥ, free AR; load frozen LM, patch+measure KL@p.
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

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fsdp import CFG, KitftAR, build_av_prompt_embeds, build_ar_inputs_from_rollout

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
KITFT_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
KITFT_AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"


def make_patch_state():
    return {"h_hat": None, "positions": None}


def make_patch_hook(state):
    def hook(module, inputs, output):
        if state["h_hat"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        idx = torch.arange(h.shape[0], device=h.device)
        h[idx, state["positions"]] = state["h_hat"].to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def tokenize_contexts(tok, texts, positions, max_ctx, device):
    B = len(texts)
    pad_id = tok.pad_token_id
    ctx_ids = torch.full((B, max_ctx), pad_id, dtype=torch.long, device=device)
    ctx_mask = torch.zeros((B, max_ctx), dtype=torch.long, device=device)
    pos_in_ctx = torch.zeros(B, dtype=torch.long, device=device)
    for i, (text, p) in enumerate(zip(texts, positions)):
        ids = tok(text, add_special_tokens=False).input_ids
        end = min(len(ids), max_ctx)
        ctx_ids[i, :end] = torch.tensor(ids[:end], dtype=torch.long, device=device)
        ctx_mask[i, :end] = 1
        pos_in_ctx[i] = min(int(p), end - 1)
    return ctx_ids, ctx_mask, pos_in_ctx


@torch.no_grad()
def hf_generate(model, inputs_embeds, attention_mask, max_new_tokens, eos_id, pad_id,
                temperature, top_p):
    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        return_dict_in_generate=False,
    )
    return out  # [B, max_new_tokens] (generate skips the inputs_embeds prefix when given embeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-init", required=True)
    ap.add_argument("--ar-init", required=True)
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-records", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Records: same eval slice as training (last 5%)
    print(f"Loading activations: {args.activations}")
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
    print(f"  using {len(eval_idx)} eval records, d={d}")

    # === Phase 1: AV → text ===
    print(f"\n[Phase 1] Loading AV: {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    sd = torch.load(args.av_init, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()

    all_gen_ids = []   # list of [B, T_new]
    all_gen_mask = []
    pbar = tqdm(range(0, len(eval_idx), args.batch_size), desc="AV generate")
    for b0 in pbar:
        b1 = min(b0 + args.batch_size, len(eval_idx))
        ids = eval_idx[b0:b1]
        h_batch = activations[ids].to(device).to(dtype)
        embeds, attn_mask, _ = build_av_prompt_embeds(av, tok, h_batch, device, dtype)
        torch.manual_seed(args.seed + b0)
        gen = hf_generate(av, embeds, attn_mask,
                          max_new_tokens=args.max_new_tokens,
                          eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                          temperature=args.rollout_temperature, top_p=args.rollout_top_p)
        # gen shape: [B, max_new_tokens] (only new tokens when inputs_embeds is used)
        mask = (gen == tok.eos_token_id).cumsum(1) <= 1
        # Pad to max_new_tokens consistency
        if gen.shape[1] < args.max_new_tokens:
            pad = torch.full((gen.shape[0], args.max_new_tokens - gen.shape[1]),
                             tok.pad_token_id, dtype=gen.dtype, device=device)
            gen = torch.cat([gen, pad], dim=1)
            mask = torch.cat([mask, torch.zeros((mask.shape[0], args.max_new_tokens - mask.shape[1]),
                                                 dtype=mask.dtype, device=device)], dim=1)
        all_gen_ids.append(gen.cpu())
        all_gen_mask.append(mask.cpu())
    del av
    torch.cuda.empty_cache()

    # === Phase 2: AR → ĥ ===
    print(f"\n[Phase 2] Loading AR: {args.ar_init}")
    ar = KitftAR(KITFT_AR_REPO, dtype=dtype).to(device)
    sd_ar = torch.load(args.ar_init, map_location=device, weights_only=False)
    sd_ar = sd_ar.get("state_dict", sd_ar)
    ar_state = ar.state_dict()
    missing = [k for k in ar_state if k not in sd_ar]
    extra = [k for k in sd_ar if k not in ar_state]
    print(f"  AR keys: missing {len(missing)} extra {len(extra)}")
    ar.load_state_dict({k: v for k, v in sd_ar.items() if k in ar_state}, strict=False)
    ar.eval()

    preds = []
    h_batches_dev = []
    pbar = tqdm(range(0, len(eval_idx), args.batch_size), desc="AR reconstruct")
    for bi, b0 in enumerate(pbar):
        b1 = min(b0 + args.batch_size, len(eval_idx))
        ids = eval_idx[b0:b1]
        h_batch = activations[ids].to(device).to(dtype)
        gen_b = all_gen_ids[bi].to(device)
        mask_b = all_gen_mask[bi].to(device)
        ar_ids, ar_mask, _, _ = build_ar_inputs_from_rollout(tok, gen_b, mask_b.long(), device)
        pred = ar(input_ids=ar_ids, attention_mask=ar_mask)  # [B, d]
        preds.append(pred.float().cpu())
        h_batches_dev.append(h_batch.float().cpu())
    del ar
    torch.cuda.empty_cache()

    # MSE / FVE
    sc = CFG["mse_scale"]
    mu = activations[: n - n_eval].mean(dim=0).float()
    preds_t = torch.cat(preds, dim=0).float()       # [N, d]
    h_t     = torch.cat(h_batches_dev, dim=0).float()  # [N, d]
    p_n = F.normalize(preds_t, dim=-1) * sc
    g_n = F.normalize(h_t, dim=-1) * sc
    mse_per = ((p_n - g_n) ** 2).mean(dim=-1)
    mu_n = F.normalize(mu, dim=-1) * sc
    base_mse = ((mu_n.unsqueeze(0) - g_n) ** 2).mean(dim=-1).mean().item()
    mse = mse_per.mean().item()
    fve = 1.0 - mse / base_mse
    print(f"\nMSE={mse:.5f}  baseline_MSE={base_mse:.5f}  FVE={fve:.4f}")

    # === Phase 3: LM with hook → KL@p ===
    print(f"\n[Phase 3] Loading frozen LM: {BASE_MODEL}")
    lm = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    lm.eval()
    patch_state = make_patch_state()
    handle = lm.model.layers[CFG["extraction_layer"] - 1].register_forward_hook(make_patch_hook(patch_state))

    totals = {"kl_avg": 0.0, "kl_at_p": 0.0, "kl_p_baseline": 0.0, "n": 0}
    pbar = tqdm(range(0, len(eval_idx), args.batch_size), desc="LM patch")
    for bi, b0 in enumerate(pbar):
        b1 = min(b0 + args.batch_size, len(eval_idx))
        ids = eval_idx[b0:b1]
        text_batch = [texts[i] for i in ids]
        pos_batch = positions[ids]
        h_batch = activations[ids].to(device)

        ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
            tok, text_batch, pos_batch, args.max_ctx_tokens, device
        )
        T_ctx = ctx_ids.shape[1]
        pred = preds[bi].to(device)

        with torch.no_grad():
            # orig
            patch_state["h_hat"] = None
            orig_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            orig_lp = F.log_softmax(orig_out.logits.float(), dim=-1)
            del orig_out
            # patched
            patch_state["h_hat"] = pred.to(dtype)
            patch_state["positions"] = pos_in_ctx
            pat_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            patch_state["h_hat"] = None
            pat_lp = F.log_softmax(pat_out.logits.float(), dim=-1)
            del pat_out
            # baseline: predict-mean
            mu_b = F.normalize(mu, dim=-1).to(device) * h_batch.norm(dim=-1, keepdim=True).mean()
            patch_state["h_hat"] = mu_b.unsqueeze(0).expand_as(pred).to(dtype)
            patch_state["positions"] = pos_in_ctx
            mb_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            patch_state["h_hat"] = None
            mb_lp = F.log_softmax(mb_out.logits.float(), dim=-1)
            del mb_out

            pos_idx = torch.arange(T_ctx, device=device).unsqueeze(0)
            future = (pos_idx >= pos_in_ctx.unsqueeze(1)) & ctx_mask.bool()
            kl_pp = (orig_lp.exp() * (orig_lp - pat_lp)).sum(-1)
            kl_avg = (kl_pp * future.float()).sum(1) / future.float().sum(1).clamp_min(1.0)
            kl_at_p = kl_pp.gather(1, pos_in_ctx.unsqueeze(1)).squeeze(1)
            kl_pp_mu = (orig_lp.exp() * (orig_lp - mb_lp)).sum(-1)
            kl_at_p_mu = kl_pp_mu.gather(1, pos_in_ctx.unsqueeze(1)).squeeze(1)

        totals["kl_avg"] += kl_avg.sum().item()
        totals["kl_at_p"] += kl_at_p.sum().item()
        totals["kl_p_baseline"] += kl_at_p_mu.sum().item()
        totals["n"] += kl_at_p.shape[0]
        pbar.set_postfix(kl_p=f"{totals['kl_at_p']/totals['n']:.3f}",
                         kl_p_mu=f"{totals['kl_p_baseline']/totals['n']:.3f}")
    handle.remove()
    del lm
    torch.cuda.empty_cache()

    n_done = totals["n"]
    res = {
        "mse": mse,
        "base_mse": base_mse,
        "fve": fve,
        "kl_future_avg": totals["kl_avg"] / n_done,
        "kl_at_p_recon": totals["kl_at_p"] / n_done,
        "kl_at_p_predict_mean": totals["kl_p_baseline"] / n_done,
        "n": n_done,
        "av_init": args.av_init,
        "ar_init": args.ar_init,
    }
    print("\n=== Result ===")
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
