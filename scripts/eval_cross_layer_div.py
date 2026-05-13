"""Cross-layer residual divergence eval — does the patched-ĥ residual look in-
distribution to the upper layers, or does the perturbation get amplified?

Protocol (per Braun e2e SAE paper, adapted):
  1. AV → text → AR → ĥ for each record (same pipeline as eval_downstream_kl)
  2. Run frozen LM on ctx, capturing hidden_states at every upper layer
     (output of layers 16..23, i.e. hidden_states[17..24])
  3. Run frozen LM with ĥ patched at layer 16 position p; capture same hidden_states
  4. Report per-layer normalized divergence at position p:
        div[L] = || h_orig[L, p] − h_patched[L, p] || / || h_orig[L, p] ||
     across the layer stack — does ĥ-induced perturbation grow, shrink, or stabilize?
  5. Compare vanilla GRPO ckpt vs e2e step-300.

Quick smoke: ~30s on a single GPU for 128 records.
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
    ap.add_argument("--n-records", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm-dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    print(f"Loading AV: {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    sd = torch.load(args.av_init, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()

    print(f"Loading AR: {args.ar_init}")
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    sd_ar = torch.load(args.ar_init, map_location=device, weights_only=False)
    sd_ar = sd_ar.get("state_dict", sd_ar)
    ar_state = ar.state_dict()
    ar.load_state_dict({k: v for k, v in sd_ar.items() if k in ar_state}, strict=False)
    ar.eval()

    lm_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[args.lm_dtype]
    print(f"Loading LM (frozen, {args.lm_dtype})")
    lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=lm_dtype).to(device)
    lm.eval()
    patch_state = make_patch_state()
    patch_handle = lm.model.layers[cfg.layer - 1].register_forward_hook(make_patch_hook(patch_state))

    print("Loading records")
    t = pq.read_table(args.activations)
    n = len(t)
    flat = np.asarray(t["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = t["text"].to_pylist()
    positions = np.asarray(t["position"].to_pylist(), dtype=np.int64)
    n_eval = max(1, n // 20)
    eval_idx_all = list(range(n - n_eval, n))
    eval_idx = [i for i in eval_idx_all if positions[i] + 8 < args.max_ctx_tokens]
    eval_idx = eval_idx[: args.n_records]
    print(f"  {len(eval_idx)} eval records")

    # Layers we want to capture (post-block hidden states 17..24).
    # In HF, hidden_states[i] = output of layers[i-1] (0-indexed).
    # We need outputs of layers L through 23 (i.e. hidden_states[L+1..24]).
    L = cfg.layer  # 16
    capture_layers = list(range(L - 1, 24))  # 15..23 (indices into model.layers)
    # We'll also need hidden_states[16] (output of layer 15) as the comparison base — same
    # as the activation we patch. orig vs patched at L=16 should diverge by exactly ĥ vs h.

    # Stats accumulator: per layer-index, sum of ||delta||/||orig|| at position p
    n_layers = len(capture_layers)
    sum_div = np.zeros(n_layers)
    sum_n = 0
    # Also track unrelated-position divergence as a control: ||delta||/||orig||
    # averaged over positions NOT equal to p (should stay tiny — only attention propagation).
    sum_div_other = np.zeros(n_layers)
    sum_n_other = 0

    pbar = tqdm(range(0, len(eval_idx), args.batch_size))
    for b0 in pbar:
        b1 = min(b0 + args.batch_size, len(eval_idx))
        ids = eval_idx[b0:b1]
        h_batch = activations[ids].to(device)
        text_batch = [texts[i] for i in ids]
        pos_batch = positions[ids]

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

            ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
                tok, text_batch, pos_batch, args.max_ctx_tokens, device,
            )
            T_ctx = ctx_ids.shape[1]
            B = ctx_ids.shape[0]

            # Orig forward (no patch), grab hidden states
            patch_state["h_hat"] = None
            orig_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False,
                          output_hidden_states=True)
            orig_hs = orig_out.hidden_states  # tuple, indices 0..24 (25 entries)

            # Patched forward
            patch_state["h_hat"] = pred.to(lm_dtype)
            patch_state["positions"] = pos_in_ctx
            pat_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False,
                         output_hidden_states=True)
            patch_state["h_hat"] = None
            pat_hs = pat_out.hidden_states

            # Compare at hidden_states[L..24] (i.e. output of layers L-1..23).
            # Note: HF's @check_model_inputs captures hidden_states BEFORE forward
            # hooks fire (see transformers/utils/generic.py wrapped_forward), so
            # pat_hs[L] at position p shows the pre-hook value (= original h) — NOT
            # the patched ĥ. The patched ĥ is `pred` directly. Downstream layers
            # (L+1 onwards) capture correctly since they see the patched residual
            # as their input.
            for li, layer_idx in enumerate(capture_layers):
                hs_idx = layer_idx + 1  # output of layers[layer_idx] = hidden_states[layer_idx+1]
                if hs_idx == L:
                    # Layer-L capture is broken by hook ordering — use ĥ directly.
                    o_at_p = h_batch.float()  # [B, d]
                    p_at_p = pred.float()
                else:
                    o = orig_hs[hs_idx].float()  # [B, T_ctx, d]
                    p = pat_hs[hs_idx].float()
                    o_at_p = o.gather(1, pos_in_ctx.view(-1, 1, 1).expand(-1, 1, d)).squeeze(1)
                    p_at_p = p.gather(1, pos_in_ctx.view(-1, 1, 1).expand(-1, 1, d)).squeeze(1)
                pos_diff = p_at_p - o_at_p
                div_p = pos_diff.norm(dim=-1) / o_at_p.norm(dim=-1).clamp_min(1e-6)  # [B]
                sum_div[li] += div_p.sum().item()
                # At "other" positions: only meaningful at hs[L+1] onwards where the
                # captured value reflects propagation. At hs[L] the captured pat_hs
                # equals orig (pre-hook) so we just record 0 — actual other-position
                # divergence at hs[L] IS 0 since patch only modifies position p.
                if hs_idx == L:
                    pass  # leave sum_div_other[li] at 0
                else:
                    diff_full = p - o  # [B, T_ctx, d]
                    row_idx = torch.arange(B, device=device)
                    valid_other = ctx_mask.bool().clone()
                    valid_other[row_idx, pos_in_ctx] = False
                    ratio = diff_full.norm(dim=-1) / o.norm(dim=-1).clamp_min(1e-6)
                    m_f = valid_other.float()
                    per_row = (ratio * m_f).sum(1) / m_f.sum(1).clamp_min(1.0)
                    sum_div_other[li] += per_row.sum().item()
        sum_n += B
        sum_n_other += B
        pbar.set_postfix(at_p=f"{sum_div[-1]/sum_n:.3f}", other=f"{sum_div_other[-1]/sum_n_other:.4f}")

    mean_div_at_p = sum_div / sum_n
    mean_div_other = sum_div_other / sum_n_other

    res = {
        "ar_init": args.ar_init,
        "av_init": args.av_init,
        "n_records": sum_n,
        "layer_indices_0based": capture_layers,  # 15..23 → outputs are hidden_states[16..24]
        "hidden_state_indices": [li + 1 for li in capture_layers],  # 16..24
        "div_at_p": mean_div_at_p.tolist(),
        "div_other_positions": mean_div_other.tolist(),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"saved → {args.out}")
    patch_handle.remove()


if __name__ == "__main__":
    main()
