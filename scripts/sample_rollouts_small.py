"""Sample rollouts from a 0.5B AV checkpoint with α-injection at the marker.

Single-GPU, reads activations + their source texts from a parquet, generates
explanations, and prints source-tail / rollout side-by-side. Optionally evals
reconstruction via a 0.5B AR checkpoint (ARModel).
"""
import argparse
import re

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import NLAConfig
from nla.model import ARModel
from nla.prompts import build_av_messages


def load_sd(path, device):
    x = torch.load(path, map_location=device, weights_only=False)
    return x["state_dict"] if isinstance(x, dict) and "state_dict" in x else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-checkpoint", required=True)
    ap.add_argument("--ar-checkpoint", default=None,
                    help="If supplied, also reports per-rollout reconstruction MSE/FVE.")
    ap.add_argument("--base-model", default=None,
                    help="Override cfg.base_model (e.g. 'Qwen/Qwen2.5-0.5B').")
    ap.add_argument("--no-prompt", action="store_true",
                    help="Skip chat template; AV input is just [α·ĥ] at position 0.")
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--use-eval-set", action="store_true",
                    help="Sample from held-out top 5% (matches training eval set).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = NLAConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    device, dtype = args.device, torch.float32
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    print(f"Loading AV: base={cfg.base_model} + {args.av_checkpoint}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).to(device).eval()
    av.load_state_dict(load_sd(args.av_checkpoint, device))

    ar = None
    if args.ar_checkpoint:
        print(f"Loading AR: identity-init + {args.ar_checkpoint}")
        ar = ARModel(cfg, dtype=dtype).to(device).eval()
        ar.load_state_dict(load_sd(args.ar_checkpoint, device))

    print(f"Loading activations: {args.activations}")
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = table["text"].to_pylist() if "text" in table.column_names else [None] * n
    doc_ids = table["doc_id"].to_pylist()
    positions = table["position"].to_pylist()

    n_eval = max(1, n // 20)
    pool = list(range(n - n_eval, n)) if args.use_eval_set else list(range(n))
    idxs = rng.choice(pool, size=args.n_samples, replace=False).tolist()
    h_batch = activations[idxs].to(device)

    # Build AV input
    B = h_batch.shape[0]
    h_unit = F.normalize(h_batch.float(), dim=-1)
    inj = (cfg.alpha * h_unit).to(dtype)
    if args.no_prompt:
        embeds = inj.unsqueeze(1)  # [B, 1, d]
        attn_mask = torch.ones((B, 1), dtype=torch.long, device=device)
    else:
        msgs = build_av_messages(cfg.marker_token)
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
        pos = (prompt_ids_b == marker_id).float().argmax(dim=1)
        embeds = av.get_input_embeddings()(prompt_ids_b).clone()
        embeds[torch.arange(B, device=device), pos] = inj
        attn_mask = torch.ones_like(prompt_ids_b)

    print(f"\nGenerating {B} rollouts (max_new_tokens={args.max_new_tokens})...")
    with torch.no_grad():
        gen = av.generate(
            inputs_embeds=embeds, attention_mask=attn_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
        )
    eos_id = tok.eos_token_id
    rollout_texts = []
    rollout_lens = []
    for i in range(B):
        ids = gen[i].tolist()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id) + 1]
            rollout_lens.append(ids.index(eos_id) + 1)
        else:
            rollout_lens.append(len(ids))
        rollout_texts.append(tok.decode(ids, skip_special_tokens=True))

    print(f"  rollout lengths: min={min(rollout_lens)} max={max(rollout_lens)} mean={sum(rollout_lens)/len(rollout_lens):.1f}")

    # Optional: reconstruction via AR
    fve_per = None
    if ar is not None:
        AR_PREFIX = "Summary of the following text: <text>"
        AR_SUFFIX = "</text> <summary>"
        sc = cfg.mse_norm
        ar_texts = [f"{AR_PREFIX}{r}{AR_SUFFIX}" for r in rollout_texts]
        enc = tok(ar_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ar_ids = enc.input_ids.to(device)
        ar_mask = enc.attention_mask.to(device)
        with torch.no_grad():
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask)
        p_norm = F.normalize(pred.float(), dim=-1) * sc
        g_norm = F.normalize(h_batch.float(), dim=-1) * sc
        per_mse = ((p_norm - g_norm) ** 2).mean(dim=-1)
        mu = activations.mean(dim=0).to(device).to(dtype)
        base_p = F.normalize(mu.expand_as(h_batch).float(), dim=-1) * sc
        base_per = ((base_p - g_norm) ** 2).mean(dim=-1)
        fve_per = 1 - per_mse / base_per
        print(f"  overall: mean_mse={per_mse.mean().item():.5f}  mean_fve={fve_per.mean().item():+.3f}")

    print("\n=== Side-by-side ===")
    for i in range(B):
        idx = idxs[i]
        src_text = texts[idx]
        msg = f"\n[{i}] doc={doc_ids[idx]} pos={positions[idx]} len={rollout_lens[i]}"
        if fve_per is not None:
            msg += f" fve={fve_per[i].item():+.3f}"
        print(msg)
        if src_text:
            print(f"  source tail: …{src_text[-180:]!r}")
        print(f"  AV rollout:  {rollout_texts[i][:500]!r}")


if __name__ == "__main__":
    main()
