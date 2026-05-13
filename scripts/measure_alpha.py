"""Measure α: the 75th-percentile L2 norm of layer-ℓ residual activations,
sampled from a pretraining-like corpus.

Why: α is the injection scale that the AV uses to splice an activation into
its prompt (replacing one token's input embedding with α · ĥ). The paper
appendix recommends α ≈ Quantile_0.75(‖h_ℓ‖) over a corpus.

Outputs:
  - prints 50/75/90/95/99 percentiles + std
  - writes a histogram + the chosen alpha to data/alpha_L{layer}.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="(unused — defaults from NLAConfig)")
    ap.add_argument("--num-docs", type=int, default=2000)
    ap.add_argument("--max-tokens-per-doc", type=int, default=512,
                    help="truncate long docs to keep forward cheap")
    ap.add_argument("--vectors-per-doc", type=int, default=5)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb",
                    help="HF dataset id; uses streaming")
    ap.add_argument("--dataset-config", default="sample-10BT",
                    help="dataset config name (set to '' for default)")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = NLAConfig()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {cfg.base_model} on {args.device} ({args.dtype})")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).to(args.device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    assert cfg.layer <= n_layers, f"layer={cfg.layer} > num_hidden_layers={n_layers}"
    assert cfg.d_model == d_model, f"config d_model={cfg.d_model} != model {d_model}"
    print(f"  {n_layers} layers, d_model={d_model}, extracting at hidden_states[{cfg.layer}]")

    print(f"Streaming {args.dataset} ({args.dataset_config or 'default'})")
    ds_kwargs = {"streaming": True, "split": "train"}
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, **ds_kwargs)
    else:
        ds = load_dataset(args.dataset, **ds_kwargs)

    norms = []
    pbar = tqdm(total=args.num_docs, desc="docs")
    n_seen = 0
    with torch.no_grad():
        for ex in ds:
            if n_seen >= args.num_docs:
                break
            text = ex.get(args.text_key)
            if not text or len(text) < 200:
                continue
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_tokens_per_doc).input_ids.to(args.device)
            seq_len = ids.shape[1]
            if seq_len <= cfg.min_position + 1:
                continue
            out = model(ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[cfg.layer][0]  # [seq, d]
            valid = torch.arange(cfg.min_position, seq_len, device=h.device)
            if valid.numel() == 0:
                continue
            k = min(args.vectors_per_doc, valid.numel())
            picks = valid[torch.from_numpy(rng.choice(valid.numel(), size=k, replace=False)).to(h.device)]
            sel = h[picks].float()  # [k, d]
            norms.append(sel.norm(dim=-1).cpu().numpy())
            n_seen += 1
            pbar.update(1)
    pbar.close()

    norms = np.concatenate(norms)
    pcts = {q: float(np.quantile(norms, q)) for q in (0.50, 0.75, 0.90, 0.95, 0.99)}
    stats = {
        "base_model": cfg.base_model,
        "layer": cfg.layer,
        "n_vectors": int(norms.size),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "percentiles": pcts,
        "alpha_recommended": pcts[0.75],
    }
    print("\n=== Activation norm percentiles at layer", cfg.layer, "===")
    for q, v in pcts.items():
        print(f"  p{int(q*100):02d}: {v:.4f}")
    print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}  n={stats['n_vectors']}")
    print(f"\n=> alpha = p75 = {stats['alpha_recommended']:.4f}")

    out_dir = Path(cfg.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alpha_L{cfg.layer}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    np.save(out_dir / f"norms_L{cfg.layer}.npy", norms)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
