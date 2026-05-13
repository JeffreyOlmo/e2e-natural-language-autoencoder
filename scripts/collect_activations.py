"""Collect (doc_id, position, text, activation@layer) records into a parquet.

Invariants (from paper repo):
  - Activations stored RAW (not L2-normalized). Norm happens at injection
    (α·ĥ) and at loss (mse_norm). Don't pre-normalize here.
  - min_position = 50: earlier positions decode to noise (insufficient context).
  - Per-doc keyed RNG via np.random.SeedSequence([seed, doc_id]) so the same
    (seed, doc_id) deterministically picks the same positions regardless of
    process count or stream order. This is what makes future multi-GPU
    collection bit-reproducible without coordination.
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=5000)
    ap.add_argument("--max-tokens-per-doc", type=int, default=1024)
    ap.add_argument("--vectors-per-doc", type=int, default=5)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--dataset-config", default="sample-10BT")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-model", default=None, help="override NLAConfig.base_model")
    ap.add_argument("--layer", type=int, default=None, help="override NLAConfig.layer")
    ap.add_argument("--row-start", type=int, default=0, help="for sharded multi-GPU runs (doc index range)")
    ap.add_argument("--row-end", type=int, default=None)
    args = ap.parse_args()

    cfg = NLAConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    if args.layer is not None:
        cfg.layer = args.layer
    out_path = Path(args.out) if args.out else Path(cfg.data_dir) / f"activations_L{cfg.layer}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.base_model} on {args.device} ({args.dtype})")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).to(args.device)
    model.eval()
    d = model.config.hidden_size
    cfg.d_model = d  # auto-update; we may have overridden base_model

    print(f"Streaming {args.dataset} ({args.dataset_config or 'default'})")
    ds_kwargs = {"streaming": True, "split": "train"}
    ds = (
        load_dataset(args.dataset, args.dataset_config, **ds_kwargs)
        if args.dataset_config
        else load_dataset(args.dataset, **ds_kwargs)
    )

    doc_ids, positions, seq_lens, texts, activations = [], [], [], [], []
    doc_id = -1
    n_collected = 0
    target_start = max(0, args.row_start)
    target_end = args.row_end if args.row_end is not None else args.num_docs
    pbar = tqdm(total=target_end - target_start, desc=f"docs[{target_start},{target_end})")
    with torch.no_grad():
        for ex in ds:
            text = ex.get(args.text_key)
            if not text or len(text) < 200:
                continue
            doc_id += 1
            if doc_id >= target_end:
                break
            if doc_id < target_start:
                continue
            ids = tok(
                text, return_tensors="pt", truncation=True, max_length=args.max_tokens_per_doc
            ).input_ids.to(args.device)
            seq_len = int(ids.shape[1])
            valid_lo, valid_hi = cfg.min_position, seq_len - 1
            if valid_hi <= valid_lo:
                continue
            n_valid = valid_hi - valid_lo + 1
            k = min(args.vectors_per_doc, n_valid)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, doc_id]))
            picks = np.sort(rng.choice(n_valid, size=k, replace=False) + valid_lo)
            out = model(ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[cfg.layer][0]  # [seq, d]
            doc_text = tok.decode(ids[0, :seq_len], skip_special_tokens=False)
            for p in picks:
                doc_ids.append(doc_id)
                positions.append(int(p))
                seq_lens.append(seq_len)
                texts.append(doc_text)
                activations.append(h[p].float().cpu().numpy())
            n_collected += 1
            pbar.update(1)
    pbar.close()

    n = len(doc_ids)
    print(f"Collected {n} vectors from {n_collected} docs")
    if n == 0:
        print("No vectors — check dataset/min_position/max_tokens.")
        return

    # Pack activations as a fixed-size list of float32 — efficient, decodes back to [N, d]
    acts_np = np.stack(activations, axis=0).astype(np.float32)  # [N, d]
    flat = pa.array(acts_np.reshape(-1), type=pa.float32())
    fsl = pa.FixedSizeListArray.from_arrays(flat, d)
    table = pa.table({
        "doc_id": pa.array(doc_ids, type=pa.int64()),
        "position": pa.array(positions, type=pa.int32()),
        "seq_len": pa.array(seq_lens, type=pa.int32()),
        "text": pa.array(texts, type=pa.string()),
        "activation": fsl,
    })
    pq.write_table(table, out_path, compression="zstd")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")
    print(f"  schema: {table.schema}")
    print(f"  activation norms: mean={np.linalg.norm(acts_np, axis=1).mean():.3f} "
          f"std={np.linalg.norm(acts_np, axis=1).std():.3f}")


if __name__ == "__main__":
    main()
