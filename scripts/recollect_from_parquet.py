"""Re-collect activations using docs from an existing parquet (avoids HF API).

Reads `data/activations_L16.parquet`, extracts unique documents (one per doc_id),
re-tokenizes with the new base model's tokenizer, samples positions, and writes
a new activation parquet.
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-parquet", default="data/activations_L16.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--max-tokens-per-doc", type=int, default=1024)
    ap.add_argument("--vectors-per-doc", type=int, default=5)
    ap.add_argument("--min-position", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--row-start", type=int, default=0)
    ap.add_argument("--row-end", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading source parquet: {args.source_parquet}")
    src = pq.read_table(args.source_parquet)
    df = src.select(["doc_id", "text"]).to_pandas().drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    print(f"  {len(df)} unique docs in source")

    start = max(0, args.row_start)
    end = args.row_end if args.row_end is not None else len(df)
    df = df.iloc[start:end].reset_index(drop=True)
    print(f"  this shard: docs [{start}, {end}) = {len(df)}")

    print(f"Loading {args.base_model} ({args.dtype})")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(args.device)
    model.eval()
    d = model.config.hidden_size
    print(f"  d_model = {d}, layer = {args.layer}")

    doc_ids, positions, seq_lens, texts, activations = [], [], [], [], []
    pbar = tqdm(total=len(df), desc=f"docs[{start},{end})")
    with torch.no_grad():
        for _, row in df.iterrows():
            text = row["text"]
            doc_id = int(row["doc_id"])
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_tokens_per_doc, add_special_tokens=False).input_ids.to(args.device)
            seq_len = int(ids.shape[1])
            valid_lo, valid_hi = args.min_position, seq_len - 1
            if valid_hi <= valid_lo:
                pbar.update(1)
                continue
            n_valid = valid_hi - valid_lo + 1
            k = min(args.vectors_per_doc, n_valid)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, doc_id]))
            picks = np.sort(rng.choice(n_valid, size=k, replace=False) + valid_lo)
            out = model(ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer][0]
            doc_text = tok.decode(ids[0, :seq_len], skip_special_tokens=False)
            for p in picks:
                doc_ids.append(doc_id)
                positions.append(int(p))
                seq_lens.append(seq_len)
                texts.append(doc_text)
                activations.append(h[p].float().cpu().numpy())
            pbar.update(1)
    pbar.close()

    n = len(doc_ids)
    print(f"Collected {n} vectors")
    if n == 0:
        return

    acts_np = np.stack(activations, axis=0).astype(np.float32)
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
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  activation norms: mean={np.linalg.norm(acts_np, axis=1).mean():.2f} "
          f"p75={np.quantile(np.linalg.norm(acts_np, axis=1), 0.75):.2f}")


if __name__ == "__main__":
    main()
