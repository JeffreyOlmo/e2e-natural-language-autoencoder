"""Generate kitft-quality summaries by running kitft 7B AV with α-injection on
activations_L20.parquet. Each (doc_id, position) gets one summary, paired
implicitly with the 0.5B activation in activations_L16.parquet (same docs/positions).

Output: data/summaries_kitft.parquet with columns (doc_id, position, summary, raw_rollout).

Run with torchrun for data-parallel sharding (each rank handles 1/world of rows):
  torchrun --standalone --nproc_per_node=N scripts/generate_kitft_summaries.py [args]
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


KITFT_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
CFG = {
    "alpha": 150.0,
    "marker_token": "㈎",
    "marker_token_id": 149705,
    "av_template": (
        "You are a meticulous AI researcher conducting an important investigation into "
        "activation vectors from a language model. Your overall task is to describe the "
        "semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. You must "
        "then produce an explanation for the vector, enclosed within <explanation> tags. "
        "The explanation consists of 2-3 text snippets describing that vector.\n\n"
        "Here is the vector:\n\n"
        "<concept>{marker}</concept>\n\n"
        "Please provide an explanation."
    ),
}


def extract_explanation(text):
    m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--out", default="data/summaries_kitft.parquet")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Lower temperature for cleaner / more reliable outputs.")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=None,
                    help="Manual rank (skip torch.distributed). Pairs with --world.")
    ap.add_argument("--world", type=int, default=None,
                    help="Manual world size (skip torch.distributed). Pairs with --rank.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip generation, just merge existing per-rank shards into --out.")
    args = ap.parse_args()

    if args.merge_only:
        out_path = Path(args.out)
        parts = sorted(out_path.parent.glob(f"{out_path.stem}.rank*{out_path.suffix}"))
        print(f"Merging {len(parts)} shards → {out_path}")
        merged = pa.concat_tables([pq.read_table(p) for p in parts])
        pq.write_table(merged, out_path, compression="zstd")
        print(f"  total rows: {len(merged)}")
        return

    use_dist = args.rank is None
    if use_dist:
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        local = int(os.environ.get("LOCAL_RANK", 0))
        if world > 1:
            dist.init_process_group(backend="nccl")
    else:
        rank = args.rank
        world = args.world
        local = 0  # CUDA_VISIBLE_DEVICES already restricts to one GPU per process
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    torch.manual_seed(args.seed + rank)
    dtype = torch.float16

    is_main = (rank == 0)
    if is_main:
        print(f"Generating kitft summaries: world_size={world}")

    # Load activations
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d_act = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d_act).copy())
    doc_ids = table["doc_id"].to_pylist()
    positions = table["position"].to_pylist()

    # Shard rows across ranks
    rows_per_rank = (n + world - 1) // world
    start = rank * rows_per_rank
    end = min((rank + 1) * rows_per_rank, n)
    indices = list(range(start, end))
    if is_main:
        print(f"  n_total={n}, rows_per_rank={rows_per_rank}, this rank: [{start}, {end})")

    # Load kitft AV
    if is_main:
        print(f"Loading {KITFT_AV_REPO}")
    tok = AutoTokenizer.from_pretrained(KITFT_AV_REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(KITFT_AV_REPO, torch_dtype=dtype).to(device).eval()

    # Pre-build prompt
    chat_msgs = [{"role": "user", "content": CFG["av_template"].format(marker=CFG["marker_token"])}]
    prompt_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids_single = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    P_marker = (prompt_ids_single == CFG["marker_token_id"]).float().argmax(dim=1).item()
    if is_main:
        print(f"  prompt length: {prompt_ids_single.shape[1]}, marker pos: {P_marker}")

    # Output buffers (per-rank)
    out_doc_ids, out_positions, out_summaries, out_raw = [], [], [], []

    bs = args.batch_size
    pbar = tqdm(range(0, len(indices), bs), desc=f"rank{rank}", disable=not is_main)
    for batch_start in pbar:
        batch_idxs = indices[batch_start : batch_start + bs]
        B = len(batch_idxs)
        h_batch = activations[batch_idxs].to(device).to(dtype)

        prompt_ids_b = prompt_ids_single.expand(B, -1).contiguous()
        h_unit = F.normalize(h_batch.float(), dim=-1)
        inj = (CFG["alpha"] * h_unit).to(dtype)
        embeds = av.get_input_embeddings()(prompt_ids_b).clone()
        embeds[torch.arange(B, device=device), P_marker] = inj
        attn_mask = torch.ones_like(prompt_ids_b)

        with torch.no_grad():
            gen = av.generate(
                inputs_embeds=embeds, attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
            )
        eos_id = tok.eos_token_id
        for i in range(B):
            ids = gen[i].tolist()
            if eos_id in ids:
                ids = ids[: ids.index(eos_id) + 1]
            text = tok.decode(ids, skip_special_tokens=True)
            summary = extract_explanation(text)
            out_doc_ids.append(doc_ids[batch_idxs[i]])
            out_positions.append(positions[batch_idxs[i]])
            out_summaries.append(summary)
            out_raw.append(text)

    # Save per-rank parquet
    out_path = Path(args.out)
    rank_path = out_path.parent / f"{out_path.stem}.rank{rank}{out_path.suffix}"
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    table_out = pa.table({
        "doc_id": pa.array(out_doc_ids, type=pa.int64()),
        "position": pa.array(out_positions, type=pa.int32()),
        "summary": pa.array(out_summaries, type=pa.string()),
        "raw_rollout": pa.array(out_raw, type=pa.string()),
    })
    pq.write_table(table_out, rank_path, compression="zstd")
    if is_main:
        print(f"\n  rank 0 wrote {rank_path} ({len(out_doc_ids)} rows)")

    if use_dist and world > 1:
        dist.barrier()

    # Merge on rank 0 (only when using torch.distributed; otherwise merge externally with --merge-only)
    if is_main and use_dist:
        parts = sorted(out_path.parent.glob(f"{out_path.stem}.rank*{out_path.suffix}"))
        print(f"Merging {len(parts)} shards → {out_path}")
        merged = pa.concat_tables([pq.read_table(p) for p in parts])
        pq.write_table(merged, out_path, compression="zstd")
        print(f"  total rows: {len(merged)}")
        for p in parts:
            p.unlink()

    if use_dist and world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
