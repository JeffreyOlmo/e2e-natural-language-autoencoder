"""Generate matched-pair (source, explanation_A, explanation_B) records for grading.

For each eval record, produce explanations from two AV checkpoints (control and e2e)
on the SAME activation, with anonymized A/B labels per record. Saves:
  - /tmp/grading_pairs.json   — list of {id, source_window, expl_A, expl_B}
  - /tmp/grading_mapping.json — {id -> {'A': ckpt_name, 'B': ckpt_name}}

Subagent grades by ID without knowing which is which.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl import build_av_prompt_embeds, ddp_generate


def build_source_window(text, position, tok, window_pre=120, window_post=40):
    ids = tok(text, add_special_tokens=False).input_ids
    start = max(0, position - window_pre)
    end = min(len(ids), position + window_post)
    pre = tok.decode(ids[start:position], skip_special_tokens=True)
    post = tok.decode(ids[position : end], skip_special_tokens=True)
    return f"{pre} <<HERE>> {post}"


def gen_explanations(av_path, h_batches, tok, marker_id, alpha, device, max_new_tokens=130, seed=0):
    print(f"  loading {av_path}")
    av = AutoModelForCausalLM.from_pretrained(NLAConfig().base_model, torch_dtype=torch.float32).to(device)
    sd = torch.load(av_path, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()
    out = []
    for h_batch in tqdm(h_batches, desc=f"AV {Path(av_path).parent.name}"):
        emb, mask = build_av_prompt_embeds(av, tok, h_batch.to(device), marker_id, alpha, device, torch.float32)
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = ddp_generate(av, emb, mask, max_new_tokens=max_new_tokens,
                               eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                               temperature=1.0, top_p=0.95)
        gen_mask = (gen == tok.eos_token_id).cumsum(1) <= 1
        for bi in range(gen.shape[0]):
            text = tok.decode(gen[bi, : int(gen_mask[bi].sum().item())], skip_special_tokens=True)
            out.append(text)
    del av
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", default="checkpoints/rl_small_grpo_control300/av_step_300.pt",
                    help="First checkpoint to compare")
    ap.add_argument("--ckpt-b", default="checkpoints/rl_small_grpo_e2e/av_step_300.pt",
                    help="Second checkpoint to compare")
    ap.add_argument("--label-a", default="control")
    ap.add_argument("--label-b", default="e2e")
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-pairs", default="/tmp/grading_pairs.json")
    ap.add_argument("--out-mapping", default="/tmp/grading_mapping.json")
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

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
    eval_idx = [i for i in eval_idx_all if positions[i] + 8 < args.max_ctx_tokens][: args.n]
    print(f"  {len(eval_idx)} records")

    # Batch the activations
    h_batches = []
    for i in range(0, len(eval_idx), args.batch_size):
        ids = eval_idx[i : i + args.batch_size]
        h_batches.append(activations[ids])

    # Generate explanations from both
    print("\nGenerating explanations from CKPT-A")
    exps_a = gen_explanations(args.ckpt_a, h_batches, tok, marker_id, cfg.alpha, device,
                              args.max_new_tokens, args.seed)
    print("\nGenerating explanations from CKPT-B")
    exps_b = gen_explanations(args.ckpt_b, h_batches, tok, marker_id, cfg.alpha, device,
                              args.max_new_tokens, args.seed)

    # Build source windows
    sources = [build_source_window(texts[i], int(positions[i]), tok, 120, 40)
               for i in eval_idx]

    # Anonymize per record: randomly assign A/B → (control, e2e) or (e2e, control)
    rng = random.Random(args.seed)
    pairs = []
    mapping = {}
    for k, (rec_id, src, ea, eb) in enumerate(zip(eval_idx, sources, exps_a, exps_b)):
        rid = f"rec_{k:03d}"
        # 50/50 randomize whether A in the file = ckpt-a or ckpt-b
        flip = rng.random() < 0.5
        if flip:
            file_a, file_b = eb, ea
            mapping[rid] = {"A": args.label_b, "B": args.label_a}
        else:
            file_a, file_b = ea, eb
            mapping[rid] = {"A": args.label_a, "B": args.label_b}
        pairs.append({
            "id": rid,
            "record_index": int(rec_id),
            "position": int(positions[rec_id]),
            "source_window": src,
            "expl_A": file_a,
            "expl_B": file_b,
        })

    Path(args.out_pairs).write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    Path(args.out_mapping).write_text(json.dumps(mapping, indent=2))
    print(f"\nsaved {len(pairs)} pairs → {args.out_pairs}")
    print(f"saved mapping → {args.out_mapping}")


if __name__ == "__main__":
    main()
