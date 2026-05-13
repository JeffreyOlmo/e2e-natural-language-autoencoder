"""Two-hop bridge-entity NLA eval.

For each record with a verified first-hop-capable 0.5B subject, test three prompts:
  (A) subject_cut: "The author of Nineteen Eighty-Four"           -- first-hop description
  (B) full two-hop: "...was born in the city of"                  -- bridge is implicit
  (C) cot: "The author of N.E.F. is George Orwell. ...born in..." -- bridge made explicit

Collect activation at last position, run AV → explanation, check if explanation
mentions the bridge entity (alias-aware fuzzy match).

Metrics per ckpt × condition:
  bridge_mention_rate = fraction of explanations that mention the bridge
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl import build_av_prompt_embeds, ddp_generate


def check_bridge_mention(text, aliases):
    """Case-insensitive whole-word match against any alias."""
    text_lc = text.lower()
    for a in aliases:
        a = a.strip()
        if not a:
            continue
        # whole-word match — accept partial token for multi-word names
        # (e.g. "Orwell" alone counts as a match for "George Orwell")
        parts = a.lower().split()
        # The MOST distinctive token in the alias is usually the surname / last token.
        # Match if any 2+ char token from the alias appears whole-word in the text.
        for p in parts:
            if len(p) < 4:
                continue
            if re.search(rf"\b{re.escape(p)}\b", text_lc):
                return True, a
        # Also try full alias as substring
        if a.lower() in text_lc:
            return True, a
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", default="checkpoints/rl_small_grpo_control300/av_step_300.pt")
    ap.add_argument("--ckpt-b", default="checkpoints/rl_small_grpo_e2e/av_step_300.pt")
    ap.add_argument("--label-a", default="control")
    ap.add_argument("--label-b", default="e2e")
    ap.add_argument("--capable-records", default="/tmp/twohop_capable.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/twohop_eval.json")
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    torch.manual_seed(args.seed)

    capable = json.loads(Path(args.capable_records).read_text())
    print(f"loaded {len(capable)} first-hop-capable records")

    ds = load_dataset("soheeyang/TwoHopFact", split="train")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    # Build the 3 prompts per record
    records = []
    for rec in capable:
        i = rec["index"]
        row = ds[i]
        e2_first = eval(row["e2.aliases"])[0][0]
        records.append({
            "index": i,
            "type": rec["type"],
            "bridge_aliases": rec["bridge_aliases"],
            "answer_aliases": rec["answer_aliases"],
            "prompts": {
                "subj_cut": row["r1(e1).subject_cut.prompt"],
                "two_hop":  row["r2(r1(e1)).prompt"],
                "cot":      row["cot.r1(e1).prompt"],
            },
        })

    # --- Collect activations from subject for each (record, condition) ---
    print(f"Loading subject: {cfg.base_model}")
    subject = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    subject.eval()

    conds = ["subj_cut", "two_hop", "cot"]
    h_per_cond = {c: [] for c in conds}
    for r in tqdm(records, desc="collect h"):
        for c in conds:
            prompt = r["prompts"][c]
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = subject(input_ids=ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[cfg.layer][0, -1].cpu()
            h_per_cond[c].append(h)
    del subject; torch.cuda.empty_cache()

    # --- Generate AV explanations ---
    def gen_av(ckpt_path, h_list):
        av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        av.load_state_dict(sd.get("state_dict", sd))
        av.eval()
        H = torch.stack(h_list).to(device)
        emb, mask = build_av_prompt_embeds(av, tok, H, marker_id, cfg.alpha, device, torch.float32)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            gen = ddp_generate(av, emb, mask, max_new_tokens=args.max_new_tokens,
                               eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                               temperature=1.0, top_p=0.95)
        gen_mask = (gen == tok.eos_token_id).cumsum(1) <= 1
        del av; torch.cuda.empty_cache()
        return [tok.decode(gen[i, : int(gen_mask[i].sum().item())], skip_special_tokens=True)
                for i in range(gen.shape[0])]

    results = {(ck, cd): [] for ck in (args.label_a, args.label_b) for cd in conds}
    for ck, path in ((args.label_a, args.ckpt_a), (args.label_b, args.ckpt_b)):
        for cd in conds:
            print(f"\nAV {ck} on {cd}")
            results[(ck, cd)] = gen_av(path, h_per_cond[cd])

    # --- Grade: check bridge mention ---
    summary = {}
    detail = []
    for ck in (args.label_a, args.label_b):
        for cd in conds:
            n_total = 0
            n_mention = 0
            for r, expl in zip(records, results[(ck, cd)]):
                hit, matched = check_bridge_mention(expl, r["bridge_aliases"])
                n_total += 1
                if hit:
                    n_mention += 1
            rate = n_mention / max(1, n_total)
            summary[f"{ck}_{cd}_mention_rate"] = rate
            summary[f"{ck}_{cd}_n_mention"] = n_mention
            summary[f"{ck}_{cd}_n_total"] = n_total

    for i, r in enumerate(records):
        d = {"index": r["index"], "type": r["type"], "bridge_aliases": r["bridge_aliases"]}
        for ck in (args.label_a, args.label_b):
            for cd in conds:
                e = results[(ck, cd)][i]
                hit, matched = check_bridge_mention(e, r["bridge_aliases"])
                d[f"{ck}_{cd}_explanation"] = e[:400]
                d[f"{ck}_{cd}_bridge_hit"] = hit
                d[f"{ck}_{cd}_match"] = matched or ""
        detail.append(d)

    out = {"summary": summary, "detail": detail,
           "records": len(records),
           "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== Bridge-mention summary ===")
    for ck in (args.label_a, args.label_b):
        for cd in conds:
            r = summary[f"{ck}_{cd}_mention_rate"]
            n = summary[f"{ck}_{cd}_n_mention"]
            t = summary[f"{ck}_{cd}_n_total"]
            print(f"  {ck:<10} {cd:<10}  {n}/{t} = {r:.3f}")
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
