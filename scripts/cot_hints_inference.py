"""Run CoT Hints inference: collect activations from Qwen2.5-0.5B-Instruct on each
transcript, then generate NLA explanations from control and e2e AV checkpoints.

Inputs:
  /tmp/cot_hints_dataset.json — list of 20 transcripts
  --ckpt-a, --ckpt-b — two AV checkpoints (matched-training pair)
Outputs:
  /tmp/cot_hints_explanations.json — each entry has transcript_id, regime, expl_A, expl_B
  /tmp/cot_hints_mapping.json — A/B → ckpt name
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl import build_av_prompt_embeds, ddp_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", default="checkpoints/rl_small_grpo_control300/av_step_300.pt")
    ap.add_argument("--ckpt-b", default="checkpoints/rl_small_grpo_e2e/av_step_300.pt")
    ap.add_argument("--label-a", default="control")
    ap.add_argument("--label-b", default="e2e")
    ap.add_argument("--dataset", default="/tmp/cot_hints_dataset.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/cot_hints_explanations.json")
    ap.add_argument("--mapping-out", default="/tmp/cot_hints_mapping.json")
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    torch.manual_seed(args.seed)

    transcripts = json.loads(Path(args.dataset).read_text())
    print(f"loaded {len(transcripts)} transcripts")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    # --- 1. Collect activations from subject (Qwen2.5-0.5B-Instruct) ---
    print(f"Loading subject: {cfg.base_model}")
    subject = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    subject.eval()

    activations = []  # [d] per transcript
    print("Collecting activations at the last prompt position")
    for t in tqdm(transcripts):
        # Tokenize with raw text (no chat template; the prompt IS the demonstration block + final Q)
        ids = tok(t["final_prompt"], return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
        with torch.no_grad():
            out = subject(input_ids=ids, output_hidden_states=True, use_cache=False)
        # Last token's hidden state at layer 16
        h = out.hidden_states[cfg.layer][0, -1]  # [d]
        activations.append(h.cpu().clone())
        # Also peek at what the subject's prediction is (sanity: does it match the bullet/answer letter?)
        logits_last = out.logits[0, -1].float()
        top_id = int(logits_last.argmax().item())
        top_tok = tok.decode([top_id])
        t["_subject_top_token"] = top_tok
    del subject
    torch.cuda.empty_cache()
    H = torch.stack(activations)  # [N, d]
    print(f"  shape: {H.shape}")

    # --- 2. Generate AV explanations from both checkpoints ---
    def gen(ckpt_path):
        av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        av.load_state_dict(sd.get("state_dict", sd))
        av.eval()
        h_dev = H.to(device)
        emb, mask = build_av_prompt_embeds(av, tok, h_dev, marker_id, cfg.alpha, device, torch.float32)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            gen_ids = ddp_generate(av, emb, mask, max_new_tokens=args.max_new_tokens,
                                   eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                                   temperature=1.0, top_p=0.95)
        gen_mask = (gen_ids == tok.eos_token_id).cumsum(1) <= 1
        del av
        torch.cuda.empty_cache()
        return [tok.decode(gen_ids[i, : int(gen_mask[i].sum().item())], skip_special_tokens=True)
                for i in range(gen_ids.shape[0])]

    print(f"\nGenerating from CKPT-A: {args.ckpt_a}")
    exps_a = gen(args.ckpt_a)
    print(f"\nGenerating from CKPT-B: {args.ckpt_b}")
    exps_b = gen(args.ckpt_b)

    # --- 3. Anonymize and save ---
    rng = random.Random(args.seed)
    out_records = []
    mapping = {}
    for t, ea, eb in zip(transcripts, exps_a, exps_b):
        rid = t["transcript_id"]
        flip = rng.random() < 0.5
        if flip:
            mapping[rid] = {"A": args.label_b, "B": args.label_a}
            f_a, f_b = eb, ea
        else:
            mapping[rid] = {"A": args.label_a, "B": args.label_b}
            f_a, f_b = ea, eb
        out_records.append({
            "id": rid,
            "regime": t["regime"],
            "final_question": t["final_question"],
            "subject_top_token": t.get("_subject_top_token", ""),
            "expl_A": f_a,
            "expl_B": f_b,
        })

    Path(args.out).write_text(json.dumps(out_records, indent=2, ensure_ascii=False))
    Path(args.mapping_out).write_text(json.dumps(mapping, indent=2))
    print(f"\nsaved {len(out_records)} → {args.out}")
    print(f"saved mapping → {args.mapping_out}")

    # Quick sanity print
    from collections import Counter
    print(f"regimes: {Counter(r['regime'] for r in out_records)}")
    print(f"subject top tokens: {Counter(r['subject_top_token'] for r in out_records)}")


if __name__ == "__main__":
    main()
