"""Factual-accuracy / confabulation eval, NLA-paper adapted.

For each (h, source_context, position) record:
  1. Generate AV explanation from the activation.
  2. Build a grader prompt with: source text (window around p) + AV explanation.
  3. Ask Qwen2.5-7B-Instruct grader to enumerate distinct claims and verdict each as
     SUPPORTED / CONTRADICTED / NEUTRAL given the context.
  4. Parse JSON; aggregate.

Metrics (mean across records):
  - claims_per_expl: total claims found per explanation
  - frac_supported, frac_contradicted, frac_neutral
  - supported_claims_per_expl: claims_per_expl × frac_supported
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl import build_av_prompt_embeds, ddp_generate


GRADER_SYSTEM = (
    "You are a meticulous, careful evaluator. You will be given a piece of source text "
    "and an 'explanation' that describes the text around a specific position in it. Your job "
    "is to enumerate the distinct factual claims in the explanation and verdict each one against "
    "the source text. Output ONLY JSON."
)

GRADER_USER_TEMPLATE = """Source text (the actual document; a '<<HERE>>' marker shows the position whose surrounding content the explanation is about):

<source>
{source}
</source>

Explanation to evaluate:

<explanation>
{explanation}
</explanation>

Instructions:
1. Read the explanation. List the distinct factual claims it makes about the source text. A factual claim is a specific assertion about: topic, format, genre, named entities, structural elements, surrounding content, the next likely token/phrase, or specific descriptive properties. Skip vague filler ("informal tone", "structured layout") unless they are clearly assertions.
2. For each claim, decide its verdict against the source text:
   - "supported" — the claim is directly evident or clearly inferable from the source.
   - "contradicted" — the source text clearly shows the opposite.
   - "neutral" — neither supported nor contradicted; you can't tell from the source.
3. Output a single JSON object — no prose, no markdown fences:

{{"claims": [{{"claim": "<the claim>", "verdict": "supported|contradicted|neutral"}}, ...]}}

If the explanation contains no claims, output {{"claims": []}}.
"""


def build_source_window(text, position, tok, window_pre=120, window_post=40):
    """Return a string that contains tokens around position p with a marker at p."""
    ids = tok(text, add_special_tokens=False).input_ids
    start = max(0, position - window_pre)
    end = min(len(ids), position + window_post)
    pre = tok.decode(ids[start:position], skip_special_tokens=True)
    post = tok.decode(ids[position : end], skip_special_tokens=True)
    # Mark position p with <<HERE>>
    return f"{pre} <<HERE>> {post}"


def parse_grader_output(s):
    """Robust JSON parse from grader output. Returns claims list or None."""
    s = s.strip()
    # Strip markdown fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Find the first { and last } to bracket the JSON
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j < 0 or j <= i:
        return None
    blob = s[i : j + 1]
    try:
        d = json.loads(blob)
        claims = d.get("claims", [])
        # Validate shape
        out = []
        for c in claims:
            if isinstance(c, dict) and "verdict" in c:
                v = c["verdict"].lower().strip()
                if v in {"supported", "contradicted", "neutral"}:
                    out.append({"claim": str(c.get("claim", ""))[:200], "verdict": v})
        return out
    except json.JSONDecodeError:
        return None


def grade_batch(grader, grader_tok, sources, explanations, device, max_new_tokens=1024,
                batch_size=2):
    """Run the grader over (source, explanation) pairs. Returns list[list[claim_dict] | None]."""
    results = []
    for i in tqdm(range(0, len(sources), batch_size), desc="grader"):
        batch_src = sources[i : i + batch_size]
        batch_exp = explanations[i : i + batch_size]
        prompts = []
        for s, e in zip(batch_src, batch_exp):
            msgs = [
                {"role": "system", "content": GRADER_SYSTEM},
                {"role": "user", "content": GRADER_USER_TEMPLATE.format(source=s, explanation=e)},
            ]
            p = grader_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
        enc = grader_tok(prompts, return_tensors="pt", padding=True, truncation=True,
                         max_length=4096).to(device)
        with torch.no_grad():
            out = grader.generate(**enc, max_new_tokens=max_new_tokens,
                                  do_sample=False, temperature=1.0,
                                  pad_token_id=grader_tok.pad_token_id)
        for bi in range(out.shape[0]):
            full = out[bi]
            prompt_len = enc.attention_mask[bi].sum().item()
            gen_ids = full[prompt_len:]
            text = grader_tok.decode(gen_ids, skip_special_tokens=True)
            claims = parse_grader_output(text)
            results.append(claims)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-init", required=True)
    ap.add_argument("--av-init", required=True)
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grader-device", default=None,
                    help="separate device for grader (default: same as AV/AR)")
    ap.add_argument("--grader-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--grader-dtype", default="bfloat16")
    ap.add_argument("--n-records", type=int, default=64)
    ap.add_argument("--av-batch-size", type=int, default=8)
    ap.add_argument("--grader-batch-size", type=int, default=2)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--source-window-pre", type=int, default=120)
    ap.add_argument("--source-window-post", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    grader_device = args.grader_device or device
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    # ---- AV ----
    print(f"Loading AV: {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    sd = torch.load(args.av_init, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()

    # ---- Activations ----
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
    eval_idx = [i for i in eval_idx_all if positions[i] + 8 < args.max_ctx_tokens][: args.n_records]
    print(f"  {len(eval_idx)} records")

    # ---- Generate AV explanations ----
    print(f"Generating AV explanations")
    explanations = []
    pbar = tqdm(range(0, len(eval_idx), args.av_batch_size), desc="AV")
    for i in pbar:
        ids = eval_idx[i : i + args.av_batch_size]
        h_batch = activations[ids].to(device)
        emb, mask = build_av_prompt_embeds(av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            gen = ddp_generate(av, emb, mask, max_new_tokens=args.max_new_tokens,
                               eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                               temperature=1.0, top_p=0.95)
        gen_mask = (gen == tok.eos_token_id).cumsum(1) <= 1
        for bi in range(gen.shape[0]):
            e = tok.decode(gen[bi, : int(gen_mask[bi].sum().item())], skip_special_tokens=True)
            explanations.append(e)

    # Free AV before loading grader
    del av
    torch.cuda.empty_cache()

    # ---- Build sources ----
    sources = [build_source_window(texts[i], int(positions[i]), tok,
                                   args.source_window_pre, args.source_window_post)
               for i in eval_idx]

    # ---- Load grader ----
    print(f"Loading grader: {args.grader_model} ({args.grader_dtype})")
    grader_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                    "float32": torch.float32}[args.grader_dtype]
    grader_tok = AutoTokenizer.from_pretrained(args.grader_model)
    if grader_tok.pad_token is None:
        grader_tok.pad_token = grader_tok.eos_token
    grader = AutoModelForCausalLM.from_pretrained(args.grader_model, torch_dtype=grader_dtype).to(grader_device)
    grader.eval()

    # ---- Grade ----
    graded = grade_batch(grader, grader_tok, sources, explanations, grader_device,
                         max_new_tokens=1024, batch_size=args.grader_batch_size)

    # ---- Aggregate ----
    n_with_parse = sum(1 for c in graded if c is not None)
    n_with_claims = sum(1 for c in graded if c)
    all_claim_counts = []
    all_supp_fracs = []
    all_contr_fracs = []
    all_neut_fracs = []
    n_supp_total, n_contr_total, n_neut_total = 0, 0, 0
    n_claims_total = 0
    detail = []
    for i, claims in enumerate(graded):
        if claims is None:
            detail.append({"explanation": explanations[i][:200], "claims": None, "parse_failed": True})
            continue
        n = len(claims)
        n_supp = sum(1 for c in claims if c["verdict"] == "supported")
        n_contr = sum(1 for c in claims if c["verdict"] == "contradicted")
        n_neut = sum(1 for c in claims if c["verdict"] == "neutral")
        all_claim_counts.append(n)
        if n > 0:
            all_supp_fracs.append(n_supp / n)
            all_contr_fracs.append(n_contr / n)
            all_neut_fracs.append(n_neut / n)
        n_supp_total += n_supp
        n_contr_total += n_contr
        n_neut_total += n_neut
        n_claims_total += n
        detail.append({
            "explanation": explanations[i][:300],
            "source_window": sources[i][:300],
            "claims": claims[:10],
        })

    res = {
        "av_init": args.av_init,
        "n_records": len(eval_idx),
        "n_with_parse": n_with_parse,
        "n_with_claims": n_with_claims,
        "claims_per_expl_mean": float(np.mean(all_claim_counts)) if all_claim_counts else 0,
        "claims_per_expl_median": float(np.median(all_claim_counts)) if all_claim_counts else 0,
        "supported_per_expl_mean": float(n_supp_total / max(1, len(all_claim_counts))),
        "contradicted_per_expl_mean": float(n_contr_total / max(1, len(all_claim_counts))),
        "neutral_per_expl_mean": float(n_neut_total / max(1, len(all_claim_counts))),
        "frac_supported_micro": n_supp_total / max(1, n_claims_total),
        "frac_contradicted_micro": n_contr_total / max(1, n_claims_total),
        "frac_neutral_micro": n_neut_total / max(1, n_claims_total),
        "frac_supported_macro": float(np.mean(all_supp_fracs)) if all_supp_fracs else 0,
        "frac_contradicted_macro": float(np.mean(all_contr_fracs)) if all_contr_fracs else 0,
        "frac_neutral_macro": float(np.mean(all_neut_fracs)) if all_neut_fracs else 0,
        "samples": detail[:8],
    }
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=2))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
