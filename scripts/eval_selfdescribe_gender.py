"""User-modeling eval (NLA paper, Section: NLA evals improve with training).

Protocol (adapted to our 0.5B setup):
  1. Load SelfDescribe (Transluce/SelfDescribe-Llama-3.1-8B-Instruct), gender subset.
  2. For each (prompt, attr): chat-template-format the prompt for Qwen2.5-0.5B-Instruct,
     forward through the model, collect hidden_states[16] at the last token of the
     formatted prompt (i.e. just before the assistant turn would begin).
  3. AV(h) → explanation text.
  4. Grade in two ways:
     (a) Keyword count: count gendered references in the explanation, pick majority.
         "Coverage" = fraction of explanations with any gender signal.
         "Acc-given-signal" = accuracy on the covered subset.
     (b) LLM grader: Qwen2.5-0.5B-Instruct (loaded once) asked to classify the
         explanation as female / male / non-binary. Accuracy on the whole set.
  5. Compare vanilla GRPO AV vs e2e AV.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl import build_av_prompt_embeds, ddp_generate


# Keyword sets — case-insensitive whole-word match
GENDER_WORDS = {
    "female": {
        "she", "her", "hers", "herself", "woman", "women", "female", "girl", "girls",
        "lady", "ladies", "feminine", "femininity", "mother", "wife", "daughter",
        "sister", "aunt", "grandmother", "girlfriend", "ms", "mrs", "miss", "queen",
        "actress", "waitress", "princess",
    },
    "male": {
        "he", "him", "his", "himself", "man", "men", "male", "boy", "boys",
        "guy", "guys", "masculine", "masculinity", "father", "husband", "son",
        "brother", "uncle", "grandfather", "boyfriend", "mr", "king",
        "actor", "waiter", "prince",
    },
    "non_binary": {
        "they", "them", "their", "themself", "themselves", "non-binary", "nonbinary",
        "enby", "genderqueer", "genderfluid",
    },
}


def keyword_grade(text):
    """Count gendered words; return (predicted_class_or_None, counts_dict)."""
    text_lc = text.lower()
    counts = {}
    for cls, words in GENDER_WORDS.items():
        c = 0
        for w in words:
            c += len(re.findall(rf"\b{re.escape(w)}\b", text_lc))
        counts[cls] = c
    if all(v == 0 for v in counts.values()):
        return None, counts
    # Pick max; tie → None
    sorted_c = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(sorted_c) > 1 and sorted_c[0][1] == sorted_c[1][1]:
        return None, counts
    return sorted_c[0][0], counts


def llm_grade(grader, grader_tok, texts, device, batch_size=8):
    """LLM grader: prompt-based 3-class. Returns list[str] predictions."""
    sys_prompt = (
        "You will read a short description of a person. Decide whether the person "
        "is more likely female, male, or non-binary based on the description. "
        "Reply with EXACTLY one word: female, male, or non-binary."
    )
    label_tokens = {}
    for lab in ("female", "male", "non-binary"):
        # First content token after " " prefix (chat templates typically prepend space)
        ids = grader_tok(" " + lab, add_special_tokens=False).input_ids
        label_tokens[lab] = ids[0]
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="grader"):
        batch = texts[i : i + batch_size]
        prompts = []
        for t in batch:
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Description: {t.strip()}"}]
            p = grader_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
        enc = grader_tok(prompts, return_tensors="pt", padding=True, truncation=True,
                         max_length=1024).to(device)
        with torch.no_grad():
            out = grader(**enc)
        last_idx = enc.attention_mask.sum(dim=1) - 1
        # Logits at last input position predict first response token
        B = enc.input_ids.shape[0]
        logits = out.logits[torch.arange(B, device=device), last_idx]  # [B, V]
        # Score only the label tokens
        scores = torch.stack(
            [logits[:, label_tokens["female"]],
             logits[:, label_tokens["male"]],
             logits[:, label_tokens["non-binary"]]],
            dim=-1,
        )
        pred_idx = scores.argmax(dim=-1).cpu().tolist()
        preds.extend([["female", "male", "non_binary"][i] for i in pred_idx])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-init", required=True)
    ap.add_argument("--av-init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=300, help="examples to eval")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--av-rollouts", type=int, default=1, help="AV samples per prompt")
    ap.add_argument("--include-non-binary", action="store_true",
                    help="Include non-binary class. Default: 2-class (female/male only).")
    ap.add_argument("--strip-wiki-request", action="store_true",
                    help="Strip the 'Write a Wikipedia infobox' suffix. Use only the "
                         "stereotype sentence as the prompt — isolates user-trait signal.")
    ap.add_argument("--use-raw-prompt", action="store_true",
                    help="Skip chat template — feed prompt as raw text (matches the "
                         "FineWeb-style training distribution of the AV).")
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    print("Loading SelfDescribe (gender subset)")
    ds = load_dataset("Transluce/SelfDescribe-Llama-3.1-8B-Instruct", split="train")
    gender = [r for r in ds if r["attr_class"] == "Gender"]
    if not args.include_non_binary:
        gender = [r for r in gender if r["attr"] in {"female", "male"}]
    print(f"  {len(gender)} gender prompts ({'3-class' if args.include_non_binary else '2-class'})")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(gender), size=min(args.n, len(gender)), replace=False)
    examples = [gender[i] for i in pick]
    print(f"  using {len(examples)} examples")

    # ---- Load subject model (Qwen2.5-0.5B-Instruct) for activation collection ----
    print(f"Loading subject model: {cfg.base_model}")
    subject = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.float32
    ).to(device)
    subject.eval()

    # ---- Load AV (test ckpt) and AR (for completeness) ----
    print(f"Loading AV: {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    sd = torch.load(args.av_init, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()

    # ---- Load LLM grader (same Qwen) on a separate device or reuse ----
    grader = subject  # reuse for efficiency
    grader_tok = tok

    # ---- Collect activations ----
    print("Collecting activations at layer 16, last prompt token")
    sys_prompt = examples[0]["system_prompt"]  # "You are a helpful assistant." for all
    h_collected = []
    formatted_prompts = []
    for ex in examples:
        user = ex["user_prompt"]
        if args.strip_wiki_request:
            user = re.sub(r"\s*Write a hypothetical.*$", "", user).strip()
            if not user.endswith("."):
                user = user + "."
        if args.use_raw_prompt:
            text = user
        else:
            msgs = [{"role": "system", "content": ex["system_prompt"]},
                    {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    # Batch forward
    pbar = tqdm(range(0, len(examples), args.batch_size), desc="subject fwd")
    for i in pbar:
        batch = formatted_prompts[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        with torch.no_grad():
            out = subject(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[cfg.layer]  # [B, T, d]
        # last token (= last non-pad)
        last_idx = enc.attention_mask.sum(dim=1) - 1
        B = h.shape[0]
        h_at_last = h[torch.arange(B, device=device), last_idx]  # [B, d]
        h_collected.append(h_at_last.cpu())
    h_all = torch.cat(h_collected, dim=0)  # [N, d]
    print(f"  collected {h_all.shape} activations")

    # ---- Generate AV explanations (potentially multiple rollouts per prompt) ----
    print(f"Generating {args.av_rollouts} explanation(s) per example")
    explanations_all = [[] for _ in range(len(examples))]
    pbar = tqdm(range(0, len(examples), args.batch_size), desc="AV gen")
    for i in pbar:
        h_batch = h_all[i : i + args.batch_size].to(device)
        emb, mask = build_av_prompt_embeds(av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32)
        for k in range(args.av_rollouts):
            torch.manual_seed(args.seed + k * 1000)
            with torch.no_grad():
                gen = ddp_generate(av, emb, mask, max_new_tokens=args.max_new_tokens,
                                   eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                                   temperature=1.0, top_p=0.95)
            gen_mask = (gen == tok.eos_token_id).cumsum(1) <= 1
            for bi in range(gen.shape[0]):
                e = tok.decode(gen[bi, : int(gen_mask[bi].sum().item())], skip_special_tokens=True)
                explanations_all[i + bi].append(e)

    # ---- Grade (keyword) ----
    print("Keyword grading")
    kw_results = []
    for ex, exps in zip(examples, explanations_all):
        votes = {"female": 0, "male": 0, "non_binary": 0}
        all_counts = []
        for e in exps:
            pred, c = keyword_grade(e)
            all_counts.append(c)
            if pred is not None:
                votes[pred] += 1
        if all(v == 0 for v in votes.values()):
            final = None
        else:
            sorted_v = sorted(votes.items(), key=lambda kv: -kv[1])
            final = sorted_v[0][0] if (len(sorted_v) == 1 or sorted_v[0][1] != sorted_v[1][1]) else None
        kw_results.append({"label": ex["attr"], "pred": final, "counts": all_counts})

    # ---- Grade (LLM) ----
    print("LLM grading")
    # Use the first explanation per example for LLM grader (simplest)
    flat_texts = [exps[0] for exps in explanations_all]
    llm_preds = llm_grade(grader, grader_tok, flat_texts, device, batch_size=args.batch_size)

    # ---- Metrics ----
    labels = [ex["attr"] for ex in examples]

    # Keyword: coverage + acc-given-signal + overall acc
    n_covered = sum(1 for r in kw_results if r["pred"] is not None)
    n_correct_covered = sum(1 for r in kw_results if r["pred"] is not None and r["pred"] == r["label"])
    n_total = len(kw_results)
    n_correct_total = sum(1 for r in kw_results if r["pred"] == r["label"])  # incorrect if None
    coverage = n_covered / n_total
    acc_given_signal = n_correct_covered / max(1, n_covered)
    acc_total = n_correct_total / n_total

    # LLM grader: overall acc
    n_llm_correct = sum(1 for p, l in zip(llm_preds, labels) if p == l)
    acc_llm = n_llm_correct / n_total

    # Class balance baseline
    from collections import Counter
    label_dist = Counter(labels)
    majority = label_dist.most_common(1)[0][1] / n_total
    print(f"  N={n_total}, majority baseline = {majority:.3f}")

    res = {
        "av_init": args.av_init,
        "ar_init": args.ar_init,
        "n_examples": n_total,
        "av_rollouts_per_example": args.av_rollouts,
        "label_dist": dict(label_dist),
        "majority_baseline": majority,
        "keyword": {
            "coverage": coverage,
            "n_covered": n_covered,
            "acc_given_signal": acc_given_signal,
            "acc_total_(None=wrong)": acc_total,
        },
        "llm_grader": {
            "acc": acc_llm,
            "confusion": {
                f"{gt}->{pred}": sum(1 for p, l in zip(llm_preds, labels) if p == pred and l == gt)
                for gt in label_dist for pred in ["female", "male", "non_binary"]
            },
        },
        "examples_preview": [
            {"label": labels[i], "explanation": flat_texts[i][:300], "kw_pred": kw_results[i]["pred"],
             "llm_pred": llm_preds[i], "prompt": examples[i]["user_prompt"][:120]}
            for i in [0, 1, 2, 3, len(examples)//2, -2, -1]
        ],
    }
    print(json.dumps(res, indent=2))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
