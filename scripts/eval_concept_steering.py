"""Concept-edit steering eval (NLA paper Section: steering via edit-and-reconstruct).

For each record:
  1. AV(h) → explanation text
  2. Apply a defined concept-edit (e.g., "humorous" → "serious") via regex
     substitution. Skip records where no edit word is found.
  3. AR(orig) → ĥ_orig;  AR(edited) → ĥ_edit
  4. Encoding response: ||ĥ_edit − ĥ_orig|| / ||ĥ_orig||  (how much edit moved encoding)
  5. Behavioral response: KL between logits at position p when patching ĥ_orig
     vs ĥ_edit into source context at layer 16. Bigger = encoding's behavioral
     contrast is sharper.
  6. Steered-completion test: greedy-decode 5 tokens after p with each patch;
     report whether the EDITED word (or a synonym) appears more often.

Compare vanilla GRPO vs e2e ckpt: e2e should produce encodings whose response to
text-level concept edits is BEHAVIORALLY larger (downstream KL bigger), even if
geometric MSE/FVE looks similar.
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
from nla.model import ARModel

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_small_rl_e2e import make_patch_state, make_patch_hook, tokenize_contexts
from train_small_rl import (
    AR_PREFIX, AR_SUFFIX, build_ar_inputs, build_av_prompt_embeds, ddp_generate,
)


# Antonym pairs covering sentiment / tone / formality / age. Bi-directional;
# whichever word appears first in the explanation gets replaced by the other.
CONCEPT_PAIRS = [
    ("professional", "amateur"),
    ("formal", "casual"),
    ("humorous", "serious"),
    ("authoritative", "uncertain"),
    ("technical", "casual"),
    ("academic", "casual"),
    ("modern", "traditional"),
    ("happy", "sad"),
    ("positive", "negative"),
    ("good", "bad"),
    ("simple", "complex"),
    ("old", "new"),
    ("large", "small"),
    ("strong", "weak"),
    ("public", "private"),
    ("warm", "cold"),
    ("religious", "secular"),
    ("polite", "rude"),
    ("safe", "dangerous"),
    ("rich", "poor"),
]


def try_edit_explanation(text):
    """Return (edited_text, orig_word, edit_word) or None if no concept word found."""
    for w1, w2 in CONCEPT_PAIRS:
        # whole-word match, case-insensitive
        m1 = re.search(rf"\b{re.escape(w1)}\b", text, re.IGNORECASE)
        m2 = re.search(rf"\b{re.escape(w2)}\b", text, re.IGNORECASE)
        if m1 and not m2:
            edited = re.sub(rf"\b{re.escape(w1)}\b", w2, text, count=0, flags=re.IGNORECASE)
            return edited, w1, w2
        if m2 and not m1:
            edited = re.sub(rf"\b{re.escape(w2)}\b", w1, text, count=0, flags=re.IGNORECASE)
            return edited, w2, w1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-init", required=True)
    ap.add_argument("--av-init", required=True)
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-records", type=int, default=256, help="records to try (some skipped)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm-dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = NLAConfig()
    device = args.device
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    print(f"Loading AV: {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    sd = torch.load(args.av_init, map_location=device, weights_only=False)
    av.load_state_dict(sd.get("state_dict", sd))
    av.eval()

    print(f"Loading AR: {args.ar_init}")
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    sd_ar = torch.load(args.ar_init, map_location=device, weights_only=False)
    sd_ar = sd_ar.get("state_dict", sd_ar)
    ar_state = ar.state_dict()
    ar.load_state_dict({k: v for k, v in sd_ar.items() if k in ar_state}, strict=False)
    ar.eval()

    lm_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[args.lm_dtype]
    print(f"Loading LM (frozen, {args.lm_dtype})")
    lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=lm_dtype).to(device)
    lm.eval()
    patch_state = make_patch_state()
    handle = lm.model.layers[cfg.layer - 1].register_forward_hook(make_patch_hook(patch_state))

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
    eval_idx = [i for i in eval_idx_all if positions[i] + 8 < args.max_ctx_tokens]
    eval_idx = eval_idx[: args.n_records]
    print(f"  {len(eval_idx)} candidate records (will skip those without editable concept word)")

    # Stage 1: generate explanations for all candidates, find which ones are editable.
    accum = {
        "encoding_response": [],   # ||ĥ_edit - ĥ_orig|| / ||ĥ_orig||
        "encoding_cos": [],         # cos(ĥ_edit, ĥ_orig)
        "downstream_kl": [],        # KL @ p between orig-patched and edit-patched
        "downstream_kl_unpatched": [],  # KL @ p between unpatched and edit-patched (for context)
        "edit_token_prob_gain": [], # P(edit_word|ĥ_edit at p) - P(edit_word|ĥ_orig at p) at logits[p]
        "orig_token_prob_loss": [], # P(orig_word|ĥ_orig at p) - P(orig_word|ĥ_edit at p) at logits[p]
        "edits_used": [],           # for inspection
    }

    pbar = tqdm(range(0, len(eval_idx), args.batch_size))
    for b0 in pbar:
        b1 = min(b0 + args.batch_size, len(eval_idx))
        ids = eval_idx[b0:b1]
        h_batch = activations[ids].to(device)
        text_batch = [texts[i] for i in ids]
        pos_batch = positions[ids]

        # AV → text
        emb, mask = build_av_prompt_embeds(av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32)
        with torch.no_grad():
            gen = ddp_generate(av, emb, mask, max_new_tokens=args.max_new_tokens,
                               eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                               temperature=1.0, top_p=0.95)
            gen_mask = (gen == tok.eos_token_id).cumsum(1) <= 1

        # Decode each, attempt edit
        edited_records = []  # list of (b_idx, orig_text, edited_text, orig_word, edit_word)
        for bi in range(gen.shape[0]):
            g = gen[bi]
            text = tok.decode(g[: gen_mask[bi].sum().item()], skip_special_tokens=True)
            res = try_edit_explanation(text)
            if res is None:
                continue
            edit_text, orig_word, edit_word = res
            edited_records.append((bi, text, edit_text, orig_word, edit_word))

        if not edited_records:
            continue

        # Tokenize orig and edited explanations together as a batch for AR
        all_texts = []
        b_indices = []
        for bi, ot, et, ow, ew in edited_records:
            all_texts.append(ot)
            b_indices.append(bi)
        for bi, ot, et, ow, ew in edited_records:
            all_texts.append(et)
            b_indices.append(bi)

        # Build AR inputs from text strings (re-tokenize)
        pre = tok(AR_PREFIX, add_special_tokens=False).input_ids
        suf = tok(AR_SUFFIX, add_special_tokens=False).input_ids
        pad_id = tok.pad_token_id
        rows, lengths = [], []
        for txt in all_texts:
            mid = tok(txt, add_special_tokens=False).input_ids
            seq = pre + mid + suf
            rows.append(seq); lengths.append(len(seq))
        max_len = max(lengths)
        ar_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
        ar_mask = torch.zeros_like(ar_ids)
        for i, r in enumerate(rows):
            ar_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=device)
            ar_mask[i, : len(r)] = 1
        with torch.no_grad():
            pred_all = ar(input_ids=ar_ids, attention_mask=ar_mask)  # [2*N, d]
        N = len(edited_records)
        pred_orig = pred_all[:N]   # [N, d]
        pred_edit = pred_all[N:]   # [N, d]

        # Encoding response
        diff = pred_edit - pred_orig
        cos = F.cosine_similarity(pred_edit, pred_orig, dim=-1)
        resp = diff.norm(dim=-1) / pred_orig.norm(dim=-1).clamp_min(1e-6)

        # Downstream KL — build per-record contexts (each at its own pos)
        h_sel = h_batch[[bi for bi, *_ in edited_records]]
        text_sel = [text_batch[bi] for bi, *_ in edited_records]
        pos_sel = pos_batch[[bi for bi, *_ in edited_records]]
        ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
            tok, text_sel, pos_sel, args.max_ctx_tokens, device,
        )
        T_ctx = ctx_ids.shape[1]

        with torch.no_grad():
            # Unpatched
            patch_state["h_hat"] = None
            unp = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            unp_lp = F.log_softmax(unp.logits.float(), dim=-1)
            # Orig-patched
            patch_state["h_hat"] = pred_orig.to(lm_dtype)
            patch_state["positions"] = pos_in_ctx
            o = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            o_lp = F.log_softmax(o.logits.float(), dim=-1)
            # Edit-patched
            patch_state["h_hat"] = pred_edit.to(lm_dtype)
            patch_state["positions"] = pos_in_ctx
            e = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
            e_lp = F.log_softmax(e.logits.float(), dim=-1)
            patch_state["h_hat"] = None

        # KL at position p (orig-patched vs edit-patched). High = encoding's
        # behavioral contrast is sharp.
        B = o_lp.shape[0]
        o_at_p = o_lp.gather(1, pos_in_ctx.view(-1, 1, 1).expand(-1, 1, o_lp.shape[-1])).squeeze(1)
        e_at_p = e_lp.gather(1, pos_in_ctx.view(-1, 1, 1).expand(-1, 1, e_lp.shape[-1])).squeeze(1)
        unp_at_p = unp_lp.gather(1, pos_in_ctx.view(-1, 1, 1).expand(-1, 1, unp_lp.shape[-1])).squeeze(1)
        kl_o_vs_e = (o_at_p.exp() * (o_at_p - e_at_p)).sum(-1)
        kl_unp_vs_e = (unp_at_p.exp() * (unp_at_p - e_at_p)).sum(-1)

        # Edit-word and orig-word token probabilities at position p
        for i, (_, _, _, ow, ew) in enumerate(edited_records):
            # Get token ids — leading space matters for tokenizer
            ow_ids = tok(" " + ow, add_special_tokens=False).input_ids
            ew_ids = tok(" " + ew, add_special_tokens=False).input_ids
            ow_id = ow_ids[0] if ow_ids else 0
            ew_id = ew_ids[0] if ew_ids else 0
            p_ew_edit = e_at_p[i].exp()[ew_id].item()
            p_ew_orig = o_at_p[i].exp()[ew_id].item()
            p_ow_orig = o_at_p[i].exp()[ow_id].item()
            p_ow_edit = e_at_p[i].exp()[ow_id].item()
            accum["edit_token_prob_gain"].append(p_ew_edit - p_ew_orig)
            accum["orig_token_prob_loss"].append(p_ow_orig - p_ow_edit)

        accum["encoding_response"].extend(resp.tolist())
        accum["encoding_cos"].extend(cos.tolist())
        accum["downstream_kl"].extend(kl_o_vs_e.tolist())
        accum["downstream_kl_unpatched"].extend(kl_unp_vs_e.tolist())
        for (_, _, _, ow, ew) in edited_records:
            accum["edits_used"].append(f"{ow}->{ew}")

        pbar.set_postfix(
            n_edits=len(accum["encoding_response"]),
            cos=f"{np.mean(accum['encoding_cos']):.3f}",
            kl=f"{np.mean(accum['downstream_kl']):.3f}",
        )

    res = {
        "ar_init": args.ar_init,
        "av_init": args.av_init,
        "n_candidate_records": len(eval_idx),
        "n_records_edited": len(accum["encoding_response"]),
        "encoding_response_mean": float(np.mean(accum["encoding_response"])),
        "encoding_cos_mean": float(np.mean(accum["encoding_cos"])),
        "downstream_kl_orig_vs_edit_mean": float(np.mean(accum["downstream_kl"])),
        "downstream_kl_unp_vs_edit_mean": float(np.mean(accum["downstream_kl_unpatched"])),
        "edit_token_prob_gain_mean": float(np.mean(accum["edit_token_prob_gain"])),
        "orig_token_prob_loss_mean": float(np.mean(accum["orig_token_prob_loss"])),
        "edit_counts": dict(sorted(
            {k: accum["edits_used"].count(k) for k in set(accum["edits_used"])}.items(),
            key=lambda x: -x[1],
        )),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"saved → {args.out}")
    handle.remove()


if __name__ == "__main__":
    main()
