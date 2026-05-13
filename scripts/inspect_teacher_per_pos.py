"""Inspect the per-position grad-distill teacher distribution.

Like inspect_teacher.py, but:
  - AR returns reconstructions at EVERY input position (value_head applied at all pos).
  - L_total = mean over positions of per-position MSE.
  - g_s at each input position s is the CUMULATIVE gradient ∂L_total/∂e_s.
    For causal AR, this equals (1/T) Σ_{t≥s} ∂L_t/∂e_s.

For a chosen subset of rollout positions, prints:
  rollout_token | top-5 π_AV | top-5 teacher q | ||g_s|| | KL(q||π)
"""
import argparse
import re

import numpy as np
import pyarrow.parquet as pq
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


KITFT_CFG = {
    "alpha": 150.0,
    "mse_scale": 59.86651818838306,
    "marker_token": "㈎",
    "marker_token_id": 149705,
    "extraction_layer": 20,
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
    "ar_template": "Summary of the following text: <text>{explanation}</text> <summary>",
}


def load_ar(repo_id, device, dtype):
    body = AutoModel.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()
    vh_path = hf_hub_download(repo_id=repo_id, filename="value_head.safetensors")
    vh_state = safetensors.torch.load_file(vh_path)
    d = body.config.hidden_size
    value_head = nn.Linear(d, d, bias=False, dtype=dtype).to(device)
    for k, v in vh_state.items():
        if v.shape == (d, d):
            value_head.weight.data = v.to(dtype).to(device)
            break
    value_head.eval()
    return body, value_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-repo", default="kitft/nla-qwen2.5-7b-L20-av")
    ap.add_argument("--ar-repo", default="kitft/nla-qwen2.5-7b-L20-ar")
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--show-positions", default="0,5,15,40,80",
                    help="comma-separated rollout positions to show")
    args = ap.parse_args()

    device = args.device
    dtype = torch.float16
    torch.manual_seed(args.seed)
    show_positions = [int(x) for x in args.show_positions.split(",")]

    print("Loading models")
    tok = AutoTokenizer.from_pretrained(args.av_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(args.av_repo, torch_dtype=dtype).to(device).eval()
    ar_body, value_head = load_ar(args.ar_repo, device, dtype)

    print("Loading activations")
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = table["text"].to_pylist()
    doc_ids = table["doc_id"].to_pylist()
    positions = table["position"].to_pylist()

    if args.idx is None:
        rng = np.random.default_rng(args.seed)
        args.idx = int(rng.integers(0, n))
    print(f"Row {args.idx} (doc_id={doc_ids[args.idx]} pos={positions[args.idx]})")
    print(f"  source tail: ...{texts[args.idx][-200:]!r}")

    h = activations[args.idx : args.idx + 1].to(device).to(dtype)

    # AV rollout
    chat = [{"role": "user", "content": KITFT_CFG["av_template"].format(marker=KITFT_CFG["marker_token"])}]
    pt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    pi_ids = tok(pt, return_tensors="pt").input_ids.to(device)
    h_unit = F.normalize(h, dim=-1)
    inj = (KITFT_CFG["alpha"] * h_unit).to(dtype)
    pos_mark = (pi_ids == KITFT_CFG["marker_token_id"]).float().argmax(dim=1)
    embeds = av.get_input_embeddings()(pi_ids).clone()
    embeds[0, pos_mark] = inj
    attn = torch.ones_like(pi_ids)

    print("\nGenerating rollout...")
    with torch.no_grad():
        gen = av.generate(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    rollout_ids = gen[0]
    eos_id = tok.eos_token_id
    if eos_id in rollout_ids.tolist():
        end = rollout_ids.tolist().index(eos_id) + 1
        rollout_ids = rollout_ids[:end]
    rollout_text = tok.decode(rollout_ids, skip_special_tokens=False)
    print(f"Rollout ({len(rollout_ids)} tokens):\n  {rollout_text!r}\n")

    m = re.search(r"<explanation>(.*?)</explanation>", rollout_text, re.DOTALL)
    explanation = m.group(1).strip() if m else rollout_text.strip()

    # Build AR input (prefix + explanation + suffix)
    prefix_text = "Summary of the following text: <text>"
    suffix_text = "</text> <summary>"
    pre_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    suf_ids = tok(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    expl_ids = tok(explanation, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    ar_ids = torch.cat([pre_ids, expl_ids, suf_ids]).unsqueeze(0)
    ar_mask = torch.ones_like(ar_ids)
    expl_offset_in_ar = len(pre_ids)
    n_expl = len(expl_ids)
    print(f"AR input: {ar_ids.shape[1]} tokens; explanation at [{expl_offset_in_ar}..{expl_offset_in_ar+n_expl})")

    # === PER-POSITION AR forward + backward ===
    ar_embeds = ar_body.embed_tokens(ar_ids).clone().requires_grad_(True)
    out = ar_body(inputs_embeds=ar_embeds, attention_mask=ar_mask, output_hidden_states=True)
    h_ar = out.hidden_states[KITFT_CFG["extraction_layer"]]  # [1, T_ar, d]
    pred_all = value_head(h_ar)  # [1, T_ar, d] — per-position recon
    p_norm = F.normalize(pred_all.float(), dim=-1) * KITFT_CFG["mse_scale"]
    h_norm = F.normalize(h.float(), dim=-1).unsqueeze(1).expand_as(p_norm) * KITFT_CFG["mse_scale"]
    per_pos_mse = ((p_norm - h_norm) ** 2).mean(dim=-1)  # [1, T_ar]
    L_total = per_pos_mse.mean()
    # last-position MSE for reference (this is what eval reports)
    last_mse = per_pos_mse[0, -1].item()
    print(f"per-position mean MSE = {L_total.item():.4f}, last-position MSE = {last_mse:.4f}")
    print(f"per-position MSE quartiles: min={per_pos_mse.min().item():.3f} "
          f"med={per_pos_mse.median().item():.3f} max={per_pos_mse.max().item():.3f}")

    g_at_input = torch.autograd.grad(L_total, ar_embeds)[0].detach()  # [1, T_ar, d]
    g_norms_all = g_at_input.norm(dim=-1)[0]  # [T_ar]

    # Slice to rollout (explanation) positions
    g_resp = g_at_input[0, expl_offset_in_ar : expl_offset_in_ar + n_expl]  # [n_expl, d]
    g_norms_resp = g_norms_all[expl_offset_in_ar : expl_offset_in_ar + n_expl]

    # === AV teacher-force forward to get logits at rollout positions ===
    response_embeds_av = av.get_input_embeddings()(rollout_ids.unsqueeze(0))
    full_embeds = torch.cat([embeds, response_embeds_av], dim=1)
    full_attn = torch.ones((1, full_embeds.shape[1]), dtype=torch.long, device=device)
    T_pre = embeds.shape[1]
    T_resp = rollout_ids.shape[0]
    with torch.no_grad():
        av_out = av(inputs_embeds=full_embeds, attention_mask=full_attn)
    av_logits = av_out.logits[0, T_pre - 1 : T_pre - 1 + T_resp]  # [T_resp, V]

    # Limit to positions present in both views (the AV's rollout_ids and AR's expl_ids
    # may differ slightly in tokenization). Show min length.
    n_show = min(n_expl, T_resp)
    g_resp = g_resp[:n_show]
    g_norms_resp = g_norms_resp[:n_show]
    av_logits_show = av_logits[:n_show]
    rollout_show = rollout_ids[:n_show]

    # Per-position teacher
    g_hat = g_resp / g_resp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e_v = av.get_input_embeddings().weight.to(g_hat.dtype)
    raw_scores = -(g_hat @ e_v.T)  # [n_show, V]
    log_pi = F.log_softmax(av_logits_show.float(), dim=-1)
    scores = raw_scores.float() / args.tau
    log_q = F.log_softmax(log_pi + scores, dim=-1)
    q = log_q.exp()
    kl_per = (q * (log_q - log_pi)).sum(-1)  # [n_show]

    print(f"\n=== Per-position gradient-norm profile ||g_s|| ===")
    # Show binned stats
    bins = [
        ("first 10%", g_norms_resp[: max(1, n_show // 10)]),
        ("first 25%", g_norms_resp[: max(1, n_show // 4)]),
        ("middle 20%", g_norms_resp[n_show * 2 // 5 : n_show * 3 // 5]),
        ("last 25%", g_norms_resp[-max(1, n_show // 4):]),
        ("last 10%", g_norms_resp[-max(1, n_show // 10):]),
    ]
    for lbl, x in bins:
        print(f"  {lbl:<12s} mean={x.mean().item():.4e}  max={x.max().item():.4e}")
    print(f"  ratio first10%/last10% = {bins[0][1].mean().item() / max(bins[4][1].mean().item(), 1e-12):.2f}x")

    print(f"\n=== Per-position KL(q || π) profile (τ={args.tau}) ===")
    bins_kl = [
        ("first 10%", kl_per[: max(1, n_show // 10)]),
        ("first 25%", kl_per[: max(1, n_show // 4)]),
        ("middle 20%", kl_per[n_show * 2 // 5 : n_show * 3 // 5]),
        ("last 25%", kl_per[-max(1, n_show // 4):]),
        ("last 10%", kl_per[-max(1, n_show // 10):]),
    ]
    for lbl, x in bins_kl:
        print(f"  {lbl:<12s} mean={x.mean().item():.3f}")
    print(f"  overall mean KL = {kl_per.mean().item():.3f} nats")

    # Detailed display at selected positions
    print(f"\n=== Detail at selected positions (τ={args.tau}) ===")
    valid_positions = [p for p in show_positions if 0 <= p < n_show]
    for p in valid_positions:
        tok_str = repr(tok.decode([rollout_show[p].item()]))
        print(f"\n  --- pos {p}: token={tok_str}  ||g||={g_norms_resp[p].item():.3e}  KL={kl_per[p].item():.3f} ---")
        pi_top = log_pi[p].topk(args.top_k)
        print(f"  top-{args.top_k} π_AV:")
        for v, i in zip(pi_top.values, pi_top.indices):
            print(f"    {repr(tok.decode([i.item()])):<18s}  p={v.exp().item():.3f}  s[v]={raw_scores[p, i].item():+.3e}")
        q_top = q[p].topk(args.top_k)
        print(f"  top-{args.top_k} teacher q:")
        for v, i in zip(q_top.values, q_top.indices):
            print(f"    {repr(tok.decode([i.item()])):<18s}  p={v.item():.3f}  s[v]={raw_scores[p, i].item():+.3e}")


if __name__ == "__main__":
    main()
