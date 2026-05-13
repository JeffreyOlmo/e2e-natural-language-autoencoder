"""For a single (h, rollout) pair, show what the grad-distill teacher
distribution q_t looks like at each rollout position vs the policy π_AV.

Output per position:
  rollout_token  | top-5 in π_AV (with logprobs) | top-5 in q (with logprobs) | KL(q||π)

Intuition test: at high AR quality (kitft, FVE ~0.79), the teacher should
suggest semantically-related alternatives to the rollout token, not random.
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

# Hardcoded from kitft sidecars
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
    ap.add_argument("--idx", type=int, default=None, help="row index of the activation to inspect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--target-kl", type=float, default=0.1, help="auto-tune τ to hit this per-position KL")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    device = args.device
    dtype = torch.float16
    torch.manual_seed(args.seed)

    print(f"Loading models")
    tok = AutoTokenizer.from_pretrained(args.av_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(args.av_repo, torch_dtype=dtype).to(device).eval()
    ar_body, value_head = load_ar(args.ar_repo, device, dtype)

    print(f"Loading activations")
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
    print(f"Inspecting row {args.idx} (doc_id={doc_ids[args.idx]} pos={positions[args.idx]})")
    print(f"  source tail: …{texts[args.idx][-200:]!r}")

    h = activations[args.idx : args.idx + 1].to(device).to(dtype)  # [1, d]

    # AV rollout
    chat_msgs = [{"role": "user", "content": KITFT_CFG["av_template"].format(marker=KITFT_CFG["marker_token"])}]
    prompt_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    h_unit = F.normalize(h, dim=-1)
    inj = (KITFT_CFG["alpha"] * h_unit).to(dtype)
    pos = (prompt_ids == KITFT_CFG["marker_token_id"]).float().argmax(dim=1)
    embeds = av.get_input_embeddings()(prompt_ids).clone()
    embeds[0, pos] = inj
    attn = torch.ones_like(prompt_ids)

    print(f"\nGenerating rollout...")
    with torch.no_grad():
        gen = av.generate(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    rollout_ids = gen[0]  # only new tokens with inputs_embeds
    eos_id = tok.eos_token_id
    if eos_id in rollout_ids.tolist():
        end = rollout_ids.tolist().index(eos_id) + 1
        rollout_ids = rollout_ids[:end]
    rollout_text = tok.decode(rollout_ids, skip_special_tokens=False)
    print(f"Rollout ({len(rollout_ids)} tokens):\n  {rollout_text!r}\n")

    m = re.search(r"<explanation>(.*?)</explanation>", rollout_text, re.DOTALL)
    explanation = m.group(1).strip() if m else rollout_text.strip()

    # Build AR input
    ar_text = KITFT_CFG["ar_template"].format(explanation=explanation)
    ar_enc = tok(ar_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    ar_ids = ar_enc.input_ids
    ar_mask = ar_enc.attention_mask

    # Find where the rollout (explanation) tokens appear in ar_ids
    # Easier: tokenize the prefix "Summary of the following text: <text>" and the suffix "</text> <summary>"
    prefix_text = "Summary of the following text: <text>"
    suffix_text = f"</text> <summary>"
    pre_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    suf_ids = tok(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    expl_token_ids = tok(explanation, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    rebuilt = torch.cat([pre_ids, expl_token_ids, suf_ids])

    # Sanity: check rebuilt matches ar_ids modulo BOS
    ar_ids_flat = ar_ids[0]
    if ar_ids_flat[0] != rebuilt[0]:
        # AR text has BOS; rebuilt doesn't
        rebuilt = torch.cat([ar_ids_flat[:1], rebuilt])
    print(f"AR input: {len(ar_ids_flat)} tokens (rebuilt {len(rebuilt)} tokens)")

    expl_offset_in_ar = len(ar_ids_flat) - len(suf_ids) - len(expl_token_ids)
    print(f"Explanation tokens at offset {expl_offset_in_ar}..{expl_offset_in_ar + len(expl_token_ids)} in AR input")

    # AR forward + backward to get g_t at AR's input embeddings
    ar_embeds = ar_body.embed_tokens(ar_ids).clone().requires_grad_(True)
    out = ar_body(inputs_embeds=ar_embeds, attention_mask=ar_mask, output_hidden_states=True)
    h_ar = out.hidden_states[KITFT_CFG["extraction_layer"]]
    last_idx = ar_mask.sum(1) - 1
    last = h_ar[torch.arange(1, device=device), last_idx]
    pred = value_head(last)
    p_norm = F.normalize(pred.float(), dim=-1) * KITFT_CFG["mse_scale"]
    g_norm = F.normalize(h.float(), dim=-1) * KITFT_CFG["mse_scale"]
    mse = ((p_norm - g_norm) ** 2).mean()
    print(f"AR reconstruction MSE: {mse.item():.4f}")

    g_at_input = torch.autograd.grad(mse, ar_embeds)[0].detach()  # [1, T_ar, d]

    # Get π_AV logits at each rollout position via teacher-forcing forward
    # AV input: prompt_embeds + response_embeds
    response_embeds_av = av.get_input_embeddings()(rollout_ids.unsqueeze(0))
    full_embeds = torch.cat([embeds, response_embeds_av], dim=1)
    full_attn = torch.ones((1, full_embeds.shape[1]), dtype=torch.long, device=device)
    T_pre = embeds.shape[1]
    T_resp = rollout_ids.shape[0]
    with torch.no_grad():
        av_out = av(inputs_embeds=full_embeds, attention_mask=full_attn)
    av_logits = av_out.logits[0, T_pre - 1 : T_pre - 1 + T_resp]  # [T_resp, V]

    # Extract g at the rollout/explanation positions from AR input
    # The rollout_ids may differ from expl_token_ids slightly (whitespace tokenization).
    # Use the explanation token positions in AR input as the proxy.
    n_show = min(T_resp, len(expl_token_ids))
    g_resp = g_at_input[0, expl_offset_in_ar : expl_offset_in_ar + n_show]  # [n_show, d]
    av_logits = av_logits[:n_show]
    rollout_show = rollout_ids[:n_show]

    # Per-position normalize g
    g_hat = g_resp / g_resp.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # Compute scores at all τ values; pick the one closest to target_kl
    e_v = av.get_input_embeddings().weight.to(g_hat.dtype)  # [V, d]
    raw_scores = -(g_hat @ e_v.T)  # [n_show, V]

    log_pi = F.log_softmax(av_logits.float(), dim=-1)
    pi_argmax = log_pi.argmax(-1)

    candidates = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0]
    print(f"\nτ sweep (mean per-position KL):")
    for tau in candidates:
        scores = raw_scores.float() / tau
        log_q = F.log_softmax(log_pi + scores, dim=-1)
        q = log_q.exp()
        kl = (q * (log_q - log_pi)).sum(-1).mean().item()
        print(f"  τ={tau:>6.3f}  mean KL = {kl:.4f}")

    # Auto-pick the τ closest to target_kl
    best = min(candidates, key=lambda t: abs(((F.log_softmax(log_pi + raw_scores.float()/t, dim=-1).exp() *
                                                (F.log_softmax(log_pi + raw_scores.float()/t, dim=-1) - log_pi)).sum(-1).mean().item()) - args.target_kl))
    tau = best
    scores = raw_scores.float() / tau
    log_q = F.log_softmax(log_pi + scores, dim=-1)
    q = log_q.exp()

    print(f"\nUsing τ = {tau} (mean KL ≈ {(q * (log_q - log_pi)).sum(-1).mean().item():.3f} nats)\n")
    print(f"{'pos':>3s} {'token':<25s} | {'top-5 π_AV':<60s} | {'top-5 q (teacher)':<60s} | {'KL':>6s}")
    print("-" * 165)

    for t in range(n_show):
        tok_str = repr(tok.decode([rollout_show[t].item()]))
        # top-5 in π
        pi_top = log_pi[t].topk(args.top_k)
        pi_str = " ".join(f"{repr(tok.decode([i.item()])):>10s}({v.exp().item():.2f})"
                           for v, i in zip(pi_top.values, pi_top.indices))
        # top-5 in q
        q_top = q[t].topk(args.top_k)
        q_str = " ".join(f"{repr(tok.decode([i.item()])):>10s}({v.item():.2f})"
                          for v, i in zip(q_top.values, q_top.indices))
        kl_t = (q[t] * (log_q[t] - log_pi[t])).sum().item()
        print(f"{t:>3d} {tok_str[:25]:<25s} | {pi_str[:60]:<60s} | {q_str[:60]:<60s} | {kl_t:>6.3f}")


if __name__ == "__main__":
    main()
