"""Evaluate kitft/nla-qwen2.5-7b-L20-{av,ar} on our newly-collected 7B@L20 activations.

Validates the 7B pivot:
  1. Load Anthropic-released AV/AR
  2. Sample rollouts from AV with α-injection
  3. Run AR on rollouts; compute FVE
  4. Print sample rollouts so we can see what FVE 0.7 quality looks like

Sidecar values are hardcoded from the fetched nla_meta.yaml files.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


# Hardcoded from kitft sidecars (nla_meta.yaml on HF)
KITFT_CFG = {
    "alpha": 150.0,
    "mse_scale": 59.86651818838306,  # sqrt(3584)
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


def load_ar(repo_id: str, device, dtype):
    """Load kitft AR: truncated Qwen2 body + value_head from separate safetensors file."""
    print(f"  AR body from {repo_id}")
    body = AutoModel.from_pretrained(repo_id, torch_dtype=dtype).to(device)
    body.eval()

    print(f"  value_head from {repo_id}/value_head.safetensors")
    vh_path = hf_hub_download(repo_id=repo_id, filename="value_head.safetensors")
    vh_state = safetensors.torch.load_file(vh_path)
    print(f"    keys: {list(vh_state.keys())}; shapes: {[tuple(v.shape) for v in vh_state.values()]}")

    d = body.config.hidden_size
    value_head = nn.Linear(d, d, bias=False, dtype=dtype).to(device)
    # Find a [d,d] tensor among the keys
    found = False
    for k, v in vh_state.items():
        if v.shape == (d, d):
            value_head.weight.data = v.to(dtype).to(device)
            print(f"    loaded weight from key {k!r}")
            found = True
            break
    if not found:
        raise ValueError(f"No [{d}, {d}] weight found in value_head.safetensors")
    value_head.eval()
    return body, value_head


def ar_forward(body, value_head, input_ids, attention_mask, extraction_layer):
    out = body(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h = out.hidden_states[extraction_layer]
    bsz = h.shape[0]
    last_idx = attention_mask.sum(1) - 1
    last = h[torch.arange(bsz, device=h.device), last_idx]
    return value_head(last)


def extract_explanation(text: str) -> str:
    m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-repo", default="kitft/nla-qwen2.5-7b-L20-av")
    ap.add_argument("--ar-repo", default="kitft/nla-qwen2.5-7b-L20-ar")
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = args.device
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print(f"Loading activations parquet: {args.activations}")
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d))
    texts = table["text"].to_pylist()
    doc_ids = table["doc_id"].to_pylist()
    positions = table["position"].to_pylist()
    print(f"  {n} vectors, d={d}")
    print(f"  norms: mean={activations.norm(dim=-1).mean().item():.2f} std={activations.norm(dim=-1).std().item():.2f}")

    indices = np_rng.choice(n, size=args.n_samples, replace=False).tolist()
    h_batch = activations[indices].to(device).to(dtype)

    print(f"\nLoading AV: {args.av_repo}")
    tok = AutoTokenizer.from_pretrained(args.av_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(args.av_repo, torch_dtype=dtype).to(device)
    av.eval()

    print(f"\nLoading AR: {args.ar_repo}")
    ar_body, value_head = load_ar(args.ar_repo, device, dtype)

    # Build AV prompt
    chat_msgs = [{"role": "user", "content": KITFT_CFG["av_template"].format(marker=KITFT_CFG["marker_token"])}]
    av_prompt_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    av_prompt_ids = tok(av_prompt_text, return_tensors="pt").input_ids.to(device)
    print(f"\nAV prompt: {av_prompt_ids.shape[1]} tokens")
    marker_count = (av_prompt_ids == KITFT_CFG["marker_token_id"]).sum().item()
    print(f"  marker tokens in prompt: {marker_count}")
    assert marker_count == 1, f"Expected 1 marker, got {marker_count}"

    B = h_batch.shape[0]
    prompt_ids_b = av_prompt_ids.expand(B, -1).contiguous()
    h_unit = F.normalize(h_batch, dim=-1)
    inj = (KITFT_CFG["alpha"] * h_unit).to(dtype)
    pos = (prompt_ids_b == KITFT_CFG["marker_token_id"]).float().argmax(dim=1)
    embeds = av.get_input_embeddings()(prompt_ids_b).clone()
    embeds[torch.arange(B, device=device), pos] = inj
    attn_mask = torch.ones_like(prompt_ids_b)

    print(f"\nGenerating {B} rollouts (max_new_tokens={args.max_new_tokens})...")
    with torch.no_grad():
        gen = av.generate(
            inputs_embeds=embeds, attention_mask=attn_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
        )
    rollout_ids = gen  # [B, T_resp] (only new tokens with inputs_embeds)
    eos_id = tok.eos_token_id

    # Truncate at first eos and decode
    rollout_text_list = []
    for i in range(B):
        ids = rollout_ids[i].tolist()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id) + 1]
        text = tok.decode(ids, skip_special_tokens=True)
        rollout_text_list.append(text)
    explanations = [extract_explanation(t) for t in rollout_text_list]

    # Build AR inputs
    ar_prompts = [KITFT_CFG["ar_template"].format(explanation=e) for e in explanations]
    ar_enc = tok(ar_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    ar_ids = ar_enc.input_ids.to(device)
    ar_mask = ar_enc.attention_mask.to(device)
    print(f"AR input shape: {tuple(ar_ids.shape)}")

    print("Running AR...")
    with torch.no_grad():
        pred = ar_forward(ar_body, value_head, ar_ids, ar_mask, KITFT_CFG["extraction_layer"])

    # Compute MSE & FVE
    sc = KITFT_CFG["mse_scale"]
    p_norm = F.normalize(pred.float(), dim=-1) * sc
    g_norm = F.normalize(h_batch.float(), dim=-1) * sc
    per_mse = ((p_norm - g_norm) ** 2).mean(dim=-1)
    print(f"\nPer-sample MSE: mean={per_mse.mean().item():.4f} std={per_mse.std().item():.4f} "
          f"min={per_mse.min().item():.4f} max={per_mse.max().item():.4f}")

    # Baseline: predict-mean (using all activations as reference for mean)
    mu = activations.mean(dim=0).to(device).to(dtype)
    mu_b = mu.expand_as(h_batch)
    base_p = F.normalize(mu_b.float(), dim=-1) * sc
    base_mse_per = ((base_p - g_norm) ** 2).mean(dim=-1)
    base_mse = base_mse_per.mean().item()
    print(f"Predict-mean baseline MSE: {base_mse:.4f}")

    fve_per = 1 - per_mse / base_mse_per
    fve_overall = 1 - per_mse.mean().item() / base_mse
    print(f"Per-sample FVE: mean={fve_per.mean().item():.4f}  overall (mean MSE / baseline) = {fve_overall:.4f}")

    print(f"\n=== {min(10, B)} sample rollouts ===")
    for k in range(min(10, B)):
        i = indices[k]
        print(f"\n[{k}] doc={doc_ids[i]} pos={positions[i]} fve={fve_per[k].item():.3f}")
        print(f"  source tail: …{texts[i][-180:]!r}")
        print(f"  AV explanation: {explanations[k][:350]!r}")


if __name__ == "__main__":
    main()
