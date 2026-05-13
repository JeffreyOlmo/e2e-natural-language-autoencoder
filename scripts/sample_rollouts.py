"""Sample rollouts from AV (kitft baseline or trained checkpoint).

Single-GPU. Loads AV, samples N activations, generates with α-injection,
decodes the explanations, runs AR for reconstruction MSE/FVE, prints
side-by-side.
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


KITFT_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
KITFT_AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
CFG = {
    "alpha": 150.0,
    "mse_scale": 59.86651818838306,
    "marker_token_id": 149705,
    "marker_token": "㈎",
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
    "ar_prefix": "Summary of the following text: <text>",
    "ar_suffix": "</text> <summary>",
}


def load_ar(repo_id, device, dtype):
    body = AutoModel.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()
    d = body.config.hidden_size
    vh = nn.Linear(d, d, bias=False, dtype=dtype).to(device)
    vh_path = hf_hub_download(repo_id=repo_id, filename="value_head.safetensors")
    state = safetensors.torch.load_file(vh_path)
    for _, v in state.items():
        if v.shape == (d, d):
            vh.weight.data = v.to(dtype).to(device)
            break
    vh.eval()
    return body, vh


def ar_forward(body, vh, input_ids, attention_mask, layer):
    with torch.no_grad():
        out = body(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h = out.hidden_states[layer]
    last_idx = attention_mask.sum(1) - 1
    last = h[torch.arange(h.shape[0], device=h.device), last_idx]
    return vh(last)


def extract_explanation(text):
    m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-checkpoint", default=None,
                    help="Path to a trained AV state_dict (.pt). If None, uses kitft baseline.")
    ap.add_argument("--ar-checkpoint", default=None,
                    help="Path to a trained AR state_dict (.pt) — KitftAR with body+value_head. "
                         "If None, uses kitft baseline AR.")
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-eval-set", action="store_true",
                    help="Sample from held-out top 5% (matches training eval set).")
    args = ap.parse_args()

    device = args.device
    dtype = torch.float16
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading AV from {KITFT_AV_REPO}{' + checkpoint' if args.av_checkpoint else ''}")
    tok = AutoTokenizer.from_pretrained(KITFT_AV_REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(KITFT_AV_REPO, torch_dtype=dtype).to(device).eval()
    if args.av_checkpoint:
        sd = torch.load(args.av_checkpoint, map_location=device)
        av.load_state_dict(sd, strict=False)
        del sd
        torch.cuda.empty_cache()
        print(f"  loaded {args.av_checkpoint}")

    print(f"Loading AR from {KITFT_AR_REPO}{' + checkpoint' if args.ar_checkpoint else ''}")
    ar_body, ar_vh = load_ar(KITFT_AR_REPO, device, dtype)
    if args.ar_checkpoint:
        # mmap=True keeps tensors out of GPU during load; we copy key-by-key into
        # the already-allocated AR params, avoiding double-allocation.
        sd = torch.load(args.ar_checkpoint, map_location="cpu", mmap=True)
        body_sd = {k[len("body."):]: v for k, v in sd.items() if k.startswith("body.")}
        body_state = ar_body.state_dict()
        n_loaded = 0
        for name, p in body_sd.items():
            if name in body_state:
                body_state[name].copy_(p.to(device).to(body_state[name].dtype))
                n_loaded += 1
        if "value_head.weight" in sd:
            ar_vh.weight.data.copy_(sd["value_head.weight"].to(device).to(dtype))
            n_loaded += 1
        del sd
        torch.cuda.empty_cache()
        print(f"  loaded {args.ar_checkpoint} ({n_loaded} tensors)")
    ar_emb = ar_body.embed_tokens.weight  # for soft eval if needed

    print(f"Loading activations: {args.activations}")
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = table["text"].to_pylist()
    doc_ids = table["doc_id"].to_pylist()
    positions = table["position"].to_pylist()
    print(f"  {n} vectors")

    if args.use_eval_set:
        # Match training script's split: top 5% by index
        n_eval = max(1, n // 20)
        pool = list(range(n - n_eval, n))
    else:
        pool = list(range(n))
    indices = rng.choice(pool, size=args.n_samples, replace=False).tolist()
    h_batch = activations[indices].to(device).to(dtype)

    # Build AV prompt
    chat_msgs = [{"role": "user", "content": CFG["av_template"].format(marker=CFG["marker_token"])}]
    prompt_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    B = h_batch.shape[0]
    prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
    h_unit = F.normalize(h_batch.float(), dim=-1)
    inj = (CFG["alpha"] * h_unit).to(dtype)
    pos = (prompt_ids_b == CFG["marker_token_id"]).float().argmax(dim=1)
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
    eos_id = tok.eos_token_id
    rollout_texts = []
    for i in range(B):
        ids = gen[i].tolist()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id) + 1]
        rollout_texts.append(tok.decode(ids, skip_special_tokens=True))
    explanations = [extract_explanation(t) for t in rollout_texts]

    # AR reconstruction
    ar_prompts = [f"{CFG['ar_prefix']}{e}{CFG['ar_suffix']}" for e in rollout_texts]
    enc = tok(ar_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    ar_ids = enc.input_ids.to(device)
    ar_mask = enc.attention_mask.to(device)
    pred = ar_forward(ar_body, ar_vh, ar_ids, ar_mask, CFG["extraction_layer"])

    sc = CFG["mse_scale"]
    p_norm = F.normalize(pred.float(), dim=-1) * sc
    g_norm = F.normalize(h_batch.float(), dim=-1) * sc
    per_mse = ((p_norm - g_norm) ** 2).mean(dim=-1)
    # baseline = predict-mean MSE on full set
    mu = activations.mean(dim=0).to(device).to(dtype)
    base_p = F.normalize(mu.expand_as(h_batch).float(), dim=-1) * sc
    base_per = ((base_p - g_norm) ** 2).mean(dim=-1)
    fve_per = 1 - per_mse / base_per

    print(f"\n=== overall: mean_mse={per_mse.mean().item():.4f} mean_fve={fve_per.mean().item():.4f} ===\n")
    for i in range(B):
        idx = indices[i]
        print(f"[{i}] doc={doc_ids[idx]} pos={positions[idx]} fve={fve_per[i].item():+.3f} mse={per_mse[i].item():.3f}")
        print(f"  source tail: …{texts[idx][-180:]!r}")
        print(f"  AV rollout:  {rollout_texts[i][:400]!r}")
        print()


if __name__ == "__main__":
    main()
