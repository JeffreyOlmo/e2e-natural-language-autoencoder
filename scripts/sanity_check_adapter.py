"""Sanity-check the adapter: feed A(h_0.5B) and the true h_7B through kitft AV
side-by-side. If A is good enough, both descriptions should describe the same
source text.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


KITFT_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
CFG = {
    "alpha": 150.0,
    "marker_token": "㈎",
    "marker_token_id": 149705,
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
}


class MLPAdapter(nn.Module):
    def __init__(self, d_in, d_out, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class LinearAdapter(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.lin(x)


def extract_explanation(text):
    m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def load_acts(path):
    t = pq.read_table(path)
    n = len(t)
    flat = np.asarray(t["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    h = torch.from_numpy(flat.reshape(n, d).copy())
    keys = list(zip(t["doc_id"].to_pylist(), t["position"].to_pylist()))
    texts = t["text"].to_pylist() if "text" in t.column_names else [None] * n
    return h, keys, texts


def run_av(av, tok, h_batch, device, dtype, max_new_tokens, temperature, top_p):
    B = h_batch.shape[0]
    chat_msgs = [{"role": "user", "content": CFG["av_template"].format(marker=CFG["marker_token"])}]
    prompt_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
    h_unit = F.normalize(h_batch.float(), dim=-1)
    inj = (CFG["alpha"] * h_unit).to(dtype)
    pos = (prompt_ids_b == CFG["marker_token_id"]).float().argmax(dim=1)
    embeds = av.get_input_embeddings()(prompt_ids_b).clone()
    embeds[torch.arange(B, device=device), pos] = inj
    attn_mask = torch.ones_like(prompt_ids_b)
    with torch.no_grad():
        gen = av.generate(
            inputs_embeds=embeds, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    eos = tok.eos_token_id
    out = []
    for i in range(B):
        ids = gen[i].tolist()
        if eos in ids:
            ids = ids[: ids.index(eos) + 1]
        out.append(tok.decode(ids, skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/adapter_05B_to_7B/adapter.pt")
    ap.add_argument("--src", default="data/activations_L16.parquet")
    ap.add_argument("--tgt", default="data/activations_L20.parquet")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device, dtype = args.device, torch.float16
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.adapter, map_location=device)
    arch = ck["arch"]
    if arch == "linear":
        adapter = LinearAdapter(ck["d_in"], ck["d_out"]).to(device)
    else:
        adapter = MLPAdapter(ck["d_in"], ck["d_out"], hidden=ck["hidden"]).to(device)
    adapter.load_state_dict(ck["state_dict"])
    adapter.eval()
    print(f"Loaded {arch} adapter (best epoch {ck.get('best_epoch')} val_cos_sim={ck.get('best_val_cos_sim'):.4f})")

    h_src, keys_src, texts = load_acts(args.src)
    h_tgt, keys_tgt, _ = load_acts(args.tgt)
    assert keys_src == keys_tgt
    n = len(h_src)
    # Use held-out tail (paralleling the adapter's val split: it used a permutation,
    # but here we just pick random indices for qualitative comparison).
    idxs = rng.choice(n, size=args.n, replace=False).tolist()
    h_src_b = h_src[idxs].to(device)
    h_tgt_b = h_tgt[idxs].to(device)

    with torch.no_grad():
        h_adapt = adapter(h_src_b)
    # Cosine similarity diagnostic
    cs = F.cosine_similarity(h_adapt.float(), h_tgt_b.float(), dim=-1)
    print(f"  per-sample cos_sim(A(h_0.5B), h_7B): {[round(c.item(), 3) for c in cs]}")

    print(f"\nLoading {KITFT_AV_REPO}")
    tok = AutoTokenizer.from_pretrained(KITFT_AV_REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(KITFT_AV_REPO, torch_dtype=dtype).to(device).eval()

    print("\nGenerating via adapter input...")
    out_adapt = run_av(av, tok, h_adapt.to(dtype), device, dtype,
                       args.max_new_tokens, args.temperature, args.top_p)
    print("Generating via true h_7B input...")
    out_true = run_av(av, tok, h_tgt_b.to(dtype), device, dtype,
                      args.max_new_tokens, args.temperature, args.top_p)

    print("\n=== Side-by-side ===")
    for i in range(args.n):
        ci = cs[i].item()
        di, pi = keys_src[idxs[i]]
        src_text = texts[idxs[i]]
        print(f"\n[{i}] doc={di} pos={pi}  cos(adapt,true)={ci:+.3f}")
        if src_text:
            print(f"  source: …{src_text[-180:]!r}")
        print(f"  via ADAPTER A(h_0.5B): {extract_explanation(out_adapt[i])[:400]!r}")
        print(f"  via TRUE    h_7B    : {extract_explanation(out_true[i])[:400]!r}")


if __name__ == "__main__":
    main()
