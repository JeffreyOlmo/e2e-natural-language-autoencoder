"""Generate summaries for each (doc, position) using Qwen2.5-0.5B-Instruct as
self-teacher. Output parquet has (doc_id, position, prefix_text, summary).

Note: 0.5B is a weak summarizer. Expect noisy outputs and some extraction
failures. The summary text stored excludes the wrapping tags so it can be
slotted into either the AV target (`<explanation>{summary}</explanation>`)
or the AR input (`<text>{summary}</text> <summary>`) at use-site.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import NLAConfig


SYSTEM_PROMPT = "You are a literary analyst who writes detailed but compact summaries."

USER_TEMPLATE = """Read the snippet below and produce 3-5 short phrases (separated by semicolons) describing the text. Cover: what it is about, the genre or style, what kind of content immediately precedes the end, and what is likely to come next. Aim for 60-100 words total. No preface, no markdown headers, no quotes — just the phrases separated by semicolons.

<text>
{prefix}
</text>"""


SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
# Strip leading markdown headers, "Summary:", quote marks, etc.
PREAMBLE_RE = re.compile(
    r"^\s*(?:\*+\s*)?(?:<?/?summary>?|text\s*summary|here(?:'s)?\s+(?:a\s+)?(?:brief\s+)?summary|the\s+text\s+(?:is\s+about|discusses|describes))?\s*[:\-—]?\s*\*+?\s*",
    re.IGNORECASE,
)
QUOTE_TRIM_RE = re.compile(r'^["\'`]+\s*|\s*["\'`]+$')


def extract_summary(text: str) -> str | None:
    """Permissive extraction: prefer <summary>X</summary> if present and non-trivial,
    else clean the raw response (strip markdown, common preambles, surrounding quotes)."""
    text = text.strip()
    if not text:
        return None
    m = SUMMARY_RE.search(text)
    if m:
        inner = m.group(1).strip()
        if len(inner) >= 10:
            return _clean(inner)
        text = (text[: m.start()] + " " + text[m.end():]).strip()
    text = re.sub(r"</?summary>", "", text, flags=re.IGNORECASE).strip()
    text = _clean(text)
    if 10 <= len(text) <= 800:
        return text
    return None


def _clean(s: str) -> str:
    s = s.strip()
    # Strip up to two leading preambles (e.g., "**Text Summary:**\n")
    for _ in range(2):
        new = PREAMBLE_RE.sub("", s).strip()
        if new == s:
            break
        s = new
    s = QUOTE_TRIM_RE.sub("", s).strip()
    return s


def build_chat(tok, prefix: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(prefix=prefix)},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def derive_prefix(tok, text: str, position: int, max_input_tokens: int) -> str:
    """Decode tokens[:position+1] of text. Truncate-from-left if too long."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    end = min(position + 1, len(ids))
    if end > max_input_tokens:
        # Keep the tail (most recent context — that's what the activation reflects).
        ids_keep = ids[end - max_input_tokens : end]
    else:
        ids_keep = ids[:end]
    return tok.decode(ids_keep, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=None,
                    help="input activations parquet; default data/activations_L{layer}.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-input-tokens", type=int, default=768,
                    help="max prefix tokens kept (truncate-from-left if longer)")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap rows for debugging")
    ap.add_argument("--row-start", type=int, default=0, help="for sharded multi-GPU runs")
    ap.add_argument("--row-end", type=int, default=None)
    args = ap.parse_args()

    cfg = NLAConfig()
    in_path = Path(args.in_path) if args.in_path else Path(cfg.data_dir) / f"activations_L{cfg.layer}.parquet"
    out_path = Path(args.out) if args.out else Path(cfg.data_dir) / f"summaries_L{cfg.layer}.parquet"

    print(f"Reading {in_path}")
    table = pq.read_table(in_path)
    total_rows = len(table)
    start = max(0, args.row_start)
    end = total_rows if args.row_end is None else min(args.row_end, total_rows)
    if args.limit:
        end = min(end, start + args.limit)
    table = table.slice(start, end - start)
    n_rows = len(table)
    print(f"  rows [{start}, {end}) = {n_rows}")

    print(f"Loading {cfg.base_model} (teacher) on {args.device} fp16")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float16).to(args.device)
    model.eval()

    torch.manual_seed(args.seed)

    doc_ids = table["doc_id"].to_pylist()
    positions = table["position"].to_pylist()
    texts = table["text"].to_pylist()

    out_doc, out_pos, out_prefix, out_summary, out_raw = [], [], [], [], []
    n_failed = 0

    pbar = tqdm(total=n_rows, desc="summaries")
    with torch.no_grad():
        for start in range(0, n_rows, args.batch_size):
            end = min(start + args.batch_size, n_rows)
            prefixes = [
                derive_prefix(tok, texts[i], positions[i], args.max_input_tokens)
                for i in range(start, end)
            ]
            chats = [build_chat(tok, p) for p in prefixes]
            enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_input_tokens + 200).to(args.device)
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
            )
            new_tokens = gen[:, enc.input_ids.shape[1]:]
            decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for i, raw in enumerate(decoded):
                summary = extract_summary(raw)
                if summary is None:
                    n_failed += 1
                    summary = ""  # keep row but mark empty
                out_doc.append(doc_ids[start + i])
                out_pos.append(positions[start + i])
                out_prefix.append(prefixes[i])
                out_summary.append(summary)
                out_raw.append(raw.strip())
            pbar.update(end - start)
    pbar.close()

    print(f"\n{n_failed}/{n_rows} extractions failed ({100*n_failed/n_rows:.1f}%)")
    summary_lens = [len(s.split()) for s in out_summary if s]
    if summary_lens:
        print(f"Summary length (words): mean={np.mean(summary_lens):.1f} "
              f"median={np.median(summary_lens):.0f} p95={np.percentile(summary_lens, 95):.0f}")
    print("\n=== 5 random examples ===")
    rng = np.random.default_rng(args.seed)
    for i in rng.choice(n_rows, size=min(5, n_rows), replace=False):
        i = int(i)
        print(f"\n[{i}] doc_id={out_doc[i]} pos={out_pos[i]}")
        print(f"  prefix tail: …{out_prefix[i][-150:]!r}")
        print(f"  summary    : {out_summary[i]!r}")

    table_out = pa.table({
        "doc_id": pa.array(out_doc, type=pa.int64()),
        "position": pa.array(out_pos, type=pa.int32()),
        "prefix_text": pa.array(out_prefix, type=pa.string()),
        "summary": pa.array(out_summary, type=pa.string()),
        "raw_response": pa.array(out_raw, type=pa.string()),
    })
    pq.write_table(table_out, out_path, compression="zstd")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
