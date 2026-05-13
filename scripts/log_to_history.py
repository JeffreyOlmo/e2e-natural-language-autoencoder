"""Parse a tqdm log mid-run into a history.json-shaped object."""
import argparse
import json
import re
import sys
from pathlib import Path


# Generic pattern: fve=NUM, kl=NUM, ...
# We extract all key=val pairs from postfix sections
KV_RE = re.compile(r"(\w+)=([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
EVAL_RE = re.compile(r"\[eval @ (\d+)\] FVE=(-?\d+\.\d+), MSE=(-?\d+\.\d+)")
STEP_RE = re.compile(r"(\d+)/\d+ \[")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out")
    args = ap.parse_args()

    text = Path(args.log).read_text().replace("\r", "\n")
    train = []
    evals = []
    seen_steps = set()

    for line in text.split("\n"):
        m_eval = EVAL_RE.search(line)
        if m_eval:
            evals.append({"step": int(m_eval.group(1)), "fve": float(m_eval.group(2)), "mse": float(m_eval.group(3))})
            continue
        m_step = STEP_RE.search(line)
        if not m_step:
            continue
        step = int(m_step.group(1))
        # Extract postfix kv pairs
        idx = line.find(",")
        if idx == -1:
            continue
        kv = dict(KV_RE.findall(line))
        if not kv:
            continue
        # We only want updates that include the train postfix, not just the step counter
        if "fve" not in kv and "FVE" not in kv:
            continue
        if step in seen_steps:
            # take the latest (last) occurrence per step
            train = [t for t in train if t["step"] != step]
        seen_steps.add(step)
        # Map known keys generically
        entry = {"step": step}
        for k, v in kv.items():
            try:
                entry[k] = float(v)
            except ValueError:
                pass
        # Synthesize 'last_pos_fve' (and 'last_pos_mse' if MSE present, else NaN)
        if "fve" in entry:
            entry["last_pos_fve"] = entry["fve"]
        if "kl" in entry:
            entry["kl_ref"] = entry["kl"]
        # Remap short grad-norm postfix names → schema keys used by plot_compare
        if "avg" in entry:
            entry["av_grad_norm"] = entry["avg"]
        if "arg" in entry:
            entry["ar_grad_norm"] = entry["arg"]
        train.append(entry)

    history = train + [{"eval": e} for e in evals]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(history, f, indent=2)
        print(f"wrote {args.out}: {len(train)} train + {len(evals)} eval entries")
    else:
        json.dump(history, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
