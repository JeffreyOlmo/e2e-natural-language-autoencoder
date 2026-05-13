"""Replot: GRPO full trajectory (PG+orig+cont then extended @ 260 tokens).
Global step axis: orig 0-249 → cont 250-749 (offset +250) → len260 750-1249 (offset +750).
"""
import json
import subprocess
from pathlib import Path


def offset_history(path, offset):
    h = json.load(open(path))
    out = []
    for e in h:
        if "eval" in e:
            ev = dict(e["eval"]); ev["step"] += offset
            out.append({"eval": ev})
        else:
            e2 = dict(e); e2["step"] = e2.get("step", 0) + offset
            out.append(e2)
    return out


def truncate(history, max_step):
    out = []
    for e in history:
        if "eval" in e:
            if e["eval"]["step"] <= max_step:
                out.append(e)
        elif e.get("step", 0) <= max_step:
            out.append(e)
    return out


def main():
    subprocess.run(
        ["python3", "scripts/log_to_history.py",
         "logs/rl_small_grpo_len260.log", "--out", "/tmp/grpo_len260.json"],
        check=True, capture_output=True,
    )

    orig = truncate(json.load(open("checkpoints/rl_small_grpo/history.json")), 249)
    cont = truncate(offset_history("checkpoints/rl_small_grpo_cont/history.json", 250), 749)
    len260 = offset_history("/tmp/grpo_len260.json", 750)

    merged = orig + cont + len260
    Path("/tmp/grpo_extended_len260.json").write_text(json.dumps(merged))
    n_train = sum(1 for e in merged if "eval" not in e)
    evals = [e["eval"] for e in merged if "eval" in e]

    subprocess.run(
        ["python3", "scripts/plot_compare.py",
         "--runs",
         "/tmp/grpo_extended_len260.json:GRPO(orig+cont+len260)",
         "--out", "/tmp/grpo_len260.png"],
        check=True, capture_output=True,
    )

    latest = evals[-1] if evals else None
    msg = f"merged={n_train} train, {len(evals)} evals"
    if latest:
        msg += f"; latest: @{latest['step']}={latest['fve']:.4f}"
    print(msg + "; plot → /tmp/grpo_len260.png")


if __name__ == "__main__":
    main()
