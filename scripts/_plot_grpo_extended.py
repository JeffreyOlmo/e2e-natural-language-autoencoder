"""Helper: re-parse rl_small_grpo_cont log, concat with orig GRPO 0-249, plot.
Called by a Monitor on every new eval line.
"""
import json
import subprocess
from pathlib import Path

LOG = "logs/rl_small_grpo_cont.log"
OUT_PNG = "/tmp/grpo_extended.png"
OFFSET = 250  # cont starts from step-250 checkpoint of orig


def main():
    # 1. Parse cont log
    subprocess.run(
        ["python3", "scripts/log_to_history.py", LOG, "--out", "/tmp/grpo_cont.json"],
        check=True, capture_output=True,
    )

    # 2. Merge orig (truncated to <=249) with offset cont
    orig = json.load(open("checkpoints/rl_small_grpo/history.json"))
    cont = json.load(open("/tmp/grpo_cont.json"))

    merged = []
    for e in orig:
        if "eval" in e:
            if e["eval"]["step"] <= 249:
                merged.append(e)
        elif e.get("step", 0) <= 249:
            merged.append(e)

    for e in cont:
        if "eval" in e:
            ev = dict(e["eval"]); ev["step"] += OFFSET
            merged.append({"eval": ev})
        else:
            e2 = dict(e); e2["step"] = e2.get("step", 0) + OFFSET
            merged.append(e2)

    Path("/tmp/grpo_full.json").write_text(json.dumps(merged))
    n_train = sum(1 for e in merged if "eval" not in e)
    evals = [e["eval"] for e in merged if "eval" in e]

    # 3. Plot
    subprocess.run(
        ["python3", "scripts/plot_compare.py",
         "--runs",
         "checkpoints/rl_small_pg/history.json:PG",
         "/tmp/grpo_full.json:GRPO(orig+cont)",
         "--out", OUT_PNG],
        check=True, capture_output=True,
    )

    print(f"merged={n_train} train, {len(evals)} evals; "
          f"latest eval: @{evals[-1]['step']}={evals[-1]['fve']:.4f}; "
          f"plot → {OUT_PNG}")


if __name__ == "__main__":
    main()
