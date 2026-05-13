"""Replot cold-start GRPO trajectory."""
import subprocess
from pathlib import Path
import json


def main():
    subprocess.run(
        ["python3", "scripts/log_to_history.py",
         "logs/rl_small_grpo_cold.log", "--out", "/tmp/grpo_cold.json"],
        check=True, capture_output=True,
    )
    h = json.load(open("/tmp/grpo_cold.json"))
    n_train = sum(1 for e in h if "eval" not in e)
    evals = [e["eval"] for e in h if "eval" in e]

    subprocess.run(
        ["python3", "scripts/plot_compare.py",
         "--runs",
         "/tmp/grpo_cold.json:GRPO-cold",
         "--baseline-fve", "0.645",
         "--out", "/tmp/grpo_cold.png"],
        check=True, capture_output=True,
    )

    latest = evals[-1] if evals else None
    msg = f"cold: {n_train} train, {len(evals)} evals"
    if latest:
        msg += f"; latest: @{latest['step']}={latest['fve']:.4f}"
    print(msg + "; plot → /tmp/grpo_cold.png")


if __name__ == "__main__":
    main()
