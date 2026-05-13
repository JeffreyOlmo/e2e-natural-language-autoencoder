"""Replot pure GD trajectory alongside GRPO baseline."""
import json
import subprocess


def main():
    subprocess.run(
        ["python3", "scripts/log_to_history.py",
         "logs/rl_pure_gd.log", "--out", "/tmp/gd.json"],
        check=True, capture_output=True,
    )
    h = json.load(open("/tmp/gd.json"))
    n_train = sum(1 for e in h if "eval" not in e)
    evals = [e["eval"] for e in h if "eval" in e]

    subprocess.run(
        ["python3", "scripts/plot_compare.py",
         "--runs",
         "/tmp/grpo_full.json:GRPO(orig+cont)",
         "/tmp/gd.json:pure-GD",
         "--baseline-fve", "0.645",
         "--out", "/tmp/gd_vs_grpo.png"],
        check=True, capture_output=True,
    )

    msg = f"GD: {n_train} train, {len(evals)} evals"
    if evals:
        msg += f"; latest: @{evals[-1]['step']}={evals[-1]['fve']:.4f}"
    print(msg + "; plot → /tmp/gd_vs_grpo.png")


if __name__ == "__main__":
    main()
