"""Replot e2e (downstream-KL) vs vanilla GRPO."""
import json
import subprocess


def main():
    subprocess.run(
        ["python3", "scripts/log_to_history.py",
         "logs/rl_e2e.log", "--out", "/tmp/e2e.json"],
        check=True, capture_output=True,
    )
    h = json.load(open("/tmp/e2e.json"))
    n_train = sum(1 for e in h if "eval" not in e)
    evals = [e["eval"] for e in h if "eval" in e]

    subprocess.run(
        ["python3", "scripts/plot_compare.py",
         "--runs",
         "/tmp/grpo_full.json:GRPO(vanilla)",
         "/tmp/e2e.json:GRPO+e2e-KL",
         "--baseline-fve", "0.645",
         "--out", "/tmp/e2e_vs_grpo.png"],
        check=True, capture_output=True,
    )

    msg = f"E2E: {n_train} train, {len(evals)} evals"
    if evals:
        msg += f"; latest: @{evals[-1]['step']}={evals[-1]['fve']:.4f}"
    print(msg + "; plot → /tmp/e2e_vs_grpo.png")


if __name__ == "__main__":
    main()
