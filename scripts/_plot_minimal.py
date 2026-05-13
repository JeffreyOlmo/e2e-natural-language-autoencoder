"""Replot minimum-prior cold-start GRPO trajectory."""
import json
import subprocess


def main():
    subprocess.run(
        ["python3", "scripts/log_to_history.py",
         "logs/rl_minimal_prior.log", "--out", "/tmp/minimal.json"],
        check=True, capture_output=True,
    )
    h = json.load(open("/tmp/minimal.json"))
    n_train = sum(1 for e in h if "eval" not in e)
    evals = [e["eval"] for e in h if "eval" in e]

    subprocess.run(
        ["python3", "scripts/plot_compare.py",
         "--runs",
         "/tmp/minimal.json:minimum-prior",
         "--baseline-fve", "0.645",
         "--out", "/tmp/minimal.png"],
        check=True, capture_output=True,
    )

    msg = f"minimal: {n_train} train, {len(evals)} evals"
    if evals:
        msg += f"; latest: @{evals[-1]['step']}={evals[-1]['fve']:.4f}"
    print(msg + "; plot → /tmp/minimal.png")


if __name__ == "__main__":
    main()
