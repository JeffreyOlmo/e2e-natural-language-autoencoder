"""Live plot of a training run from its tqdm log.

Parses lines like:
  ... coef=+0.000, fve=0.764, gn=0.01, kl=0.000, rl=130
and the eval lines:
  [eval @ 49] FVE=0.8512, MSE=0.1374
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_RE = re.compile(r"coef=([+-]?\d+\.\d+), fve=(-?\d+\.\d+), gn=(-?\d+\.\d+), kl=(-?\d+\.\d+), rl=(\d+)")
EVAL_RE = re.compile(r"\[eval @ (\d+)\] FVE=(-?\d+\.\d+), MSE=(-?\d+\.\d+)")


def parse(log_path):
    text = Path(log_path).read_text()
    # tqdm uses \r — split on both \r and \n
    text = text.replace("\r", "\n")
    train = []
    evals = []
    seen_step = -1
    for line in text.split("\n"):
        m = TRAIN_RE.search(line)
        if m:
            # multiple postfix updates per step happen as tqdm refreshes — keep last
            train.append([float(m.group(1)), float(m.group(2)), float(m.group(3)),
                          float(m.group(4)), int(m.group(5))])
        m = EVAL_RE.search(line)
        if m:
            evals.append([int(m.group(1)), float(m.group(2)), float(m.group(3))])
    return np.array(train), np.array(evals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="One or more log files to overlay.")
    ap.add_argument("--out", default="checkpoints/_running_plot.png")
    ap.add_argument("--baseline-fve", type=float, default=0.788)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = ["C0", "C1", "C2", "C3", "C4"]

    for idx, log_path in enumerate(args.logs):
        name = Path(log_path).stem
        col = colors[idx % len(colors)]
        train, evals = parse(log_path)
        if len(train) == 0:
            print(f"  {name}: no train rows yet")
            continue
        steps = np.arange(len(train))
        coef = train[:, 0]
        fve_train = train[:, 1]
        gn = train[:, 2]
        kl = train[:, 3]

        # smooth
        def smooth(x, w=10):
            if len(x) < w: return x, np.arange(len(x))
            sm = np.convolve(x, np.ones(w)/w, mode="valid")
            return sm, np.arange(len(sm)) + (w - 1) // 2

        ax = axes[0, 0]
        ax.scatter(steps, fve_train, s=3, alpha=0.2, color=col)
        sm, sx = smooth(fve_train)
        ax.plot(sx, sm, color=col, lw=1.2, label=f"{name} train (smoothed w=10)")
        if len(evals) > 0:
            ax.plot(evals[:, 0], evals[:, 1], "o-", color=col, lw=2, ms=8, label=f"{name} eval")
        ax.axhline(args.baseline_fve, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_title("FVE"); ax.set_xlabel("step"); ax.set_ylabel("FVE")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="lower right")

        ax = axes[0, 1]
        sm, sx = smooth(gn)
        ax.plot(sx, sm, color=col, lw=1.2, label=name)
        ax.scatter(steps, gn, s=3, alpha=0.2, color=col)
        ax.set_title("AV grad norm"); ax.set_xlabel("step"); ax.set_ylabel("‖∇‖_2")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

        ax = axes[1, 0]
        sm, sx = smooth(kl)
        ax.plot(sx, sm, color=col, lw=1.2, label=name)
        ax.scatter(steps, kl, s=3, alpha=0.2, color=col)
        ax.set_title("KL(π_AV ‖ π_ref)"); ax.set_xlabel("step"); ax.set_ylabel("KL")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

        ax = axes[1, 1]
        sm, sx = smooth(coef)
        ax.plot(sx, sm, color=col, lw=1.2, label=name)
        ax.scatter(steps, coef, s=3, alpha=0.2, color=col)
        ax.set_title("RELAX coef = s[z] - π·s   (≈0 means π is at its 'right' mass)")
        ax.set_xlabel("step"); ax.set_ylabel("coef")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

        print(f"\n{name}: {len(train)} train steps, {len(evals)} evals")
        if len(evals) > 0:
            print(f"  evals: " + " ".join(f"@{int(s)}={f:.4f}" for s, f, _ in evals))
        print(f"  fve_train last 20: mean={fve_train[-20:].mean():.4f}")
        print(f"  gn last 20: mean={gn[-20:].mean():.4f}")
        print(f"  kl last 20: mean={kl[-20:].mean():.4f}")
        print(f"  coef last 20: mean={coef[-20:].mean():.4f}  abs_max={np.abs(coef).max():.4f}")

    fig.suptitle("Running training comparison", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
