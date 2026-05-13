"""Overlay multiple training runs' diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    h = json.load(open(path))
    train = [e for e in h if "eval" not in e]
    evals = [e["eval"] for e in h if "eval" in e]
    return train, evals


def smooth(x, w=10):
    if len(x) < w:
        return x, np.arange(len(x))
    sm = np.convolve(x, np.ones(w) / w, mode="valid")
    return sm, np.arange(len(sm)) + (w - 1) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="path1:label1 path2:label2 ...")
    ap.add_argument("--out", default="/tmp/compare_plot.png")
    ap.add_argument("--baseline-fve", type=float, default=0.788)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["C0", "C1", "C2", "C3", "C4"]

    for idx, spec in enumerate(args.runs):
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, Path(spec).parent.name
        col = colors[idx % len(colors)]
        train, evals = load(path)
        if len(train) == 0:
            continue

        # detect schema (v4 uses last_pos_fve key; older uses ar_fve)
        if "last_pos_fve" in train[0]:
            fve_key = "last_pos_fve"
            mse_key = "last_pos_mse"
        else:
            fve_key = "ar_fve"
            mse_key = "L_z" if "L_z" in train[0] else "ar_mse"
        kl_key = "kl_ref"

        steps = np.arange(len(train))
        fve_train = np.array([e[fve_key] for e in train])
        kl = np.array([e[kl_key] for e in train])

        eval_steps = np.array([e["step"] for e in evals])
        eval_fve = np.array([e["fve"] for e in evals])

        ax = axes[0, 0]
        sm, sx = smooth(fve_train, 10)
        ax.plot(sx, sm, color=col, lw=1.0, alpha=0.5, label=f"{label} train (smoothed)")
        if len(evals) > 0:
            ax.plot(eval_steps, eval_fve, "o-", color=col, lw=2, ms=7, label=f"{label} eval")
        ax = axes[0, 1]
        ax.scatter(steps, fve_train, s=3, alpha=0.15, color=col)
        sm, sx = smooth(fve_train, 10)
        ax.plot(sx, sm, color=col, lw=1.5, label=label)

        ax = axes[1, 0]
        sm, sx = smooth(kl, 10)
        ax.plot(sx, sm, color=col, lw=1.5, label=label)

        ax = axes[1, 1]
        if "av_grad_norm" in train[0]:
            ag = np.array([e["av_grad_norm"] for e in train])
            sm, sx = smooth(ag, 10)
            ax.plot(sx, sm, color=col, lw=1.5, label=f"{label} AV grad")

        print(f"\n{label}: {len(train)} train, {len(evals)} evals")
        if len(evals):
            print(f"  evals: " + ", ".join(f"@{int(s)}={f:.4f}" for s, f in zip(eval_steps, eval_fve)))
        print(f"  fve_train start (0-19)={fve_train[:20].mean():.4f}  end (last20)={fve_train[-20:].mean():.4f}")
        print(f"  kl_ref start={kl[:20].mean():.4f}  end={kl[-20:].mean():.4f}")

    axes[0, 0].axhline(args.baseline_fve, color="k", ls="--", lw=1, alpha=0.5)
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("FVE")
    axes[0, 0].set_title("Last-position FVE (eval = ●, train = lines)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axhline(args.baseline_fve, color="k", ls="--", lw=1, alpha=0.5)
    axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("FVE (training-time, single rollout)")
    axes[0, 1].set_title("Per-step train FVE (smoothed, with raw scatter)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("KL")
    axes[1, 0].set_title("KL(π_AV || π_ref) — drift from SFT init")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("‖∇‖_2")
    axes[1, 1].set_title("AV grad norm (smoothed)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
