"""Plot training-history diagnostics for an FSDP run."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="checkpoints/rl_fsdp_v2/history.json")
    ap.add_argument("--out", default="checkpoints/rl_fsdp_v2/plots.png")
    ap.add_argument("--baseline-fve", type=float, default=0.788)
    args = ap.parse_args()

    h = json.load(open(args.history))
    train = [e for e in h if "eval" not in e]
    evals = [e["eval"] for e in h if "eval" in e]

    steps = np.array([e["step"] for e in train])
    ar_fve = np.array([e["ar_fve"] for e in train])
    gd_kl = np.array([e["gd_kl"] for e in train])
    kl_ref = np.array([e["kl_ref"] for e in train])
    av_loss = np.array([e["av_loss"] for e in train])
    grad_norm = np.array([e["av_grad_norm"] for e in train])
    skipped = np.array([e.get("av_skipped", 0.0) for e in train])
    rollout_len = np.array([e["rollout_len_mean"] for e in train])

    eval_steps = np.array([e["step"] for e in evals])
    eval_fve = np.array([e["fve"] for e in evals])

    def smooth(x, w=10):
        if len(x) < w: return x
        return np.convolve(x, np.ones(w)/w, mode="valid")
    sm_steps = steps[len(steps) - len(smooth(ar_fve)):]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.scatter(steps, ar_fve, s=3, alpha=0.3, color="C0", label="train (per-step)")
    ax.plot(sm_steps, smooth(ar_fve), color="C0", lw=1.5, label="train smoothed (w=10)")
    ax.plot(eval_steps, eval_fve, "o-", color="C3", lw=2, ms=8, label="eval")
    ax.axhline(args.baseline_fve, color="k", ls="--", lw=1, label=f"kitft baseline {args.baseline_fve}")
    ax.set_xlabel("step"); ax.set_ylabel("FVE")
    ax.set_title("FVE trajectory")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(sm_steps, smooth(gd_kl), color="C1", lw=1.5, label="grad-distill KL (smoothed)")
    ax.scatter(steps, gd_kl, s=3, alpha=0.3, color="C1")
    ax.set_xlabel("step"); ax.set_ylabel("KL(q || π_AV)")
    ax.set_title("Grad-distill loss (per-position avg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(sm_steps, smooth(kl_ref), color="C2", lw=1.5, label="KL(π_AV || π_ref) smoothed")
    ax.scatter(steps, kl_ref, s=3, alpha=0.3, color="C2")
    ax.set_xlabel("step"); ax.set_ylabel("KL")
    ax.set_title("Drift from SFT init")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(sm_steps, smooth(av_loss), color="C4", lw=1.5, label="total av_loss smoothed")
    ax.plot(sm_steps, smooth(gd_kl), color="C1", lw=1, alpha=0.7, label="gd_kl")
    ax.plot(sm_steps, smooth(0.05 * kl_ref), color="C2", lw=1, alpha=0.7, label="0.05 · kl_ref")
    ax.set_xlabel("step"); ax.set_ylabel("loss components")
    ax.set_title("AV loss decomposition")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(sm_steps, smooth(grad_norm), color="C5", lw=1.5, label="grad norm (smoothed)")
    ax.scatter(steps, grad_norm, s=3, alpha=0.3, color="C5")
    n_skipped = int(skipped.sum())
    ax.set_xlabel("step"); ax.set_ylabel("‖∇‖_2")
    ax.set_title(f"AV grad norm  ({n_skipped} steps skipped due to NaN/inf)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(steps, rollout_len, color="C6", lw=1, label="rollout_len mean")
    ax.set_xlabel("step"); ax.set_ylabel("tokens")
    ax.set_title("Rollout length (mean over batch)")
    ax.set_ylim(0, max(rollout_len.max() + 5, 140))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Training diagnostics — {Path(args.history).parent.name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")

    print("\n=== summary stats ===")
    print(f"baseline FVE: {args.baseline_fve}")
    print(f"eval FVE: min={eval_fve.min():.4f} max={eval_fve.max():.4f} final={eval_fve[-1]:.4f}")
    print(f"train ar_fve (last 20 steps): mean={ar_fve[-20:].mean():.4f}")
    print(f"gd_kl: start={gd_kl[:10].mean():.4f}  end={gd_kl[-10:].mean():.4f}  Δ={gd_kl[-10:].mean()-gd_kl[:10].mean():+.4f}")
    print(f"kl_ref: start={kl_ref[:10].mean():.4f}  end={kl_ref[-10:].mean():.4f}  Δ={kl_ref[-10:].mean()-kl_ref[:10].mean():+.4f}")
    print(f"grad_norm: start={grad_norm[:10].mean():.3f}  end={grad_norm[-10:].mean():.3f}")
    print(f"steps skipped: {n_skipped}/{len(steps)}")


if __name__ == "__main__":
    main()
