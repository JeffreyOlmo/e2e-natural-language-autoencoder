"""Plot per-position-grad-distill training diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-fve", type=float, default=0.788)
    args = ap.parse_args()

    h = json.load(open(args.history))
    train = [e for e in h if "eval" not in e]
    evals = [e["eval"] for e in h if "eval" in e]

    steps = np.array([e["step"] for e in train])
    L_total = np.array([e["L_total"] for e in train])
    last_mse = np.array([e["last_pos_mse"] for e in train])
    last_fve = np.array([e["last_pos_fve"] for e in train])
    gd_kl = np.array([e["gd_kl"] for e in train])
    kl_ref = np.array([e["kl_ref"] for e in train])
    av_loss = np.array([e["av_loss"] for e in train])
    gn_early = np.array([e["g_norm_early"] for e in train])
    gn_mid = np.array([e["g_norm_mid"] for e in train])
    gn_late = np.array([e["g_norm_late"] for e in train])
    gn_ratio = np.array([e["g_norm_ratio_early_late"] for e in train])
    av_gn = np.array([e["av_grad_norm"] for e in train])
    ar_gn = np.array([e["ar_grad_norm"] for e in train])
    eval_steps = np.array([e["step"] for e in evals])
    eval_fve = np.array([e["fve"] for e in evals])

    def smooth(x, w=10):
        if len(x) < w: return x, np.arange(len(x))
        sm = np.convolve(x, np.ones(w)/w, mode="valid")
        return sm, np.arange(len(sm)) + (w - 1) // 2

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # FVE / loss
    ax = axes[0, 0]
    sm, sx = smooth(last_fve)
    ax.plot(sx, sm, color="C0", lw=1.5, label="last-pos FVE (training, smoothed)")
    ax.scatter(steps, last_fve, s=3, alpha=0.2, color="C0")
    if len(evals) > 0:
        ax.plot(eval_steps, eval_fve, "o-", color="C3", lw=2, ms=7, label="eval FVE (150 held out)")
    ax.axhline(args.baseline_fve, color="k", ls="--", lw=1, alpha=0.6, label=f"kitft baseline ({args.baseline_fve})")
    ax.set_xlabel("step"); ax.set_ylabel("FVE")
    ax.set_title("Last-position FVE — the eval metric")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sm, sx = smooth(L_total)
    ax.plot(sx, sm, color="C1", lw=1.5, label="L_total = mean over positions")
    ax.scatter(steps, L_total, s=3, alpha=0.2, color="C1")
    sm2, sx2 = smooth(last_mse)
    ax.plot(sx2, sm2, color="C2", lw=1.5, label="last-pos MSE")
    ax.set_xlabel("step"); ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE — training objective vs eval-equivalent")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # losses
    ax = axes[1, 0]
    sm, sx = smooth(gd_kl)
    ax.plot(sx, sm, color="C4", lw=1.5, label="grad-distill KL (smoothed)")
    ax.scatter(steps, gd_kl, s=3, alpha=0.2, color="C4")
    ax.set_xlabel("step"); ax.set_ylabel("KL(q || π)")
    ax.set_title("Grad-distill teacher KL (= the actual training loss)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    sm, sx = smooth(kl_ref)
    ax.plot(sx, sm, color="C5", lw=1.5, label="KL(π_AV ‖ π_ref) smoothed")
    ax.scatter(steps, kl_ref, s=3, alpha=0.2, color="C5")
    ax.set_xlabel("step"); ax.set_ylabel("KL")
    ax.set_title("Drift from SFT init")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # grad norms across positions
    ax = axes[2, 0]
    for name, arr, c in [("early ||g||", gn_early, "C0"), ("mid ||g||", gn_mid, "C1"), ("late ||g||", gn_late, "C2")]:
        sm, sx = smooth(arr)
        ax.plot(sx, sm, color=c, lw=1.3, label=name)
    ax.set_xlabel("step"); ax.set_ylabel("||g_s||")
    ax.set_title("Per-rollout-position gradient norms (binned)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    sm, sx = smooth(gn_ratio)
    ax.plot(sx, sm, color="C6", lw=1.5, label="early/late ratio (smoothed)")
    ax.scatter(steps, gn_ratio, s=3, alpha=0.2, color="C6")
    ax.axhline(10, color="r", ls="--", lw=1, alpha=0.6, label="concern threshold (10x)")
    ax.set_xlabel("step"); ax.set_ylabel("ratio")
    ax.set_title("Gradient-norm early/late ratio")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"per-position grad-distill — {Path(args.history).parent.name}", fontsize=13)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.history).parent / "plots.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")

    print(f"\n=== summary ===")
    print(f"baseline FVE: {args.baseline_fve}")
    print(f"eval FVE: first={eval_fve[0]:.4f} last={eval_fve[-1]:.4f} peak={eval_fve.max():.4f} (step {eval_steps[eval_fve.argmax()]})")
    print(f"L_total: start={L_total[:20].mean():.4f} end={L_total[-20:].mean():.4f}  Δ={100*(L_total[-20:].mean() - L_total[:20].mean())/L_total[:20].mean():+.1f}%")
    print(f"last_mse: start={last_mse[:20].mean():.4f} end={last_mse[-20:].mean():.4f}  Δ={100*(last_mse[-20:].mean() - last_mse[:20].mean())/last_mse[:20].mean():+.1f}%")
    print(f"gd_kl:   start={gd_kl[:20].mean():.4f} end={gd_kl[-20:].mean():.4f}")
    print(f"kl_ref:  start={kl_ref[:20].mean():.4f} end={kl_ref[-20:].mean():.4f}")
    print(f"early/late g-norm ratio: start={gn_ratio[:20].mean():.2f} end={gn_ratio[-20:].mean():.2f}")


if __name__ == "__main__":
    main()
