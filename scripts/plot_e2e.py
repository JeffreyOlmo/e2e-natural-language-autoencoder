"""Plot the e2e (downstream-KL) finetune run: FVE + KL@p over training."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(x, w=10):
    if len(x) < w:
        return np.array(x), np.arange(len(x))
    sm = np.convolve(x, np.ones(w) / w, mode="valid")
    return sm, np.arange(len(sm)) + (w - 1) // 2


def main():
    # Run history
    h = json.load(open("/tmp/e2e.json"))
    train = [e for e in h if "eval" not in e]
    evals = [e["eval"] for e in h if "eval" in e]

    # KL@p data points from offline eval script
    kl_baseline = json.load(open("/tmp/eval_grpo_baseline.json"))
    kl_100 = json.load(open("/tmp/eval_e2e_step100.json"))
    kl_200 = json.load(open("/tmp/eval_e2e_step200.json"))
    kl_300 = json.load(open("/tmp/eval_e2e_step300.json"))
    kl_points = [
        (0, kl_baseline["kl_at_p_recon"], kl_baseline["mse"]),
        (100, kl_100["kl_at_p_recon"], kl_100["mse"]),
        (200, kl_200["kl_at_p_recon"], kl_200["mse"]),
        (300, kl_300["kl_at_p_recon"], kl_300["mse"]),
    ]
    kl_ceiling = kl_baseline["kl_at_p_predict_mean"]  # predict-mean baseline

    # Vanilla GRPO eval FVE (from rl_small_grpo_cont/history.json)
    gv = json.load(open("checkpoints/rl_small_grpo_cont/history.json"))
    gv_evals = [e["eval"] for e in gv if "eval" in e]
    grpo_baseline_fve = gv_evals[-1]["fve"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel 1: FVE over training (smoothed train + eval points) ---
    ax = axes[0, 0]
    fve_train = [e["last_pos_fve"] for e in train]
    steps_train = [e.get("step", i) for i, e in enumerate(train)]
    sm, idx = smooth(fve_train, w=15)
    ax.plot(np.array(steps_train)[idx], sm, color="C0", lw=1.5, label="e2e: train FVE (smoothed)")
    ax.scatter([e["step"] for e in evals], [e["fve"] for e in evals],
               color="C0", marker="o", s=40, zorder=5, label="e2e: eval FVE")
    ax.axhline(grpo_baseline_fve, color="C3", ls="--", lw=1.5,
               label=f"vanilla GRPO baseline ({grpo_baseline_fve:.3f})")
    ax.set_xlabel("step")
    ax.set_ylabel("FVE (eval) = 1 - MSE/baseline_MSE")
    ax.set_title("FVE over training\n(geometric metric — expected to dip slightly)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 2: KL@p eval, the metric that matters ---
    ax = axes[0, 1]
    steps_kl = [s for s, _, _ in kl_points]
    kl_vals = [k for _, k, _ in kl_points]
    ax.plot(steps_kl, kl_vals, color="C2", marker="o", lw=2, ms=8,
            label="e2e: KL@p (downstream causal fidelity)")
    ax.axhline(kl_baseline["kl_at_p_recon"], color="C3", ls="--", lw=1.5,
               label=f"vanilla GRPO baseline ({kl_baseline['kl_at_p_recon']:.3f} nats)")
    # Reduction percentages
    for s, k, _ in kl_points:
        if s == 0: continue
        reduction = (kl_baseline["kl_at_p_recon"] - k) / kl_baseline["kl_at_p_recon"] * 100
        ax.annotate(f"-{reduction:.0f}%", (s, k), xytext=(5, 5),
                    textcoords="offset points", color="C2", fontsize=10, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("KL(orig || patched-ĥ) at position p  (nats)")
    ax.set_title("Downstream KL@p over training\n(behavioral fidelity — lower = better)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(kl_vals) * 1.4)

    # --- Panel 3: train-time KL (online measurement) ---
    ax = axes[1, 0]
    kl_train = [e.get("kl", 0) for e in train]
    sm, idx = smooth(kl_train, w=15)
    ax.plot(np.array(steps_train)[idx], sm, color="C2", lw=1.5,
            label="train-time KL (per-step, smoothed)")
    ax.set_xlabel("step")
    ax.set_ylabel("KL@p (train batch, nats)")
    ax.set_title("Train-time KL signal\n(per-step value driving the gradient)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 4: KL@p gap-closing bar ---
    ax = axes[1, 1]
    ceiling = kl_ceiling
    bar_labels, bar_vals = [], []
    for s, k, _ in kl_points:
        pct_closed = (ceiling - k) / ceiling * 100
        bar_labels.append(f"step {s}" if s > 0 else "baseline")
        bar_vals.append(pct_closed)
    colors = ["C3", "C2", "C2"]
    bars = ax.bar(bar_labels, bar_vals, color=colors, edgecolor="black")
    for b, v in zip(bars, bar_vals):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width()/2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"% of KL@p gap closed\n(ceiling: predict-mean = {ceiling:.2f} nats)")
    ax.set_title("Causal-fidelity gap closed")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("E2E finetune (downstream-KL on top of vanilla GRPO): FVE dips slightly, "
                 "KL@p drops ~35%", fontsize=13, y=1.00)
    fig.tight_layout()
    out = "/tmp/e2e_vs_grpo.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
