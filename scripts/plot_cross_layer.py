"""Plot cross-layer residual divergence: vanilla GRPO vs e2e step-300."""
import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    g = json.load(open("/tmp/cross_layer_grpo.json"))
    e = json.load(open("/tmp/cross_layer_e2e.json"))
    hs = g["hidden_state_indices"]
    gp = np.array(g["div_at_p"])
    ep = np.array(e["div_at_p"])
    go = np.array(g["div_other_positions"])
    eo = np.array(e["div_other_positions"])
    ratio = ep / np.maximum(gp, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: divergence at position p
    ax = axes[0]
    ax.plot(hs, gp, "o-", color="C3", lw=2, ms=8, label="vanilla GRPO")
    ax.plot(hs, ep, "o-", color="C2", lw=2, ms=8, label="e2e step-300")
    ax.set_xlabel("hidden_states[L]  (output of layer L−1)")
    ax.set_ylabel("|| h_orig − h_patched || / || h_orig ||  at position p")
    ax.set_title("Cross-layer divergence AT position p\ne2e is ~9% closer to in-distribution at every layer")
    ax.axvline(16, color="gray", ls="--", alpha=0.5, label="patch site (L=16)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Annotate % reduction
    for h, g_v, e_v in zip(hs, gp, ep):
        red = (g_v - e_v) / g_v * 100
        ax.annotate(f"-{red:.0f}%", (h, e_v), xytext=(0, -14), ha="center",
                    textcoords="offset points", color="C2", fontsize=8)

    # Panel 2: divergence at OTHER positions (attention propagation)
    ax = axes[1]
    ax.plot(hs, go, "o-", color="C3", lw=2, ms=8, label="vanilla GRPO")
    ax.plot(hs, eo, "o-", color="C2", lw=2, ms=8, label="e2e step-300")
    ax.set_xlabel("hidden_states[L]")
    ax.set_ylabel("mean || h_orig − h_patched || / || h_orig ||  at other positions")
    ax.set_title("Divergence at OTHER positions (attention spread)\ne2e ≈ GRPO — collateral damage is unchanged")
    ax.axvline(16, color="gray", ls="--", alpha=0.5, label="patch site (L=16)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("Cross-layer residual divergence — e2e produces a more behaviorally faithful ĥ", y=1.02, fontsize=13)
    fig.tight_layout()
    out = "/tmp/cross_layer.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
