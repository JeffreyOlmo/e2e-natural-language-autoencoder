"""Single source of truth for all NLA hyperparameters.

Convention: `layer` indexes into HF `output_hidden_states` (so `hidden_states[layer]`
is the residual stream after `layer` transformer blocks). Qwen2.5-0.5B has 24
blocks; layer=16 ≈ 2/3 depth, matching the paper's open-model layer choices.
"""
from dataclasses import dataclass, field


@dataclass
class NLAConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    layer: int = 16
    d_model: int = 896

    # Injection: AV replaces marker_token's embedding with alpha * (h / ||h||).
    # alpha is the 75th-percentile L2 norm of layer-`layer` activations,
    # measured on FineWeb (see scripts/measure_alpha.py, data/alpha_L16.json).
    alpha: float = 22.37  # Qwen2.5-0.5B-Instruct @ L16, n=10000 vectors
    marker_token: str = "<|fim_pad|>"  # single special token id 151662, never appears in normal text

    # AR = first `layer` transformer blocks of base + Linear(d_model, d_model)
    # value head, identity-init. Reads at last-token suffix-anchored position.
    # Loss is MSE on L2-normalized vectors (so MSE = 2(1−cos)).
    mse_norm: float = 1.0  # both pred and gold normalized to this norm before MSE

    # Activation-collection
    min_position: int = 50  # paper invariant — earlier positions decode to noise
    max_context: int = 4096
    vectors_per_doc: int = 5

    # Training (placeholders — tune later)
    sft_lr: float = 2e-5
    sft_batch_size: int = 256
    sft_micro_batch_size: int = 16

    rl_actor_lr: float = 1e-5
    rl_critic_lr: float = 5e-5
    rl_batch_size: int = 128

    # Grad-distill specific
    # τ in `q_t ∝ π_AV · exp(s/τ)` IS the implicit β for the per-step closed-form
    # optimum (KL-regularized RL with π_ref = current AV). Score spread scales as
    # ~‖e_v‖/τ ≈ 0.5/τ; we need it ~2-3 nats to reorder a peaked AV. Hence τ≈0.1.
    tau: float = 0.1
    # Static anchor toward SFT-init (or base for cold-start). Distinct from τ:
    # τ bounds per-step movement, kl_to_ref bounds cumulative drift.
    kl_to_sft_coef: float = 0.05
    pi_ref_mode: str = "current_av"  # "uniform" or "current_av"

    # Paths
    data_dir: str = "data"
    ckpt_dir: str = "checkpoints"

    def __post_init__(self):
        # K+1 layer convention check: hidden_states[layer] is the activation we
        # reconstruct; AR truncates to the first `layer` transformer blocks.
        assert 1 <= self.layer, "layer must be ≥ 1"
