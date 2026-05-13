# e2e Natural Language Autoencoders

This repository is a replication and extension of Anthropic's Natural Language
Autoencoders (NLA). An NLA consists of two adapters trained on top of a frozen
subject LM (here, Qwen2.5 base): an **AV** (verbalizer) that takes a hidden-state
activation from some layer of the subject model and emits a short natural-language
explanation of what that activation represents, and an **AR** (reconstructor) that
reads the explanation and reconstructs the activation. Together (AV then AR) they
form an autoencoder whose latent code is a human-readable English sentence, which
is what makes NLAs interesting as an interpretability tool.

The extension implemented here is an **end-to-end downstream-KL training
objective** (the `_e2e` variants). In addition to the geometric MSE that
matches the AR's reconstruction to the original activation, we patch the
reconstructed activation back into the frozen subject LM at the source layer
and add a KL term between the subject LM's next-token distribution on the
original vs. reconstructed activation. This pushes the autoencoder to preserve
not just the geometry of the activation but its *causal effect on downstream
predictions*, in the spirit of Karvonen et al. (arXiv:2503.17272) and
Braun et al.'s end-to-end SAE work. The replication uses a softmax
gradient-distillation teacher (`q_t = softmax(-<g_t, e_v>/tau)`) on the AV
side rather than the GRPO objective in the original NLA paper.

The repo includes both **0.5B** (single-GPU) and **7B** (FSDP, multi-GPU)
training pipelines, downstream-KL evals, and the behavioral evals from the NLA
paper — SelfDescribe (gender), factuality grading by an LLM subagent, CoT-Hints,
and two-hop reasoning — which we use to compare an e2e-trained NLA against a
matched control trained on geometric MSE alone.

## Layout

- `nla/` — core modules: `model.py` (AV/AR adapters), `injection.py`
  (patching reconstructed activations back into the subject LM),
  `grad_distill.py` (softmax gradient-distillation teacher), `loss.py`,
  `rollout.py`, `data.py`, `prompts.py`, `config.py`, `relax.py`.
- `scripts/` — training, eval, and plotting scripts.
  - Training: `train_small_rl_e2e.py` (0.5B + e2e KL), `train_fsdp_grpo_e2e.py`
    (7B FSDP + e2e KL), plus matched non-e2e baselines (`train_small_rl.py`,
    `train_fsdp_grpo.py`-style), SFT warmups (`sft_av.py`, `sft_ar.py`).
  - Evals: `eval_downstream_kl.py` / `eval_downstream_kl_7b.py`,
    `eval_selfdescribe_gender.py`, `eval_factual_accuracy.py`,
    `cot_hints_inference.py`, `eval_two_hop.py`, `eval_cross_layer_div.py`,
    `eval_concept_steering.py`, `eval_kitft.py`.
  - Plots: `plot_e2e.py`, `plot_compare.py`, `plot_history.py`,
    `plot_4way.py`, `plot_per_pos.py`, etc.
- `requirements.txt` — pinned-ish deps (torch >=2.7, transformers >=4.54,
  datasets, accelerate, pyarrow).

## Running a training

0.5B / single-GPU e2e training:

```bash
python scripts/train_small_rl_e2e.py
```

7B / FSDP e2e training (launch with `torchrun` across the GPUs you have):

```bash
torchrun --nproc_per_node=8 scripts/train_fsdp_grpo_e2e.py
```

Both scripts read run-specific knobs (subject model, layer index, KL weight,
teacher temperature `tau`, learning rate, batch size, checkpoint dir) from
flags / `nla/config.py`. Activation corpora are produced ahead of time with
`scripts/collect_activations.py`. Behavioral evals expect a trained AV/AR
checkpoint and run against the same frozen subject model used for training.

## Notes

- Checkpoints, activation parquets, and training logs are excluded from the
  repo (see `.gitignore`); they are large and regenerable from the scripts here.
- Inference and training are intended to run on GPU; CPU paths are not
  exercised.
