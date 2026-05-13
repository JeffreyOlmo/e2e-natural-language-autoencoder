"""E2E (downstream-KL-augmented) GRPO finetune for the 0.5B NLA setup.

Builds on vanilla GRPO (`train_small_rl.py --method grpo`) by augmenting the
training signal with a downstream KL penalty in the spirit of:

  - Braun et al. 2024 (e2e SAE training)
  - Karvonen et al. 2025 (https://arxiv.org/pdf/2503.17272)

Setup:
  - AR's reconstruction ĥ is patched into the FULL frozen Qwen2.5-0.5B-Instruct
    at layer 16, the SAME position p where the activation was originally
    collected, in the SAME source `text`.
  - We compare patched logits (with ĥ swapped in) vs original logits (with the
    natural h, i.e. unhooked forward) at downstream positions (p+1 .. end of ctx).
  - Per-batch dynamic balancing: α_KL = (MSE / (KL + eps)).detach() — so MSE
    and α_KL·KL have equal magnitude per the paper's recipe.
  - AR gradient: L_ar = MSE + α_KL · KL (single backward; KL grad flows back to
    AR via the LM's upper layers + the patch hook).
  - AV (GRPO) advantage: standardize -(MSE + α_KL·KL) within each K-group →
    scalar advantage per rollout, broadcast over tokens.
  - LR-decay on top of the warm-started vanilla-GRPO checkpoint.

The frozen "downstream LM" is the same Qwen2.5-0.5B-Instruct the activations
were collected from — so patching ĥ at layer 16 position p is the *exact*
counterfactual: "what does the LM predict next if we replace this position's
residual with the reconstruction?"

Run with torchrun:
  torchrun --standalone --nproc-per-node=8 scripts/train_small_rl_e2e.py \\
      --ar-init checkpoints/rl_small_grpo_cont/ar_step_500.pt \\
      --av-init checkpoints/rl_small_grpo_cont/av_step_500.pt \\
      --av-ref  checkpoints/av_sft_kitft/av.pt \\
      --out checkpoints/rl_small_grpo_e2e --steps 300
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import NLAConfig
from nla.model import ARModel

# Reuse helpers from the main RL script
from scripts.train_small_rl import (
    AR_PREFIX,
    AR_SUFFIX,
    build_ar_inputs,
    build_av_prompt_embeds,
    ddp_generate,
    is_main,
    kl_to_ref,
    teacher_force_logits,
)


def load_records_with_text(path):
    """Load activations parquet → (acts [N,d], texts [N], positions [N])."""
    t = pq.read_table(path)
    n = len(t)
    flat = np.asarray(
        t["activation"].combine_chunks().values.to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )
    d = flat.shape[0] // n
    acts = torch.from_numpy(flat.reshape(n, d).copy())
    texts = t["text"].to_pylist()
    positions = np.asarray(t["position"].to_pylist(), dtype=np.int64)
    return acts, texts, positions


def tokenize_contexts(tok, texts, positions, max_ctx, device):
    """Tokenize raw context texts to [B, T_ctx] (left-truncated if needed to
    keep position p in range)."""
    B = len(texts)
    pad_id = tok.pad_token_id
    rows = []
    pos_in_ctx = []
    for i in range(B):
        enc = tok(texts[i], add_special_tokens=False).input_ids
        ids = enc[:max_ctx]
        p = int(positions[i])
        rows.append(ids)
        pos_in_ctx.append(p)
    T = max(len(r) for r in rows)
    ctx_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    ctx_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        ctx_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        ctx_mask[i, : len(r)] = 1
    return ctx_ids, ctx_mask, torch.tensor(pos_in_ctx, device=device, dtype=torch.long)


def make_patch_state():
    """Returns a dict that the hook reads. Set 'h_hat' [B,d] and 'positions' [B]
    before the LM forward; the hook will scatter h_hat[b] into the layer's
    output at position positions[b]. Set 'h_hat' to None to disable patching.
    """
    return {"h_hat": None, "positions": None}


def make_patch_hook(state):
    def hook(module, inputs, output):
        if state["h_hat"] is None:
            return output  # passthrough
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        B = h.shape[0]
        idx = torch.arange(B, device=h.device)
        new_h = h.clone()
        # Cast ĥ to layer-output dtype so the hook is dtype-safe even if LM is fp16.
        new_h[idx, state["positions"]] = state["h_hat"].to(new_h.dtype)
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--ar-init", required=True)
    ap.add_argument("--av-init", required=True)
    ap.add_argument(
        "--av-ref",
        default=None,
        help="Frozen AV used for KL-to-ref anchor. Defaults to --av-init.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--per-rank-h", type=int, default=1)
    ap.add_argument("--rollouts-per-h", type=int, default=4)
    ap.add_argument("--av-lr", type=float, default=2e-6)
    ap.add_argument("--ar-lr", type=float, default=1e-5)
    ap.add_argument("--lr-final-frac", type=float, default=0.0,
                    help="Linear decay LR to this fraction of initial by end of run.")
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.05)
    ap.add_argument("--downstream-kl-coef", type=float, default=1.0,
                    help="Global multiplier on the downstream-KL term (on TOP of the "
                         "dynamic α_KL = MSE/(KL+eps).detach()). 0 disables (= vanilla GRPO).")
    ap.add_argument("--max-ctx-tokens", type=int, default=256,
                    help="Max tokens of source context used for downstream KL forward.")
    ap.add_argument("--kl-min-future", type=int, default=1,
                    help="Filter dataset to records with `p + kl_min_future < max_ctx_tokens` "
                         "so position p fits inside the context window.")
    ap.add_argument("--lm-dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=10)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    torch.manual_seed(args.seed + rank)

    cfg = NLAConfig()
    sc = cfg.mse_norm
    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"DDP GRPO+E2E  world={world}")
        print(f"  layer={cfg.layer} d={cfg.d_model} α={cfg.alpha}")
        print(f"  per_rank_h={args.per_rank_h} K={args.rollouts_per_h}")
        print(f"  downstream_kl_coef={args.downstream_kl_coef}  max_ctx={args.max_ctx_tokens}")

    # ---- Activations + texts ----
    activations, texts, positions = load_records_with_text(args.activations)
    n = len(texts)
    n_eval = max(1, n // 20)
    train_idx_all = list(range(n - n_eval))
    eval_idx_all = list(range(n - n_eval, n))

    # Filter train to records where (position) leaves at least kl_min_future
    # positions within max_ctx for downstream KL signal.
    def keep(i):
        p = int(positions[i])
        return p + args.kl_min_future < args.max_ctx_tokens

    train_indices = [i for i in train_idx_all if keep(i)]
    eval_indices = eval_idx_all  # eval uses MSE only; no filter needed
    if is_main():
        print(f"  total={n}, train_filtered={len(train_indices)}/{len(train_idx_all)}, eval={len(eval_indices)}")

    # Baseline MSE
    h_eval = activations[eval_indices]
    mu = activations[train_indices].mean(dim=0)
    p_b = F.normalize(mu.expand_as(h_eval).float(), dim=-1) * sc
    g_b = F.normalize(h_eval.float(), dim=-1) * sc
    base_mse = ((p_b - g_b) ** 2).mean(dim=-1).mean().item()
    if is_main():
        print(f"  predict-mean baseline MSE = {base_mse:.5f}")

    # ---- Tokenizer ----
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)

    def load_sd(path):
        x = torch.load(path, map_location=device, weights_only=False)
        return x["state_dict"] if isinstance(x, dict) and "state_dict" in x else x

    # ---- AV (full Qwen) ----
    if is_main():
        print(f"Loading AV: {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    av.load_state_dict(load_sd(args.av_init))
    av.train()
    av = DDP(av, device_ids=[local])

    # AV_ref (frozen)
    use_av_ref = args.kl_to_ref_coef > 0
    if use_av_ref:
        ref_path = args.av_ref or args.av_init
        if is_main():
            print(f"Loading AV_ref (frozen): {ref_path}")
        av_ref = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=torch.float32
        ).to(device)
        av_ref.load_state_dict(load_sd(ref_path))
        av_ref.eval()
        for p in av_ref.parameters():
            p.requires_grad = False
    else:
        av_ref = None

    # ---- AR (truncated body + value_head) ----
    if is_main():
        print(f"Loading AR: {args.ar_init}")
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    sd_ar = load_sd(args.ar_init)
    ar_state = ar.state_dict()
    sd_ar_filtered = {k: v for k, v in sd_ar.items() if k in ar_state}
    ar.load_state_dict(sd_ar_filtered, strict=False)
    ar.train()
    ar = DDP(ar, device_ids=[local])

    # ---- Downstream LM (full Qwen, frozen) ----
    if args.downstream_kl_coef > 0:
        lm_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                    "float32": torch.float32}[args.lm_dtype]
        if is_main():
            print(f"Loading downstream LM (frozen, dtype={args.lm_dtype}): {cfg.base_model}")
        lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=lm_dtype).to(device)
        lm.eval()
        for p in lm.parameters():
            p.requires_grad = False
        # Hook layer 15 (0-indexed) — its output = hidden_states[16] = our activation
        patch_state = make_patch_state()
        hook_handle = lm.model.layers[cfg.layer - 1].register_forward_hook(make_patch_hook(patch_state))
    else:
        lm = None
        patch_state = None
        hook_handle = None

    av_opt = torch.optim.AdamW(av.parameters(), lr=args.av_lr, betas=(0.9, 0.95),
                               eps=1e-8, weight_decay=0.01)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95),
                               eps=1e-8, weight_decay=0.01)

    def lr_at(step):
        progress = step / max(1, args.steps - 1)
        return 1.0 - (1.0 - args.lr_final_frac) * min(progress, 1.0)

    rng = np.random.default_rng(args.seed + rank * 1000)
    K = args.rollouts_per_h
    B_h = args.per_rank_h
    B_eff = B_h * K

    def train_step():
        # 1. Sample h, text, position
        idxs = rng.choice(len(train_indices), size=B_h, replace=False)
        rec_ids = [train_indices[i] for i in idxs]
        h_batch = activations[rec_ids].to(device)
        text_batch = [texts[i] for i in rec_ids]
        pos_batch = positions[rec_ids]
        h_expanded = h_batch.repeat_interleave(K, dim=0) if K > 1 else h_batch

        # 2. K rollouts
        prompt_embeds, prompt_mask = build_av_prompt_embeds(
            av, tok, h_expanded, marker_id, cfg.alpha, device, torch.float32,
            no_prompt=False,
        )
        av.eval()
        rollout_ids = ddp_generate(
            av, prompt_embeds, prompt_mask,
            max_new_tokens=args.max_new_tokens,
            eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
            temperature=args.rollout_temperature, top_p=args.rollout_top_p,
        )
        av.train()
        is_eos = rollout_ids == tok.eos_token_id
        rollout_mask = (is_eos.cumsum(dim=1) <= 1).long()
        T_resp = rollout_ids.shape[1]
        mask_f = rollout_mask.float()
        n_valid = mask_f.sum().clamp_min(1.0)

        # 3. AR forward — last-position MSE
        ar_ids, ar_mask, offsets, lengths = build_ar_inputs(
            tok, rollout_ids, rollout_mask, device, AR_PREFIX, AR_SUFFIX
        )
        pred_last = ar(input_ids=ar_ids, attention_mask=ar_mask)  # [B_eff, d]
        p_norm = F.normalize(pred_last.float(), dim=-1) * sc
        g_norm_tgt = F.normalize(h_expanded.float(), dim=-1) * sc
        L_per_rollout = ((p_norm - g_norm_tgt) ** 2).mean(dim=-1)  # [B_eff]
        mse_mean = L_per_rollout.mean()

        # 4. Downstream KL (if enabled)
        if lm is not None:
            ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
                tok, text_batch, pos_batch, args.max_ctx_tokens, device
            )
            T_ctx = ctx_ids.shape[1]
            # 4a. Original LM forward (no patch, no grad)
            with torch.no_grad():
                patch_state["h_hat"] = None  # disable patch
                orig_out = lm(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
                orig_log_probs = F.log_softmax(orig_out.logits.float(), dim=-1)  # [B_h, T_ctx, V]
            # 4b. Patched LM forward (with grad through pred_last)
            ctx_ids_e = ctx_ids.repeat_interleave(K, dim=0)  # [B_eff, T_ctx]
            ctx_mask_e = ctx_mask.repeat_interleave(K, dim=0)
            pos_e = pos_in_ctx.repeat_interleave(K, dim=0)  # [B_eff]
            patch_state["h_hat"] = pred_last  # [B_eff, d]; has grad
            patch_state["positions"] = pos_e
            patched_out = lm(input_ids=ctx_ids_e, attention_mask=ctx_mask_e, use_cache=False)
            patch_state["h_hat"] = None  # disable post-forward
            patched_log_probs = F.log_softmax(patched_out.logits.float(), dim=-1)  # [B_eff, T_ctx, V]

            # 4c. KL at position p only — the direct analog of Braun-style e2e CE
            # loss for our one-position-per-record setup. logits[p] predicts t_{p+1}
            # and is the output where residual@p has its causal effect; downstream
            # positions are diluted by attention (verified by diag_e2e_kl.py:
            # KL@p ≈ 1.26 nats vs ≈ 0.005 at p+1).
            orig_lp_e = orig_log_probs.repeat_interleave(K, dim=0)  # [B_eff, T_ctx, V]
            kl_pp = (orig_lp_e.exp() * (orig_lp_e - patched_log_probs)).sum(-1)  # [B_eff, T_ctx]
            KL_per_rollout = kl_pp.gather(1, pos_e.unsqueeze(1)).squeeze(1)  # [B_eff]
            kl_mean = KL_per_rollout.mean()

            # 4d. Dynamic α_KL (detached scalar)
            alpha_kl = (mse_mean.detach() / (kl_mean.detach() + 1e-8))
            # 4e. AR loss = (MSE + α_KL · KL) * 0.5
            L_ar = (mse_mean + args.downstream_kl_coef * alpha_kl * kl_mean) * 0.5
        else:
            alpha_kl = torch.zeros((), device=device)
            kl_mean = torch.zeros((), device=device)
            KL_per_rollout = torch.zeros_like(L_per_rollout)
            L_ar = mse_mean
        last_mse_per = L_per_rollout.detach()
        L_ar.backward()

        # 5. GRPO advantage on combined per-rollout cost
        with torch.no_grad():
            cost_per = (L_per_rollout
                        + args.downstream_kl_coef * alpha_kl * KL_per_rollout).detach()  # [B_eff]
            C_grouped = cost_per.view(B_h, K)
            mu_c = C_grouped.mean(dim=1, keepdim=True)
            std_c = C_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
            A_grouped = -(C_grouped - mu_c) / std_c  # high A = low cost = good
            A = A_grouped.view(B_eff).unsqueeze(1).expand(B_eff, T_resp)

        # 6. AV teacher-force + PG loss
        av_logits = teacher_force_logits(av, prompt_embeds, prompt_mask,
                                         rollout_ids, rollout_mask)
        log_pi = F.log_softmax(av_logits.float(), dim=-1)
        log_pi_z = log_pi.gather(-1, rollout_ids.unsqueeze(-1)).squeeze(-1)
        pg_loss = -(A * log_pi_z * mask_f).sum() / n_valid

        # 7. KL-to-ref anchor on AV
        if use_av_ref:
            with torch.no_grad():
                ref_prompt_embeds, _ = build_av_prompt_embeds(
                    av_ref, tok, h_expanded, marker_id, cfg.alpha, device, torch.float32,
                    no_prompt=False,
                )
                ref_logits = teacher_force_logits(av_ref, ref_prompt_embeds, prompt_mask,
                                                  rollout_ids, rollout_mask)
            kl_r = kl_to_ref(av_logits, ref_logits, rollout_mask)
            av_loss = pg_loss + args.kl_to_ref_coef * kl_r
            kl_r_v = kl_r.item()
        else:
            av_loss = pg_loss
            kl_r_v = 0.0
        av_loss.backward()

        return {
            "last_pos_mse": last_mse_per.mean().item(),
            "last_pos_fve": 1.0 - last_mse_per.mean().item() / base_mse,
            "kl_downstream": kl_mean.item(),
            "alpha_kl": alpha_kl.item(),
            "pg_loss": pg_loss.item(),
            "kl_ref": kl_r_v,
            "A_abs": A.abs().mean().item(),
            "rollout_len_mean": mask_f.sum(1).mean().item(),
            "log_pi_z_mean": (log_pi_z * mask_f).sum().item() / n_valid.item(),
            "cost_mean": cost_per.mean().item(),
        }

    @torch.no_grad()
    def evaluate():
        av.eval(); ar.eval()
        total_mse, n_total = 0.0, 0
        per_rank = args.eval_batches * B_h
        rank_offset = rank * per_rank
        for batch_i in range(args.eval_batches):
            base = rank_offset + batch_i * B_h
            idxs = [eval_indices[i] for i in range(base, base + B_h) if i < len(eval_indices)]
            if not idxs:
                break
            h_batch = activations[idxs].to(device)
            prompt_embeds, prompt_mask = build_av_prompt_embeds(
                av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32,
                no_prompt=False,
            )
            gen = ddp_generate(
                av, prompt_embeds, prompt_mask,
                max_new_tokens=args.max_new_tokens,
                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            )
            mask = (gen == tok.eos_token_id).cumsum(1) <= 1
            ar_ids, ar_mask, _, _ = build_ar_inputs(tok, gen, mask.long(), device, AR_PREFIX, AR_SUFFIX)
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask)
            p = F.normalize(pred.float(), dim=-1) * sc
            g = F.normalize(h_batch.float(), dim=-1) * sc
            per = ((p - g) ** 2).mean(dim=-1)
            total_mse += per.sum().item()
            n_total += per.numel()
        av.train(); ar.train()
        m = total_mse / max(1, n_total)
        if world > 1:
            t = torch.tensor(m, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            m = t.item()
        return m, 1.0 - m / base_mse

    def save_ckpt(model, path):
        if is_main():
            sd = (model.module if isinstance(model, DDP) else model).state_dict()
            torch.save(sd, path)
            print(f"  saved {path}")

    history = []
    pbar = tqdm(range(args.steps), desc="train e2e", disable=not is_main())
    for step in pbar:
        av_opt.zero_grad(set_to_none=True)
        ar_opt.zero_grad(set_to_none=True)
        log = train_step()
        av_norm = torch.nn.utils.clip_grad_norm_(av.parameters(), args.grad_clip)
        ar_norm = torch.nn.utils.clip_grad_norm_(ar.parameters(), args.grad_clip)
        lr_mult = lr_at(step)
        for pg in av_opt.param_groups: pg["lr"] = args.av_lr * lr_mult
        for pg in ar_opt.param_groups: pg["lr"] = args.ar_lr * lr_mult
        if torch.isfinite(av_norm).all():
            av_opt.step()
        if torch.isfinite(ar_norm).all():
            ar_opt.step()
        log["av_grad_norm"] = av_norm.item() if torch.isfinite(av_norm).all() else float("nan")
        log["ar_grad_norm"] = ar_norm.item() if torch.isfinite(ar_norm).all() else float("nan")
        log["lr_mult"] = lr_mult
        if world > 1:
            for k in list(log.keys()):
                t = torch.tensor(log[k], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                log[k] = t.item()
        if is_main():
            log["step"] = step
            history.append(log)
            if step % args.log_every == 0:
                pbar.set_postfix(
                    fve=f"{log['last_pos_fve']:.3f}",
                    kl=f"{log['kl_downstream']:.3f}",
                    aK=f"{log['alpha_kl']:.2e}",
                    pg=f"{log['pg_loss']:+.3f}",
                    A=f"{log['A_abs']:.2f}",
                    klr=f"{log['kl_ref']:.3f}",
                    avg=f"{log['av_grad_norm']:.2f}",
                    arg=f"{log['ar_grad_norm']:.2f}",
                )
        if (step + 1) % args.eval_every == 0:
            m, fve = evaluate()
            if is_main():
                history.append({"eval": {"step": step, "mse": m, "fve": fve}})
                print(f"\n  [eval @ {step}] FVE={fve:.4f}, MSE={m:.5f}")
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_ckpt(av, out_dir / f"av_step_{step+1}.pt")
            save_ckpt(ar, out_dir / f"ar_step_{step+1}.pt")

    if is_main():
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if hook_handle is not None:
        hook_handle.remove()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
