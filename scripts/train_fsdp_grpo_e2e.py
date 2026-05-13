"""7B FSDP GRPO + e2e downstream-KL training.

Mirrors `train_small_rl_e2e.py` (which trained on 0.5B) but at 7B scale via FSDP.

Per step (each rank):
  1. Sample h_batch (per-rank-h activations), expand by K rollouts.
  2. AV.generate → rollout_ids per rollout.
  3. AR forward → pred_last at suffix-anchored position → MSE per rollout.
  4. If downstream_kl_coef > 0:
       Tokenize the source text (a window around each activation's position).
       Run frozen Qwen2.5-7B-Instruct (`lm_down`) on this context:
         - once unpatched → orig_logits
         - once with ĥ patched at layer 19 (0-indexed) position p → patched_logits
       Compute KL(orig || patched) at position p (the single-position substitution
       analog of Braun-style e2e CE loss).
       Dynamic α_KL = (MSE / (KL + ε)).detach() — paper formula.
       AR loss = MSE_mean + α_KL · KL_mean (single backward).
  5. GRPO advantage on per-rollout cost (MSE + α_KL · KL): standardize within K.
  6. AV teacher-force → log_pi; pg_loss = -mean(A · log_pi_z · mask).
  7. Optional KL-to-AV_ref anchor.

Warm start: kitft's HuggingFace post-RL NLA — kitft/nla-qwen2.5-7b-L20-{av,ar}.

Launch:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 \\
      torchrun --standalone --nproc-per-node=8 scripts/train_fsdp_grpo_e2e.py \\
          --method grpo --downstream-kl-coef 1.0 --out checkpoints/rl_7b_e2e

For matched control: same command with --downstream-kl-coef 0.0 (no LM loaded).
"""
import argparse
import functools
import json
import os
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the heavy lifting from train_fsdp.py
from scripts.train_fsdp import (
    CFG, KITFT_AV_REPO, KITFT_AR_REPO,
    KitftAR, is_main, fsdp_wrap, enable_activation_checkpointing,
    build_av_prompt_embeds, build_ar_inputs_from_rollout,
    fsdp_generate, teacher_force_logits, kl_to_ref,
)


# ---- Downstream-KL hook machinery (mirrors train_small_rl_e2e.py) ----

def make_patch_state():
    return {"h_hat": None, "positions": None}


def make_patch_hook(state):
    def hook(module, inputs, output):
        if state["h_hat"] is None:
            return output
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        B = h.shape[0]
        idx = torch.arange(B, device=h.device)
        new_h = h.clone()
        new_h[idx, state["positions"]] = state["h_hat"].to(new_h.dtype)
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h
    return hook


def tokenize_contexts(tok, texts, positions, max_ctx, device):
    B = len(texts)
    pad_id = tok.pad_token_id
    rows = []
    pos_in_ctx = []
    for i in range(B):
        enc = tok(texts[i], add_special_tokens=False).input_ids
        ids = enc[:max_ctx]
        rows.append(ids)
        pos_in_ctx.append(int(positions[i]))
    T = max(len(r) for r in rows)
    ctx_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    ctx_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        ctx_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        ctx_mask[i, : len(r)] = 1
    return ctx_ids, ctx_mask, torch.tensor(pos_in_ctx, device=device, dtype=torch.long)


def save_fsdp_state_dict(model, path, rank0_only=True):
    """Save FSDP-wrapped model's state dict (gathered on rank 0)."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
        sd = model.state_dict()
    if (not rank0_only) or is_main():
        torch.save(sd, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--per-rank-h", type=int, default=1)
    ap.add_argument("--rollouts-per-h", type=int, default=4)
    ap.add_argument("--av-lr", type=float, default=5e-6)
    ap.add_argument("--ar-lr", type=float, default=2e-5)
    ap.add_argument("--lr-final-frac", type=float, default=0.0)
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.05)
    ap.add_argument("--downstream-kl-coef", type=float, default=1.0,
                    help="0 = control (no downstream-KL, LM not loaded). >0 = e2e.")
    ap.add_argument("--max-ctx-tokens", type=int, default=256)
    ap.add_argument("--kl-min-future", type=int, default=1)
    ap.add_argument("--lm-down-dtype", default="bfloat16",
                    choices=["float16", "bfloat16"])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=2)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    torch.manual_seed(args.seed + rank)

    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"FSDP GRPO+e2e  world={world}  downstream_kl_coef={args.downstream_kl_coef}")
        print(f"  per_rank_h={args.per_rank_h} K={args.rollouts_per_h}  "
              f"max_ctx={args.max_ctx_tokens}")

    # ---- Activations + texts ----
    if is_main():
        print(f"Loading activations: {args.activations}")
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    texts = table["text"].to_pylist()
    positions = np.asarray(table["position"].to_pylist(), dtype=np.int64)
    if is_main():
        print(f"  n={n}, d={d}")

    n_eval = max(1, n // 20)
    train_idx_all = list(range(n - n_eval))
    eval_idx_all = list(range(n - n_eval, n))

    # Filter train to records where p fits in source ctx window
    train_indices = [i for i in train_idx_all
                     if positions[i] + args.kl_min_future < args.max_ctx_tokens]
    eval_indices = eval_idx_all
    if is_main():
        print(f"  train_filtered={len(train_indices)}/{len(train_idx_all)}, eval={len(eval_indices)}")

    # Predict-mean baseline
    h_eval = activations[eval_indices]
    mu = activations[train_indices].mean(dim=0)
    sc = CFG["mse_scale"]
    p_b = F.normalize(mu.expand_as(h_eval).float(), dim=-1) * sc
    g_b = F.normalize(h_eval.float(), dim=-1) * sc
    base_mse = ((p_b - g_b) ** 2).mean(dim=-1).mean().item()
    if is_main():
        print(f"  predict-mean baseline MSE = {base_mse:.4f}")

    # ---- Models ----
    tok = AutoTokenizer.from_pretrained(KITFT_AV_REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if is_main():
        print(f"Loading + FSDP-wrapping AV (fp16)  warm start: {KITFT_AV_REPO}")
    av = AutoModelForCausalLM.from_pretrained(
        KITFT_AV_REPO, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    av.train()
    av_embed_module = av.model.embed_tokens.to(device).to(torch.float16)
    av = fsdp_wrap(av, ignored_modules=[av_embed_module])
    enable_activation_checkpointing(av)

    if is_main():
        print(f"Loading + FSDP-wrapping AR (fp16)  warm start: {KITFT_AR_REPO}")
    ar = KitftAR(KITFT_AR_REPO, dtype=torch.float16)
    ar.train()
    ar_embed_module = ar.body.embed_tokens.to(device).to(torch.float16)
    ar = fsdp_wrap(ar, ignored_modules=[ar_embed_module])
    enable_activation_checkpointing(ar)

    # AV_ref (frozen) — anchors AV close to its starting policy
    use_av_ref = args.kl_to_ref_coef > 0
    if use_av_ref:
        if is_main():
            print(f"Loading AV_ref (frozen fp16)")
        av_ref = AutoModelForCausalLM.from_pretrained(
            KITFT_AV_REPO, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        av_ref.eval()
        for p in av_ref.parameters():
            p.requires_grad = False
        av_ref_embed_module = av_ref.model.embed_tokens.to(device).to(torch.float16)
        av_ref = fsdp_wrap(av_ref, ignored_modules=[av_ref_embed_module])
        enable_activation_checkpointing(av_ref)
    else:
        av_ref = None

    # Downstream LM (only if e2e). Use full Qwen2.5-7B-Instruct.
    if args.downstream_kl_coef > 0:
        lm_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.lm_down_dtype]
        if is_main():
            print(f"Loading downstream LM (frozen {args.lm_down_dtype}): Qwen/Qwen2.5-7B-Instruct")
        lm_down = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", torch_dtype=lm_dtype, low_cpu_mem_usage=True
        )
        lm_down.eval()
        for p in lm_down.parameters():
            p.requires_grad = False
        lm_embed_module = lm_down.model.embed_tokens.to(device).to(lm_dtype)
        lm_down = fsdp_wrap(lm_down, ignored_modules=[lm_embed_module],
                            dtype_param=lm_dtype, dtype_reduce=torch.float32)
        enable_activation_checkpointing(lm_down)
        # Hook on layer (extraction_layer - 1) (0-indexed) = output of that layer matches hidden_states[extraction_layer]
        patch_state = make_patch_state()
        target_layer = lm_down.module.model.layers[CFG["extraction_layer"] - 1]
        hook_handle = target_layer.register_forward_hook(make_patch_hook(patch_state))
    else:
        lm_down = None
        patch_state = None
        hook_handle = None

    av_opt = torch.optim.AdamW(av.parameters(), lr=args.av_lr, betas=(0.9, 0.95),
                               eps=1e-4, weight_decay=0.01)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95),
                               eps=1e-4, weight_decay=0.01)

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
        prompt_embeds, prompt_mask, _ = build_av_prompt_embeds(
            av, tok, h_expanded, device, dtype=torch.float16)
        av.eval()
        rollout_ids = fsdp_generate(
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
        ar_ids, ar_mask, offsets, lengths = build_ar_inputs_from_rollout(
            tok, rollout_ids, rollout_mask, device)
        pred_last = ar(input_ids=ar_ids, attention_mask=ar_mask)  # [B_eff, d]
        p_norm = F.normalize(pred_last.float(), dim=-1) * sc
        g_norm_tgt = F.normalize(h_expanded.float(), dim=-1) * sc
        L_per_rollout = ((p_norm - g_norm_tgt) ** 2).mean(dim=-1)  # [B_eff]
        mse_mean = L_per_rollout.mean()

        # 4. Downstream KL (if e2e)
        if lm_down is not None:
            ctx_ids, ctx_mask, pos_in_ctx = tokenize_contexts(
                tok, text_batch, pos_batch, args.max_ctx_tokens, device,
            )
            B_ctx = ctx_ids.shape[0]
            # 4a. orig (no patch). Extract log_probs at position p only (single row,
            # vs [B,T,V] which is 1GB+ at 7B vocab).
            with torch.no_grad():
                patch_state["h_hat"] = None
                orig_out = lm_down(input_ids=ctx_ids, attention_mask=ctx_mask, use_cache=False)
                row_idx = torch.arange(B_ctx, device=device)
                orig_logits_at_p = orig_out.logits[row_idx, pos_in_ctx].float()  # [B_ctx, V]
                orig_lp_at_p = F.log_softmax(orig_logits_at_p, dim=-1)
                del orig_out
            # 4b. patched (per-rollout): expand by K
            ctx_ids_e = ctx_ids.repeat_interleave(K, dim=0)
            ctx_mask_e = ctx_mask.repeat_interleave(K, dim=0)
            pos_e = pos_in_ctx.repeat_interleave(K, dim=0)
            patch_state["h_hat"] = pred_last
            patch_state["positions"] = pos_e
            patched_out = lm_down(input_ids=ctx_ids_e, attention_mask=ctx_mask_e, use_cache=False)
            patch_state["h_hat"] = None
            row_e = torch.arange(pos_e.shape[0], device=device)
            patched_logits_at_p = patched_out.logits[row_e, pos_e].float()  # [B_eff, V]
            patched_lp_at_p = F.log_softmax(patched_logits_at_p, dim=-1)
            orig_lp_e = orig_lp_at_p.repeat_interleave(K, dim=0)  # [B_eff, V]
            KL_per_rollout = (orig_lp_e.exp() * (orig_lp_e - patched_lp_at_p)).sum(-1)  # [B_eff]
            kl_mean = KL_per_rollout.mean()

            alpha_kl = (mse_mean.detach() / (kl_mean.detach() + 1e-8))
            L_ar = (mse_mean + args.downstream_kl_coef * alpha_kl * kl_mean) * 0.5
        else:
            alpha_kl = torch.zeros((), device=device)
            kl_mean = torch.zeros((), device=device)
            KL_per_rollout = torch.zeros_like(L_per_rollout)
            L_ar = mse_mean
        last_mse_per = L_per_rollout.detach()
        L_ar.backward()

        # 5. GRPO advantage on combined cost
        with torch.no_grad():
            cost_per = (L_per_rollout
                        + args.downstream_kl_coef * alpha_kl * KL_per_rollout).detach()
            C_grouped = cost_per.view(B_h, K)
            mu_c = C_grouped.mean(dim=1, keepdim=True)
            std_c = C_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
            A_grouped = -(C_grouped - mu_c) / std_c
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
                ref_prompt_embeds = build_av_prompt_embeds(av_ref, tok, h_expanded, device,
                                                            dtype=torch.float16)[0]
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
            prompt_embeds, prompt_mask, _ = build_av_prompt_embeds(
                av, tok, h_batch, device, dtype=torch.float16)
            gen = fsdp_generate(av, prompt_embeds, prompt_mask,
                                 max_new_tokens=args.max_new_tokens,
                                 eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                                 temperature=args.rollout_temperature, top_p=args.rollout_top_p)
            eos = (gen == tok.eos_token_id)
            mask = (eos.cumsum(1) <= 1).long()
            ar_ids, ar_mask, _, _ = build_ar_inputs_from_rollout(tok, gen, mask, device)
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

    history = []
    pbar = tqdm(range(args.steps), desc="train e2e", disable=not is_main())
    for step in pbar:
        av_opt.zero_grad(set_to_none=True)
        ar_opt.zero_grad(set_to_none=True)
        log = train_step()
        av_norm = av.clip_grad_norm_(args.grad_clip)
        ar_norm = ar.clip_grad_norm_(args.grad_clip)
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
                )
        if (step + 1) % args.eval_every == 0:
            m, fve = evaluate()
            if is_main():
                history.append({"eval": {"step": step, "mse": m, "fve": fve}})
                print(f"\n  [eval @ {step}] FVE={fve:.4f}, MSE={m:.4f}")
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_fsdp_state_dict(av, out_dir / f"av_step_{step+1}.pt")
            save_fsdp_state_dict(ar, out_dir / f"ar_step_{step+1}.pt")
            if is_main():
                print(f"  saved step {step+1}")

    if is_main():
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if hook_handle is not None:
        hook_handle.remove()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
