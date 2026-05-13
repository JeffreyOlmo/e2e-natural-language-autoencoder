"""Gumbel-softmax RL training for the small (0.5B) NLA setup with SFT init.

Two methods, selectable via --method:

  st_gs   Straight-Through Gumbel-Softmax. Forward = discrete one-hot
          (matches inference, no side channel); backward = soft-Gumbel grad.

  soft_gs Pure soft Gumbel-Softmax. Forward = continuous mixture y_soft · E
          (introduces a training-time side channel that vanishes at τ → 0);
          backward = same soft gradient.

Both modes:
  1. Sample discrete rollout from AV (no grad) — same as PG.
  2. Teacher-force AV on the rollout to get per-position logits.
  3. Apply Gumbel-softmax to logits (soft or ST) → effective embeddings e_t.
  4. Feed [AR_PREFIX_embeds, e_rollout, AR_SUFFIX_embeds] to AR (inputs_embeds).
  5. AR predicts reconstruction; MSE loss is computed and backpropagated
     end-to-end. Gradient flows: MSE → AR params + AR inputs → y → AV logits
     → AV params. Includes optional KL-to-ref anchor (frozen SFT-init AV).

No annealing — τ is fixed for the whole run.

Loads SFT init from checkpoints/av_sft_kitft/av.pt and
checkpoints/ar_sft_kitft/ar.pt.
"""
import argparse
import json
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


AR_PREFIX = "Summary of the following text: <text>"
AR_SUFFIX = "</text> <summary>"


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


@torch.no_grad()
def ddp_generate(model, inputs_embeds, attention_mask, max_new_tokens,
                 eos_id, pad_id, temperature=1.0, top_p=0.95):
    inner = model.module if isinstance(model, DDP) else model
    embed = inner.get_input_embeddings()
    B = inputs_embeds.shape[0]
    device = inputs_embeds.device
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated = []
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
    past_kv = out.past_key_values
    cur_mask = attention_mask
    for _ in range(max_new_tokens):
        logits = out.logits[:, -1, :].float()
        if temperature != 1.0:
            logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, sorted_idx, keep)
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
        generated.append(next_token)
        finished = finished | (next_token == eos_id)
        next_embed = embed(next_token).unsqueeze(1)
        cur_mask = torch.cat([cur_mask, torch.ones((B, 1), dtype=cur_mask.dtype, device=device)], dim=1)
        out = model(inputs_embeds=next_embed, attention_mask=cur_mask,
                    past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
    return torch.stack(generated, dim=1)


def build_av_prompt_embeds(av, tok, h_batch, marker_token_id, alpha, device, dtype):
    from nla.prompts import build_av_messages
    msgs = build_av_messages(tok.decode([marker_token_id]))
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    B = h_batch.shape[0]
    prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
    h_unit = F.normalize(h_batch.float(), dim=-1)
    inj = (alpha * h_unit).to(dtype)
    pos = (prompt_ids_b == marker_token_id).float().argmax(dim=1)
    embed_layer = av.module.get_input_embeddings() if isinstance(av, DDP) else av.get_input_embeddings()
    embeds = embed_layer(prompt_ids_b).clone()
    embeds[torch.arange(B, device=device), pos] = inj
    attn_mask = torch.ones_like(prompt_ids_b)
    return embeds, attn_mask


def teacher_force_logits(model, prompt_embeds, prompt_mask, response_ids, response_mask):
    inner = model.module if isinstance(model, DDP) else model
    response_embeds = inner.get_input_embeddings()(response_ids)
    full_embeds = torch.cat([prompt_embeds, response_embeds], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    T_pre = prompt_embeds.shape[1]
    T_resp = response_ids.shape[1]
    return out.logits[:, T_pre - 1 : T_pre - 1 + T_resp, :].contiguous()


def kl_to_ref(av_logits, ref_logits, rollout_mask):
    log_pi = F.log_softmax(av_logits.float(), dim=-1)
    log_ref = F.log_softmax(ref_logits.float(), dim=-1)
    pi = log_pi.exp()
    kl = (pi * (log_pi - log_ref)).sum(dim=-1)
    mask_f = rollout_mask.float()
    return (kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def gumbel_to_embeds(logits, E_table, tau, hard, rollout_mask, pad_embed):
    """logits: [B, T, V]; E_table: [V, d]; rollout_mask: [B, T]; pad_embed: [d]
    Returns: e: [B, T, d] (continuous in soft mode; one-hot in ST mode forward,
    but with soft-grad for backward). Masked positions overwritten with pad_embed.
    """
    y = F.gumbel_softmax(logits.float(), tau=tau, hard=hard, dim=-1)  # [B, T, V]
    e = y @ E_table.float()  # [B, T, d]
    # Mask out positions past EOS — replace with pad embedding (still differentiable
    # but doesn't carry signal we want).
    mask_f = rollout_mask.float().unsqueeze(-1)  # [B, T, 1]
    e = e * mask_f + pad_embed.view(1, 1, -1) * (1 - mask_f)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["st_gs", "soft_gs"], required=True)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Gumbel-softmax temperature (fixed, no annealing).")
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--ar-init", default="checkpoints/ar_sft_kitft/ar.pt")
    ap.add_argument("--av-init", default="checkpoints/av_sft_kitft/av.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--per-rank-h", type=int, default=1)
    ap.add_argument("--rollouts-per-h", type=int, default=4)
    ap.add_argument("--av-lr", type=float, default=5e-6)
    ap.add_argument("--ar-lr", type=float, default=2.5e-5)
    ap.add_argument("--warmup-steps", type=int, default=20,
                    help="Linear LR warmup for both AV and AR over this many steps.")
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=10)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    torch.manual_seed(args.seed + rank)

    cfg = NLAConfig()
    sc = cfg.mse_norm
    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"DDP {args.method.upper()} world_size={world} rank={rank}  τ={args.tau}")
        print(f"  base={cfg.base_model} layer={cfg.layer} d={cfg.d_model} α={cfg.alpha}")
        print(f"  per_rank_h={args.per_rank_h} K={args.rollouts_per_h}")

    # ---- Activations ----
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d_act = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d_act).copy())
    n_eval = max(1, n // 20)
    train_indices = list(range(n - n_eval))
    eval_indices = list(range(n - n_eval, n))

    h_eval = activations[eval_indices]
    mu = activations[train_indices].mean(dim=0)
    p_b = F.normalize(mu.expand_as(h_eval).float(), dim=-1) * sc
    g_b = F.normalize(h_eval.float(), dim=-1) * sc
    base_mse = ((p_b - g_b) ** 2).mean(dim=-1).mean().item()
    if is_main():
        print(f"  predict-mean baseline MSE = {base_mse:.5f}")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker_id = tok.convert_tokens_to_ids(cfg.marker_token)
    pad_id = tok.pad_token_id

    if is_main():
        print(f"Loading AV from {args.av_init}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    av_sft = torch.load(args.av_init, map_location=device, weights_only=False)
    av.load_state_dict(av_sft["state_dict"])
    del av_sft
    av.train()
    av = DDP(av, device_ids=[local])

    use_av_ref = args.kl_to_ref_coef > 0
    if use_av_ref:
        if is_main():
            print(f"Loading AV_ref (frozen)")
        av_ref = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
        sd_ref = torch.load(args.av_init, map_location=device, weights_only=False)
        av_ref.load_state_dict(sd_ref["state_dict"])
        del sd_ref
        av_ref.eval()
        for p in av_ref.parameters():
            p.requires_grad = False
    else:
        av_ref = None

    if is_main():
        print(f"Loading AR from {args.ar_init}")
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    ar_sft = torch.load(args.ar_init, map_location=device, weights_only=False)
    ar.load_state_dict(ar_sft["state_dict"])
    del ar_sft
    ar.train()
    ar = DDP(ar, device_ids=[local])

    av_opt = torch.optim.AdamW(av.parameters(), lr=args.av_lr, betas=(0.9, 0.95),
                               eps=1e-8, weight_decay=0.01)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95),
                               eps=1e-8, weight_decay=0.01)

    # Pre-tokenize AR_PREFIX / AR_SUFFIX
    pre_ids_cpu = tok(AR_PREFIX, add_special_tokens=False, return_tensors="pt").input_ids[0]
    suf_ids_cpu = tok(AR_SUFFIX, add_special_tokens=False, return_tensors="pt").input_ids[0]
    pre_ids = pre_ids_cpu.to(device)
    suf_ids = suf_ids_cpu.to(device)

    rng = np.random.default_rng(args.seed + rank * 1000)
    K = args.rollouts_per_h
    B_h = args.per_rank_h
    B_eff = B_h * K

    method = args.method
    hard = (method == "st_gs")

    def train_step():
        # 1. Sample h batch, K rollouts each
        idxs = rng.choice(len(train_indices), size=B_h, replace=False)
        h_batch = activations[[train_indices[i] for i in idxs]].to(device)
        h_expanded = h_batch.repeat_interleave(K, dim=0) if K > 1 else h_batch

        # 2. Discrete rollouts (no grad)
        prompt_embeds, prompt_mask = build_av_prompt_embeds(
            av, tok, h_expanded, marker_id, cfg.alpha, device, torch.float32)
        av.eval()
        rollout_ids = ddp_generate(
            av, prompt_embeds, prompt_mask,
            max_new_tokens=args.max_new_tokens,
            eos_id=tok.eos_token_id, pad_id=pad_id,
            temperature=args.rollout_temperature, top_p=args.rollout_top_p,
        )
        av.train()
        is_eos = rollout_ids == tok.eos_token_id
        rollout_mask = (is_eos.cumsum(dim=1) <= 1).long()
        T_resp = rollout_ids.shape[1]
        mask_f = rollout_mask.float()
        n_valid = mask_f.sum().clamp_min(1.0)

        # 3. AV teacher-force on the discrete rollout → per-position logits (with grad)
        av_logits = teacher_force_logits(av, prompt_embeds, prompt_mask, rollout_ids, rollout_mask)
        # [B_eff, T_resp, V]

        # 4. Gumbel-softmax → effective rollout embeddings (ST: discrete forward; Soft: mixture)
        ar_inner = ar.module if isinstance(ar, DDP) else ar
        E_ar = ar_inner.get_input_embeddings().weight  # [V, d]
        pad_embed = E_ar[pad_id].detach()
        e_rollout = gumbel_to_embeds(av_logits, E_ar, args.tau, hard, rollout_mask, pad_embed)
        # [B_eff, T_resp, d]

        # 5. Build full AR input: [pre, rollout, suf]
        B = B_eff
        pre_emb = E_ar[pre_ids].unsqueeze(0).expand(B, -1, -1)  # [B, T_pre, d]
        suf_emb = E_ar[suf_ids].unsqueeze(0).expand(B, -1, -1)  # [B, T_suf, d]
        ar_embeds = torch.cat([pre_emb, e_rollout, suf_emb], dim=1)
        T_pre = pre_ids.shape[0]
        T_suf = suf_ids.shape[0]
        ar_attn = torch.cat([
            torch.ones((B, T_pre), dtype=torch.long, device=device),
            rollout_mask,
            torch.ones((B, T_suf), dtype=torch.long, device=device),
        ], dim=1)

        # 6. AR forward + MSE
        pred = ar(inputs_embeds=ar_embeds, attention_mask=ar_attn)  # [B, d]
        p_norm = F.normalize(pred.float(), dim=-1) * sc
        g_norm = F.normalize(h_expanded.float(), dim=-1) * sc
        per_mse = ((p_norm - g_norm) ** 2).mean(dim=-1)
        L = per_mse.mean()

        # 7. KL-to-ref anchor (optional)
        if use_av_ref:
            with torch.no_grad():
                ref_prompt_embeds, _ = build_av_prompt_embeds(
                    av_ref, tok, h_expanded, marker_id, cfg.alpha, device, torch.float32)
                ref_logits = teacher_force_logits(av_ref, ref_prompt_embeds, prompt_mask,
                                                  rollout_ids, rollout_mask)
            kl_r = kl_to_ref(av_logits, ref_logits, rollout_mask)
            total = L + args.kl_to_ref_coef * kl_r
            kl_r_v = kl_r.item()
        else:
            total = L
            kl_r_v = 0.0

        total.backward()

        return {
            "L_total": L.item(),
            "last_pos_mse": per_mse.detach().mean().item(),
            "last_pos_fve": 1.0 - per_mse.detach().mean().item() / base_mse,
            "kl_ref": kl_r_v,
            "rollout_len_mean": mask_f.sum(1).mean().item(),
        }

    @torch.no_grad()
    def evaluate():
        """Eval = discrete rollout + discrete AR forward (matches inference, both methods)."""
        av.eval(); ar.eval()
        total_mse, n_total = 0.0, 0
        per_rank = args.eval_batches * B_h
        rank_offset = rank * per_rank
        ar_inner = ar.module if isinstance(ar, DDP) else ar
        for batch_i in range(args.eval_batches):
            base = rank_offset + batch_i * B_h
            idxs = list(range(base, base + B_h))
            idxs = [eval_indices[i] for i in idxs if i < len(eval_indices)]
            if not idxs:
                break
            h_batch = activations[idxs].to(device)
            prompt_embeds, prompt_mask = build_av_prompt_embeds(
                av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32)
            gen = ddp_generate(
                av, prompt_embeds, prompt_mask,
                max_new_tokens=args.max_new_tokens,
                eos_id=tok.eos_token_id, pad_id=pad_id,
                temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            )
            mask = ((gen == tok.eos_token_id).cumsum(1) <= 1).long()
            # Discrete AR forward — through input_ids
            B = gen.shape[0]
            T_resp = gen.shape[1]
            ar_ids_rows = []
            ar_attn_rows = []
            max_T = T_pre + T_resp + T_suf
            for i in range(B):
                n_resp = int(mask[i].sum().item())
                ids = torch.cat([pre_ids, gen[i, :n_resp], suf_ids])
                ar_ids_rows.append(ids)
                ar_attn_rows.append(torch.ones(ids.shape[0], dtype=torch.long, device=device))
            max_T = max(r.shape[0] for r in ar_ids_rows)
            ar_ids = torch.full((B, max_T), pad_id, dtype=torch.long, device=device)
            ar_attn = torch.zeros((B, max_T), dtype=torch.long, device=device)
            for i, (r, m) in enumerate(zip(ar_ids_rows, ar_attn_rows)):
                ar_ids[i, :r.shape[0]] = r
                ar_attn[i, :m.shape[0]] = m
            pred = ar(input_ids=ar_ids, attention_mask=ar_attn)
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

    T_pre = pre_ids.shape[0]
    T_suf = suf_ids.shape[0]

    def save_ckpt(model, path):
        if is_main():
            sd = (model.module if isinstance(model, DDP) else model).state_dict()
            torch.save(sd, path)
            print(f"  saved {path}")

    def lr_scale(step):
        if args.warmup_steps <= 0 or step >= args.warmup_steps:
            return 1.0
        return (step + 1) / args.warmup_steps

    history = []
    pbar = tqdm(range(args.steps), desc=f"train {args.method}", disable=not is_main())
    for step in pbar:
        scale = lr_scale(step)
        for pg in av_opt.param_groups:
            pg["lr"] = args.av_lr * scale
        for pg in ar_opt.param_groups:
            pg["lr"] = args.ar_lr * scale
        av_opt.zero_grad(set_to_none=True)
        ar_opt.zero_grad(set_to_none=True)
        log = train_step()
        av_norm = torch.nn.utils.clip_grad_norm_(av.parameters(), args.grad_clip)
        ar_norm = torch.nn.utils.clip_grad_norm_(ar.parameters(), args.grad_clip)
        if torch.isfinite(av_norm).all():
            av_opt.step()
        if torch.isfinite(ar_norm).all():
            ar_opt.step()
        log["av_grad_norm"] = av_norm.item() if torch.isfinite(av_norm).all() else float("nan")
        log["ar_grad_norm"] = ar_norm.item() if torch.isfinite(ar_norm).all() else float("nan")
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
                    L=f"{log['L_total']:.4f}",
                    kl=f"{log['kl_ref']:.3f}",
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

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
