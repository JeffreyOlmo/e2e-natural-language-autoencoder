"""RL training for the smaller (0.5B) NLA setup with SFT init.

Two methods, selectable via --method:

  pg    Causal AR with per-position value_head; per-step rewards
        r_s = L_{s-1} - L_s; return-to-go G_s = L_{prefix to s-1} - L_final;
        group-relative advantage (GRPO-style baseline at each position s);
        REINFORCE loss = -mean(A_s · log π(z_s | z_<s)).

  grpo  Standard NLA/Anthropic recipe. AR outputs only at last token (one
        reconstruction). Single L_final per rollout → single advantage per
        rollout broadcast uniformly to all tokens.

Both: AR trained jointly via its MSE objective. KL-to-SFT anchor on AV.
Architecture: DDP (0.5B is small; FSDP overhead unnecessary).

Loads SFT init from checkpoints/ar_sft/ar.pt and checkpoints/av_sft/av.pt.
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
from nla.grad_distill import build_ar_inputs_from_rollout, compute_g_at_rollout, grad_distill_loss
from nla.injection import build_av_inputs_embeds, build_av_prompt_ids
from nla.model import ARModel


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


@torch.no_grad()
def ddp_generate(model, inputs_embeds, attention_mask, max_new_tokens,
                 eos_id, pad_id, temperature=1.0, top_p=0.95):
    """Manual sampling — same control flow on every rank (no early break)."""
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


def build_av_prompt_embeds(av, tok, h_batch, marker_token_id, alpha, device, dtype,
                            no_prompt=False):
    """Build AV prompt with α-injection.

    Standard mode: chat-formatted prompt with the marker token replaced by α·ĥ.
    no_prompt mode: returns just a single position whose embedding is α·ĥ (no
    instruction, no chat formatting). For minimum-prior cold-start experiments.
    """
    B = h_batch.shape[0]
    h_unit = F.normalize(h_batch.float(), dim=-1)
    inj = (alpha * h_unit).to(dtype)  # [B, d]
    if no_prompt:
        embeds = inj.unsqueeze(1)  # [B, 1, d]
        attn_mask = torch.ones((B, 1), dtype=torch.long, device=device)
        return embeds, attn_mask
    from nla.prompts import build_av_messages
    msgs = build_av_messages(tok.decode([marker_token_id]))
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
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


def build_ar_inputs(tok, rollout_ids, rollout_mask, device, ar_prefix_text, ar_suffix_text):
    pre = tok(ar_prefix_text, add_special_tokens=False).input_ids
    suf = tok(ar_suffix_text, add_special_tokens=False).input_ids
    pad_id = tok.pad_token_id
    rows, masks, offsets, lengths = [], [], [], []
    B = rollout_ids.shape[0]
    for i in range(B):
        n = int(rollout_mask[i].sum().item())
        seq = pre + rollout_ids[i, :n].tolist() + suf
        rows.append(seq)
        masks.append([1] * len(seq))
        offsets.append(len(pre))
        lengths.append(n)
    max_len = max(len(r) for r in rows)
    ar_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    ar_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, (r, m) in enumerate(zip(rows, masks)):
        ar_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        ar_mask[i, :len(m)] = torch.tensor(m, dtype=torch.long, device=device)
    return ar_ids, ar_mask, torch.tensor(offsets, device=device), torch.tensor(lengths, device=device)


def ar_forward_per_position(ar, ar_ids, ar_mask):
    """Returns [B, T, d] — value_head applied at every position."""
    inner = ar.module if isinstance(ar, DDP) else ar
    body_out = inner.body(input_ids=ar_ids, attention_mask=ar_mask, use_cache=False)
    h = body_out.last_hidden_state  # [B, T, d]
    return inner.value_head(h)


def ar_forward_last(ar, ar_ids, ar_mask):
    """Returns [B, d] — value_head at last unmasked position only."""
    return ar(input_ids=ar_ids, attention_mask=ar_mask)


def kl_to_ref(av_logits, ref_logits, rollout_mask):
    log_pi = F.log_softmax(av_logits.float(), dim=-1)
    log_ref = F.log_softmax(ref_logits.float(), dim=-1)
    pi = log_pi.exp()
    kl = (pi * (log_pi - log_ref)).sum(dim=-1)
    mask_f = rollout_mask.float()
    return (kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)


AR_PREFIX = "Summary of the following text: <text>"
AR_SUFFIX = "</text> <summary>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pg", "grpo", "pg_aux"], required=True)
    ap.add_argument("--activations", default="data/activations_L16.parquet")
    ap.add_argument("--ar-init", default="checkpoints/ar_sft/ar.pt")
    ap.add_argument("--av-init", default="checkpoints/av_sft/av.pt")
    ap.add_argument("--av-ref", default=None,
                    help="Frozen AV used as KL-to-ref anchor. Defaults to --av-init.")
    ap.add_argument("--cold-start", action="store_true",
                    help="Skip AV and AR checkpoint loading: AV = base Qwen, AR = "
                         "identity-init head + base Qwen body. KL-to-ref anchor (if "
                         "any) uses base Qwen as well.")
    ap.add_argument("--base-model", default=None,
                    help="Override cfg.base_model for AV/AR backbones (e.g. "
                         "'Qwen/Qwen2.5-0.5B' for the non-Instruct base).")
    ap.add_argument("--no-av-prompt", action="store_true",
                    help="Skip the AV chat template entirely. AV's input is just "
                         "[α·ĥ] at position 0 — no instruction, no chat formatting. "
                         "For minimum-prior cold-start experiments.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--per-rank-h", type=int, default=1)
    ap.add_argument("--rollouts-per-h", type=int, default=4)
    ap.add_argument("--av-lr", type=float, default=5e-6)
    ap.add_argument("--ar-lr", type=float, default=2.5e-5)
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.05)
    ap.add_argument("--pg-loss-coef", type=float, default=1.0,
                    help="Coefficient on the REINFORCE term. Set to 0 for pure "
                         "grad-distill (works with --grad-distill-coef>0).")
    ap.add_argument("--grad-distill-coef", type=float, default=0.0,
                    help="Coefficient λ for per-position grad-distill aux loss on AV "
                         "(KL(q_t||π_AV) where q_t = softmax(log π_AV + scores/τ), "
                         "scores = -<ĝ_t, e_v>). 0 disables. GRPO method only.")
    ap.add_argument("--tau", type=float, default=0.01,
                    help="Grad-distill softmax temperature. Lower = sharper teacher.")
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
    if args.base_model is not None:
        cfg.base_model = args.base_model
    sc = cfg.mse_norm  # 1.0 for our small model — MSE is on normalized vectors
    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"DDP {args.method.upper()} world_size={world} rank={rank}")
        print(f"  base={cfg.base_model} layer={cfg.layer} d={cfg.d_model} α={cfg.alpha}")
        print(f"  per_rank_h={args.per_rank_h} K={args.rollouts_per_h}")
        print(f"  no_av_prompt={args.no_av_prompt} cold_start={args.cold_start}")

    # ---- Activations ----
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d_act = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d_act).copy())
    n_eval = max(1, n // 20)
    train_indices = list(range(n - n_eval))
    eval_indices = list(range(n - n_eval, n))

    # Baseline: predict-mean MSE on normalized vectors
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

    # ---- AV (full Qwen) ----
    def load_sd(path):
        x = torch.load(path, map_location=device, weights_only=False)
        return x["state_dict"] if isinstance(x, dict) and "state_dict" in x else x

    if is_main():
        print(f"Loading AV: base={cfg.base_model}"
              + ("" if args.cold_start else f", checkpoint={args.av_init}"))
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    if not args.cold_start:
        av.load_state_dict(load_sd(args.av_init))
    av.train()
    av = DDP(av, device_ids=[local])

    # AV_ref (frozen KL anchor); cold-start → base Qwen
    use_av_ref = args.kl_to_ref_coef > 0
    if use_av_ref:
        ref_path = None if args.cold_start else (args.av_ref or args.av_init)
        if is_main():
            print(f"Loading AV_ref (frozen): "
                  + ("base Qwen (cold start)" if ref_path is None else ref_path))
        av_ref = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
        if ref_path is not None:
            av_ref.load_state_dict(load_sd(ref_path))
        av_ref.eval()
        for p in av_ref.parameters():
            p.requires_grad = False
    else:
        av_ref = None

    # ---- AR (truncated body + value_head) ----
    use_aux_head = (args.method == "pg_aux")
    if is_main():
        print(f"Loading AR: identity-init head + truncated body"
              + ("" if args.cold_start else f", checkpoint={args.ar_init}")
              + (" [+ aux_head for pg_aux]" if use_aux_head else ""))
    ar = ARModel(cfg, dtype=torch.float32, with_aux_head=use_aux_head).to(device)
    if not args.cold_start:
        # Filter checkpoint to weights ARModel expects (aux_head not in SFT ckpts)
        sd_ar = load_sd(args.ar_init)
        ar_state = ar.state_dict()
        sd_ar_filtered = {k: v for k, v in sd_ar.items() if k in ar_state}
        ar.load_state_dict(sd_ar_filtered, strict=False)
    # Initialize aux_head from value_head's (now-loaded) weights so per-position
    # predictions start at the same quality as the main head.
    if use_aux_head:
        with torch.no_grad():
            ar.aux_head.weight.copy_(ar.value_head.weight)
        if is_main():
            print("  copied value_head weights → aux_head (warm start)")
    ar.train()
    ar = DDP(ar, device_ids=[local])

    av_opt = torch.optim.AdamW(av.parameters(), lr=args.av_lr, betas=(0.9, 0.95),
                               eps=1e-8, weight_decay=0.01)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95),
                               eps=1e-8, weight_decay=0.01)

    rng = np.random.default_rng(args.seed + rank * 1000)
    K = args.rollouts_per_h
    B_h = args.per_rank_h
    B_eff = B_h * K

    method = args.method

    def train_step():
        # 1. Sample h batch, expand by K
        idxs = rng.choice(len(train_indices), size=B_h, replace=False)
        h_batch = activations[[train_indices[i] for i in idxs]].to(device)
        h_expanded = h_batch.repeat_interleave(K, dim=0) if K > 1 else h_batch

        # 2. K rollouts
        prompt_embeds, prompt_mask = build_av_prompt_embeds(
            av, tok, h_expanded, marker_id, cfg.alpha, device, torch.float32,
            no_prompt=args.no_av_prompt)
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

        # 3. AR forward (per-position for pg, last-only for grpo)
        ar_ids, ar_mask, offsets, lengths = build_ar_inputs(
            tok, rollout_ids, rollout_mask, device, AR_PREFIX, AR_SUFFIX)
        P = int(offsets[0].item())

        if method == "pg":
            pred_all = ar_forward_per_position(ar, ar_ids, ar_mask)  # [B_eff, T_ar, d]
            p_norm = F.normalize(pred_all.float(), dim=-1) * sc
            h_b = h_expanded.float().unsqueeze(1)
            g_norm_tgt = F.normalize(h_b, dim=-1).expand_as(p_norm) * sc
            per_pos_mse = ((p_norm - g_norm_tgt) ** 2).mean(dim=-1)  # [B_eff, T_ar]
            mask_f_ar = ar_mask.float()
            L_total = (per_pos_mse * mask_f_ar).sum() / mask_f_ar.sum().clamp_min(1.0)
            last_idx_ar = ar_mask.sum(1) - 1
            last_mse_per = per_pos_mse[torch.arange(B_eff, device=device), last_idx_ar].detach()
            L_total.backward()

            # Returns: G_s = L_{P+s-1} - L_final
            with torch.no_grad():
                L_det = per_pos_mse.detach()
                prefix_L = L_det[:, P - 1 : P - 1 + T_resp]  # [B_eff, T_resp]
                L_final = L_det[torch.arange(B_eff, device=device), last_idx_ar].unsqueeze(1)
                G = prefix_L - L_final  # [B_eff, T_resp]
                G_grouped = G.view(B_h, K, T_resp)
                mu_s = G_grouped.mean(dim=1, keepdim=True)
                std_s = G_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
                A_grouped = (G_grouped - mu_s) / std_s
                A = A_grouped.view(B_eff, T_resp)
        elif method == "pg_aux":
            # Main path: last-position MSE updates backbone + value_head (matches vanilla GRPO).
            # Aux path: per-position MSE through a separate Linear head that sees a
            # DETACHED hidden state; gradient flows only into aux_head. Per-position
            # MSE from aux_head provides credit-assignment signal for AV.
            ar_inner = ar.module if isinstance(ar, DDP) else ar
            pred_last, pred_per_pos = ar_inner.forward_with_aux(
                input_ids=ar_ids, attention_mask=ar_mask)
            # Main loss: last-position MSE
            p_norm_main = F.normalize(pred_last.float(), dim=-1) * sc
            g_norm_tgt_main = F.normalize(h_expanded.float(), dim=-1) * sc
            L_per_rollout = ((p_norm_main - g_norm_tgt_main) ** 2).mean(dim=-1)
            L_main = L_per_rollout.mean()
            last_mse_per = L_per_rollout.detach()
            # Aux loss: per-position MSE (flows only into aux_head via detach)
            p_norm_aux = F.normalize(pred_per_pos.float(), dim=-1) * sc
            h_b = h_expanded.float().unsqueeze(1)
            g_norm_tgt_aux = F.normalize(h_b, dim=-1).expand_as(p_norm_aux) * sc
            per_pos_mse_aux = ((p_norm_aux - g_norm_tgt_aux) ** 2).mean(dim=-1)  # [B_eff, T_ar]
            mask_f_ar = ar_mask.float()
            L_aux = (per_pos_mse_aux * mask_f_ar).sum() / mask_f_ar.sum().clamp_min(1.0)
            L_total = L_main + L_aux
            L_total.backward()

            # Credit assignment: prefix_L comes from aux head's per-position MSE,
            # but L_final comes from the MAIN head — it's the actual reconstruction
            # we optimize for and is more accurate than aux's last-position prediction.
            with torch.no_grad():
                L_det = per_pos_mse_aux.detach()
                last_idx_ar = ar_mask.sum(1) - 1
                prefix_L = L_det[:, P - 1 : P - 1 + T_resp]  # [B_eff, T_resp]
                L_final_main = L_per_rollout.detach().unsqueeze(1)  # [B_eff, 1]
                G = prefix_L - L_final_main  # [B_eff, T_resp]
                G_grouped = G.view(B_h, K, T_resp)
                mu_s = G_grouped.mean(dim=1, keepdim=True)
                std_s = G_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
                A_grouped = (G_grouped - mu_s) / std_s
                A = A_grouped.view(B_eff, T_resp)
        else:  # grpo
            pred_last = ar_forward_last(ar, ar_ids, ar_mask)  # [B_eff, d]
            p_norm = F.normalize(pred_last.float(), dim=-1) * sc
            g_norm_tgt = F.normalize(h_expanded.float(), dim=-1) * sc
            L_per_rollout = ((p_norm - g_norm_tgt) ** 2).mean(dim=-1)  # [B_eff]
            L_total = L_per_rollout.mean()
            last_mse_per = L_per_rollout.detach()
            L_total.backward()

            # GRPO advantage: standardize within group; broadcast scalar to all tokens
            with torch.no_grad():
                L_grouped = L_per_rollout.detach().view(B_h, K)
                mu = L_grouped.mean(dim=1, keepdim=True)
                std = L_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
                # A_k = -(L_k - mu) / std   (negative L = good reward; high A = good)
                A_grouped = -(L_grouped - mu) / std
                A_scalar = A_grouped.view(B_eff)  # [B_eff]
                A = A_scalar.unsqueeze(1).expand(B_eff, T_resp)  # broadcast

        # 4. AV teacher-force (with grad)
        av_logits = teacher_force_logits(av, prompt_embeds, prompt_mask, rollout_ids, rollout_mask)
        log_pi = F.log_softmax(av_logits.float(), dim=-1)
        log_pi_z = log_pi.gather(-1, rollout_ids.unsqueeze(-1)).squeeze(-1)

        pg_loss = -(A * log_pi_z * mask_f).sum() / n_valid

        # 4b. Grad-distill aux loss (GRPO only). Recomputes AR forward with
        # inputs_embeds to backprop ∂MSE/∂e_t at every rollout position; builds
        # per-position teacher q_t = softmax(log π_AV + scores/τ).
        gd_loss = torch.zeros((), device=device)
        gd_log = {"gd_loss": 0.0, "gd_top1_agree": 0.0, "gd_grad_norm_mean": 0.0,
                  "gd_teacher_ent": 0.0, "gd_av_ent": 0.0}
        if method == "grpo" and args.grad_distill_coef > 0:
            ar_in = build_ar_inputs_from_rollout(tok, rollout_ids.cpu(), rollout_mask.cpu())
            g_at_resp = compute_g_at_rollout(
                ar.module if isinstance(ar, DDP) else ar,
                ar_in["ar_ids"], ar_in["ar_mask"],
                ar_in["rollout_offsets"], ar_in["rollout_lengths"],
                h_expanded, mse_norm=sc,
            )
            av_inner = av.module if isinstance(av, DDP) else av
            E_av = av_inner.get_input_embeddings().weight.detach()
            # Pad/truncate g_at_resp to match av_logits' T_resp
            T_av = av_logits.shape[1]
            T_g = g_at_resp.shape[1]
            if T_g < T_av:
                pad = torch.zeros(g_at_resp.shape[0], T_av - T_g, g_at_resp.shape[2],
                                  device=g_at_resp.device, dtype=g_at_resp.dtype)
                g_at_resp = torch.cat([g_at_resp, pad], dim=1)
            elif T_g > T_av:
                g_at_resp = g_at_resp[:, :T_av]
            gd_out = grad_distill_loss(av_logits, rollout_mask, g_at_resp, E_av,
                                       tau=args.tau, pi_ref_mode="current_av")
            gd_loss = gd_out.loss
            gd_log = {
                "gd_loss": gd_out.loss.item(),
                "gd_top1_agree": gd_out.top1_agree.item(),
                "gd_grad_norm_mean": gd_out.grad_norm_mean.item(),
                "gd_teacher_ent": gd_out.teacher_ent.item(),
                "gd_av_ent": gd_out.av_ent.item(),
            }

        # 5. KL-to-ref anchor
        if use_av_ref:
            with torch.no_grad():
                ref_prompt_embeds, _ = build_av_prompt_embeds(
                    av_ref, tok, h_expanded, marker_id, cfg.alpha, device, torch.float32,
                    no_prompt=args.no_av_prompt)
                ref_logits = teacher_force_logits(av_ref, ref_prompt_embeds, prompt_mask,
                                                  rollout_ids, rollout_mask)
            kl_r = kl_to_ref(av_logits, ref_logits, rollout_mask)
            av_loss = (args.pg_loss_coef * pg_loss
                       + args.grad_distill_coef * gd_loss
                       + args.kl_to_ref_coef * kl_r)
            kl_r_v = kl_r.item()
        else:
            av_loss = args.pg_loss_coef * pg_loss + args.grad_distill_coef * gd_loss
            kl_r_v = 0.0

        av_loss.backward()

        return {
            "L_total": L_total.item(),
            "last_pos_mse": last_mse_per.mean().item(),
            "last_pos_fve": 1.0 - last_mse_per.mean().item() / base_mse,
            "pg_loss": pg_loss.item(),
            "kl_ref": kl_r_v,
            "A_abs": A.abs().mean().item(),
            "rollout_len_mean": mask_f.sum(1).mean().item(),
            "log_pi_z_mean": (log_pi_z * mask_f).sum().item() / n_valid.item(),
            **gd_log,
        }

    @torch.no_grad()
    def evaluate():
        av.eval(); ar.eval()
        total_mse, n_total = 0.0, 0
        per_rank = args.eval_batches * B_h
        rank_offset = rank * per_rank
        for batch_i in range(args.eval_batches):
            base = rank_offset + batch_i * B_h
            idxs = list(range(base, base + B_h))
            idxs = [eval_indices[i] for i in idxs if i < len(eval_indices)]
            if not idxs:
                break
            h_batch = activations[idxs].to(device)
            prompt_embeds, prompt_mask = build_av_prompt_embeds(
                av, tok, h_batch, marker_id, cfg.alpha, device, torch.float32,
                no_prompt=args.no_av_prompt)
            gen = ddp_generate(
                av, prompt_embeds, prompt_mask,
                max_new_tokens=args.max_new_tokens,
                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            )
            mask = (gen == tok.eos_token_id).cumsum(1) <= 1
            ar_ids, ar_mask, _, _ = build_ar_inputs(tok, gen, mask.long(), device, AR_PREFIX, AR_SUFFIX)
            # eval always uses last-position (consistent metric across both methods)
            pred = ar_forward_last(ar, ar_ids, ar_mask)
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
    pbar = tqdm(range(args.steps), desc=f"train {args.method}", disable=not is_main())
    for step in pbar:
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
                    pg=f"{log['pg_loss']:+.3f}",
                    A=f"{log['A_abs']:.2f}",
                    kl=f"{log['kl_ref']:.3f}",
                    avg=f"{log['av_grad_norm']:.2f}",
                    arg=f"{log['ar_grad_norm']:.2f}",
                    gd=f"{log.get('gd_loss', 0):.2f}",
                    agree=f"{log.get('gd_top1_agree', 0):.2f}",
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
