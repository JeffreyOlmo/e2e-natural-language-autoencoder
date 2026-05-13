"""FSDP REINFORCE training with per-step rewards from causal-AR.

Per-step reward at rollout position s:
    r_s = L_{P+s-1} - L_{P+s}        (decrease in AR loss from adding z_s)

Return-to-go (telescoping):
    G_s = sum_{t>=s} r_t = L_{P+s-1} - L_final

where L_t is the AR's per-position reconstruction MSE, P = prefix length, and
L_final is the AR loss at the very last AR position (post-suffix) — i.e., the
SAME metric we evaluate on.

So: maximizing E[G_s] for all s directly minimizes E[L_final].

Policy gradient with group baseline:
    A_s^k = G_s^k - mean_k(G_s^k)
    pg_loss = -mean over (k, s) of  A_s^k · log π(z_s^k | z_<s^k)

K rollouts sampled per h for variance reduction (GRPO-style).

AR trained jointly via L_total backward (per-position MSE objective).
AV trained via pg_loss + λ KL(π || π_ref).
"""
import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


KITFT_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
KITFT_AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
CFG = {
    "alpha": 150.0,
    "mse_scale": 59.86651818838306,
    "marker_token": "㈎",
    "marker_token_id": 149705,
    "extraction_layer": 20,
    "av_template": (
        "You are a meticulous AI researcher conducting an important investigation into "
        "activation vectors from a language model. Your overall task is to describe the "
        "semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. You must "
        "then produce an explanation for the vector, enclosed within <explanation> tags. "
        "The explanation consists of 2-3 text snippets describing that vector.\n\n"
        "Here is the vector:\n\n"
        "<concept>{marker}</concept>\n\n"
        "Please provide an explanation."
    ),
    "ar_prefix": "Summary of the following text: <text>",
    "ar_suffix": "</text> <summary>",
}


class KitftAR(nn.Module):
    """kitft AR with per-position value_head output."""

    def __init__(self, repo_id: str, dtype=torch.float16):
        super().__init__()
        self.body = AutoModel.from_pretrained(repo_id, torch_dtype=dtype, low_cpu_mem_usage=True)
        d = self.body.config.hidden_size
        self.value_head = nn.Linear(d, d, bias=False, dtype=dtype)
        vh_path = hf_hub_download(repo_id=repo_id, filename="value_head.safetensors")
        vh_state = safetensors.torch.load_file(vh_path)
        for _, v in vh_state.items():
            if v.shape == (d, d):
                self.value_head.weight.data = v.to(dtype)
                break
        else:
            raise ValueError("No [d,d] tensor in value_head.safetensors")
        self.extraction_layer = CFG["extraction_layer"]

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, per_position=False):
        out = self.body(input_ids=input_ids, inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.extraction_layer]  # [B, T, d]
        if per_position:
            return self.value_head(h)  # [B, T, d]
        bsz = h.shape[0]
        last_idx = attention_mask.sum(1) - 1
        last = h[torch.arange(bsz, device=h.device), last_idx]
        return self.value_head(last)


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def fsdp_wrap(model, ignored_modules=None):
    auto_wrap = functools.partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={Qwen2DecoderLayer})
    mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.float16)
    return FSDP(model, auto_wrap_policy=auto_wrap, mixed_precision=mp,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
                ignored_modules=ignored_modules,
                sync_module_states=True, forward_prefetch=True, backward_prefetch=None)


def enable_activation_checkpointing(model):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, Qwen2DecoderLayer),
    )


def build_av_prompt_embeds(av, tok, h_batch, device, dtype):
    chat_msgs = [{"role": "user", "content": CFG["av_template"].format(marker=CFG["marker_token"])}]
    prompt_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    B = h_batch.shape[0]
    prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
    h_unit = F.normalize(h_batch.float(), dim=-1)
    inj = (CFG["alpha"] * h_unit).to(dtype)
    pos = (prompt_ids_b == CFG["marker_token_id"]).float().argmax(dim=1)
    embed_layer = av.module.get_input_embeddings() if isinstance(av, FSDP) else av.get_input_embeddings()
    embeds = embed_layer(prompt_ids_b).clone()
    embeds[torch.arange(B, device=device), pos] = inj
    attn_mask = torch.ones_like(prompt_ids_b)
    return embeds, attn_mask


@torch.no_grad()
def fsdp_generate(model, inputs_embeds, attention_mask, max_new_tokens, eos_id,
                  pad_id, temperature=1.0, top_p=0.95):
    inner = model.module if isinstance(model, FSDP) else model
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


def teacher_force_logits(model, prompt_embeds, prompt_mask, response_ids, response_mask):
    inner = model.module if isinstance(model, FSDP) else model
    response_embeds = inner.get_input_embeddings()(response_ids)
    full_embeds = torch.cat([prompt_embeds, response_embeds], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    T_pre = prompt_embeds.shape[1]
    T_resp = response_ids.shape[1]
    return out.logits[:, T_pre - 1 : T_pre - 1 + T_resp, :].contiguous()


def build_ar_inputs(tok, rollout_ids, rollout_mask, device):
    pre = tok(CFG["ar_prefix"], add_special_tokens=False).input_ids
    suf = tok(CFG["ar_suffix"], add_special_tokens=False).input_ids
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


def kl_to_ref(av_logits, ref_logits, rollout_mask):
    log_pi = F.log_softmax(av_logits.float(), dim=-1)
    log_ref = F.log_softmax(ref_logits.float(), dim=-1)
    pi = log_pi.exp()
    kl = (pi * (log_pi - log_ref)).sum(dim=-1)
    mask_f = rollout_mask.float()
    return (kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def save_checkpoint(model, path, is_main_rank):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    if is_main_rank:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, path)
        print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--out", default="checkpoints/rl_pg_v1")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--per-rank-h", type=int, default=1,
                    help="distinct activations per rank per step (effective batch = world * per_rank_h)")
    ap.add_argument("--rollouts-per-h", type=int, default=4,
                    help="K — number of rollouts per activation (for group baseline)")
    ap.add_argument("--av-lr", type=float, default=5e-6)
    ap.add_argument("--ar-lr", type=float, default=2.5e-5)
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
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

    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"FSDP policy-gradient (per-step rewards) world_size={world}, rank={rank}")
        print(f"  per_rank_h={args.per_rank_h}, K={args.rollouts_per_h} → effective B={args.per_rank_h*args.rollouts_per_h} per rank")

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
    sc = CFG["mse_scale"]
    p_b = F.normalize(mu.expand_as(h_eval).float(), dim=-1) * sc
    g_b = F.normalize(h_eval.float(), dim=-1) * sc
    base_mse = ((p_b - g_b) ** 2).mean(dim=-1).mean().item()
    if is_main():
        print(f"  predict-mean baseline MSE = {base_mse:.4f}")

    tok = AutoTokenizer.from_pretrained(KITFT_AV_REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if is_main():
        print(f"Loading + FSDP-wrapping AV (fp16)")
    av = AutoModelForCausalLM.from_pretrained(KITFT_AV_REPO, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    av.train()
    av_embed_module = av.model.embed_tokens.to(device).to(torch.float16)
    av = fsdp_wrap(av, ignored_modules=[av_embed_module])
    enable_activation_checkpointing(av)

    if is_main():
        print(f"Loading + FSDP-wrapping AR (fp16)")
    ar = KitftAR(KITFT_AR_REPO, dtype=torch.float16)
    ar.train()
    ar_embed_module = ar.body.embed_tokens.to(device).to(torch.float16)
    ar = fsdp_wrap(ar, ignored_modules=[ar_embed_module])
    enable_activation_checkpointing(ar)

    use_av_ref = args.kl_to_ref_coef > 0
    if use_av_ref:
        if is_main():
            print(f"Loading + FSDP-wrapping AV_ref (fp16)")
        av_ref = AutoModelForCausalLM.from_pretrained(KITFT_AV_REPO, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        av_ref.eval()
        for p in av_ref.parameters():
            p.requires_grad = False
        av_ref_embed_module = av_ref.model.embed_tokens.to(device).to(torch.float16)
        av_ref = fsdp_wrap(av_ref, ignored_modules=[av_ref_embed_module])
        enable_activation_checkpointing(av_ref)
    else:
        av_ref = None

    av_opt = torch.optim.AdamW(av.parameters(), lr=args.av_lr, betas=(0.9, 0.95),
                               eps=1e-4, weight_decay=0.01)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95),
                               eps=1e-4, weight_decay=0.01)

    rng = np.random.default_rng(args.seed + rank * 1000)
    K = args.rollouts_per_h
    B_h = args.per_rank_h  # distinct h's per rank
    B_eff = B_h * K        # rollouts per rank

    def train_step():
        # 1. Sample h batch, expand by K
        idxs = rng.choice(len(train_indices), size=B_h, replace=False)
        h_batch = activations[[train_indices[i] for i in idxs]].to(device)  # [B_h, d]
        h_expanded = h_batch.repeat_interleave(K, dim=0)                    # [B_h*K, d]

        # 2. K rollouts per h (one batched call; multinomial gives K different samples)
        prompt_embeds, prompt_mask = build_av_prompt_embeds(av, tok, h_expanded, device, torch.float16)
        av.eval()
        rollout_ids = fsdp_generate(
            av, prompt_embeds, prompt_mask,
            max_new_tokens=args.max_new_tokens,
            eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
            temperature=args.rollout_temperature, top_p=args.rollout_top_p,
        )
        av.train()
        is_eos = rollout_ids == tok.eos_token_id
        rollout_mask = (is_eos.cumsum(dim=1) <= 1).long()  # [B_eff, T_resp]

        # 3. AR per-position forward (with grad through AR params)
        ar_ids, ar_mask, offsets, lengths = build_ar_inputs(tok, rollout_ids, rollout_mask, device)
        # P = prefix length (same for every row)
        P = int(offsets[0].item())
        pred_all = ar(input_ids=ar_ids, attention_mask=ar_mask, per_position=True)
        p_norm = F.normalize(pred_all.float(), dim=-1) * sc
        h_b = h_expanded.float().unsqueeze(1)
        g_norm_tgt = F.normalize(h_b, dim=-1).expand_as(p_norm) * sc
        per_pos_mse = ((p_norm - g_norm_tgt) ** 2).mean(dim=-1)  # [B_eff, T_ar]
        mask_f_ar = ar_mask.float()
        L_total = (per_pos_mse * mask_f_ar).sum() / mask_f_ar.sum().clamp_min(1.0)
        # last-position MSE per row (eval-equivalent), for diagnostics
        last_idx = ar_mask.sum(1) - 1
        last_mse_per = per_pos_mse[torch.arange(B_eff, device=device), last_idx].detach()
        last_mse = last_mse_per.mean()

        # 4. AR backward (this updates AR's params; embed_tokens.weight too)
        L_total.backward()

        # 5. Compute per-rollout-position returns from DETACHED per_pos_mse.
        # G_s = L_at_AR_pos(P+s-1) - L_final
        # rollout token z_s sits at AR position (P + s). Just before z_s, AR has
        # seen tokens 0..P+s-1, so L_just_before_z_s = per_pos_mse[:, P+s-1].
        # L_final is L at the last valid AR position (after all suffix tokens).
        with torch.no_grad():
            L_det = per_pos_mse.detach()  # [B_eff, T_ar]
            T_resp = rollout_ids.shape[1]
            # prefix-L for each rollout step: indices P-1, P, P+1, ..., P+T_resp-2
            prefix_L = L_det[:, P - 1 : P - 1 + T_resp]  # [B_eff, T_resp]
            L_final = L_det[torch.arange(B_eff, device=device), last_idx].unsqueeze(1)  # [B_eff, 1]
            G = prefix_L - L_final  # [B_eff, T_resp]
            # Group baseline: mean over K rollouts within same h
            G_grouped = G.view(B_h, K, T_resp)
            mu_s = G_grouped.mean(dim=1, keepdim=True)
            A_grouped = G_grouped - mu_s
            A = A_grouped.view(B_eff, T_resp)  # [B_eff, T_resp]
            # Normalize A by std within group for stable gradient magnitudes
            # (standard in modern GRPO implementations; without it, advantages can
            # be tiny or huge depending on rollout-variance scale).
            std_s = G_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
            A_norm = (A_grouped / std_s).view(B_eff, T_resp)

        # 6. AV teacher-force forward (with grad through AV)
        av_logits = teacher_force_logits(av, prompt_embeds, prompt_mask, rollout_ids, rollout_mask)
        log_pi = F.log_softmax(av_logits.float(), dim=-1)
        log_pi_z = log_pi.gather(-1, rollout_ids.unsqueeze(-1)).squeeze(-1)  # [B_eff, T_resp]

        # 7. Policy-gradient loss (REINFORCE with group baseline)
        mask_f = rollout_mask.float()
        n_valid = mask_f.sum().clamp_min(1.0)
        # Loss to MINIMIZE; we want to MAXIMIZE return → minimize -A·log π
        pg_loss = -(A_norm * log_pi_z * mask_f).sum() / n_valid

        # 8. KL-to-ref anchor
        if use_av_ref:
            with torch.no_grad():
                ref_prompt_embeds, _ = build_av_prompt_embeds(av_ref, tok, h_expanded, device, torch.float16)
                ref_logits = teacher_force_logits(av_ref, ref_prompt_embeds, prompt_mask, rollout_ids, rollout_mask)
            kl_r = kl_to_ref(av_logits, ref_logits, rollout_mask)
            av_loss = pg_loss + args.kl_to_ref_coef * kl_r
            kl_r_v = kl_r.item()
        else:
            av_loss = pg_loss
            kl_r_v = 0.0

        # 9. AV backward
        av_loss.backward()

        # Diagnostics
        with torch.no_grad():
            r_per = torch.zeros_like(prefix_L)
            # r_s = L_{P+s-1} - L_{P+s} = prefix_L[:,s] - per_pos_mse[:, P+s]
            r_per[:, : T_resp] = prefix_L - L_det[:, P : P + T_resp]
            r_early = (r_per[:, : T_resp // 4] * mask_f[:, : T_resp // 4]).sum() / mask_f[:, : T_resp // 4].sum().clamp_min(1.0)
            r_late = (r_per[:, -T_resp // 4 :] * mask_f[:, -T_resp // 4 :]).sum() / mask_f[:, -T_resp // 4 :].sum().clamp_min(1.0)
            G_mean = (G * mask_f).sum() / n_valid
            G_std = G.std()
            A_mean = (A * mask_f).sum() / n_valid
            A_abs = (A.abs() * mask_f).sum() / n_valid

        return {
            "L_total": L_total.item(),
            "last_pos_mse": last_mse.item(),
            "last_pos_fve": 1.0 - last_mse.item() / base_mse,
            "pg_loss": pg_loss.item(),
            "kl_ref": kl_r_v,
            "av_loss": av_loss.item(),
            "rollout_len_mean": mask_f.sum(1).mean().item(),
            "G_mean": G_mean.item(),
            "G_std": G_std.item(),
            "A_mean": A_mean.item(),
            "A_abs": A_abs.item(),
            "r_early_mean": r_early.item(),
            "r_late_mean": r_late.item(),
            "log_pi_z_mean": (log_pi_z * mask_f).sum().item() / n_valid.item(),
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
            prompt_embeds, prompt_mask = build_av_prompt_embeds(av, tok, h_batch, device, torch.float16)
            gen = fsdp_generate(
                av, prompt_embeds, prompt_mask,
                max_new_tokens=args.max_new_tokens,
                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            )
            mask = (gen == tok.eos_token_id).cumsum(1) <= 1
            ar_ids, ar_mask, _, _ = build_ar_inputs(tok, gen, mask.long(), device)
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask, per_position=False)
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
    pbar = tqdm(range(args.steps), desc="train", disable=not is_main())
    for step in pbar:
        av_opt.zero_grad(set_to_none=True)
        ar_opt.zero_grad(set_to_none=True)
        log = train_step()
        av_norm = av.clip_grad_norm_(args.grad_clip)
        av_skip = not torch.isfinite(av_norm).all()
        if not av_skip:
            av_opt.step()
        log["av_grad_norm"] = av_norm.item() if torch.isfinite(av_norm).all() else float("nan")
        log["av_skipped"] = float(av_skip)
        ar_norm = ar.clip_grad_norm_(args.grad_clip)
        ar_skip = not torch.isfinite(ar_norm).all()
        if not ar_skip:
            ar_opt.step()
        log["ar_grad_norm"] = ar_norm.item() if torch.isfinite(ar_norm).all() else float("nan")
        log["ar_skipped"] = float(ar_skip)
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
                    rE=f"{log['r_early_mean']:+.3f}",
                    rL=f"{log['r_late_mean']:+.3f}",
                )
        if (step + 1) % args.eval_every == 0:
            m, fve = evaluate()
            if is_main():
                history.append({"eval": {"step": step, "mse": m, "fve": fve}})
                print(f"\n  [eval @ {step}] FVE={fve:.4f}, MSE={m:.4f}")
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_checkpoint(av, out_dir / f"av_step_{step+1}.pt", is_main())
            save_checkpoint(ar, out_dir / f"ar_step_{step+1}.pt", is_main())

    if is_main():
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
