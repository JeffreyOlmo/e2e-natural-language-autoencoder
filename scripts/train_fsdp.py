"""FSDP-based joint training for grad-distill RL on 7B with kitft init.

Per step (each rank):
  1. AV rollout with α-injection.
  2. AR forward+backward → MSE for AR training, autograd.grad for g_t at AR input.
  3. AV teacher-force forward (with grad) for π_AV logits at rollout positions.
  4. AV_ref teacher-force (no grad) for π_ref.
  5. Loss = grad_distill(τ, π_AV, g_t, e_v) + α_kl·KL(π_AV ‖ π_ref).
  6. backward → optimizer step.

FSDP sharding strategy:
  - FULL_SHARD on AV, AR, AV_ref decoder layers.
  - AV's embed_tokens kept unsharded on every rank (ignored_modules) so we can
    cheaply access e_v for the grad-distill score matmul.
  - Mixed precision: fp16 params, fp32 reduce, fp32 master optimizer states.
  - Activation checkpointing on each transformer layer for memory.

Launch:
  PYTHONPATH=. torchrun --standalone --nproc_per_node=N scripts/train_fsdp.py [args]
"""
import argparse
import functools
import json
import os
import re
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
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
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
    """kitft AR: truncated Qwen2 body + Linear(d,d) value_head loaded from
    separate value_head.safetensors. Reads at last unmasked token."""

    def __init__(self, repo_id: str, dtype=torch.float32):
        super().__init__()
        self.body = AutoModel.from_pretrained(repo_id, torch_dtype=dtype)
        d = self.body.config.hidden_size
        self.value_head = nn.Linear(d, d, bias=False, dtype=dtype)
        vh_path = hf_hub_download(repo_id=repo_id, filename="value_head.safetensors")
        vh_state = safetensors.torch.load_file(vh_path)
        for _, v in vh_state.items():
            if v.shape == (d, d):
                self.value_head.weight.data = v.to(dtype)
                break
        else:
            raise ValueError(f"No [d,d] tensor in value_head.safetensors")
        self.extraction_layer = CFG["extraction_layer"]

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        out = self.body(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[self.extraction_layer]
        bsz = h.shape[0]
        last_idx = attention_mask.sum(1) - 1
        last = h[torch.arange(bsz, device=h.device), last_idx]
        return self.value_head(last)


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def fsdp_wrap(model, ignored_modules=None, sharding=ShardingStrategy.FULL_SHARD,
              dtype_param=torch.float16, dtype_reduce=torch.float32):
    auto_wrap = functools.partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={Qwen2DecoderLayer})
    mp = MixedPrecision(param_dtype=dtype_param, reduce_dtype=dtype_reduce, buffer_dtype=dtype_param)
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        sharding_strategy=sharding,
        device_id=torch.cuda.current_device(),
        ignored_modules=ignored_modules,
        sync_module_states=True,
        forward_prefetch=True,
        backward_prefetch=None,
    )


def enable_activation_checkpointing(model, layer_cls=Qwen2DecoderLayer):
    """Wrap each layer's forward in torch.utils.checkpoint."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda m: isinstance(m, layer_cls),
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
    embed_layer = av.get_input_embeddings() if not isinstance(av, FSDP) else av.module.get_input_embeddings()
    embeds = embed_layer(prompt_ids_b).clone()
    embeds[torch.arange(B, device=device), pos] = inj
    attn_mask = torch.ones_like(prompt_ids_b)
    return embeds, attn_mask, prompt_ids_b.shape[1]


def build_ar_inputs_from_rollout(tok, rollout_ids, rollout_mask, device):
    """Per-row: prefix + valid_rollout_tokens + suffix, padded right."""
    pre = tok(CFG["ar_prefix"], add_special_tokens=False).input_ids
    suf = tok(CFG["ar_suffix"], add_special_tokens=False).input_ids
    pad_id = tok.pad_token_id
    rows, masks, offsets, lengths = [], [], [], []
    B = rollout_ids.shape[0]
    for i in range(B):
        n = int(rollout_mask[i].sum().item())
        if n > 0:
            seq = pre + rollout_ids[i, :n].tolist() + suf
        else:
            seq = pre + suf
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


def grad_distill(av_logits, rollout_mask, g_at_resp, e_v, tau):
    """KL(q || π_AV) averaged over valid rollout positions; π_ref = current AV."""
    g_norm = g_at_resp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    g_hat = g_at_resp / g_norm
    scores = -torch.einsum("btd,vd->btv", g_hat.to(e_v.dtype), e_v) / tau
    log_pi = F.log_softmax(av_logits, dim=-1)
    log_q = F.log_softmax(log_pi.detach() + scores, dim=-1)
    q = log_q.exp()
    kl = (q * (log_q - log_pi)).sum(dim=-1)  # [B, T]
    mask_f = rollout_mask.float()
    n = mask_f.sum().clamp_min(1.0)
    return (kl * mask_f).sum() / n, q.detach(), log_pi.detach()


def kl_to_ref(av_logits, ref_logits, rollout_mask):
    log_pi = F.log_softmax(av_logits, dim=-1)
    log_ref = F.log_softmax(ref_logits, dim=-1)
    pi = log_pi.exp()
    kl = (pi * (log_pi - log_ref)).sum(dim=-1)
    mask_f = rollout_mask.float()
    return (kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)


@torch.no_grad()
def fsdp_generate(model, inputs_embeds, attention_mask, max_new_tokens, eos_id,
                  pad_id, temperature=1.0, top_p=0.95):
    """Manual sampling via FSDP forward — avoids summon_full_params bugs.
    Each forward step calls model(...) so FSDP's pre-fwd hook all-gathers per layer.
    Uses KV cache; cache state lives on each rank.
    """
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
        # No early break: every rank must do the same number of forward calls,
        # else FSDP all-gather collectives mismatch across ranks (some ranks
        # diverge in control flow because their batch finished sooner).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--out", default="checkpoints/rl_fsdp_v1")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--per-rank-batch", type=int, default=1)
    ap.add_argument("--av-lr", type=float, default=1e-5)
    ap.add_argument("--ar-lr", type=float, default=5e-5)
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.0,
                    help="0 disables AV_ref (saves a 7B model worth of memory).")
    ap.add_argument("--freeze-ar", action="store_true",
                    help="Don't train AR (kitft init was already optimized). "
                         "Backward still flows through AR's frozen body so g_t at "
                         "rollout positions can still be computed.")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="L2 norm clip threshold for both AV and AR.")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
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
        print(f"FSDP world_size={world}, rank={rank}")

    # ---- Load activations ----
    if is_main():
        print(f"Loading activations: {args.activations}")
    table = pq.read_table(args.activations)
    n = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    activations = torch.from_numpy(flat.reshape(n, d).copy())
    if is_main():
        print(f"  n={n}, d={d}")

    # Held-out for eval: top 5% by index
    n_eval = max(1, n // 20)
    train_indices = list(range(n - n_eval))
    eval_indices = list(range(n - n_eval, n))

    # Predict-mean baseline on eval
    h_eval = activations[eval_indices]
    mu = activations[train_indices].mean(dim=0)
    sc = CFG["mse_scale"]
    p_b = F.normalize(mu.expand_as(h_eval).float(), dim=-1) * sc
    g_b = F.normalize(h_eval.float(), dim=-1) * sc
    base_mse = ((p_b - g_b) ** 2).mean(dim=-1).mean().item()
    if is_main():
        print(f"  predict-mean baseline MSE = {base_mse:.4f}")

    # ---- Tokenizer + models ----
    if is_main():
        print(f"Loading models")
    tok = AutoTokenizer.from_pretrained(KITFT_AV_REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load + wrap each model in turn so peak GPU memory is one model at a time.
    # Keep each model's embed_tokens UN-sharded (in fp16) so:
    #   (1) we can cheaply access e_v for grad-distill scoring
    #   (2) direct embed_tokens(...) calls don't need summon_full_params
    # Cost: 3 × 1GB = 3GB per rank. Worth it for code simplicity.
    if is_main():
        print(f"Loading + FSDP-wrapping AV (fp16)")
    av = AutoModelForCausalLM.from_pretrained(
        KITFT_AV_REPO, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    av.train()
    av_embed_module = av.model.embed_tokens.to(device).to(torch.float16)
    av = fsdp_wrap(av, ignored_modules=[av_embed_module])
    enable_activation_checkpointing(av)

    if is_main():
        print(f"Loading + FSDP-wrapping AR (fp16){' [FROZEN]' if args.freeze_ar else ''}")
    ar = KitftAR(KITFT_AR_REPO, dtype=torch.float16)
    if args.freeze_ar:
        ar.eval()
        for p in ar.parameters():
            p.requires_grad = False
    else:
        ar.train()
    ar_embed_module = ar.body.embed_tokens.to(device).to(torch.float16)
    ar = fsdp_wrap(ar, ignored_modules=[ar_embed_module])
    enable_activation_checkpointing(ar)

    use_av_ref = args.kl_to_ref_coef > 0
    if use_av_ref:
        if is_main():
            print(f"Loading + FSDP-wrapping AV_ref (fp16)")
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
        if is_main():
            print(f"Skipping AV_ref load (kl-to-ref-coef=0)")
        av_ref = None

    # eps=1e-4 (vs default 1e-8) is critical for fp16 master weights:
    # AdamW's 1/(sqrt(v)+eps) with eps=1e-8 underflows in fp16 (min normal ≈ 6e-5),
    # so the denominator can hit 0 → inf → NaN params after one step.
    av_opt = torch.optim.AdamW(av.parameters(), lr=args.av_lr, betas=(0.9, 0.95),
                               eps=1e-4, weight_decay=0.01)
    ar_opt = (None if args.freeze_ar else
              torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95),
                                eps=1e-4, weight_decay=0.01))

    e_v = av_embed_module.weight  # unsharded; full [V, d] on every rank

    def sample_indices(rng, n_per_rank):
        return rng.choice(len(train_indices), size=n_per_rank, replace=False)

    rng = np.random.default_rng(args.seed + rank * 1000)

    def rollout_then_train_step():
        idxs = sample_indices(rng, args.per_rank_batch)
        h_batch = activations[[train_indices[i] for i in idxs]].to(device)

        # AV rollout (eval mode for sampling). Manual generate calls FSDP forward
        # per step so all-gather happens via pre-fwd hooks; no summon needed.
        prompt_embeds, prompt_mask, T_pre = build_av_prompt_embeds(av, tok, h_batch, device, dtype=torch.float16)
        av.eval()
        gen = fsdp_generate(
            av, prompt_embeds, prompt_mask,
            max_new_tokens=args.max_new_tokens,
            eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
            temperature=args.rollout_temperature, top_p=args.rollout_top_p,
        )
        av.train()
        rollout_ids = gen
        eos_id = tok.eos_token_id
        is_eos = rollout_ids == eos_id
        cum = is_eos.cumsum(dim=1)
        rollout_mask = (cum <= 1).long()

        # AR forward+backward.
        # FSDP frees unsharded params after the 1st backward, so retain_graph=True
        # → 2nd backward fails. Instead, make ar_embeds a leaf (detach + requires_grad)
        # and do one backward — populates BOTH ar_embeds.grad (for grad-distill) AND
        # AR's body/value_head param grads. Embed_tokens.weight isn't trained in RL
        # so detaching it from the graph is safe.
        ar_ids, ar_mask, offsets, lengths = build_ar_inputs_from_rollout(tok, rollout_ids, rollout_mask, device)
        if not args.freeze_ar:
            ar.train()
        ar_inner = ar.module
        ar_embed_layer = ar_inner.body.embed_tokens
        with torch.no_grad():
            ar_embeds_raw = ar_embed_layer(ar_ids)
        ar_embeds = ar_embeds_raw.detach().requires_grad_(True)
        pred = ar(inputs_embeds=ar_embeds, attention_mask=ar_mask)
        p_norm = F.normalize(pred.float(), dim=-1) * CFG["mse_scale"]
        g_norm = F.normalize(h_batch.float(), dim=-1) * CFG["mse_scale"]
        ar_mse = ((p_norm - g_norm) ** 2).mean()
        ar_mse.backward()
        g_at_input = ar_embeds.grad.detach()

        # Extract g at rollout positions
        B, _, dd = g_at_input.shape
        T_resp = rollout_ids.shape[1]
        g_at_resp = torch.zeros(B, T_resp, dd, device=device, dtype=g_at_input.dtype)
        for i in range(B):
            n_v = int(lengths[i].item())
            if n_v > 0:
                off = int(offsets[i].item())
                g_at_resp[i, :n_v] = g_at_input[i, off : off + n_v]

        # AV teacher-force logits with grad
        av_logits = teacher_force_logits(av, prompt_embeds, prompt_mask, rollout_ids, rollout_mask)

        gd_loss, q, log_pi = grad_distill(av_logits, rollout_mask, g_at_resp, e_v, tau=args.tau)
        if use_av_ref:
            with torch.no_grad():
                ref_prompt_embeds = build_av_prompt_embeds(av_ref, tok, h_batch, device, dtype=torch.float16)[0]
                ref_logits = teacher_force_logits(av_ref, ref_prompt_embeds, prompt_mask, rollout_ids, rollout_mask)
            kl_ref = kl_to_ref(av_logits, ref_logits, rollout_mask)
            av_loss = gd_loss + args.kl_to_ref_coef * kl_ref
            kl_ref_v = kl_ref.item()
        else:
            av_loss = gd_loss
            kl_ref_v = 0.0
        av_loss.backward()

        return {
            "ar_mse": ar_mse.item(),
            "ar_fve": 1.0 - ar_mse.item() / base_mse,
            "gd_kl": gd_loss.item(),
            "kl_ref": kl_ref_v,
            "av_loss": av_loss.item(),
            "rollout_len_mean": rollout_mask.float().sum(1).mean().item(),
        }

    @torch.no_grad()
    def evaluate():
        av.eval()
        ar.eval()
        total_mse = 0.0
        n_total = 0
        for batch_i in range(args.eval_batches):
            idxs = list(range(batch_i * args.per_rank_batch, (batch_i + 1) * args.per_rank_batch))
            idxs = [eval_indices[i] for i in idxs if i < len(eval_indices)]
            if not idxs:
                break
            h_batch = activations[idxs].to(device)
            prompt_embeds, prompt_mask, _ = build_av_prompt_embeds(av, tok, h_batch, device, dtype=torch.float16)
            gen = fsdp_generate(
                av, prompt_embeds, prompt_mask,
                max_new_tokens=args.max_new_tokens,
                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            )
            eos = (gen == tok.eos_token_id)
            mask = (eos.cumsum(1) <= 1).long()
            ar_ids, ar_mask, _, _ = build_ar_inputs_from_rollout(tok, gen, mask, device)
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask)
            p = F.normalize(pred.float(), dim=-1) * CFG["mse_scale"]
            g = F.normalize(h_batch.float(), dim=-1) * CFG["mse_scale"]
            per = ((p - g) ** 2).mean(dim=-1)
            total_mse += per.sum().item()
            n_total += per.numel()
        if not args.freeze_ar:
            ar.train()
        av.train()
        m = total_mse / max(1, n_total)
        # All-reduce mean across ranks
        if world > 1:
            t = torch.tensor(m, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            m = t.item()
        return m, 1.0 - m / base_mse

    history = []
    pbar = tqdm(range(args.steps), desc="train", disable=not is_main())
    for step in pbar:
        av_opt.zero_grad(set_to_none=True)
        if ar_opt is not None:
            ar_opt.zero_grad(set_to_none=True)
        log = rollout_then_train_step()
        av_norm = av.clip_grad_norm_(args.grad_clip)
        av_skip = not torch.isfinite(av_norm).all()
        if not av_skip:
            av_opt.step()
        log["av_grad_norm"] = av_norm.item() if torch.isfinite(av_norm).all() else float("nan")
        log["av_skipped"] = float(av_skip)
        if ar_opt is not None:
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
                    fve=f"{log['ar_fve']:.3f}",
                    gd=f"{log['gd_kl']:.3f}",
                    kl=f"{log['kl_ref']:.3f}",
                    rl=f"{log['rollout_len_mean']:.0f}",
                )
        if (step + 1) % args.eval_every == 0:
            m, fve = evaluate()
            if is_main():
                history.append({"eval": {"step": step, "mse": m, "fve": fve}})
                print(f"\n  [eval @ {step}] FVE={fve:.4f}, MSE={m:.4f}")

    if is_main():
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
