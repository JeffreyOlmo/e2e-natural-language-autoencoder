"""FSDP RELAX gradient-baseline estimator for AV on 7B (kitft init).

Estimator (the "no-soft-AR" specialization of RELAX):
    ĝ = [L(z) - b_t] · ∇φ log π_φ(z_t)  +  ∇φ b_t

Per-token baseline:
    b_t = L(z) + Σ_v π_φ(v|z_<t) · ⟨g_t, e_v - e_{z_t}⟩

i.e. the first-order Taylor expansion of L around the hard sample, evaluated
at the expected AR-input embedding under π_φ.

Why this is good for NLA:
  - L(z) cancels in [L(z) - b_t], so we never compute L on soft tokens
  - AR sees only hard tokens (no off-distribution queries)
  - Compute = 1 hard AR fwd+bwd  +  1 AV teacher-force fwd+bwd  +  one cheap
    [B,T,d]·[V,d]^T matmul. Same overall cost as grad-distill.

Concretely:
    s[v]  = ⟨g_t, e_v⟩                              [B, T, V]   (constant in φ)
    s[z]  = s.gather(z_t)                            [B, T]      (constant in φ)
    π·s   = (softmax(av_logits) * s.detach()).sum    [B, T]      (diff in φ)
    coef  = (s[z] - π·s).detach()                    [B, T]      (constant)
    log_π_z = log_softmax(av_logits)[z_t]            [B, T]      (diff in φ)

    loss_t = coef · log_π_z + π·s

Backward gives:
    coef · ∇φ log_π_z   (REINFORCE-with-baseline)
    + ∇φ (π·s)          (baseline gradient — pulls π toward low-s tokens)

Note: e_v is the AR's embed_tokens row (since g_t is a gradient at the AR's
input embeddings). We feed those AR embedding rows into the matmul.
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
def fsdp_generate(model, inputs_embeds, attention_mask, max_new_tokens,
                  eos_id, pad_id, temperature=1.0, top_p=0.95):
    """Manual sampling via FSDP forward — same control flow on every rank."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--out", default="checkpoints/rl_relax_v1")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--per-rank-batch", type=int, default=1)
    ap.add_argument("--av-lr", type=float, default=1e-5)
    ap.add_argument("--ar-lr", type=float, default=5e-5,
                    help="Only used if --train-ar is set.")
    ap.add_argument("--train-ar", action="store_true",
                    help="Also train AR with standard MSE backward. By default AR is frozen.")
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.05)
    ap.add_argument("--loss-scale", type=float, default=1.0,
                    help="Multiply the total loss by this factor before backward. "
                         "Needed because the linearized RELAX objective has tiny magnitude "
                         "(~0.02 per token), so v=g² underflows fp16 (smallest normal ~6e-5) "
                         "in the optimizer state, breaking AdamW's normalization. "
                         "Scaling lifts v above eps so step ≈ LR rather than LR·g/eps.")
    ap.add_argument("--max-new-tokens", type=int, default=130)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=2.0)
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
        print(f"FSDP RELAX-grad-baseline world_size={world}, rank={rank}")

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

    if is_main():
        print(f"Loading models")
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
        print(f"Loading + FSDP-wrapping AR (fp16){'' if args.train_ar else ' [FROZEN]'}")
    ar = KitftAR(KITFT_AR_REPO, dtype=torch.float16)
    if args.train_ar:
        ar.train()
    else:
        ar.eval()
        for p in ar.parameters():
            p.requires_grad = False
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
    ar_opt = (torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95),
                                eps=1e-4, weight_decay=0.01)
              if args.train_ar else None)

    rng = np.random.default_rng(args.seed + rank * 1000)
    ar_embed_weight = ar_embed_module.weight  # [V, d], unsharded fp16, on device

    def relax_step():
        # 1. Sample h
        idxs = rng.choice(len(train_indices), size=args.per_rank_batch, replace=False)
        h_batch = activations[[train_indices[i] for i in idxs]].to(device)

        # 2. Hard rollout
        prompt_embeds, prompt_mask = build_av_prompt_embeds(av, tok, h_batch, device, torch.float16)
        av.eval()
        z_hard = fsdp_generate(av, prompt_embeds, prompt_mask,
                               max_new_tokens=args.max_new_tokens,
                               eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                               temperature=args.rollout_temperature, top_p=args.rollout_top_p)
        av.train()
        eos_id = tok.eos_token_id
        is_eos = z_hard == eos_id
        cum = is_eos.cumsum(dim=1)
        rollout_mask = (cum <= 1).long()

        # 3. AR forward+backward on hard z to extract g_t at each rollout position.
        # AR is frozen → no AR param updates; ar_embeds_in is a leaf with grad.
        ar_ids, ar_mask, offsets, lengths = build_ar_inputs(tok, z_hard, rollout_mask, device)
        ar_inner = ar.module
        ar_embed_layer = ar_inner.body.embed_tokens
        with torch.no_grad():
            ar_embeds_raw = ar_embed_layer(ar_ids)
        ar_embeds_in = ar_embeds_raw.detach().requires_grad_(True)
        pred = ar(inputs_embeds=ar_embeds_in, attention_mask=ar_mask)
        p_norm = F.normalize(pred.float(), dim=-1) * sc
        g_norm_target = F.normalize(h_batch.float(), dim=-1) * sc
        L_z_per = ((p_norm - g_norm_target) ** 2).mean(dim=-1)  # [B]
        L_z = L_z_per.mean()
        L_z.backward()  # populates ar_embeds_in.grad
        g_at_input = ar_embeds_in.grad.detach()

        # Extract g at rollout positions: g_at_resp[i, t] = g_at_input[i, offset_i + t]
        B = z_hard.shape[0]
        T_resp = z_hard.shape[1]
        d = g_at_input.shape[-1]
        g_at_resp = torch.zeros(B, T_resp, d, device=device, dtype=g_at_input.dtype)
        for i in range(B):
            n_v = int(lengths[i].item())
            if n_v > 0:
                off = int(offsets[i].item())
                g_at_resp[i, :n_v] = g_at_input[i, off : off + n_v]

        # 4. AV teacher-force forward (with grad through AV)
        av_logits = teacher_force_logits(av, prompt_embeds, prompt_mask, z_hard, rollout_mask)
        # [B, T_resp, V]

        # 5. Per-vocab linearized score: s[v] = ⟨g_t, e_v_AR⟩
        # Detach to keep s constant w.r.t. AV (it already is — g and E_AR have no AV grad).
        s = torch.matmul(g_at_resp, ar_embed_weight.T.contiguous()).detach()  # [B, T, V]

        # 6. Baseline components
        # log π_φ(z_t) per position (diff in φ via av_logits)
        log_pi = F.log_softmax(av_logits.float(), dim=-1)
        log_pi_z = log_pi.gather(-1, z_hard.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # π·s = expected score under π — diff in φ via av_logits
        pi = F.softmax(av_logits.float(), dim=-1)  # [B, T, V] fp32
        score_under_pi = (pi * s.float()).sum(-1)  # [B, T]

        # s[z_t] — constant, gather from s
        score_at_z = s.gather(-1, z_hard.unsqueeze(-1)).squeeze(-1).float()  # [B, T]

        # 7. RELAX gradient-baseline loss
        # coef = (s[z_t] - π·s).detach() — the "advantage" for the REINFORCE term
        # Note: L(z) cancels in [L(z) - b_t] since b_t = L(z) + π·s - s[z_t].
        coef = (score_at_z - score_under_pi).detach()  # [B, T]
        relax_loss_per_t = coef * log_pi_z + score_under_pi  # diff in φ via both terms
        mask_f = rollout_mask.float()
        valid_n = mask_f.sum().clamp_min(1.0)
        relax_loss = (relax_loss_per_t * mask_f).sum() / valid_n

        # 8. Optional KL-to-SFT-ref anchor
        if use_av_ref:
            with torch.no_grad():
                ref_prompt_embeds, _ = build_av_prompt_embeds(av_ref, tok, h_batch, device, torch.float16)
                ref_logits = teacher_force_logits(av_ref, ref_prompt_embeds, prompt_mask,
                                                  z_hard, rollout_mask)
            log_ref = F.log_softmax(ref_logits.float(), dim=-1)
            kl = (pi * (log_pi - log_ref)).sum(-1)  # [B, T]
            kl_ref = (kl * mask_f).sum() / valid_n
            kl_ref_v = kl_ref.item()
            # Scale ONLY the relax term; kl-anchor stays at natural magnitude so
            # it doesn't blow up the gradient as AV drifts from SFT.
            total_loss = args.loss_scale * relax_loss + args.kl_to_ref_coef * kl_ref
        else:
            kl_ref_v = 0.0
            total_loss = args.loss_scale * relax_loss
        total_loss_unscaled = total_loss.item() / max(args.loss_scale, 1e-12)

        total_loss.backward()

        return {
            "L_z": L_z.item(),
            "ar_fve": 1.0 - L_z.item() / base_mse,
            "score_at_z_mean": score_at_z.mean().item(),
            "score_under_pi_mean": score_under_pi.mean().item(),
            "coef_mean": coef.mean().item(),
            "coef_abs_mean": coef.abs().mean().item(),
            "log_pi_z_mean": log_pi_z.mean().item(),
            "relax_loss": relax_loss.item(),
            "kl_ref": kl_ref_v,
            "total_loss": total_loss.item(),
            "total_loss_unscaled": total_loss_unscaled,
            "rollout_len_mean": mask_f.sum(1).mean().item(),
        }

    @torch.no_grad()
    def evaluate():
        av.eval()
        total_mse, n_total = 0.0, 0
        # Rank-shard the eval set so every rank sees DIFFERENT activations.
        # Total unique activations per call = world * eval_batches * per_rank_batch.
        per_rank = args.eval_batches * args.per_rank_batch
        rank_offset = rank * per_rank
        for batch_i in range(args.eval_batches):
            base = rank_offset + batch_i * args.per_rank_batch
            idxs = list(range(base, base + args.per_rank_batch))
            idxs = [eval_indices[i] for i in idxs if i < len(eval_indices)]
            if not idxs:
                break
            h_batch = activations[idxs].to(device)
            prompt_embeds, prompt_mask = build_av_prompt_embeds(av, tok, h_batch, device, torch.float16)
            gen = fsdp_generate(av, prompt_embeds, prompt_mask,
                                max_new_tokens=args.max_new_tokens,
                                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
                                temperature=args.rollout_temperature, top_p=args.rollout_top_p)
            eos = (gen == tok.eos_token_id)
            mask = (eos.cumsum(1) <= 1).long()
            ar_ids, ar_mask, _, _ = build_ar_inputs(tok, gen, mask, device)
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask)
            p = F.normalize(pred.float(), dim=-1) * sc
            g = F.normalize(h_batch.float(), dim=-1) * sc
            per = ((p - g) ** 2).mean(dim=-1)
            total_mse += per.sum().item()
            n_total += per.numel()
        av.train()
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
        if ar_opt is not None:
            ar_opt.zero_grad(set_to_none=True)
        log = relax_step()
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
                    coef=f"{log['coef_mean']:+.3f}",
                    gn=f"{log['av_grad_norm']:.2f}",
                    rl=f"{log['rollout_len_mean']:.0f}",
                    kl=f"{log['kl_ref']:.3f}",
                )
        if (step + 1) % args.eval_every == 0:
            m, fve = evaluate()
            if is_main():
                history.append({"eval": {"step": step, "mse": m, "fve": fve}})
                print(f"\n  [eval @ {step}] FVE={fve:.4f}, MSE={m:.4f}")
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_checkpoint(av, out_dir / f"av_step_{step+1}.pt", is_main())

    if is_main():
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if world > 1:
        dist.destroy_process_group()


def save_checkpoint(model, path, is_main_rank):
    """Gather full state dict on rank 0 and save (skip non-trainable embed_tokens)."""
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    if is_main_rank:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, path)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
