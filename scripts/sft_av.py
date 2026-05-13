"""AV SFT: train AV (full Qwen2.5-0.5B-Instruct) to generate summaries from
injected activations.

The AV gets a chat-formatted prompt with the marker token; that token's
embedding is replaced by α·ĥ. Target text: <explanation>{summary}</explanation>.
Loss: next-token CE masked to response tokens only.

Saves checkpoint to checkpoints/av_sft/.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig
from nla.data import NLADataset, collate
from nla.injection import build_av_inputs_embeds, find_marker_position
from nla.prompts import build_av_messages


def build_av_training_batch(tok, summaries: list[str], marker_token: str, max_length: int = 384):
    """Build a batch for AV SFT.
    Returns:
      input_ids:     [B, T]      — prompt + response (right-padded)
      attention_mask:[B, T]
      loss_mask:     [B, T]      — 1 on response tokens, 0 elsewhere
    """
    batch_inputs, batch_loss_mask = [], []
    for summary in summaries:
        user_msgs = build_av_messages(marker_token)
        assistant = f"<explanation>{summary}</explanation>"
        # Prefix = user turn + assistant header; response = the explanation tokens + im_end.
        prefix_text = tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(
            user_msgs + [{"role": "assistant", "content": assistant}], tokenize=False
        )
        prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        # Sanity: full_ids starts with prefix_ids
        assert full_ids[: len(prefix_ids)] == prefix_ids, (
            f"chat-template prefix mismatch (len prefix={len(prefix_ids)}, full={len(full_ids)})"
        )
        prefix_len = len(prefix_ids)
        loss_mask = [0] * prefix_len + [1] * (len(full_ids) - prefix_len)
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            loss_mask = loss_mask[:max_length]
        batch_inputs.append(full_ids)
        batch_loss_mask.append(loss_mask)

    max_len = max(len(x) for x in batch_inputs)
    pad_id = tok.pad_token_id
    input_ids, attn_mask, lmask = [], [], []
    for ids, lm in zip(batch_inputs, batch_loss_mask):
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn_mask.append([1] * len(ids) + [0] * pad)
        lmask.append(lm + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.long),
        torch.tensor(lmask, dtype=torch.long),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=None)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--out", default="checkpoints/av_sft")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-text-tokens", type=int, default=384)
    args = ap.parse_args()

    cfg = NLAConfig()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    else:
        device = args.device
    is_main = (rank == 0)
    torch.manual_seed(args.seed + rank)
    activations = Path(args.activations or f"{cfg.data_dir}/activations_L{cfg.layer}.parquet")
    summaries = Path(args.summaries or f"{cfg.data_dir}/summaries_L{cfg.layer}.parquet")
    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Loading dataset (split=av_sft) world={world}")
    full = NLADataset(activations, summaries, split="av_sft")
    n_total = len(full)
    n_eval = max(1, int(n_total * args.eval_frac))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    all_train = [full.records[i] for i in perm[n_eval:]]
    eval_records = [full.records[i] for i in perm[:n_eval]]
    # Round-robin shard train data across ranks (deterministic since perm is shared)
    train_records = all_train[rank::world]
    if is_main:
        print(f"  train_total={len(all_train)} (per-rank={len(train_records)}), eval={len(eval_records)}")

    if is_main:
        print(f"Loading AV: {cfg.base_model}")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    av.train()
    if world > 1:
        av = DDP(av, device_ids=[local])

    marker_id = tok.encode(cfg.marker_token, add_special_tokens=False)[0]
    if is_main:
        print(f"  marker={cfg.marker_token!r} id={marker_id}, alpha={cfg.alpha:.3f}")

    opt = torch.optim.AdamW(av.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def make_loader(records, shuffle):
        return DataLoader(records, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=0, generator=g if shuffle else None)

    train_loader = make_loader(train_records, shuffle=True)
    eval_loader = make_loader(eval_records, shuffle=False)
    n_train_steps = len(train_loader) * args.epochs
    n_warmup = max(1, int(n_train_steps * args.warmup_frac))

    def lr_at(step):
        if step < n_warmup:
            return step / n_warmup
        progress = (step - n_warmup) / max(1, n_train_steps - n_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def step_loss(batch):
        ids, attn, lmask = build_av_training_batch(tok, batch["summary"], cfg.marker_token, args.max_text_tokens)
        ids, attn, lmask = ids.to(device), attn.to(device), lmask.to(device)
        h = batch["h"].to(device)
        embed_layer = av.module.get_input_embeddings() if isinstance(av, DDP) else av.get_input_embeddings()
        embeds = build_av_inputs_embeds(embed_layer, ids, marker_id, h, cfg.alpha)
        out = av(inputs_embeds=embeds, attention_mask=attn, use_cache=False)
        # Predict next-token: logits[:, :-1] vs labels[:, 1:]
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        shift_lmask = lmask[:, 1:].contiguous().float()
        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        denom = shift_lmask.sum().clamp_min(1.0)
        return (per_tok * shift_lmask).sum() / denom

    @torch.no_grad()
    def evaluate():
        av.eval()
        losses = []
        for batch in eval_loader:
            losses.append(step_loss(batch).item())
        av.train()
        return sum(losses) / max(1, len(losses))

    step = 0
    history = []
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main)
        for batch in pbar:
            loss = step_loss(batch)
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            opt.step()
            step += 1
            if is_main and step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", gn=f"{grad_norm.item():.2f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        eval_loss = evaluate()
        if is_main:
            print(f"  epoch {epoch}: eval CE={eval_loss:.4f}")
            history.append({"epoch": epoch, "eval_ce": eval_loss})
        if world > 1:
            dist.barrier()

    if is_main:
        sd = (av.module if isinstance(av, DDP) else av).state_dict()
        torch.save({"state_dict": sd, "config": cfg.__dict__, "history": history},
                   out_dir / "av.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved AV checkpoint to {out_dir / 'av.pt'}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
