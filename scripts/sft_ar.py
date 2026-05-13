"""AR SFT: train ARModel on (summary → activation) pairs.

Loss: MSE on L2-normalized vectors. AR is the truncated Qwen-0.5B + Linear(d,d)
identity-init head; only the AR's params get gradients. Doc-level ar_sft split.

Saves checkpoint to checkpoints/ar_sft/.
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
from transformers import AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig
from nla.data import NLADataset, collate
from nla.loss import ar_mse_loss, predict_mean_mse, per_sample_mse
from nla.model import ARModel
from nla.prompts import build_ar_prompt


def tokenize_summaries(tok, summaries: list[str], device, max_length: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [build_ar_prompt(s) for s in summaries]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=None)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--out", default="checkpoints/ar_sft")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-frac", type=float, default=0.05, help="held-out fraction of train split for eval")
    ap.add_argument("--max-text-tokens", type=int, default=256)
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
        print(f"Loading dataset (split=ar_sft) world={world}")
    full = NLADataset(activations, summaries, split="ar_sft")
    n_total = len(full)
    n_eval = max(1, int(n_total * args.eval_frac))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    all_train = [full.records[i] for i in perm[n_eval:]]
    eval_records = [full.records[i] for i in perm[:n_eval]]
    train_records = all_train[rank::world]
    if is_main:
        print(f"  train_total={len(all_train)} (per-rank={len(train_records)}), eval={len(eval_records)}")

    # Predict-mean baseline (computed on eval set, using train-mean h)
    h_train = torch.stack([r.h for r in all_train])
    mu = h_train.mean(dim=0)
    h_eval = torch.stack([r.h for r in eval_records])
    base_mse = per_sample_mse(mu.expand_as(h_eval), h_eval, cfg.mse_norm).mean().item()
    if is_main:
        print(f"  predict-mean baseline MSE on eval = {base_mse:.5f}")

    if is_main:
        print(f"Loading AR (first {cfg.layer} layers + Linear({cfg.d_model},{cfg.d_model}) identity-init)")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    ar.train()
    if world > 1:
        ar = DDP(ar, device_ids=[local])

    opt = torch.optim.AdamW(ar.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def make_loader(records, shuffle):
        return DataLoader(
            records, batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=collate, num_workers=0, generator=g if shuffle else None,
        )

    train_loader = make_loader(train_records, shuffle=True)
    eval_loader = make_loader(eval_records, shuffle=False)
    n_train_steps = len(train_loader) * args.epochs
    n_warmup = max(1, int(n_train_steps * args.warmup_frac))

    def lr_at(step):
        if step < n_warmup:
            return step / n_warmup
        progress = (step - n_warmup) / max(1, n_train_steps - n_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    @torch.no_grad()
    def evaluate():
        ar.eval()
        mses = []
        for batch in eval_loader:
            ids, mask = tokenize_summaries(tok, batch["summary"], device, args.max_text_tokens)
            pred = ar(input_ids=ids, attention_mask=mask)
            gold = batch["h"].to(device)
            mses.append(per_sample_mse(pred, gold, cfg.mse_norm).cpu())
        ar.train()
        m = torch.cat(mses).mean().item()
        return m, 1.0 - m / base_mse

    step = 0
    history = []
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main)
        for batch in pbar:
            ids, mask = tokenize_summaries(tok, batch["summary"], device, args.max_text_tokens)
            pred = ar(input_ids=ids, attention_mask=mask)
            gold = batch["h"].to(device)
            loss = ar_mse_loss(pred, gold, cfg.mse_norm)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            opt.step()
            step += 1
            if is_main and step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", gn=f"{grad_norm.item():.2f}",
                                 fve=f"{1 - loss.item()/base_mse:.3f}")
        eval_mse, eval_fve = evaluate()
        if is_main:
            print(f"  epoch {epoch}: eval MSE={eval_mse:.5f}, FVE={eval_fve:.4f}")
            history.append({"epoch": epoch, "eval_mse": eval_mse, "eval_fve": eval_fve})
        if world > 1:
            dist.barrier()

    if is_main:
        sd = (ar.module if isinstance(ar, DDP) else ar).state_dict()
        torch.save({"state_dict": sd, "config": cfg.__dict__,
                    "base_mse": base_mse, "history": history}, out_dir / "ar.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved AR checkpoint to {out_dir / 'ar.pt'}")
        print(f"Final eval FVE: {history[-1]['eval_fve']:.4f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
