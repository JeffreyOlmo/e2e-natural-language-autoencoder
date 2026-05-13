"""AR SFT with per-position loss — capacity test.

Same as sft_ar.py but the loss is the MSE at EVERY input position (vs the
single activation target) rather than just the suffix-anchored last position.
This lets the backbone + value_head be trained for per-position reconstruction,
testing whether the model has enough capacity to predict the activation from
intermediate-position representations.

After training, we report FVE per-position bucket (early / mid / late) to see
where the AR is recovering information.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig
from nla.data import NLADataset, collate
from nla.loss import per_sample_mse
from nla.model import ARModel
from nla.prompts import build_ar_prompt


def tokenize_summaries(tok, summaries, device, max_length=256):
    texts = [build_ar_prompt(s) for s in summaries]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def per_pos_pred(ar, ids, mask):
    """Return [B, T, d] — value_head applied at every input position."""
    body_out = ar.body(input_ids=ids, attention_mask=mask, use_cache=False)
    h = body_out.last_hidden_state
    return ar.value_head(h), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=None)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--out", default="checkpoints/ar_sft_per_pos")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-text-tokens", type=int, default=256)
    args = ap.parse_args()

    cfg = NLAConfig()
    torch.manual_seed(args.seed)
    activations = Path(args.activations or f"{cfg.data_dir}/activations_L{cfg.layer}.parquet")
    summaries = Path(args.summaries or "data/summaries_kitft.parquet")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Loading dataset (split=ar_sft)")
    full = NLADataset(activations, summaries, split="ar_sft")
    n_total = len(full)
    n_eval = max(1, int(n_total * args.eval_frac))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_records = [full.records[i] for i in perm[n_eval:]]
    eval_records = [full.records[i] for i in perm[:n_eval]]
    print(f"  train={len(train_records)}, eval={len(eval_records)}")

    h_train = torch.stack([r.h for r in train_records])
    mu = h_train.mean(dim=0)
    h_eval = torch.stack([r.h for r in eval_records])
    base_mse = per_sample_mse(mu.expand_as(h_eval), h_eval, cfg.mse_norm).mean().item()
    print(f"  predict-mean baseline MSE on eval = {base_mse:.5f}")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading AR (per-position trained from scratch)")
    ar = ARModel(cfg, dtype=torch.float32).to(device)
    ar.train()

    opt = torch.optim.AdamW(ar.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sc = cfg.mse_norm

    def make_loader(records, shuffle):
        return DataLoader(records, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=0,
                          generator=g if shuffle else None)

    train_loader = make_loader(train_records, shuffle=True)
    eval_loader = make_loader(eval_records, shuffle=False)
    n_train_steps = len(train_loader) * args.epochs
    n_warmup = max(1, int(n_train_steps * args.warmup_frac))

    def lr_at(step):
        if step < n_warmup:
            return step / n_warmup
        progress = (step - n_warmup) / max(1, n_train_steps - n_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def per_pos_mse_batch(ids, mask, gold):
        """Returns per_pos_mse [B, T] and last_pos_mse [B]."""
        pred, _ = per_pos_pred(ar, ids, mask)  # [B, T, d]
        p_norm = F.normalize(pred, dim=-1) * sc
        gold_b = gold.unsqueeze(1)
        g_norm = F.normalize(gold_b, dim=-1).expand_as(p_norm) * sc
        per_pos = ((p_norm - g_norm) ** 2).mean(dim=-1)  # [B, T]
        B = ids.shape[0]
        last_idx = mask.sum(dim=1) - 1
        last_mse = per_pos[torch.arange(B, device=device), last_idx]
        return per_pos, last_mse

    @torch.no_grad()
    def evaluate():
        ar.eval()
        # Iterate batches; track per-sample stats directly (avoid concat shape mismatch)
        bucket_mse = [[] for _ in range(5)]
        all_mse_weighted_sum = 0.0
        all_mse_n = 0
        last_acc = []
        for batch in eval_loader:
            ids, mask = tokenize_summaries(tok, batch["summary"], device, args.max_text_tokens)
            gold = batch["h"].to(device)
            per_pos, last_mse = per_pos_mse_batch(ids, mask, gold)  # [B, T], [B]
            per_pos = per_pos.cpu(); mask_c = mask.cpu()
            last_acc.append(last_mse.cpu())
            mask_f = mask_c.float()
            all_mse_weighted_sum += (per_pos * mask_f).sum().item()
            all_mse_n += int(mask_f.sum().item())
            Bn = per_pos.shape[0]
            lens = mask_c.sum(dim=1)
            for i in range(Bn):
                n = int(lens[i].item())
                if n == 0:
                    continue
                for s in range(n):
                    frac = s / max(1, n - 1)
                    b = min(4, int(frac * 5))
                    bucket_mse[b].append(per_pos[i, s].item())
        ar.train()
        bucket_fve = [(1 - sum(b) / len(b) / base_mse) if b else None for b in bucket_mse]
        mean_pp_mse = all_mse_weighted_sum / max(1, all_mse_n)
        mean_last_mse = torch.cat(last_acc).mean().item()
        return mean_pp_mse, mean_last_mse, bucket_fve

    mp0, ml0, b0 = evaluate()
    print(f"\n  init eval: pp_MSE={mp0:.5f} (FVE={1 - mp0/base_mse:+.4f}) | "
          f"last_MSE={ml0:.5f} (FVE={1 - ml0/base_mse:+.4f})")
    print(f"  init per-quintile FVE: " + " ".join(f"q{i}={f:+.3f}" if f is not None else f"q{i}=?" for i, f in enumerate(b0)))

    step = 0
    history = []
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            ids, mask = tokenize_summaries(tok, batch["summary"], device, args.max_text_tokens)
            gold = batch["h"].to(device)
            per_pos, last_mse = per_pos_mse_batch(ids, mask, gold)
            mask_f = mask.float()
            loss = (per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}", gn=f"{gn.item():.2f}",
                                 fve=f"{1 - loss.item()/base_mse:+.3f}",
                                 last_fve=f"{1 - last_mse.mean().item()/base_mse:+.3f}")
        mp, ml, b = evaluate()
        print(f"  epoch {epoch}: pp_FVE={1-mp/base_mse:+.4f}  last_FVE={1-ml/base_mse:+.4f}")
        print(f"    per-quintile FVE: " + " ".join(f"q{i}={f:+.3f}" if f is not None else f"q{i}=?" for i, f in enumerate(b)))
        history.append({"epoch": epoch, "pp_mse": mp, "pp_fve": 1-mp/base_mse,
                        "last_mse": ml, "last_fve": 1-ml/base_mse,
                        "quintile_fve": b})

    torch.save({"state_dict": ar.state_dict(), "config": cfg.__dict__,
                "base_mse": base_mse, "history": history}, out_dir / "ar.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved AR → {out_dir / 'ar.pt'}")


if __name__ == "__main__":
    main()
