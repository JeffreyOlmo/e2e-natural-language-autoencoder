"""SFT just the aux head, on top of a trained AR checkpoint.

Backbone + value_head are loaded from --ar-init and FROZEN. The aux head is
initialized from value_head's weights and trained on per-position MSE (each
hidden state's reconstruction toward the activation). Aux head sees a detached
hidden state, so its gradient stays within the aux head. Saves a new AR
checkpoint with `aux_head.weight` included.

Single-GPU; small workload (just one Linear(d, d) layer training).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-init", default="checkpoints/ar_sft_kitft/ar.pt",
                    help="Existing AR checkpoint to bootstrap from (backbone + value_head).")
    ap.add_argument("--activations", default=None)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--out", default="checkpoints/ar_sft_kitft_aux",
                    help="Output dir for new checkpoint (with aux_head included).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-text-tokens", type=int, default=256)
    args = ap.parse_args()

    cfg = NLAConfig()
    torch.manual_seed(args.seed)
    device = args.device
    activations = Path(args.activations or f"{cfg.data_dir}/activations_L{cfg.layer}.parquet")
    summaries = Path(args.summaries or "data/summaries_kitft.parquet")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Loading AR + aux_head from {args.ar_init}")
    ar = ARModel(cfg, dtype=torch.float32, with_aux_head=True).to(device)
    sd = torch.load(args.ar_init, map_location=device, weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    ar_state = ar.state_dict()
    filt = {k: v for k, v in sd.items() if k in ar_state}
    ar.load_state_dict(filt, strict=False)
    # Init aux_head from value_head's trained weights (matches the warm-start
    # we use at RL time in train_small_rl.py).
    with torch.no_grad():
        ar.aux_head.weight.copy_(ar.value_head.weight)

    # Freeze backbone + value_head; train ONLY aux_head
    for name, p in ar.named_parameters():
        p.requires_grad = name.startswith("aux_head.")
    n_trainable = sum(p.numel() for p in ar.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in ar.parameters() if not p.requires_grad)
    print(f"  aux_head trainable params: {n_trainable:,}; frozen: {n_frozen:,}")
    ar.train()

    opt = torch.optim.AdamW([p for p in ar.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

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

    sc = cfg.mse_norm

    def step_mses(batch):
        """Compute per-position aux MSE (mean over valid positions) + last-position MSE
        of both heads for logging."""
        ids, mask = tokenize_summaries(tok, batch["summary"], device, args.max_text_tokens)
        gold = batch["h"].to(device)
        # Backbone forward (no grad needed for backbone since it's frozen)
        out = ar.body(input_ids=ids, attention_mask=mask, use_cache=False)
        h = out.last_hidden_state  # [B, T, d]
        B = h.shape[0]
        last_idx = mask.sum(dim=1) - 1
        last_h = h[torch.arange(B, device=device), last_idx]
        # Main head (frozen, for logging)
        with torch.no_grad():
            pred_main = ar.value_head(last_h)
            p_m = F.normalize(pred_main, dim=-1) * sc
            g_n = F.normalize(gold, dim=-1) * sc
            main_mse = ((p_m - g_n) ** 2).mean(dim=-1)
        # Aux head: per-position, h detached so backbone stays frozen
        pred_aux_all = ar.aux_head(h.detach())  # [B, T, d]
        p_a = F.normalize(pred_aux_all, dim=-1) * sc
        gold_b = gold.unsqueeze(1)
        g_n_full = F.normalize(gold_b, dim=-1).expand_as(p_a) * sc
        per_pos_mse = ((p_a - g_n_full) ** 2).mean(dim=-1)  # [B, T]
        m_f = mask.float()
        loss = (per_pos_mse * m_f).sum() / m_f.sum().clamp_min(1.0)
        # Also report aux at last position only (apples-to-apples with main)
        aux_last_mse = per_pos_mse[torch.arange(B, device=device), last_idx]
        return loss, main_mse.mean().item(), aux_last_mse.mean().item()

    @torch.no_grad()
    def evaluate():
        ar.eval()
        losses, main_mses, aux_last_mses = [], [], []
        for batch in eval_loader:
            loss, mm, alm = step_mses(batch)
            losses.append(loss.item())
            main_mses.append(mm)
            aux_last_mses.append(alm)
        ar.train()
        return (sum(losses) / max(1, len(losses)),
                sum(main_mses) / max(1, len(main_mses)),
                sum(aux_last_mses) / max(1, len(aux_last_mses)))

    eval_loss0, eval_main0, eval_aux_last0 = evaluate()
    print(f"\n  init eval: aux_pp_MSE={eval_loss0:.5f} "
          f"(FVE={1 - eval_loss0/base_mse:+.4f}) | "
          f"main_last_MSE={eval_main0:.5f} (FVE={1 - eval_main0/base_mse:+.4f}) | "
          f"aux_last_MSE={eval_aux_last0:.5f} (FVE={1 - eval_aux_last0/base_mse:+.4f})")

    step = 0
    history = []
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            loss, _, _ = step_mses(batch)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in ar.parameters() if p.requires_grad], 1.0)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}", gn=f"{gn.item():.2f}",
                                 fve=f"{1 - loss.item()/base_mse:+.3f}")
        em, mm, alm = evaluate()
        fve_aux = 1 - em / base_mse
        fve_main = 1 - mm / base_mse
        fve_aux_last = 1 - alm / base_mse
        print(f"  epoch {epoch}: aux_pp_FVE={fve_aux:+.4f}  "
              f"main_last_FVE={fve_main:+.4f}  aux_last_FVE={fve_aux_last:+.4f}")
        history.append({
            "epoch": epoch,
            "aux_per_pos_mse": em, "aux_per_pos_fve": fve_aux,
            "main_last_mse": mm, "main_last_fve": fve_main,
            "aux_last_mse": alm, "aux_last_fve": fve_aux_last,
        })

    torch.save({"state_dict": ar.state_dict(), "config": cfg.__dict__,
                "base_mse": base_mse, "history": history,
                "bootstrapped_from": args.ar_init}, out_dir / "ar.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved AR (with trained aux_head) → {out_dir / 'ar.pt'}")


if __name__ == "__main__":
    main()
