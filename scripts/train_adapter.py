"""Train an adapter A: h_0.5B → h_7B on paired activations.

Goal: make kitft AV (trained on 7B L20 representations) usable as a teacher
for 0.5B activations. We need v = A(h_0.5B) such that kitft AV(v) produces a
good description of the SAME underlying text.

Loss: cosine-distance (= MSE on L2-normalized vectors). Since kitft AV injects
α·v̂, only the direction of v matters.

Output: checkpoints/adapter_05B_to_7B/adapter.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_activations(path):
    t = pq.read_table(path)
    n = len(t)
    flat = np.asarray(t["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n
    h = torch.from_numpy(flat.reshape(n, d).copy())
    keys = list(zip(t["doc_id"].to_pylist(), t["position"].to_pylist()))
    return h, keys


class MLPAdapter(nn.Module):
    def __init__(self, d_in, d_out, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class LinearAdapter(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.lin(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/activations_L16.parquet")
    ap.add_argument("--tgt", default="data/activations_L20.parquet")
    ap.add_argument("--out", default="checkpoints/adapter_05B_to_7B")
    ap.add_argument("--arch", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.src} (0.5B src)")
    h_src, keys_src = load_activations(args.src)
    print(f"Loading {args.tgt} (7B tgt)")
    h_tgt, keys_tgt = load_activations(args.tgt)
    assert keys_src == keys_tgt, "src and tgt must share (doc_id, position) pairs in same order"
    n = len(h_src)
    d_src, d_tgt = h_src.shape[1], h_tgt.shape[1]
    print(f"  n={n}  d_src={d_src}  d_tgt={d_tgt}")

    # Train/val split
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    h_src = h_src.to(args.device)
    h_tgt = h_tgt.to(args.device)
    # Normalize targets to unit vectors (only direction matters for kitft AV)
    h_tgt_unit = F.normalize(h_tgt, dim=-1)

    if args.arch == "linear":
        adapter = LinearAdapter(d_src, d_tgt).to(args.device)
    else:
        adapter = MLPAdapter(d_src, d_tgt, hidden=args.hidden).to(args.device)
    print(f"  adapter: {args.arch}  n_params={sum(p.numel() for p in adapter.parameters())}")

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    n_train_steps = (len(train_idx) // args.batch_size + 1) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_train_steps)

    def cos_loss(pred, gold_unit):
        pred_u = F.normalize(pred, dim=-1)
        return (1 - (pred_u * gold_unit).sum(dim=-1)).mean()

    @torch.no_grad()
    def evaluate():
        adapter.eval()
        pred = adapter(h_src[val_idx])
        loss = cos_loss(pred, h_tgt_unit[val_idx])
        # Cosine similarity (1 - loss)
        cos_sim = 1 - loss.item()
        adapter.train()
        return loss.item(), cos_sim

    history = []
    init_loss, init_cos = evaluate()
    print(f"  init val: cos_dist={init_loss:.4f}  cos_sim={init_cos:.4f}")

    best_val_cos = -1.0
    best_state = None
    best_epoch = -1
    for epoch in range(args.epochs):
        perm_train = train_idx[torch.randperm(len(train_idx), generator=g)]
        n_batches = (len(perm_train) + args.batch_size - 1) // args.batch_size
        losses = []
        for b in range(n_batches):
            batch_idx = perm_train[b * args.batch_size : (b + 1) * args.batch_size]
            pred = adapter(h_src[batch_idx])
            loss = cos_loss(pred, h_tgt_unit[batch_idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        train_avg = sum(losses) / len(losses)
        val_loss, val_cos = evaluate()
        marker = ""
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            marker = " *"
        print(f"  epoch {epoch:2d}: train_cos_dist={train_avg:.4f}  val_cos_dist={val_loss:.4f}  val_cos_sim={val_cos:.4f}{marker}")
        history.append({"epoch": epoch, "train_cos_dist": train_avg, "val_cos_dist": val_loss, "val_cos_sim": val_cos})

    print(f"\nBest val: epoch={best_epoch} cos_sim={best_val_cos:.4f}")
    torch.save({"state_dict": best_state, "arch": args.arch, "hidden": args.hidden,
                "d_in": d_src, "d_out": d_tgt, "history": history,
                "best_epoch": best_epoch, "best_val_cos_sim": best_val_cos}, out_dir / "adapter.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved adapter (best epoch {best_epoch}) to {out_dir / 'adapter.pt'}")
    print(f"Best val cosine similarity: {best_val_cos:.4f}")


if __name__ == "__main__":
    main()
