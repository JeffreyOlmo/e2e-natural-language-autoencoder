"""Joint AV (grad-distill) + AR (supervised MSE) training. No GRPO.

Per step:
  1. Sample batch (h, doc_id, position) from RL split.
  2. AV rollout: sample explanations, recover policy logits via teacher-forcing.
  3. AR forward on (<text>rollout</text> <summary>): compute MSE w.r.t. h_gold.
     Extract g_t = ∂MSE/∂e_t at AR's input embeddings via autograd.grad
     (retain_graph=True), then backward through MSE for AR params.
  4. AV loss = grad_distill_loss(av_logits, ..., g_at_resp, e_v, tau)
                + kl_to_ref * KL(π_AV || π_ref)  averaged over rollout positions.
  5. Backward + step AV.

Per τ=β identity, do NOT multiply the grad-distill loss by an additional β —
τ in the per-vocab softmax IS the regularization strength.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from nla.config import NLAConfig
from nla.data import NLADataset, collate
from nla.grad_distill import build_ar_inputs_from_rollout, grad_distill_loss
from nla.injection import build_av_inputs_embeds, build_av_prompt_ids
from nla.loss import per_sample_mse
from nla.model import ARModel
from nla.rollout import av_rollout


def kl_to_ref_loss(av_logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(π_AV || π_ref) summed over response positions, normalized by valid count."""
    log_pi = F.log_softmax(av_logits, dim=-1)
    log_ref = F.log_softmax(ref_logits, dim=-1)
    pi = log_pi.exp()
    kl = (pi * (log_pi - log_ref)).sum(dim=-1)  # [B, T]
    mask_f = mask.float()
    return (kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)


@torch.no_grad()
def teacher_force_logits(model, prompt_embeds, prompt_mask, response_ids, response_mask):
    """Forward (prompt_embeds + response_embeds) through model and extract logits at
    each response position. Used for both π_AV (in train mode for grad) and π_ref."""
    response_embeds = model.get_input_embeddings()(response_ids)
    full_embeds = torch.cat([prompt_embeds, response_embeds], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    T_pre = prompt_embeds.shape[1]
    T_resp = response_ids.shape[1]
    return out.logits[:, T_pre - 1 : T_pre - 1 + T_resp, :].contiguous()


def teacher_force_logits_grad(model, prompt_embeds, prompt_mask, response_ids, response_mask):
    """Same as teacher_force_logits but WITH gradient (for π_AV)."""
    response_embeds = model.get_input_embeddings()(response_ids)
    full_embeds = torch.cat([prompt_embeds, response_embeds], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    T_pre = prompt_embeds.shape[1]
    T_resp = response_ids.shape[1]
    return out.logits[:, T_pre - 1 : T_pre - 1 + T_resp, :].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=None)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--av-ckpt", default="checkpoints/av_sft/av.pt")
    ap.add_argument("--ar-ckpt", default="checkpoints/ar_sft/ar.pt")
    ap.add_argument("--from-scratch", action="store_true",
                    help="skip loading SFT checkpoints (cold-start; needs base model only)")
    ap.add_argument("--out", default="checkpoints/rl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--av-lr", type=float, default=1e-5)
    ap.add_argument("--ar-lr", type=float, default=5e-5)
    ap.add_argument("--kl-to-ref-coef", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=None, help="override cfg.tau")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--rl-split", default="rl",
                    help='split filter for train data; one of "rl","ar_sft","av_sft","all"')
    ap.add_argument("--eval-split", default="ar_sft")
    args = ap.parse_args()

    cfg = NLAConfig()
    if args.tau is not None:
        cfg.tau = args.tau
    torch.manual_seed(args.seed)
    activations = Path(args.activations or f"{cfg.data_dir}/activations_L{cfg.layer}.parquet")
    summaries = Path(args.summaries or f"{cfg.data_dir}/summaries_L{cfg.layer}.parquet")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Datasets")
    rl_split = None if args.rl_split == "all" else args.rl_split
    eval_split = None if args.eval_split == "all" else args.eval_split
    train_ds = NLADataset(activations, summaries, split=rl_split)
    eval_ds = NLADataset(activations, summaries, split=eval_split)
    print(f"  train ({args.rl_split}): {len(train_ds)} | eval ({args.eval_split}): {len(eval_ds)}")

    h_train = torch.stack([r.h for r in train_ds.records])
    mu = h_train.mean(dim=0)
    h_eval = torch.stack([r.h for r in eval_ds.records])
    base_mse = per_sample_mse(mu.expand_as(h_eval), h_eval, cfg.mse_norm).mean().item()
    print(f"  predict-mean baseline MSE on eval = {base_mse:.5f}")

    print(f"Loading models")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    av_ref = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32).to(device)
    ar = ARModel(cfg, dtype=torch.float32).to(device)

    if not args.from_scratch:
        print(f"  AV ← {args.av_ckpt}")
        ckpt = torch.load(args.av_ckpt, map_location="cpu")
        av.load_state_dict(ckpt["state_dict"])
        av_ref.load_state_dict(ckpt["state_dict"])
        print(f"  AR ← {args.ar_ckpt}")
        ckpt = torch.load(args.ar_ckpt, map_location="cpu")
        ar.load_state_dict(ckpt["state_dict"])
    else:
        print("  cold-start: AV/AR_ref = base; AR has identity-init value head")
        # av_ref already loaded from base

    # Freeze ref
    for p in av_ref.parameters():
        p.requires_grad = False
    av_ref.eval()

    av.train(); ar.train()

    av_opt = torch.optim.AdamW(av.parameters(), lr=args.av_lr, betas=(0.9, 0.95), weight_decay=0.01)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=args.ar_lr, betas=(0.9, 0.95), weight_decay=0.01)

    marker_id = tok.encode(cfg.marker_token, add_special_tokens=False)[0]
    e_v = av.get_input_embeddings().weight  # [V, d] — shared between AV and av_ref by structure (separate tensors though)

    def make_loader(records, bs, shuffle, seed):
        g = torch.Generator().manual_seed(seed)
        return DataLoader(records, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                          num_workers=0, generator=g if shuffle else None)

    train_loader_iter = iter(make_loader(train_ds.records, args.batch_size, True, args.seed))

    def next_batch():
        nonlocal train_loader_iter
        try:
            return next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(make_loader(train_ds.records, args.batch_size, True, args.seed + 1))
            return next(train_loader_iter)

    @torch.no_grad()
    def evaluate():
        ar.eval(); av.eval()
        mses = []
        for _ in range(args.eval_batches):
            batch = next_batch()
            h = batch["h"].to(device)
            roll = av_rollout(
                av, tok, h,
                marker_token=cfg.marker_token, alpha=cfg.alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.rollout_temperature, top_p=args.rollout_top_p,
            )
            ar_inputs = build_ar_inputs_from_rollout(tok, roll["rollout_ids"], roll["rollout_mask"])
            ar_ids = ar_inputs["ar_ids"].to(device)
            ar_mask = ar_inputs["ar_mask"].to(device)
            pred = ar(input_ids=ar_ids, attention_mask=ar_mask)
            mses.append(per_sample_mse(pred, h, cfg.mse_norm).cpu())
        ar.train(); av.train()
        m = torch.cat(mses).mean().item()
        return {"eval_mse": m, "eval_fve": 1.0 - m / base_mse}

    history = []
    pbar = tqdm(range(args.steps), desc="train")
    av_opt.zero_grad(); ar_opt.zero_grad()
    accum_logs = []

    def train_micro_step():
        batch = next_batch()
        h = batch["h"].to(device)

        # 1. AV rollout
        roll = av_rollout(
            av, tok, h,
            marker_token=cfg.marker_token, alpha=cfg.alpha,
            max_new_tokens=args.max_new_tokens,
            temperature=args.rollout_temperature, top_p=args.rollout_top_p,
        )
        rollout_ids = roll["rollout_ids"]
        rollout_mask = roll["rollout_mask"]

        # 2. AR forward+backward: MSE for AR's own training, plus g_at_input for grad-distill.
        ar_inputs = build_ar_inputs_from_rollout(tok, rollout_ids, rollout_mask)
        ar_ids = ar_inputs["ar_ids"].to(device)
        ar_mask = ar_inputs["ar_mask"].to(device)
        offsets = ar_inputs["rollout_offsets"]
        lengths = ar_inputs["rollout_lengths"]

        # Don't detach: embed_layer.weight needs to stay in the graph so AR's
        # embedding table also gets trained. autograd.grad on a non-leaf tensor
        # works fine; retain_graph=True lets us .backward() afterward.
        ar_embeds = ar.get_input_embeddings()(ar_ids)
        pred = ar(inputs_embeds=ar_embeds, attention_mask=ar_mask)
        p_norm = F.normalize(pred, dim=-1) * cfg.mse_norm
        g_norm = F.normalize(h, dim=-1) * cfg.mse_norm
        ar_mse = ((p_norm - g_norm) ** 2).mean()

        # grad w.r.t. AR input embeddings (for grad-distill teacher)
        g_at_input = torch.autograd.grad(ar_mse, ar_embeds, retain_graph=True)[0].detach()

        # AR's own training: backward MSE through AR params (including embed_layer)
        (ar_mse / args.grad_accum).backward()

        # Extract g at rollout positions
        B, _, d = g_at_input.shape
        T_resp = rollout_ids.shape[1]
        g_at_resp = torch.zeros(B, T_resp, d, device=device)
        for i in range(B):
            n = int(lengths[i].item())
            if n > 0:
                off = int(offsets[i].item())
                g_at_resp[i, :n] = g_at_input[i, off : off + n]

        # 3. AV: build prompt embeds (with injection), then teacher-force logits with grad on AV.
        prompt_ids = build_av_prompt_ids(tok, cfg.marker_token).to(device).expand(B, -1).contiguous()
        prompt_mask = torch.ones_like(prompt_ids)
        prompt_embeds = build_av_inputs_embeds(av.get_input_embeddings(), prompt_ids, marker_id, h, cfg.alpha)

        av_logits = teacher_force_logits_grad(av, prompt_embeds, prompt_mask, rollout_ids.to(device), rollout_mask.to(device))

        with torch.no_grad():
            ref_prompt_embeds = build_av_inputs_embeds(av_ref.get_input_embeddings(), prompt_ids, marker_id, h, cfg.alpha)
            ref_logits = teacher_force_logits(av_ref, ref_prompt_embeds, prompt_mask, rollout_ids.to(device), rollout_mask.to(device))

        # 4. Grad-distill loss + KL anchor
        gd = grad_distill_loss(
            av_logits, rollout_mask.to(device), g_at_resp,
            av.get_input_embeddings().weight,
            tau=cfg.tau, pi_ref_mode=cfg.pi_ref_mode,
        )
        kl_ref = kl_to_ref_loss(av_logits, ref_logits, rollout_mask.to(device))
        av_loss = gd.loss + args.kl_to_ref_coef * kl_ref

        # 5. Backward AV
        (av_loss / args.grad_accum).backward()

        return {
            "ar_mse": ar_mse.item(),
            "ar_fve": 1.0 - ar_mse.item() / base_mse,
            "av_loss": av_loss.item(),
            "gd_kl": gd.teacher_kl.item(),
            "kl_ref": kl_ref.item(),
            "teacher_ent": gd.teacher_ent.item(),
            "av_ent": gd.av_ent.item(),
            "top1_agree": gd.top1_agree.item(),
            "grad_norm_mean": gd.grad_norm_mean.item(),
            "rollout_len_mean": rollout_mask.float().sum(1).mean().item(),
        }

    for step in pbar:
        for _ in range(args.grad_accum):
            log = train_micro_step()
            accum_logs.append(log)
        ar_grad_norm = torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        av_grad_norm = torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        ar_opt.step(); av_opt.step()
        ar_opt.zero_grad(); av_opt.zero_grad()

        # Aggregate logs over the grad-accum micro-steps
        agg = {k: sum(d[k] for d in accum_logs) / len(accum_logs) for k in accum_logs[0]}
        agg["step"] = step
        agg["ar_gn"] = ar_grad_norm.item()
        agg["av_gn"] = av_grad_norm.item()
        accum_logs = []
        history.append(agg)
        if step % args.log_every == 0:
            pbar.set_postfix(
                fve=f"{agg['ar_fve']:.3f}",
                gd=f"{agg['gd_kl']:.3f}",
                kl=f"{agg['kl_ref']:.3f}",
                ent=f"{agg['av_ent']:.2f}",
                rl=f"{agg['rollout_len_mean']:.0f}",
            )
        if (step + 1) % args.eval_every == 0:
            ev = evaluate()
            ev["step"] = step
            history.append({"eval": ev})
            print(f"\n  [eval @ step {step}] FVE={ev['eval_fve']:.4f}, MSE={ev['eval_mse']:.5f}")
        if (step + 1) % args.save_every == 0:
            torch.save({"av": av.state_dict(), "ar": ar.state_dict(),
                        "step": step, "history": history},
                       out_dir / f"ckpt_{step+1}.pt")

    torch.save({"av": av.state_dict(), "ar": ar.state_dict(),
                "step": args.steps, "history": history},
               out_dir / "ckpt_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Saved to {out_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
