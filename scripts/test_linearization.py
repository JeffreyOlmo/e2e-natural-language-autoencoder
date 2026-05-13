"""Test the linearization assumption underlying grad-distill.

Given a rollout z sampled from π, the per-position teacher q_t says: token v
should be more likely than π predicted, in proportion to -⟨g_t, e_v⟩/τ.
This is a first-order Taylor approximation. The TRUE claim of grad-distill
is: substituting z_t with a token sampled from q_t should reduce L (the AR
reconstruction loss).

This script tests that claim empirically:
  For N activations:
    1. Sample rollout z from kitft AV
    2. Run AR per-position forward+backward to get cumulative grad g_s at each pos
    3. Compute π_t and q_t at each rollout position
    4. Pick M random test positions per rollout
    5. For each test position t:
       - Compute baseline L = cumulative-MSE on original rollout
       - Sample K alternative tokens v from q_t; for each, compute L(z') where
         z' = z with z_t -> v
       - Same for K samples from π_t (control)
       - Δ_q = mean(L_q_alt) - L; Δ_π = mean(L_π_alt) - L
  Report E[Δ_q] vs E[Δ_π] across all (activation, position).
  If linearization is right: E[Δ_q] < 0 AND E[Δ_q] < E[Δ_π].
"""
import argparse
import re

import numpy as np
import pyarrow.parquet as pq
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


KITFT_CFG = {
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
    "ar_template": "Summary of the following text: <text>{explanation}</text> <summary>",
}


def load_ar(repo_id, device, dtype):
    body = AutoModel.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()
    vh_path = hf_hub_download(repo_id=repo_id, filename="value_head.safetensors")
    vh_state = safetensors.torch.load_file(vh_path)
    d = body.config.hidden_size
    value_head = nn.Linear(d, d, bias=False, dtype=dtype).to(device)
    for k, v in vh_state.items():
        if v.shape == (d, d):
            value_head.weight.data = v.to(dtype).to(device)
            break
    value_head.eval()
    return body, value_head


@torch.no_grad()
def ar_loss(ar_body, value_head, ar_ids, ar_mask, h_target, per_position=False, mse_scale=None):
    """Per-position MSE if per_position else last-token MSE."""
    out = ar_body(input_ids=ar_ids, attention_mask=ar_mask, output_hidden_states=True, use_cache=False)
    h_ar = out.hidden_states[KITFT_CFG["extraction_layer"]]
    if per_position:
        pred = value_head(h_ar)  # [B, T, d]
        p = F.normalize(pred.float(), dim=-1) * mse_scale
        g = F.normalize(h_target.float(), dim=-1).unsqueeze(1).expand_as(p) * mse_scale
        per_pos = ((p - g) ** 2).mean(dim=-1)  # [B, T]
        mask_f = ar_mask.float()
        return (per_pos * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)  # [B] cumulative mean
    else:
        bsz = h_ar.shape[0]
        last_idx = ar_mask.sum(1) - 1
        last = h_ar[torch.arange(bsz, device=h_ar.device), last_idx]
        pred = value_head(last)
        p = F.normalize(pred.float(), dim=-1) * mse_scale
        g = F.normalize(h_target.float(), dim=-1) * mse_scale
        return ((p - g) ** 2).mean(dim=-1)  # [B]


def build_ar_ids(tok, rollout_token_ids, device):
    pre = tok("Summary of the following text: <text>", add_special_tokens=False).input_ids
    suf = tok("</text> <summary>", add_special_tokens=False).input_ids
    seq = pre + list(rollout_token_ids) + suf
    ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(ids)
    return ids, mask, len(pre)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-repo", default="kitft/nla-qwen2.5-7b-L20-av")
    ap.add_argument("--ar-repo", default="kitft/nla-qwen2.5-7b-L20-ar")
    ap.add_argument("--activations", default="data/activations_L20.parquet")
    ap.add_argument("--n-activations", type=int, default=20)
    ap.add_argument("--m-positions-per-rollout", type=int, default=5,
                    help="how many rollout positions to test per activation")
    ap.add_argument("--k-samples", type=int, default=5,
                    help="how many alternative tokens to sample from q (and from π)")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use-eval-set", action="store_true")
    ap.add_argument("--per-pos-objective", action="store_true",
                    help="If set: use cumulative per-position MSE as L (matches train_fsdp_per_pos)."
                         " Otherwise: last-token MSE (matches original objective).")
    ap.add_argument("--normalize-embed-norm", action="store_true",
                    help="Normalize e_v in the score: s_v = ⟨g_hat, e_v/||e_v||⟩."
                         " Suppresses bias toward high-norm junk tokens (Unicode, etc.).")
    args = ap.parse_args()

    device = args.device
    dtype = torch.float16
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print("Loading models")
    tok = AutoTokenizer.from_pretrained(args.av_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(args.av_repo, torch_dtype=dtype).to(device).eval()
    ar_body, value_head = load_ar(args.ar_repo, device, dtype)

    print("Loading activations")
    table = pq.read_table(args.activations)
    n_total = len(table)
    flat = np.asarray(table["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
    d = flat.shape[0] // n_total
    activations_all = torch.from_numpy(flat.reshape(n_total, d).copy())
    if args.use_eval_set:
        n_eval = max(1, n_total // 20)
        pool = list(range(n_total - n_eval, n_total))
    else:
        pool = list(range(n_total))
    indices = np_rng.choice(pool, size=args.n_activations, replace=False).tolist()

    sc = KITFT_CFG["mse_scale"]
    e_v_all = av.get_input_embeddings().weight  # [V, d_emb]

    # Per-(activation, position) records
    results = []  # dicts with delta_q, delta_pi, baseline_L

    pbar = tqdm(indices, desc="activations")
    for idx in pbar:
        h = activations_all[idx : idx + 1].to(device).to(dtype)  # [1, d]

        # === 1. Sample rollout from AV ===
        chat = [{"role": "user", "content": KITFT_CFG["av_template"].format(marker=KITFT_CFG["marker_token"])}]
        pt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pi_ids = tok(pt, return_tensors="pt").input_ids.to(device)
        h_unit = F.normalize(h, dim=-1)
        inj = (KITFT_CFG["alpha"] * h_unit).to(dtype)
        pos_mark = (pi_ids == KITFT_CFG["marker_token_id"]).float().argmax(dim=1)
        prompt_embeds = av.get_input_embeddings()(pi_ids).clone()
        prompt_embeds[0, pos_mark] = inj
        prompt_attn = torch.ones_like(pi_ids)

        with torch.no_grad():
            gen = av.generate(
                inputs_embeds=prompt_embeds, attention_mask=prompt_attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        rollout_ids = gen[0]
        eos_id = tok.eos_token_id
        if eos_id in rollout_ids.tolist():
            end = rollout_ids.tolist().index(eos_id) + 1
            rollout_ids = rollout_ids[:end]
        T_resp = len(rollout_ids)
        if T_resp < 5:
            continue

        # === 2. Build AR input from rollout text (decoded then re-encoded) ===
        rollout_text = tok.decode(rollout_ids, skip_special_tokens=False)
        m = re.search(r"<explanation>(.*?)</explanation>", rollout_text, re.DOTALL)
        explanation = m.group(1).strip() if m else rollout_text.strip()
        # Re-tokenize the explanation for AR consistency
        expl_ids = tok(explanation, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
        ar_ids, ar_mask, expl_offset = build_ar_ids(tok, expl_ids.tolist(), device)
        n_expl = len(expl_ids)

        # === 3. Per-position AR forward + backward to get cumulative g_s ===
        ar_embeds = ar_body.embed_tokens(ar_ids).clone().requires_grad_(True)
        out = ar_body(inputs_embeds=ar_embeds, attention_mask=ar_mask, output_hidden_states=True, use_cache=False)
        h_ar = out.hidden_states[KITFT_CFG["extraction_layer"]]
        pred_all = value_head(h_ar)  # [1, T_ar, d]
        p_norm = F.normalize(pred_all.float(), dim=-1) * sc
        h_norm = F.normalize(h.float(), dim=-1).unsqueeze(1).expand_as(p_norm) * sc
        per_pos_mse = ((p_norm - h_norm) ** 2).mean(dim=-1)  # [1, T_ar]
        L_total_baseline = per_pos_mse.mean()
        last_mse_baseline = per_pos_mse[0, -1]
        g_at_input = torch.autograd.grad(L_total_baseline, ar_embeds)[0].detach()  # [1, T_ar, d]
        # gradient at explanation positions (where we'll substitute)
        g_resp = g_at_input[0, expl_offset : expl_offset + n_expl]  # [n_expl, d]

        # === 4. AV teacher-force forward to get π_t at each rollout position ===
        response_embeds_av = av.get_input_embeddings()(rollout_ids.unsqueeze(0))
        full_embeds = torch.cat([prompt_embeds, response_embeds_av], dim=1)
        full_attn = torch.ones((1, full_embeds.shape[1]), dtype=torch.long, device=device)
        T_pre = prompt_embeds.shape[1]
        with torch.no_grad():
            av_out = av(inputs_embeds=full_embeds, attention_mask=full_attn)
        av_logits = av_out.logits[0, T_pre - 1 : T_pre - 1 + T_resp]  # [T_resp, V]

        # Align logits to expl positions (rollout_ids may differ from expl_ids slightly)
        n_pos_compare = min(T_resp, n_expl)
        log_pi = F.log_softmax(av_logits[:n_pos_compare].float(), dim=-1)

        # === 5. Compute q_t at each candidate position ===
        # fp16 matmul, fp32 divisions. Do not materialize fp32 e_v matrix (OOM).
        # For --normalize-embed-norm, divide the matmul result by ||e_v|| broadcast,
        # which is mathematically equivalent to using e_v/||e_v|| as embedding.
        g_hat = g_resp[:n_pos_compare] / g_resp[:n_pos_compare].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        e_v = e_v_all.to(g_hat.dtype)  # fp16
        raw = (g_hat @ e_v.T).float()  # [n_pos, V] fp32
        if args.normalize_embed_norm:
            ev_norms = e_v.norm(dim=-1).float().clamp_min(1e-4)  # [V] fp32
            raw = raw / ev_norms.unsqueeze(0)  # broadcast: divide each column by its ||e_v||
        scores = -raw / args.tau  # [n_pos_compare, V]
        log_q = F.log_softmax(log_pi + scores, dim=-1)
        q = log_q.exp()
        pi = log_pi.exp()
        # Defensive: if any positions still have non-finite q, replace with π (no tilt)
        nonfinite_rows = ~torch.isfinite(q).all(dim=-1)
        if nonfinite_rows.any():
            q[nonfinite_rows] = pi[nonfinite_rows]
            log_q[nonfinite_rows] = log_pi[nonfinite_rows]

        # === 6. Pick M test positions and run substitutions ===
        # Choose positions where it's interesting: skip first ~3 and last ~3 (boundary effects)
        valid = list(range(3, n_pos_compare - 3))
        if not valid:
            continue
        m_actual = min(args.m_positions_per_rollout, len(valid))
        test_positions = np_rng.choice(valid, size=m_actual, replace=False).tolist()

        for t in test_positions:
            baseline_L = L_total_baseline.item() if args.per_pos_objective else last_mse_baseline.item()

            # K samples from q and K from π
            q_samples = torch.multinomial(q[t], num_samples=args.k_samples, replacement=True).tolist()
            pi_samples = torch.multinomial(pi[t], num_samples=args.k_samples, replacement=True).tolist()
            # Skip if all samples == original (would compare equal)
            original_token = expl_ids[t].item()

            def eval_substitution(new_tok):
                """Replace expl_ids[t] with new_tok, rebuild AR, compute L."""
                new_expl = expl_ids.tolist()
                new_expl[t] = new_tok
                ids_new, mask_new, _ = build_ar_ids(tok, new_expl, device)
                return ar_loss(ar_body, value_head, ids_new, mask_new, h,
                               per_position=args.per_pos_objective, mse_scale=sc).item()

            q_Ls = [eval_substitution(v) for v in q_samples]
            pi_Ls = [eval_substitution(v) for v in pi_samples]

            results.append({
                "idx": idx,
                "t": t,
                "baseline_L": baseline_L,
                "q_L_mean": float(np.mean(q_Ls)),
                "pi_L_mean": float(np.mean(pi_Ls)),
                "delta_q": float(np.mean(q_Ls)) - baseline_L,
                "delta_pi": float(np.mean(pi_Ls)) - baseline_L,
                "orig_tok": tok.decode([original_token]),
                "q_toks": [tok.decode([v]) for v in q_samples],
                "pi_toks": [tok.decode([v]) for v in pi_samples],
            })

        pbar.set_postfix(n=len(results))

    # === Aggregate ===
    print(f"\n=== Linearization test: {len(results)} (activation, position) pairs ===")
    deltas_q = np.array([r["delta_q"] for r in results])
    deltas_pi = np.array([r["delta_pi"] for r in results])

    print(f"\nObjective: {'cumulative per-position MSE' if args.per_pos_objective else 'last-position MSE'}")
    print(f"τ = {args.tau}")
    print(f"K = {args.k_samples} samples per position (per distribution)")
    print()
    print(f"Δ_q  = E[L(z with z_t<-v~q)] - L(z):  mean={deltas_q.mean():+.5f}  std={deltas_q.std():.5f}")
    print(f"Δ_π  = E[L(z with z_t<-v~π)] - L(z):  mean={deltas_pi.mean():+.5f}  std={deltas_pi.std():.5f}")
    print()
    # Negative = improvement
    n_q_neg = int((deltas_q < 0).sum())
    n_pi_neg = int((deltas_pi < 0).sum())
    print(f"Fraction with Δ < 0 (improvement):  q: {n_q_neg/len(results):.3f}  π: {n_pi_neg/len(results):.3f}")
    print(f"Mean improvement (Δ_q - Δ_π): {(deltas_q - deltas_pi).mean():+.5f}")
    print(f"   95% CI on (Δ_q - Δ_π): [{(deltas_q - deltas_pi).mean() - 1.96 * (deltas_q - deltas_pi).std()/np.sqrt(len(results)):+.5f}, "
          f"{(deltas_q - deltas_pi).mean() + 1.96 * (deltas_q - deltas_pi).std()/np.sqrt(len(results)):+.5f}]")

    # Show some examples
    print(f"\n=== 5 sample substitutions ===")
    for r in results[:5]:
        print(f"\nidx={r['idx']} pos={r['t']} original={r['orig_tok']!r}  baseline_L={r['baseline_L']:.4f}")
        print(f"  q-samples:  {r['q_toks']}  -> mean L_alt={r['q_L_mean']:.4f}  Δ_q={r['delta_q']:+.4f}")
        print(f"  π-samples:  {r['pi_toks']}  -> mean L_alt={r['pi_L_mean']:.4f}  Δ_π={r['delta_pi']:+.4f}")


if __name__ == "__main__":
    main()
