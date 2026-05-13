"""Round-trip plumbing test for AV + AR. No training.

Validates:
  1. AR truncation matches full-model hidden_states[layer] (correctness check).
  2. AV forward + generate work with inputs_embeds (injection path).
  3. AR forward+backward yields finite g_t = ∂MSE/∂e_t with the right shape.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import NLAConfig
from nla.injection import build_av_prompt_ids, build_av_inputs_embeds
from nla.prompts import build_ar_prompt
from nla.model import ARModel


def main():
    cfg = NLAConfig()
    device = "cuda:0"
    dtype = torch.float32

    print("=== Loading models ===")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    print(f"AV: {cfg.base_model}")
    av = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).to(device)
    av.eval()
    print(f"AR: first {cfg.layer} layers + Linear({cfg.d_model},{cfg.d_model}) identity-init")
    ar = ARModel(cfg, dtype=dtype).to(device)
    ar.eval()

    # ----- Test 1: AR matches full-model hidden_states[layer] before value head -----
    print("\n=== Test 1: AR truncation correctness ===")
    sentence = "The quick brown fox jumps over the lazy dog."
    ids = tok(sentence, return_tensors="pt").input_ids.to(device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        full_out = av.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        h_ref = full_out.hidden_states[cfg.layer]  # [1, T, d]
        # AR body (no value head)
        ar_body_out = ar.body(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    diff = (h_ref - ar_body_out).abs().max().item()
    print(f"  max |hidden_states[{cfg.layer}] − AR.body.last_hidden| = {diff:.2e}")
    assert diff < 1e-3, f"AR body diverges from full hidden_states[{cfg.layer}]"
    # Identity-init value head: pred at last token == AR body's last hidden state at last token
    with torch.no_grad():
        pred_init = ar(input_ids=ids, attention_mask=mask)
        last_h_full = h_ref[:, -1, :]
    init_diff = (pred_init - last_h_full).abs().max().item()
    print(f"  max |AR(init) pred − last hidden| = {init_diff:.2e}  (identity-init value head check)")
    assert init_diff < 1e-3

    # ----- Test 2: AV forward + generate with inputs_embeds -----
    print("\n=== Test 2: AV injection path ===")
    torch.manual_seed(0)
    h_real = torch.randn(1, cfg.d_model, device=device)
    h_unit = F.normalize(h_real, dim=-1)
    marker_id = tok.encode(cfg.marker_token, add_special_tokens=False)[0]
    print(f"  marker={cfg.marker_token!r} id={marker_id}, alpha={cfg.alpha:.3f}")
    av_ids = build_av_prompt_ids(tok, cfg.marker_token).to(device)
    print(f"  AV prompt: {av_ids.shape[1]} tokens")
    av_embeds = build_av_inputs_embeds(av.get_input_embeddings(), av_ids, marker_id, h_real, cfg.alpha)
    av_mask = torch.ones(av_ids.shape, dtype=torch.long, device=device)
    with torch.no_grad():
        av_fwd = av(inputs_embeds=av_embeds, attention_mask=av_mask, use_cache=False)
    print(f"  AV forward OK, logits {tuple(av_fwd.logits.shape)}")
    with torch.no_grad():
        gen = av.generate(
            inputs_embeds=av_embeds,
            attention_mask=av_mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    # generate(inputs_embeds=...) returns only the new tokens (HF convention)
    gen_text = tok.decode(gen[0], skip_special_tokens=False)
    print(f"  AV greedy gen (20 tok): {gen_text!r}")

    # ----- Test 3: AR backward — g_t = ∂MSE/∂e_t -----
    print("\n=== Test 3: AR grad through input embeddings ===")
    fake_expl = "a description of mathematical reasoning involving prime numbers"
    ar_text = build_ar_prompt(fake_expl)
    ar_ids = tok(ar_text, return_tensors="pt").input_ids.to(device)
    ar_mask = torch.ones_like(ar_ids)
    print(f"  AR prompt: {ar_ids.shape[1]} tokens, last token id={ar_ids[0,-1].item()} ({tok.decode([ar_ids[0,-1]])!r})")

    ar.train()
    # Embed input ourselves so we can backprop to the embedding outputs.
    ar_embeds = ar.get_input_embeddings()(ar_ids).detach().clone().requires_grad_(True)
    pred = ar(inputs_embeds=ar_embeds, attention_mask=ar_mask)  # [1, d]
    pred_unit = F.normalize(pred, dim=-1)
    mse = F.mse_loss(pred_unit, h_unit, reduction="mean")
    print(f"  MSE(pred_unit, h_unit) = {mse.item():.6f}  (random h_unit: expected ≈ 2/d = {2/cfg.d_model:.6f})")
    mse.backward()
    g = ar_embeds.grad  # [1, T, d]
    print(f"  g_t shape: {tuple(g.shape)}, finite: {torch.isfinite(g).all().item()}, mean|g|: {g.abs().mean().item():.2e}")

    # Per-token gradient norms — should be nonzero everywhere (gradient flows through attention)
    g_norms = g.norm(dim=-1).squeeze(0)  # [T]
    print(f"  per-token ||g_t||: min={g_norms.min().item():.2e}, max={g_norms.max().item():.2e}, last={g_norms[-1].item():.2e}")

    # Per-vocab scores at the last position (where the value head reads):
    # s_t(v) = -<g_t, e_v> / tau
    e_v = ar.get_input_embeddings().weight  # [V, d]
    g_last = g[0, -1]  # [d]
    scores = -(e_v @ g_last) / cfg.tau  # [V]
    q = torch.softmax(scores, dim=-1)
    H = -(q * (q.clamp_min(1e-30).log())).sum().item()
    print(f"  q (vocab dist at last pos) entropy = {H:.3f} nats (uniform = {torch.log(torch.tensor(float(e_v.shape[0]))).item():.3f})")
    top = q.topk(5)
    print(f"  top-5 q tokens at last pos: {[(tok.decode([i.item()]), round(p.item(), 4)) for p, i in zip(top.values, top.indices)]}")

    print("\n=== Smoke test passed ===")


if __name__ == "__main__":
    main()
