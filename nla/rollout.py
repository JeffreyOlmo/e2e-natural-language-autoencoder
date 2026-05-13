"""AV rollout helpers — sample explanations and recover on-policy logits.

The "rollout" is just `model.generate(inputs_embeds=...)` with sampling. After
sampling, we teacher-force a forward pass over the (prompt + response) sequence
to recover the policy distribution at each response position. That distribution
is what the grad-distill loss pushes around.

For Qwen2.5-0.5B this is fast enough to do inline. For larger models, swap in
vLLM via the `inputs_embeds` API — the trainer-side prep is identical.
"""
import torch
import torch.nn.functional as F

from nla.injection import build_av_prompt_ids, build_av_inputs_embeds


def av_rollout(
    av_model,
    tokenizer,
    h: torch.Tensor,
    *,
    marker_token: str,
    alpha: float,
    max_new_tokens: int = 150,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int | None = None,
) -> dict:
    """Sample, then teacher-force.

    Args:
      h: [B, d] activations (raw, not normalized — built_av_inputs_embeds normalizes).

    Returns dict:
      rollout_ids:    [B, T_resp]  — sampled token ids
      rollout_mask:   [B, T_resp]  — 1 on real tokens (up to and including first eos)
      rollout_logits: [B, T_resp, V]  — π_AV(. | prompt + rollout[:t]) for t = 0..T_resp-1
      prompt_len:     int — number of prompt tokens (constant across batch)
    """
    device = next(av_model.parameters()).device
    B = h.shape[0]
    h = h.to(device)
    marker_id = tokenizer.encode(marker_token, add_special_tokens=False)[0]

    prompt_ids_single = build_av_prompt_ids(tokenizer, marker_token).to(device)  # [1, T_pre]
    prompt_ids = prompt_ids_single.expand(B, -1).contiguous()
    prompt_mask = torch.ones_like(prompt_ids)
    T_pre = prompt_ids.shape[1]

    # Replace marker embedding with α·ĥ for each row.
    prompt_embeds = build_av_inputs_embeds(av_model.get_input_embeddings(), prompt_ids, marker_id, h, alpha)

    if seed is not None:
        torch.manual_seed(seed)
    was_training = av_model.training
    av_model.eval()
    with torch.no_grad():
        gen = av_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    if was_training:
        av_model.train()

    # With inputs_embeds, gen contains ONLY the new tokens.
    rollout_ids = gen  # [B, T_resp]

    # Mask: 1 up to and including the first eos; 0 thereafter.
    eos_id = tokenizer.eos_token_id
    is_eos = rollout_ids == eos_id
    cum_eos = is_eos.cumsum(dim=1)
    rollout_mask = (cum_eos <= 1).long()  # keeps the first eos itself

    # Teacher-force over (prompt + rollout) to get policy logits at each rollout pos.
    response_embeds = av_model.get_input_embeddings()(rollout_ids)
    full_embeds = torch.cat([prompt_embeds, response_embeds], dim=1)
    full_mask = torch.cat([prompt_mask, rollout_mask], dim=1)
    out = av_model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    # out.logits[:, T_pre - 1, :] predicts the FIRST response token (using just prompt).
    # out.logits[:, T_pre - 1 + t, :] predicts response[t+1] given prompt + response[:t+1].
    # We want π_AV(. | prompt + response[:t]) for each t in [0, T_resp), which lives at
    # logits index T_pre - 1 + t.
    T_resp = rollout_ids.shape[1]
    rollout_logits = out.logits[:, T_pre - 1 : T_pre - 1 + T_resp, :].contiguous()

    return {
        "rollout_ids": rollout_ids,
        "rollout_mask": rollout_mask,
        "rollout_logits": rollout_logits,
        "prompt_len": T_pre,
    }
