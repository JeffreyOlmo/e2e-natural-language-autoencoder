"""Build AV inputs_embeds with the marker token's embedding replaced by α·ĥ.

Preprocessing pattern: the trainer builds a [B, T, d] embedding sequence with
the activation spliced in at the marker position. The model's forward then
runs unchanged via `inputs_embeds=...`. Same pattern works for HF generate,
vLLM, and SGLang — none of them needs to know about the injection.

`h` is L2-normalized inside (do not pre-normalize). The reconstruction loss
is direction-only, so the magnitude is purely a propagation hack — a large α
keeps the injected vector dominant through the early layers until layer-ℓ-
reading machinery sees it.
"""
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from nla.prompts import build_av_messages


def find_marker_position(input_ids: torch.Tensor, marker_id: int) -> torch.Tensor:
    """Return [B] indices of the marker. Asserts exactly one marker per row."""
    matches = input_ids == marker_id
    counts = matches.sum(dim=1)
    assert (counts == 1).all(), f"each AV prompt must contain exactly one marker; got counts={counts.tolist()}"
    return matches.float().argmax(dim=1)


def build_av_inputs_embeds(
    embed_tokens: torch.nn.Embedding,
    input_ids: torch.Tensor,  # [B, T]
    marker_id: int,
    h: torch.Tensor,  # [B, d]
    alpha: float,
) -> torch.Tensor:
    """[B, T, d] embeddings; marker position replaced by alpha * ĥ."""
    embeds = embed_tokens(input_ids)  # [B, T, d]
    bsz = input_ids.shape[0]
    pos = find_marker_position(input_ids, marker_id)
    h_unit = F.normalize(h, dim=-1)
    inj = (alpha * h_unit).to(embeds.dtype)
    embeds = embeds.clone()  # don't mutate the embedding-table view
    embeds[torch.arange(bsz, device=embeds.device), pos] = inj
    return embeds


def build_av_prompt_ids(tokenizer: PreTrainedTokenizer, marker_token: str) -> torch.Tensor:
    """Tokenize the AV chat prompt with the marker placeholder. Returns [1, T]."""
    messages = build_av_messages(marker_token)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids
