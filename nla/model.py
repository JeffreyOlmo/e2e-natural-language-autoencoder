"""AR (activation reconstructor) — truncated K-layer base + identity-init value head.

Follows the paper's architecture: first `cfg.layer` transformer blocks of the
base model, no final RMSNorm, then `Linear(d, d)` with identity init reading
at the last token (suffix-anchored).

Identity init matters: kaiming would scale predictions by ~1/√3 and cost ~17%
on step-0 loss (per the Qwen2.5-7B reference run in the paper repo's notes).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from nla.config import NLAConfig


class ARModel(nn.Module):
    """First `cfg.layer` blocks of base + value head. forward(input_ids, attention_mask) -> [B, d].

    Optional `with_aux_head`: adds a separate Linear(d, d) aux head used to read
    intermediate-position reconstructions. Aux head consumes a *detached* hidden
    state, so its gradient never flows into the backbone (decoupled credit
    assignment for per-step rewards). Aux head is identity-init; the user should
    copy `value_head.weight` into it after loading a trained checkpoint.
    """

    def __init__(self, cfg: NLAConfig, dtype=torch.float32, with_aux_head: bool = False):
        super().__init__()
        full = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
        self.body = full.model
        self.body.layers = nn.ModuleList(self.body.layers[: cfg.layer])
        self.body.config.num_hidden_layers = cfg.layer
        # Skip the final RMSNorm so the body's last_hidden_state matches
        # `hidden_states[layer]` of the full model (HF convention: pre-final-norm).
        self.body.norm = nn.Identity()
        d = self.body.config.hidden_size
        self.value_head = nn.Linear(d, d, bias=False, dtype=dtype)
        with torch.no_grad():
            self.value_head.weight.copy_(torch.eye(d, dtype=dtype))
        if with_aux_head:
            self.aux_head = nn.Linear(d, d, bias=False, dtype=dtype)
            with torch.no_grad():
                self.aux_head.weight.copy_(torch.eye(d, dtype=dtype))
        else:
            self.aux_head = None

    @property
    def config(self):
        return self.body.config

    def get_input_embeddings(self):
        return self.body.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (input_ids is None) ^ (inputs_embeds is None), "give one of input_ids or inputs_embeds"
        out = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        h = out.last_hidden_state  # [B, T, d]
        bsz = h.shape[0]
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
        else:
            last_idx = torch.full((bsz,), h.shape[1] - 1, dtype=torch.long, device=h.device)
        last_h = h[torch.arange(bsz, device=h.device), last_idx]
        return self.value_head(last_h)

    def forward_with_aux(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred_last [B, d], pred_per_pos [B, T, d]).

        pred_last comes from value_head at the suffix-anchored position; its
        gradient flows into the backbone. pred_per_pos comes from aux_head
        applied to a DETACHED hidden state — its gradient updates aux_head only.
        """
        assert self.aux_head is not None, "aux_head not enabled; pass with_aux_head=True at init"
        assert (input_ids is None) ^ (inputs_embeds is None), "give one of input_ids or inputs_embeds"
        out = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        h = out.last_hidden_state  # [B, T, d]
        bsz = h.shape[0]
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
        else:
            last_idx = torch.full((bsz,), h.shape[1] - 1, dtype=torch.long, device=h.device)
        last_h = h[torch.arange(bsz, device=h.device), last_idx]
        pred_last = self.value_head(last_h)
        pred_per_pos = self.aux_head(h.detach())
        return pred_last, pred_per_pos
