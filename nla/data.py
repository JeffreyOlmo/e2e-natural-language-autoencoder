"""NLA dataset: joins activations + summaries on (doc_id, position).

Splits are doc-level per the paper invariant — never cross-doc within a row.
Default fractions (0.40, 0.40, 0.20) for (av_sft, ar_sft, rl).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


SplitName = Literal["av_sft", "ar_sft", "rl"]


def doc_split(doc_id: int, total_docs: int, fractions=(0.4, 0.4, 0.2)) -> SplitName:
    f_av, f_ar, _ = fractions
    n_av = int(total_docs * f_av)
    n_ar = int(total_docs * f_ar)
    if doc_id < n_av:
        return "av_sft"
    if doc_id < n_av + n_ar:
        return "ar_sft"
    return "rl"


@dataclass
class NLARecord:
    h: torch.Tensor  # [d_model], raw (un-normalized)
    summary: str
    doc_id: int
    position: int


class NLADataset(Dataset):
    """Yields NLARecord. Joins activations and summaries on (doc_id, position).
    `split` filters to one of {av_sft, ar_sft, rl}; None returns all rows."""

    def __init__(
        self,
        activations_path: str | Path,
        summaries_path: str | Path,
        split: SplitName | None = None,
        fractions=(0.4, 0.4, 0.2),
        require_summary: bool = True,
    ):
        acts = pq.read_table(activations_path)
        sums = pq.read_table(summaries_path)
        # Build per-row dicts. FixedSizeListArray converts to numpy list-of-arrays.
        sum_lookup = {
            (int(d), int(p)): s
            for d, p, s in zip(
                sums["doc_id"].to_pylist(),
                sums["position"].to_pylist(),
                sums["summary"].to_pylist(),
            )
        }
        records = []
        n_acts = len(acts)
        doc_ids = acts["doc_id"].to_pylist()
        positions = acts["position"].to_pylist()
        # FixedSizeList → numpy via to_numpy(zero_copy_only=False) on the values, then reshape
        flat = np.asarray(acts["activation"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32)
        d = flat.shape[0] // n_acts
        acts_2d = flat.reshape(n_acts, d)  # [N, d]

        total_docs = int(max(doc_ids)) + 1
        for i in range(n_acts):
            did = int(doc_ids[i])
            if split is not None and doc_split(did, total_docs, fractions) != split:
                continue
            s = sum_lookup.get((did, int(positions[i])))
            if s is None or (require_summary and not s):
                continue
            records.append(NLARecord(
                h=torch.from_numpy(acts_2d[i].copy()),
                summary=s,
                doc_id=did,
                position=int(positions[i]),
            ))
        self.records = records
        self.d_model = d
        self.total_docs = total_docs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int) -> NLARecord:
        return self.records[i]


def collate(batch: list[NLARecord]) -> dict:
    return {
        "h": torch.stack([r.h for r in batch]),  # [B, d]
        "summary": [r.summary for r in batch],
        "doc_id": torch.tensor([r.doc_id for r in batch]),
        "position": torch.tensor([r.position for r in batch]),
    }
