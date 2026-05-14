"""ParquetGRPODataset: yields one trajectory per item with precomputed advantages."""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class ParquetGRPODataset(Dataset):
    """Loads a parquet file produced by grpo.preprocess.build_dataset.

    Each row: input_ids, attention_mask, response_mask, advantages (per-token),
    plus task_id, seed, outcome metadata. Variable sequence length per row.
    """

    def __init__(self, parquet_path: Path):
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        self.df = pq.read_table(parquet_path).to_pandas()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.df.iloc[idx]
        return {
            "input_ids": torch.tensor(list(row["input_ids"]), dtype=torch.long),
            "attention_mask": torch.tensor(list(row["attention_mask"]), dtype=torch.long),
            "response_mask": torch.tensor(list(row["response_mask"]), dtype=torch.long),
            "advantages": torch.tensor(list(row["advantages"]), dtype=torch.float32),
            "task_id": str(row["task_id"]),
            "seed": int(row["seed"]),
        }


def collate_pad_right(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch of variable-length sequences to the max length in batch."""
    T = max(item["input_ids"].size(0) for item in batch)

    def pad(t: torch.Tensor, val) -> torch.Tensor:
        pad_len = T - t.size(0)
        if pad_len == 0:
            return t
        padding = torch.full((pad_len,), val, dtype=t.dtype)
        return torch.cat([t, padding])

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_token_id) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "response_mask": torch.stack([pad(b["response_mask"], 0) for b in batch]),
        "advantages": torch.stack([pad(b["advantages"], 0.0) for b in batch]),
        "task_ids": [b["task_id"] for b in batch],
        "seeds": [b["seed"] for b in batch],
    }
