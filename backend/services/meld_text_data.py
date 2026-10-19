"""MELD text loading: builds (utterance, dialogue-context, label) samples and a batching collate."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.models.text_model import EMOTIONS, format_context

EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}


def _clean(s: str) -> str:
    # MELD CSVs contain mis-encoded apostrophes/quotes.
    return (str(s).replace("\x92", "\x27").replace("\x91", "\x27")
            .replace("\x93", "\x22").replace("\x94", "\x22").replace("\x85", "...").strip())


def load_meld_text(root, split: str, context_turns: int = 2) -> list[dict]:
    """root = folder containing train/, dev/, test/ (i.e. MELD.Raw). split in {train, dev, test}."""
    df = pd.read_csv(Path(root) / split / f"{split}_sent_emo.csv")
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    samples = []
    for dia_id, g in df.groupby("Dialogue_ID", sort=False):
        utts = [f"{sp}: {_clean(u)}" for sp, u in zip(g["Speaker"], g["Utterance"])]
        for i, (u, emo) in enumerate(zip(utts, g["Emotion"])):
            samples.append({
                "text": u,
                "context": utts[:i],
                "label": EMO2ID[emo.lower()],
                "dialogue_id": int(dia_id),
            })
    return samples


class MELDTextDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    @property
    def labels(self):
        return [s["label"] for s in self.samples]


def make_collate(tokenizer, max_length: int = 128, context_turns: int = 2):
    def collate(batch):
        enc = tokenizer(
            [b["text"] for b in batch],
            [format_context(b["context"], context_turns) for b in batch],
            padding=True, truncation="only_second",
            max_length=max_length, return_tensors="pt",
        )
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc
    return collate