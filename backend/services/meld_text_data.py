"""MELD text loading: builds (utterance, dialogue-context, label) samples and a batching collate."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.models.text_model import EMOTIONS, encode_items
from backend.services.text_data import read_csv

logger = logging.getLogger(__name__)

EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}


def _clean(s: str) -> str:
    # MELD CSVs contain mis-encoded apostrophes/quotes (cp1252 bytes decoded as controls).
    cleaned = (str(s).replace("\x92", "'").replace("\x91", "'")
               .replace("\x93", '"').replace("\x94", '"').replace("\x85", "..."))
    return " ".join(cleaned.split())


def _resolve_csv(root, split: str) -> Path:
    """Find the split CSV.

    The MELD archive keeps ``train/train_sent_emo.csv`` but puts ``dev_sent_emo.csv`` and
    ``test_sent_emo.csv`` at the top level.  The original loader assumed the nested layout for all
    three splits and therefore crashed on dev/test.
    """
    root = Path(root)
    candidates = [
        root / split / f"{split}_sent_emo.csv",
        root / f"{split}_sent_emo.csv",
        root / "MELD.Raw" / split / f"{split}_sent_emo.csv",
        root / "MELD.Raw" / f"{split}_sent_emo.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find the {split!r} split under {root}. Looked for: "
        + ", ".join(str(c) for c in candidates)
    )


def load_meld_text(root, split: str, context_turns: int = 2,
                   include_speaker: bool = True) -> list[dict]:
    """root = folder containing train/, dev/, test/ (i.e. MELD.Raw). split in {train, dev, test}.

    ``include_speaker`` keeps the ``"Speaker: utterance"`` prefix; it helps on MELD because
    characters have distinct emotional styles, but should be disabled for anonymised mental-health
    transcripts without reliable speaker IDs.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be train/dev/test, got {split!r}")
    path = _resolve_csv(root, split)
    df = read_csv(path)
    required = {"Utterance", "Speaker", "Emotion", "Dialogue_ID", "Utterance_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    samples = []
    unknown = Counter()
    for dia_id, group in df.groupby("Dialogue_ID", sort=False):
        utterances = []
        for speaker, utterance in zip(group["Speaker"], group["Utterance"]):
            text = _clean(utterance)
            utterances.append(f"{speaker}: {text}" if include_speaker else text)

        for i, (turn, raw_emotion) in enumerate(zip(utterances, group["Emotion"])):
            emotion = str(raw_emotion).strip().lower()
            if emotion not in EMO2ID:
                unknown[emotion] += 1
                continue
            samples.append({
                "text": turn,
                "context": utterances[:i],
                "label": EMO2ID[emotion],
                "dialogue_id": int(dia_id),
            })

    if unknown:
        logger.warning("Skipped %d MELD rows with unknown labels: %s",
                       sum(unknown.values()), dict(unknown))
    if not samples:
        raise ValueError(f"No usable samples loaded from {path}")
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


def make_collate(tokenizer, max_length: int = 128, context_turns: int = 2,
                 context_max_chars: Optional[int] = 480):
    """Build the collate function used by training and evaluation.

    Uses :func:`backend.models.text_model.encode_items` so long contexts keep their most recent
    tokens instead of losing them to right-side truncation.
    """
    def collate(batch):
        enc = encode_items(
            tokenizer,
            [(b["text"], b["context"]) for b in batch],
            max_length=max_length,
            context_turns=context_turns,
            context_max_chars=context_max_chars,
        )
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc
    return collate
