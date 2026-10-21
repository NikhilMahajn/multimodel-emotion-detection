"""Generic text-emotion dataset loading for domain adaptation.

MELD is scripted TV dialogue; mental-health corpora (counselling transcripts, support-forum posts,
GoEmotions, IEMOCAP, ...) look quite different.  This loader maps common CSV layouts onto the same
``{"text", "context", "label"}`` samples used by the MELD pipeline so a checkpoint can be
fine-tuned on in-domain data without rewriting the training code.

Expected CSV columns (names are configurable):

* one text column (``text`` by default);
* one emotion label column (``label`` by default);
* optionally a speaker column;
* optionally a context column containing a JSON list or ``"turn A || turn B"`` (values are used
  as-is, so include speakers in them if you want speaker context);
* optionally dialogue/utterance ID columns, in which case the context is built from the previous
  turns of the same dialogue.

Labels are normalised through :data:`backend.models.text_model.LABEL_ALIASES` (e.g. ``happy`` ->
``joy``, ``anxious`` -> ``fear``).  Rows whose label cannot be mapped are skipped with a warning
rather than silently folded into ``neutral``.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from backend.models.text_model import EMOTIONS, try_normalize_label

logger = logging.getLogger(__name__)


def read_csv(path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
    encodings = [encoding] if encoding else ["utf-8-sig", "cp1252", "latin-1"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
    raise ValueError(f"Could not decode {path}: {last_error}")


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_text(value) -> str:
    return " ".join(str(value).replace("\x92", "'").replace("\x91", "'")
                    .replace("\x93", '"').replace("\x94", '"').split())


def _format_turn(text: str, speaker: Optional[str], include_speaker: bool) -> str:
    if include_speaker and speaker and str(speaker).strip():
        return f"{str(speaker).strip()}: {text}"
    return text


def _parse_context(value) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    for separator in ("||", "|"):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _map_label(raw, label_map: Optional[Mapping[str, str]]) -> Optional[str]:
    if label_map:
        key = str(raw).strip().lower()
        if key in label_map:
            raw = label_map[key]
    return try_normalize_label(raw)


def load_emotion_csv(
    path,
    *,
    text_col: str = "text",
    label_col: str = "label",
    context_col: Optional[str] = None,
    speaker_col: Optional[str] = None,
    dialogue_col: Optional[str] = None,
    utterance_col: Optional[str] = None,
    label_map: Optional[Mapping[str, str]] = None,
    include_speaker: bool = False,
    encoding: Optional[str] = None,
) -> list[dict]:
    """Load a CSV and return samples compatible with :class:`MELDTextDataset`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Emotion CSV not found: {path}")
    df = read_csv(path, encoding=encoding)

    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(df.columns)}")
    for col in (context_col, speaker_col, dialogue_col, utterance_col):
        if col and col not in df.columns:
            raise ValueError(f"{path} has no column {col!r}; found {list(df.columns)}")

    if dialogue_col and utterance_col:
        df = df.sort_values([dialogue_col, utterance_col]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    samples: list[dict] = []
    skipped: Counter[str] = Counter()
    group_turns: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        raw_text = row[text_col]
        if _is_missing(raw_text):
            skipped["empty_text"] += 1
            continue
        text = _clean_text(raw_text)
        if not text:
            skipped["empty_text"] += 1
            continue
        label = _map_label(row[label_col], label_map)
        if label is None:
            skipped[str(row[label_col]).strip().lower()] += 1
            continue

        speaker = row[speaker_col] if speaker_col else None
        if _is_missing(speaker):
            speaker = None
        turn = _format_turn(text, speaker, include_speaker)

        if dialogue_col and utterance_col:
            dialogue_id = str(row[dialogue_col])
            context = list(group_turns.get(dialogue_id, []))
            group_turns.setdefault(dialogue_id, []).append(turn)
        elif context_col:
            # Context-column values are treated as already-formatted turns (they may include
            # speakers); this avoids double-prefixing them with the current row's speaker.
            context = [_clean_text(t) for t in _parse_context(row[context_col])]
            dialogue_id = None
        else:
            context = []
            dialogue_id = None

        sample = {"text": turn, "context": context, "label": EMOTIONS.index(label)}
        if dialogue_id is not None:
            sample["dialogue_id"] = dialogue_id
        samples.append(sample)

    if skipped:
        logger.warning(
            "Skipped %d row(s) while loading %s: %s",
            sum(skipped.values()), path,
            dict(skipped.most_common(10)),
        )
    if not samples:
        raise ValueError(f"No usable samples in {path} (label column={label_col!r})")
    return samples


def dataset_stats(samples: Sequence[dict]) -> dict:
    """Small summary useful for spotting label/context problems before training."""
    labels = Counter(EMOTIONS[s["label"]] for s in samples)
    lengths = [len(s["text"].split()) for s in samples]
    return {
        "samples": len(samples),
        "label_counts": dict(labels),
        "label_fractions": {k: round(v / len(samples), 4) for k, v in labels.items()},
        "with_context": sum(1 for s in samples if s.get("context")),
        "mean_text_words": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "max_text_words": max(lengths) if lengths else 0,
    }
