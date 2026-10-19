"""Loads the text model once and shares it across requests.

Uses your fine-tuned checkpoint if one exists, otherwise falls back to a pretrained
Hugging Face emotion model (no training needed).
"""
import os
from functools import lru_cache
from pathlib import Path

from backend.models.text_model import (PRETRAINED_DEFAULT, PretrainedTextPredictor,
                                       TextEmotionPredictor)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_predictor():
    path = Path(os.getenv("TEXT_MODEL_PATH", PROJECT_ROOT / "experiments" / "text" / "best.pt"))
    if path.exists():
        return TextEmotionPredictor(path)
    return PretrainedTextPredictor(os.getenv("TEXT_PRETRAINED_MODEL", PRETRAINED_DEFAULT))