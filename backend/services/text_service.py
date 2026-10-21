"""Loads the text model once and shares it across requests.

Uses the fine-tuned checkpoint if one exists, otherwise falls back to a pretrained Hugging Face
emotion model (no training needed).

Environment variables
---------------------
``TEXT_MODEL_PATH``         checkpoint path (default ``experiments/text/best.pt``)
``TEXT_PRETRAINED_MODEL``   Hugging Face fallback (default ``j-hartmann/...``)
``TEXT_DEVICE``             ``cpu`` or ``cuda`` (default: auto-detect)
``TEXT_TEMPERATURE``        temperature for the pretrained fallback (default 1.0)
``TEXT_MAX_LENGTH``         token limit for the pretrained fallback (default 256)
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
    device = os.getenv("TEXT_DEVICE") or None
    if path.exists():
        return TextEmotionPredictor(path, device=device)
    return PretrainedTextPredictor(
        os.getenv("TEXT_PRETRAINED_MODEL", PRETRAINED_DEFAULT),
        device=device,
        temperature=float(os.getenv("TEXT_TEMPERATURE", "1.0")),
        max_length=int(os.getenv("TEXT_MAX_LENGTH", "256")),
    )


def get_model_info() -> dict:
    """Metadata for ``GET /text/model-info`` (also useful in logs/experiment reports)."""
    predictor = get_predictor()
    info = predictor.info() if hasattr(predictor, "info") else {}
    checkpoint = Path(os.getenv(
        "TEXT_MODEL_PATH", PROJECT_ROOT / "experiments" / "text" / "best.pt"
    ))
    info.update({
        "checkpoint_path": str(checkpoint),
        "checkpoint_used": checkpoint.exists(),
        "context_supported": bool(getattr(predictor, "cfg", {}).get("context_turns", 0)),
    })
    return info
