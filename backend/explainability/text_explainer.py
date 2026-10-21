"""Word-level explanations for the text model via occlusion.

For each word in the *current* utterance we remove it and measure how much the probability of the
target emotion drops. Positive importance = the word pushed the model toward that emotion.
Model-agnostic and needs no extra dependencies (swap in SHAP / Captum later if you want).

Occlusion is O(n) forward passes, so the default is capped at 40 words.  The word list is returned
in its original order for highlighting, plus a ``top_tokens`` list sorted by absolute effect.
"""
from __future__ import annotations

from typing import Optional, Sequence

from backend.models.text_model import EMOTIONS, TextEmotionPredictor


class TextExplainer:
    def __init__(self, predictor: TextEmotionPredictor, max_words: int = 40):
        self.predictor = predictor
        self.max_words = max_words

    def explain(self, text: str, context: Optional[Sequence[str]] = None,
                target: Optional[str] = None) -> dict:
        words = str(text).split()[: self.max_words]
        if target is not None and target not in EMOTIONS:
            raise ValueError(f"Unknown target {target!r}. Expected one of {EMOTIONS}.")

        base = self.predictor.predict_proba([(text, context)])[0]
        t_idx = EMOTIONS.index(target) if target else int(base.argmax())

        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        occluded = (self.predictor.predict_proba([(v, context) for v in variants])
                    if variants else [])

        tokens = [
            {
                "word": word,
                "importance": float(base[t_idx] - occluded[i][t_idx]),
                "direction": ("supports" if base[t_idx] - occluded[i][t_idx] >= 0
                              else "opposes"),
            }
            for i, word in enumerate(words)
        ]
        top_tokens = sorted(
            (dict(token) for token in tokens), key=lambda t: -abs(t["importance"])
        )[:10]

        return {
            "target": EMOTIONS[t_idx],
            "probability": float(base[t_idx]),
            "method": "occlusion",
            "tokens": tokens,
            "top_tokens": top_tokens,
        }
