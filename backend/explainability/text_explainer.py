"""Word-level explanations for the text model via occlusion.

For each word in the *current* utterance we remove it and measure how much the probability of the
target emotion drops. Positive importance = the word pushed the model toward that emotion.
Model-agnostic and needs no extra dependencies (swap in SHAP / Captum later if you want).
"""
from __future__ import annotations

from typing import Optional, Sequence

from backend.models.text_model import EMOTIONS, TextEmotionPredictor


class TextExplainer:
    def __init__(self, predictor: TextEmotionPredictor, max_words: int = 60):
        self.predictor = predictor
        self.max_words = max_words

    def explain(self, text: str, context: Optional[Sequence[str]] = None,
                target: Optional[str] = None) -> dict:
        words = text.split()[: self.max_words]
        base = self.predictor.predict_proba([(text, context)])[0]
        t_idx = EMOTIONS.index(target) if target else int(base.argmax())

        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        occluded = self.predictor.predict_proba([(v, context) for v in variants]) if variants else []

        tokens = [
            {"word": w, "importance": float(base[t_idx] - occluded[i][t_idx])}
            for i, w in enumerate(words)
        ]
        return {
            "target": EMOTIONS[t_idx],
            "probability": float(base[t_idx]),
            "method": "occlusion",
            "tokens": tokens,
        }