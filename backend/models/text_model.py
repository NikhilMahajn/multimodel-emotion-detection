"""Text emotion model: transformer encoder + projection + classifier, and an inference wrapper.

The projection output (`embedding`, 256-d by default) is what the fusion model will consume later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# MELD label order. Keep identical across text / audio / visual / fusion models.
EMOTIONS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]

DEFAULT_CONFIG = dict(
    model_name="roberta-base",
    num_labels=len(EMOTIONS),
    proj_dim=256,
    dropout=0.1,
    context_turns=2,   # previous utterances used as dialogue context
    max_length=128,
)

_MODEL_KEYS = ("model_name", "num_labels", "proj_dim", "dropout")


def format_context(context: Optional[Sequence[str]], k: int) -> str:
    """Join the last k previous utterances into one context string."""
    if not context or k <= 0:
        return ""
    return " | ".join(context[-k:])


class TextEmotionModel(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=7, proj_dim=256,
                 dropout=0.1, pretrained=True):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:  # architecture only; weights come from our own checkpoint
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, proj_dim)
        self.act = nn.GELU()
        self.classifier = nn.Linear(proj_dim, num_labels)

    def forward(self, input_ids, attention_mask, **_):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        z = self.act(self.proj(self.dropout(h)))
        logits = self.classifier(self.dropout(z))
        return {"logits": logits, "embedding": z}


def save_checkpoint(model: TextEmotionModel, cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


class TextEmotionPredictor:
    """Loads a checkpoint and runs inference. Used by the API, explainer and (later) fusion."""

    def __init__(self, checkpoint, device: Optional[str] = None):
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.cfg = {**DEFAULT_CONFIG, **ckpt["config"]}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        self.model = TextEmotionModel(
            **{k: self.cfg[k] for k in _MODEL_KEYS}, pretrained=False
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    def _encode(self, items):
        k = self.cfg["context_turns"]
        texts = [t for t, _ in items]
        ctxs = [format_context(c, k) for _, c in items]
        return self.tokenizer(
            texts, ctxs, padding=True, truncation="only_second",
            max_length=self.cfg["max_length"], return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def predict_proba(self, items, batch_size: int = 32, return_embedding: bool = False):
        """items: list of (text, context_list_or_None). Returns probs [N, C] (+ embeddings)."""
        probs, embs = [], []
        for i in range(0, len(items), batch_size):
            enc = self._encode(items[i:i + batch_size])
            out = self.model(**enc)
            probs.append(torch.softmax(out["logits"], dim=-1).cpu())
            embs.append(out["embedding"].cpu())
        probs = torch.cat(probs).numpy()
        if return_embedding:
            return probs, torch.cat(embs).numpy()
        return probs

    def predict(self, text: str, context: Optional[Sequence[str]] = None,
                return_embedding: bool = False) -> dict:
        probs, emb = self.predict_proba([(text, context)], return_embedding=True)
        p = probs[0]
        idx = int(np.argmax(p))
        result = {
            "label": EMOTIONS[idx],
            "confidence": float(p[idx]),
            "probabilities": {e: float(x) for e, x in zip(EMOTIONS, p)},
        }
        if return_embedding:
            result["embedding"] = emb[0].tolist()
        return result


# --------------------------------------------------------------------------- #
# Zero-training option: use an already fine-tuned emotion model from Hugging Face
# --------------------------------------------------------------------------- #
PRETRAINED_DEFAULT = "j-hartmann/emotion-english-distilroberta-base"


class PretrainedTextPredictor(TextEmotionPredictor):
    """Same interface as TextEmotionPredictor, but needs no training.

    The model was trained on single utterances, so dialogue context is ignored.
    The embedding is the encoder's final <s> hidden state (768-d, not 256-d).
    """

    def __init__(self, model_name: str = PRETRAINED_DEFAULT, device: Optional[str] = None):
        self.cfg = {"model_name": model_name, "max_length": 128}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        id2label = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        missing = [e for e in EMOTIONS if e not in id2label.values()]
        if missing:
            raise ValueError(
                f"{model_name} has labels {sorted(id2label.values())}; missing {missing}. "
                "Use a model with MELD's 7 labels or add a label mapping."
            )
        inv = {l: i for i, l in id2label.items()}
        self._order = [inv[e] for e in EMOTIONS]  # reorder model outputs to our EMOTIONS order

    @torch.no_grad()
    def predict_proba(self, items, batch_size: int = 32, return_embedding: bool = False):
        probs, embs = [], []
        for i in range(0, len(items), batch_size):
            texts = [t for t, _ in items[i:i + batch_size]]
            enc = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.cfg["max_length"], return_tensors="pt").to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            probs.append(torch.softmax(out.logits, dim=-1)[:, self._order].cpu())
            embs.append(out.hidden_states[-1][:, 0].cpu())
        probs = torch.cat(probs).numpy()
        if return_embedding:
            return probs, torch.cat(embs).numpy()
        return probs