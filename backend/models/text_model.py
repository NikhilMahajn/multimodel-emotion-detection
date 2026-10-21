"""Text emotion model: transformer encoder + projection + classifier, and inference wrappers.

Improvements over the first version, aimed at mental-health style text analysis:

* **Context window that keeps the most recent dialogue turns.**  Instead of letting the tokenizer
  cut the *end* of the context (which throws away the turns closest to the current utterance),
  `encode_items` reserves a token budget for the current utterance and keeps the tail of the
  context.
* **Configurable pooling** (``cls`` / ``mean`` / ``attention``).  Emotional cues are often spread
  across an utterance, so mean or attention pooling can help on longer, clinical-style messages.
* **Calibration.**  Checkpoints can store a temperature and a class-prior correction fitted on the
  dev set.  Calibrated probabilities matter when the output is shown to a clinician or used to
  trigger a safety workflow.
* **Richer but backwards-compatible predictions**: top-k emotions, entropy/uncertainty, derived
  sentiment and the 256-d fusion embedding are returned next to the original
  ``label``/``confidence``/``probabilities`` keys.

Labels are MELD's seven emotions.  Keep the order identical across text / audio / visual / fusion
models.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# MELD label order. Keep identical across text / audio / visual / fusion models.
EMOTIONS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]

# Common label spellings from other corpora (GoEmotions, IEMOCAP, tweet datasets, ...).
# Used by the generic CSV loader and by `try_normalize_label` so mental-health data can be
# mapped onto the shared label space without editing the CSVs.
LABEL_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "excited": "joy",
    "excitement": "joy",
    "love": "joy",
    "amused": "joy",
    "positive": "joy",
    "sad": "sadness",
    "depressed": "sadness",
    "depression": "sadness",
    "grief": "sadness",
    "hopeless": "sadness",
    "lonely": "sadness",
    "angry": "anger",
    "annoyed": "anger",
    "annoyance": "anger",
    "frustrated": "anger",
    "frustration": "anger",
    "rage": "anger",
    "fearful": "fear",
    "afraid": "fear",
    "scared": "fear",
    "anxious": "fear",
    "anxiety": "fear",
    "nervous": "fear",
    "worried": "fear",
    "disgusted": "disgust",
    "contempt": "disgust",
    "surprised": "surprise",
    "surprising": "surprise",
    "calm": "neutral",
    "none": "neutral",
    "other": "neutral",
    "others": "neutral",
    "okay": "neutral",
}

DEFAULT_CONFIG = dict(
    model_name="roberta-base",
    num_labels=len(EMOTIONS),
    labels=list(EMOTIONS),
    proj_dim=256,
    dropout=0.1,
    context_turns=2,          # previous utterances used as dialogue context
    context_max_chars=480,    # character budget for the context (keeps the most recent turns)
    max_length=128,
    pooling="cls",            # "cls" | "mean" | "attention"
    temperature=1.0,          # fitted on dev (temperature scaling)
    class_prior=None,         # train label distribution, used for prior correction
    prior_strength=0.0,       # 0 = off, 1 = fully remove the observed prior
    uncertainty_min_confidence=0.45,
    uncertainty_max_entropy=0.80,
    top_k=3,
)

# Config entries that change the architecture (and therefore the state-dict shapes).
_MODEL_KEYS = ("model_name", "num_labels", "proj_dim", "dropout", "pooling")
POOLING_CHOICES = ("cls", "mean", "attention")


def try_normalize_label(label) -> Optional[str]:
    """Map a raw label string onto one of ``EMOTIONS``. Returns None if unknown."""
    if label is None:
        return None
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    key = LABEL_ALIASES.get(key, key)
    return key if key in EMOTIONS else None


def normalize_label(label) -> str:
    normalized = try_normalize_label(label)
    if normalized is None:
        raise ValueError(f"Unknown emotion label {label!r}. Expected one of {EMOTIONS}.")
    return normalized


def format_context(context: Optional[Sequence[str]], k: int,
                   max_chars: Optional[int] = None) -> str:
    """Join the last ``k`` previous utterances into one context string.

    Most recent turns are kept when ``max_chars`` is set (the oldest turns are dropped first).
    The context is separated by newlines, which is closer to how the encoder saw MELD's
    speaker-prefixed turns than the old ``" | "`` separator.
    """
    if not context or k <= 0:
        return ""
    turns = [str(t).strip() for t in context if t is not None and str(t).strip()]
    if not turns:
        return ""
    turns = turns[-k:]
    if max_chars and max_chars > 0:
        kept: list[str] = []
        total = 0
        for turn in reversed(turns):  # add newest first
            extra = len(turn) + (1 if kept else 0)
            if kept and total + extra > max_chars:
                break
            kept.append(turn)
            total += extra
        kept.reverse()
        turns = kept
        # A single turn can still exceed the budget; keep its tail (the most recent words).
        if len(turns) == 1 and len(turns[0]) > max_chars:
            turns = [turns[0][-max_chars:]]
    return "\n".join(turns)


def normalize_items(items) -> list[tuple[str, Optional[Sequence[str]]]]:
    """Accept ``[(text, context), ...]`` or ``[text, ...]``."""
    out = []
    for item in items:
        if isinstance(item, str):
            out.append((item, None))
        else:
            text, context = item
            out.append((str(text), context))
    return out


def encode_items(tokenizer, items, max_length: int = 128, context_turns: int = 2,
                 context_max_chars: Optional[int] = 480):
    """Encode ``(text, context)`` pairs with a budget that preserves recent context.

    The current utterance is never truncated away in favour of context: we first tokenize both
    sides, then keep the *last* context tokens that fit next to the (left-truncated) utterance.

    Works with both ``transformers`` 4.x (which exposes ``build_inputs_with_special_tokens``) and
    5.x, where tokenizers no longer expose that helper; in the latter case the truncated token IDs
    are decoded and re-encoded as a pair.
    """
    items = normalize_items(items)
    texts = [text for text, _ in items]
    contexts = [
        format_context(ctx, context_turns, context_max_chars) for _, ctx in items
    ]

    text_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    ctx_ids = tokenizer(contexts, add_special_tokens=False)["input_ids"]

    budget = max(2, max_length - 4)  # 4 special tokens for a pair (RoBERTa/BERT)
    kept_text_ids, kept_ctx_ids = [], []
    for t_ids, c_ids in zip(text_ids, ctx_ids):
        if len(t_ids) > budget:
            t_ids = t_ids[:budget]
            c_ids = []
        else:
            room = budget - len(t_ids)
            if len(c_ids) > room:
                # Keep the most recent context tokens.
                c_ids = c_ids[-room:] if room > 0 else []
        kept_text_ids.append(t_ids)
        kept_ctx_ids.append(c_ids)

    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        features = []
        any_context = any(kept_ctx_ids)
        for t_ids, c_ids in zip(kept_text_ids, kept_ctx_ids):
            # Within a batch that has context, keep the pair format even for turns without
            # context so the separators stay consistent (matches the transformers 5 path).
            ids = builder(t_ids, c_ids) if (c_ids or any_context) else builder(t_ids)
            features.append({"input_ids": ids})
        return tokenizer.pad(features, padding=True, return_tensors="pt")

    # transformers >= 5: round-trip the already-truncated ids through the tokenizer.
    trimmed_texts = [tokenizer.decode(ids, skip_special_tokens=True)
                     for ids in kept_text_ids]
    trimmed_ctxs = [tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in kept_ctx_ids]
    if not any(trimmed_ctxs):
        return tokenizer(
            trimmed_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
    return tokenizer(
        trimmed_texts, trimmed_ctxs,
        padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )


def calibrate_logits(logits: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Apply saved class-prior correction and temperature scaling to logits.

    Prior correction is ``logits - alpha * log(prior)``; with ``alpha=1`` the model's effective
    prior becomes uniform, which counteracts MELD's heavy ``neutral`` majority.  Temperature
    scaling then sharpens/softens the calibrated distribution.
    """
    prior = cfg.get("class_prior")
    alpha = float(cfg.get("prior_strength", 0.0) or 0.0)
    if prior is not None and alpha > 0:
        p = torch.as_tensor(prior, dtype=logits.dtype, device=logits.device).clamp_min(1e-8)
        p = p / p.sum()
        logits = logits - alpha * torch.log(p).unsqueeze(0)

    temperature = float(cfg.get("temperature", 1.0) or 1.0)
    if temperature > 0 and abs(temperature - 1.0) > 1e-6:
        logits = logits / temperature
    return logits


def sentiment_from_probs(probs: np.ndarray, labels: Sequence[str] = EMOTIONS) -> dict:
    """Derive a coarse sentiment for trend analysis (MELD's ``Sentiment`` column is 3-way)."""
    p = {label: float(x) for label, x in zip(labels, probs)}
    surprise = p.get("surprise", 0.0)
    positive = p.get("joy", 0.0) + 0.5 * surprise
    negative = sum(p.get(e, 0.0) for e in ("sadness", "anger", "fear", "disgust")) + 0.5 * surprise
    neutral = p.get("neutral", 0.0)
    scores = {"positive": positive, "negative": negative, "neutral": neutral}
    label = max(scores, key=scores.get)
    return {"label": label, "scores": scores}


def build_prediction(cfg: dict, probs: np.ndarray,
                     embedding: Optional[np.ndarray] = None) -> dict:
    """Turn one probability vector into the API response (keeps the original keys)."""
    labels = list(cfg.get("labels") or EMOTIONS)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    order = np.argsort(-probs)
    idx = int(order[0])

    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    max_entropy = math.log(max(len(probs), 2))
    normalized_entropy = entropy / max_entropy
    min_conf = float(cfg.get("uncertainty_min_confidence", 0.45))
    max_ent = float(cfg.get("uncertainty_max_entropy", 0.80))
    is_uncertain = bool(probs[idx] < min_conf or normalized_entropy > max_ent)

    k = int(cfg.get("top_k", 3) or 3)
    top_k = [
        {"label": labels[int(i)], "probability": float(probs[int(i)])}
        for i in order[: max(1, min(k, len(labels)))]
    ]

    result = {
        "label": labels[idx],
        "confidence": float(probs[idx]),
        "probabilities": {label: float(x) for label, x in zip(labels, probs)},
        "top_k": top_k,
        "sentiment": sentiment_from_probs(probs, labels),
        "uncertainty": {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "is_uncertain": is_uncertain,
        },
        "is_uncertain": is_uncertain,
        "calibrated": bool(
            abs(float(cfg.get("temperature", 1.0) or 1.0) - 1.0) > 1e-3
            or float(cfg.get("prior_strength", 0.0) or 0.0) > 0
        ),
    }
    if embedding is not None:
        result["embedding"] = np.asarray(embedding, dtype=float).tolist()
    return result


class TextEmotionModel(nn.Module):
    """Transformer encoder -> dropout -> projection -> (dropout) -> classifier."""

    def __init__(self, model_name="roberta-base", num_labels=7, proj_dim=256,
                 dropout=0.1, pooling="cls", pretrained=True):
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"pooling must be one of {POOLING_CHOICES}, got {pooling!r}")
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:  # architecture only; weights come from our own checkpoint
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, proj_dim)
        self.act = nn.GELU()
        self.classifier = nn.Linear(proj_dim, num_labels)
        if pooling == "attention":
            self.attention_pool = nn.Linear(hidden, 1)

    def _pool(self, hidden_state, attention_mask):
        if self.pooling == "cls":
            return hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        if self.pooling == "mean":
            return (hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        scores = self.attention_pool(hidden_state).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_state * weights).sum(1)

    def forward(self, input_ids, attention_mask, **_):
        hidden_state = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = self._pool(hidden_state, attention_mask)
        z = self.act(self.proj(self.dropout(pooled)))
        logits = self.classifier(self.dropout(z))
        return {"logits": logits, "embedding": z}


def save_checkpoint(model: TextEmotionModel, cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def load_checkpoint(path, device=None) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Text checkpoint not found at {path}. Train one with "
            f"`python backend/training/train_text.py`, or unset TEXT_MODEL_PATH to use the "
            f"pretrained Hugging Face fallback."
        )
    try:
        return torch.load(path, map_location=device or "cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only
        return torch.load(path, map_location=device or "cpu")


class TextEmotionPredictor:
    """Loads a checkpoint and runs calibrated inference. Used by the API, explainer and fusion."""

    def __init__(self, checkpoint, device: Optional[str] = None):
        ckpt = load_checkpoint(checkpoint)
        self.cfg = {**DEFAULT_CONFIG, **ckpt.get("config", {})}
        self.cfg["labels"] = list(self.cfg.get("labels") or EMOTIONS)
        self.cfg["num_labels"] = len(self.cfg["labels"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        self.model = TextEmotionModel(
            **{k: self.cfg[k] for k in _MODEL_KEYS}, pretrained=False
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.labels = self.cfg["labels"]

    def _encode(self, items):
        return encode_items(
            self.tokenizer,
            items,
            max_length=self.cfg["max_length"],
            context_turns=self.cfg["context_turns"],
            context_max_chars=self.cfg.get("context_max_chars"),
        ).to(self.device)

    @torch.no_grad()
    def predict_proba(self, items, batch_size: int = 32, return_embedding: bool = False,
                      calibrated: bool = True):
        """items: list of (text, context) or list of text. Returns probs [N, C] (+ embeddings)."""
        items = normalize_items(items)
        labels = self.cfg["labels"]
        if not items:
            probs = np.zeros((0, len(labels)), dtype=float)
            if return_embedding:
                return probs, np.zeros((0, self.cfg["proj_dim"]), dtype=float)
            return probs

        probs, embs = [], []
        for i in range(0, len(items), batch_size):
            enc = self._encode(items[i:i + batch_size])
            out = self.model(**enc)
            logits = out["logits"].float().cpu()
            if calibrated:
                logits = calibrate_logits(logits, self.cfg)
            probs.append(torch.softmax(logits, dim=-1).numpy())
            embs.append(out["embedding"].float().cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        if return_embedding:
            return probs, np.concatenate(embs, axis=0)
        return probs

    def predict(self, text: str, context: Optional[Sequence[str]] = None,
                return_embedding: bool = False) -> dict:
        probs, emb = self.predict_proba([(text, context)], return_embedding=True)
        return build_prediction(self.cfg, probs[0], emb[0] if return_embedding else None)

    def info(self) -> dict:
        return {
            "backend": "finetuned-checkpoint",
            "model_name": self.cfg["model_name"],
            "labels": list(self.labels),
            "device": self.device,
            "pooling": self.cfg.get("pooling", "cls"),
            "max_length": self.cfg.get("max_length"),
            "context_turns": self.cfg.get("context_turns"),
            "calibration": {
                "temperature": float(self.cfg.get("temperature", 1.0) or 1.0),
                "prior_strength": float(self.cfg.get("prior_strength", 0.0) or 0.0),
                "has_class_prior": self.cfg.get("class_prior") is not None,
            },
        }


# --------------------------------------------------------------------------- #
# Zero-training option: use an already fine-tuned emotion model from Hugging Face
# --------------------------------------------------------------------------- #
PRETRAINED_DEFAULT = "j-hartmann/emotion-english-distilroberta-base"


class PretrainedTextPredictor(TextEmotionPredictor):
    """Same interface as TextEmotionPredictor, but needs no training.

    The model was trained on single utterances, so dialogue context is ignored.  The embedding is
    the encoder's final ``<s>`` hidden state (768-d, not 256-d).
    """

    def __init__(self, model_name: str = PRETRAINED_DEFAULT, device: Optional[str] = None,
                 temperature: float = 1.0, max_length: int = 256):
        self.cfg = {
            **DEFAULT_CONFIG,
            "model_name": model_name,
            "labels": list(EMOTIONS),
            "num_labels": len(EMOTIONS),
            "temperature": float(temperature),
            "context_turns": 0,
            "max_length": int(max_length),
        }
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.labels = list(EMOTIONS)

        id2label = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        mapping = {}
        for idx, raw in id2label.items():
            normalized = try_normalize_label(raw)
            if normalized is not None and normalized not in mapping:
                mapping[normalized] = idx
        missing = [e for e in EMOTIONS if e not in mapping]
        if missing:
            raise ValueError(
                f"{model_name} has labels {sorted(id2label.values())}; missing {missing}. "
                "Use a model with MELD's 7 labels or add a label mapping."
            )
        self._order = [mapping[e] for e in EMOTIONS]  # reorder outputs to our EMOTIONS order

    @torch.no_grad()
    def predict_proba(self, items, batch_size: int = 32, return_embedding: bool = False,
                      calibrated: bool = True):
        items = normalize_items(items)
        if not items:
            probs = np.zeros((0, len(EMOTIONS)), dtype=float)
            if return_embedding:
                return probs, np.zeros((0, self.model.config.hidden_size), dtype=float)
            return probs

        probs, embs = [], []
        for i in range(0, len(items), batch_size):
            texts = [t for t, _ in items[i:i + batch_size]]
            enc = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=self.cfg["max_length"], return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            logits = out.logits.float().cpu()[:, self._order]
            if calibrated:
                logits = calibrate_logits(logits, self.cfg)
            probs.append(torch.softmax(logits, dim=-1).numpy())
            embs.append(out.hidden_states[-1][:, 0].float().cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        if return_embedding:
            return probs, np.concatenate(embs, axis=0)
        return probs

    def info(self) -> dict:
        return {
            "backend": "pretrained-huggingface",
            "model_name": self.cfg["model_name"],
            "labels": list(EMOTIONS),
            "device": self.device,
            "context_used": False,
            "embedding_dim": int(self.model.config.hidden_size),
            "max_length": int(self.cfg["max_length"]),
            "calibration": {
                "temperature": float(self.cfg.get("temperature", 1.0) or 1.0),
                "prior_strength": 0.0,
                "has_class_prior": False,
            },
        }
