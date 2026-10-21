"""Tests for encoding, calibration and response-building helpers in text_model."""
import numpy as np
import pytest
import torch

from backend.models.text_model import (EMOTIONS, build_prediction, calibrate_logits,
                                       encode_items, format_context, sentiment_from_probs,
                                       try_normalize_label)


class FakeTokenizer:
    """Tiny tokenizer implementing the subset used by ``encode_items``."""

    pad_token_id = 99

    def __init__(self):
        self._vocab = {}
        self._reverse = {}

    def _id(self, word):
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab) + 3
            self._reverse[self._vocab[word]] = word
        return self._vocab[word]

    def build_inputs_with_special_tokens(self, ids, pair=None):
        if pair:
            return [0] + list(ids) + [2, 2] + list(pair) + [2]
        return [0] + list(ids) + [2]

    def __call__(self, texts, text_pairs=None, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(text_pairs, str):
            text_pairs = [text_pairs]
        seqs = []
        for i, text in enumerate(texts):
            ids = [self._id(w) for w in text.split()]
            pair = [self._id(w) for w in text_pairs[i].split()] if text_pairs else None
            seqs.append(self.build_inputs_with_special_tokens(ids, pair)
                        if add_special_tokens else ids)
        return {"input_ids": seqs}

    def pad(self, features, padding=True, return_tensors="pt"):
        width = max(len(f["input_ids"]) for f in features)
        ids = [f["input_ids"] + [self.pad_token_id] * (width - len(f["input_ids"]))
               for f in features]
        mask = [[1] * len(f["input_ids"]) + [0] * (width - len(f["input_ids"]))
                for f in features]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        words = [self._reverse.get(int(i), "") for i in ids]
        if skip_special_tokens:
            words = [w for w in words if w]
        return " ".join(words)


# --------------------------------------------------------------------------- #
# Context formatting
# --------------------------------------------------------------------------- #
def test_format_context_keeps_last_turns():
    assert format_context(["one", "two", "three"], 2) == "two\nthree"
    assert format_context(["one"], 0) == ""
    assert format_context(None, 2) == ""
    assert format_context(["  ", "two"], 2) == "two"


def test_format_context_drops_oldest_when_over_budget():
    context = ["a" * 100, "b" * 50]
    assert format_context(context, 2, max_chars=60) == "b" * 50
    # A single over-budget turn keeps its tail.
    assert format_context(["x" * 100], 2, max_chars=20) == "x" * 20


def test_label_aliases():
    assert try_normalize_label("Happy") == "joy"
    assert try_normalize_label("anxious") == "fear"
    assert try_normalize_label("SAD") == "sadness"
    assert try_normalize_label("not-an-emotion") is None
    assert try_normalize_label(None) is None


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def test_encode_items_preserves_current_utterance_and_recent_context():
    tokenizer = FakeTokenizer()
    enc = encode_items(
        tokenizer,
        [("hello world", ["old " + "x " * 100, "recent turn"])],
        max_length=24, context_turns=2, context_max_chars=10_000,
    )
    row = enc["input_ids"][0].tolist()
    words = [tokenizer._reverse.get(i, "") for i in row]
    assert "hello" in words and "world" in words
    assert "recent" in words
    assert "old" not in words  # oldest context tokens were dropped from the left
    assert len(row) <= 24


def test_encode_items_accepts_plain_strings_and_long_text():
    tokenizer = FakeTokenizer()
    enc = encode_items(tokenizer, ["word " * 100], max_length=16)
    assert enc["input_ids"].shape[1] <= 16
    assert enc["attention_mask"].sum().item() <= 16


def test_encode_items_handles_no_context():
    tokenizer = FakeTokenizer()
    enc = encode_items(tokenizer, [("hello", None), ("hi", [])], max_length=16)
    assert enc["input_ids"].shape[0] == 2
    assert (enc["attention_mask"][1] == 1).sum().item() == 3  # <s> hi </s>


# --------------------------------------------------------------------------- #
# Calibration / predictions
# --------------------------------------------------------------------------- #
def test_temperature_scaling_softens_probabilities():
    logits = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    raw = torch.softmax(calibrate_logits(logits, {"temperature": 1.0}), -1)
    soft = torch.softmax(calibrate_logits(logits, {"temperature": 2.0}), -1)
    assert soft.max() < raw.max()


def test_prior_correction_penalises_majority_class():
    logits = torch.tensor([[1.0, 1.0]])
    prior = [0.9, 0.1]
    adjusted = calibrate_logits(logits, {"temperature": 1.0, "class_prior": prior,
                                         "prior_strength": 1.0})
    # Rare class receives a larger boost than the majority class.
    assert adjusted[0, 1] > adjusted[0, 0]
    unadjusted = calibrate_logits(logits, {"temperature": 1.0, "class_prior": prior,
                                           "prior_strength": 0.0})
    assert unadjusted[0, 0] == unadjusted[0, 1]


def test_build_prediction_exposes_topk_and_uncertainty():
    probs = np.array([0.10, 0.60, 0.10, 0.05, 0.05, 0.05, 0.05])
    result = build_prediction({"labels": EMOTIONS, "top_k": 3}, probs,
                              embedding=np.ones(4))
    assert result["label"] == "joy"
    assert result["top_k"][0] == {"label": "joy", "probability": 0.6}
    assert len(result["top_k"]) == 3
    assert result["sentiment"]["label"] == "positive"
    assert result["is_uncertain"] is False
    assert result["embedding"] == [1.0, 1.0, 1.0, 1.0]
    assert set(result["probabilities"]) == set(EMOTIONS)


def test_build_prediction_flags_flat_distribution_as_uncertain():
    probs = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))
    result = build_prediction({"labels": EMOTIONS}, probs)
    assert result["is_uncertain"] is True
    assert result["uncertainty"]["normalized_entropy"] == pytest.approx(1.0)


def test_sentiment_from_probs():
    positive = np.array([0.05, 0.85, 0.02, 0.02, 0.02, 0.02, 0.02])
    negative = np.array([0.05, 0.02, 0.45, 0.30, 0.02, 0.08, 0.08])
    neutral = np.array([0.85, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02])
    assert sentiment_from_probs(positive)["label"] == "positive"
    assert sentiment_from_probs(negative)["label"] == "negative"
    assert sentiment_from_probs(neutral)["label"] == "neutral"
