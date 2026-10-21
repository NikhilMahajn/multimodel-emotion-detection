"""Evaluate a text emotion model on MELD (no training involved).

Works with either the fine-tuned checkpoint produced by ``train_text.py`` or the pretrained
Hugging Face fallback.  Reports macro-F1 as well as weighted-F1: macro-F1 treats sadness, fear and
disgust as equally important to neutral, which is the relevant view for mental-health screening.

Examples::

    python backend/training/evaluate_pretrained.py                      # pretrained, MELD test
    python backend/training/evaluate_pretrained.py --checkpoint experiments/text/best.pt
    python backend/training/evaluate_pretrained.py --split dev --limit 300 --errors 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                    if (p / "backend").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from tqdm import tqdm

from backend.models.text_model import (EMOTIONS, PRETRAINED_DEFAULT,
                                       PretrainedTextPredictor, TextEmotionPredictor)
from backend.services.meld_text_data import load_meld_text


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = ((confidence >= low) & (confidence <= high)) if i == n_bins - 1 \
            else ((confidence >= low) & (confidence < high))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def evaluate_probs(probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred, labels=list(range(len(EMOTIONS))), zero_division=0
    )
    top2 = np.argsort(-probs, axis=1)[:, :2]
    return {
        "macro_f1": round(float(f1_score(labels, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(labels, pred, average="weighted", zero_division=0)), 4),
        "accuracy": round(float((pred == labels).mean()), 4),
        "top2_accuracy": round(float(np.mean([y in row for y, row in zip(labels, top2)])), 4),
        "ece": round(expected_calibration_error(probs, labels), 4),
        "per_class": {
            emotion: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, emotion in enumerate(EMOTIONS)
        },
        "confusion_matrix": confusion_matrix(
            labels, pred, labels=list(range(len(EMOTIONS)))
        ).tolist(),
        "predictions": pred.tolist(),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=("train", "dev", "test"))
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "datasets" / "MELD.Raw"))
    ap.add_argument("--checkpoint", help="fine-tuned checkpoint (defaults to pretrained HF model)")
    ap.add_argument("--model", default=PRETRAINED_DEFAULT,
                    help="Hugging Face model used when --checkpoint is not given")
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N utterances")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--report", help="write the full metrics JSON here")
    ap.add_argument("--errors", type=int, default=0,
                    help="print N misclassified examples (focus on sadness/fear/anger)")
    args = ap.parse_args(argv)

    samples = load_meld_text(args.data_root, args.split, include_speaker=True)
    if args.limit:
        samples = samples[: args.limit]

    if args.checkpoint:
        predictor = TextEmotionPredictor(args.checkpoint, device=args.device)
        # The fine-tuned model was trained with speaker-prefixed text and dialogue context.
        items = [(s["text"], s["context"]) for s in samples]
        display_texts = [s["text"] for s in samples]
    else:
        predictor = PretrainedTextPredictor(args.model, device=args.device)
        # The pretrained model was trained on single utterances; drop the speaker prefix/context.
        display_texts = [s["text"].split(": ", 1)[-1] for s in samples]
        items = [(t, None) for t in display_texts]

    labels = np.asarray([s["label"] for s in samples], dtype=int)
    probs = []
    for i in tqdm(range(0, len(items), args.batch_size), desc="predicting"):
        probs.append(predictor.predict_proba(items[i:i + args.batch_size]))
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(EMOTIONS)))

    metrics = evaluate_probs(probs, labels)
    pred = np.asarray(metrics.pop("predictions"))
    print(f"\n{args.split}  samples={len(samples)}  "
          f"macro-F1={metrics['macro_f1']:.4f}  weighted-F1={metrics['weighted_f1']:.4f}  "
          f"accuracy={metrics['accuracy']:.4f}  ECE={metrics['ece']:.4f}")
    print(classification_report(labels, pred, labels=list(range(len(EMOTIONS))),
                                target_names=EMOTIONS, digits=3, zero_division=0))
    print("confusion matrix (rows=gold, cols=pred):")
    print("            " + " ".join(f"{e[:7]:>7}" for e in EMOTIONS))
    for emotion, row in zip(EMOTIONS, metrics["confusion_matrix"]):
        print(f"{emotion[:11]:>11} " + " ".join(f"{v:7d}" for v in row))

    if args.errors:
        # Error analysis matters most for the minority, mental-health-relevant classes.
        focus = {"sadness", "fear", "anger", "disgust"}
        shown = 0
        for text, gold, guess, confidence in zip(
            display_texts, labels, pred, probs.max(axis=1)
        ):
            if gold != guess and EMOTIONS[gold] in focus:
                print(f"\n[gold={EMOTIONS[gold]}  pred={EMOTIONS[guess]}  p={confidence:.2f}] {text}")
                shown += 1
                if shown >= args.errors:
                    break
        if shown == 0:
            print("\nNo misclassified minority-class examples to show.")

    report = {
        "split": args.split,
        "n": len(samples),
        "backend": predictor.info(),
        "metrics": metrics,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.report}")
    return report


if __name__ == "__main__":
    main()
