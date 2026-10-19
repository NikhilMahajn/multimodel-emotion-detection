"""Evaluate the pretrained text model on the MELD test set (no training involved).

Run from anywhere:  python training/evaluate_pretrained.py [--limit 300]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                    if (p / "backend").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from backend.models.text_model import EMOTIONS, PRETRAINED_DEFAULT, PretrainedTextPredictor
from backend.services.meld_text_data import load_meld_text

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--model", default=PRETRAINED_DEFAULT)
ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N utterances")
args = ap.parse_args()

samples = load_meld_text(PROJECT_ROOT / "datasets" / "MELD.Raw", args.split)
if args.limit:
    samples = samples[: args.limit]
texts = [s["text"].split(": ", 1)[1] for s in samples]  # drop the "Speaker: " prefix
labels = [s["label"] for s in samples]

pred = PretrainedTextPredictor(args.model)
preds = []
for i in tqdm(range(0, len(texts), 32), desc="predicting"):
    probs = pred.predict_proba([(t, None) for t in texts[i:i + 32]])
    preds += probs.argmax(-1).tolist()

print(f"\n{args.split} weighted-F1: {f1_score(labels, preds, average='weighted'):.4f}")
print(classification_report(labels, preds, labels=range(len(EMOTIONS)),
                            target_names=EMOTIONS, digits=3, zero_division=0))