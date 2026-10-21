"""Train the text emotion model with mental-health-oriented upgrades.

This is the canonical trainer (the notebook is kept as a quick-start demo).  Compared with the
original notebook it adds:

* **Macro-F1 model selection** and reporting.  MELD is ~47% neutral, so weighted-F1 hides failures
  on sadness/fear/disgust -- the classes that matter most for mental-health screening.
* **Focal loss** (``--focal-gamma``) with square-root-damped class weights and label smoothing,
  which gives minority emotions more gradient without collapsing to them.
* **Layer-wise learning-rate decay** for the encoder plus a higher LR for the new head.
* **Recent-context encoding** (long contexts keep their last turns; see
  :func:`backend.models.text_model.encode_items`).
* **Probability calibration** on the dev set (temperature scaling + optional class-prior
  correction) so downstream thresholds and the API's uncertainty flag are meaningful.
* **Early stopping** and per-class metrics (recall on sadness/fear/anger, confusion matrix, ECE).
* **Domain adaptation**: train on MELD or on any CSV of mental-health text
  (``--dataset csv``), with configurable columns and label mapping.
* Fixed MELD dev/test CSV path resolution.

Examples
--------
Fine-tune on MELD (CPU is slow; a GPU is strongly recommended)::

    python backend/training/train_text.py --dataset meld --epochs 5 --batch-size 16

Adapt a checkpoint to a counselling CSV where columns are ``message``/``emotion``::

    python backend/training/train_text.py --dataset csv \
        --train-csv data/train.csv --dev-csv data/dev.csv --test-csv data/test.csv \
        --text-col message --label-col emotion --output experiments/text-mh

Evaluate after training::

    python backend/training/evaluate_pretrained.py --checkpoint experiments/text/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (p / "backend").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from backend.models.text_model import (DEFAULT_CONFIG, EMOTIONS, TextEmotionModel,
                                       calibrate_logits, save_checkpoint)
from backend.services.meld_text_data import (MELDTextDataset, load_meld_text,
                                             make_collate)
from backend.services.text_data import dataset_stats, load_emotion_csv


# --------------------------------------------------------------------------- #
# Losses / metrics
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Weighted cross-entropy down-weighted for easy examples.

    ``gamma=0`` and ``label_smoothing=0`` reduces to ordinary (optionally weighted) CE.
    """

    def __init__(self, gamma: float = 1.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce = F.cross_entropy(
            logits, target, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        loss = (1.0 - pt).pow(self.gamma) * ce
        return loss.mean()


def class_weights_from_labels(labels, mode: str = "sqrt") -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(EMOTIONS)).astype(float)
    counts = np.maximum(counts, 1.0)
    if mode == "none":
        weights = np.ones_like(counts)
    elif mode == "inv":
        weights = counts.sum() / (len(counts) * counts)
    else:
        weights = np.sqrt(counts.sum() / (len(counts) * counts))
    weights = weights / weights.mean()  # keep the effective LR scale stable
    return torch.tensor(weights, dtype=torch.float)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if i == n_bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, cfg: dict) -> dict:
    calibrated_logits = calibrate_logits(logits.float(), cfg)
    probs = torch.softmax(calibrated_logits, dim=-1).numpy()
    y = labels.numpy()
    pred = probs.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(EMOTIONS))), zero_division=0
    )
    per_class = {
        emotion: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, emotion in enumerate(EMOTIONS)
    }
    return {
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "ece": round(expected_calibration_error(probs, y), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(EMOTIONS)))).tolist(),
        "calibration": {
            "temperature": float(cfg.get("temperature", 1.0) or 1.0),
            "prior_strength": float(cfg.get("prior_strength", 0.0) or 0.0),
        },
    }


# --------------------------------------------------------------------------- #
# Optimisation helpers
# --------------------------------------------------------------------------- #
def build_optimizer(model: TextEmotionModel, lr: float, head_lr_mult: float,
                    weight_decay: float, layer_decay: float) -> torch.optim.AdamW:
    """Head with a higher LR; encoder layers decayed from top to bottom."""
    n_layers = int(model.encoder.config.num_hidden_layers)
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}

    def add(param: nn.Parameter, param_lr: float) -> None:
        decay = 0.0 if param.ndim <= 1 else weight_decay
        groups.setdefault((round(param_lr, 10), decay), []).append(param)

    head_modules = [model.proj, model.classifier]
    if hasattr(model, "attention_pool"):
        head_modules.append(model.attention_pool)
    for module in head_modules:
        for param in module.parameters():
            add(param, lr * head_lr_mult)

    for name, param in model.encoder.named_parameters():
        if name.startswith("embeddings."):
            layer = 0
        elif name.startswith("encoder.layer."):
            layer = int(name.split(".")[2]) + 1
        else:  # pooler, final layernorm, ...
            layer = n_layers
        add(param, lr * (layer_decay ** (n_layers - layer)))

    return torch.optim.AdamW(
        [{"params": params, "lr": param_lr, "weight_decay": decay}
         for (param_lr, decay), params in groups.items()]
    )


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = dict(batch)
        y = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits.append(out["logits"].float().cpu())
        labels.append(y.cpu())
    if not logits:
        return torch.zeros((0, len(EMOTIONS))), torch.zeros((0,), dtype=torch.long)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    max_iter: int = 100) -> float:
    """Temperature scaling: fit a single scalar on dev logits (Guo et al., 2017)."""
    if len(labels) == 0:
        return 1.0
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp(min=0.05), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(0.05, 10.0).item())


def tune_prior_strength(logits: torch.Tensor, labels: torch.Tensor, prior,
                        temperature: float) -> float:
    """Pick the prior-correction strength that maximises dev macro-F1."""
    if len(labels) == 0 or prior is None:
        return 0.0
    y = labels.numpy()
    best_strength, best_score = 0.0, -1.0
    for strength in np.linspace(0.0, 1.0, 11):
        cfg = {"temperature": temperature, "class_prior": prior,
               "prior_strength": float(strength)}
        pred = calibrate_logits(logits, cfg).argmax(-1).numpy()
        score = f1_score(y, pred, average="macro", zero_division=0)
        if score > best_score:
            best_strength, best_score = float(strength), score
    return best_strength


# --------------------------------------------------------------------------- #
# Data / CLI
# --------------------------------------------------------------------------- #
def load_datasets(args):
    if args.dataset == "meld":
        include_speaker = args.include_speaker if args.include_speaker is not None else True
        data = {
            split: load_meld_text(args.data_root, split, include_speaker=include_speaker)
            for split in ("train", "dev", "test")
        }
    else:
        if not args.train_csv or not args.dev_csv:
            raise SystemExit("--dataset csv requires --train-csv and --dev-csv")
        label_map = None
        if args.label_map:
            label_map = {str(k).strip().lower(): v for k, v in
                         json.loads(Path(args.label_map).read_text()).items()}
        kwargs = dict(
            text_col=args.text_col, label_col=args.label_col,
            context_col=args.context_col, speaker_col=args.speaker_col,
            dialogue_col=args.dialogue_col, utterance_col=args.utterance_col,
            label_map=label_map,
            include_speaker=bool(args.include_speaker),
        )
        data = {
            "train": load_emotion_csv(args.train_csv, **kwargs),
            "dev": load_emotion_csv(args.dev_csv, **kwargs),
        }
        if args.test_csv:
            data["test"] = load_emotion_csv(args.test_csv, **kwargs)
        else:
            data["test"] = data["dev"]

    for split, limit in (("train", args.limit_train), ("dev", args.limit_dev),
                         ("test", args.limit_test)):
        if limit and split in data:
            data[split] = data[split][:limit]
    return data


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("meld", "csv"), default="meld")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "datasets" / "MELD.Raw"))
    ap.add_argument("--train-csv")
    ap.add_argument("--dev-csv")
    ap.add_argument("--test-csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--context-col")
    ap.add_argument("--speaker-col")
    ap.add_argument("--dialogue-col")
    ap.add_argument("--utterance-col")
    ap.add_argument("--label-map", help="JSON mapping raw labels -> MELD emotions")
    ap.add_argument("--include-speaker", action=argparse.BooleanOptionalAction, default=None)

    ap.add_argument("--output", default=str(PROJECT_ROOT / "experiments" / "text"))
    ap.add_argument("--model", default=DEFAULT_CONFIG["model_name"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr-mult", type=float, default=5.0)
    ap.add_argument("--layer-decay", type=float, default=0.95)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--context-turns", type=int, default=2)
    ap.add_argument("--context-max-chars", type=int, default=480)
    ap.add_argument("--pooling", choices=("cls", "mean", "attention"), default="cls")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--focal-gamma", type=float, default=1.0,
                    help="0 = plain cross-entropy")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--class-weighting", choices=("sqrt", "inv", "none"), default="sqrt")
    ap.add_argument("--balanced-sampler", action="store_true",
                    help="sample batches proportional to inverse class frequency")
    ap.add_argument("--select-metric", choices=("macro_f1", "weighted_f1"),
                    default="macro_f1")
    ap.add_argument("--patience", type=int, default=2,
                    help="stop after N epochs without dev improvement (0 = disable)")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit-train", type=int)
    ap.add_argument("--limit-dev", type=int)
    ap.add_argument("--limit-test", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cpu / cuda / auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    use_amp = bool(args.amp and device.startswith("cuda"))
    print(f"device={device}  amp={use_amp}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"

    data = load_datasets(args)
    stats = {split: dataset_stats(samples) for split, samples in data.items()}
    for split, summary in stats.items():
        print(f"{split}: {summary['samples']} samples, labels={summary['label_counts']}")

    cfg = {
        **DEFAULT_CONFIG,
        "model_name": args.model,
        "num_labels": len(EMOTIONS),
        "labels": list(EMOTIONS),
        "proj_dim": args.proj_dim,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "context_turns": args.context_turns,
        "context_max_chars": args.context_max_chars,
        "temperature": 1.0,
        "class_prior": None,
        "prior_strength": 0.0,
    }

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    collate = make_collate(tokenizer, cfg["max_length"], cfg["context_turns"],
                           cfg["context_max_chars"])
    datasets = {split: MELDTextDataset(samples) for split, samples in data.items()}

    train_labels = datasets["train"].labels
    sampler = None
    if args.balanced_sampler:
        counts = np.bincount(train_labels, minlength=len(EMOTIONS)).astype(float)
        sample_weights = 1.0 / np.maximum(counts, 1.0)[np.asarray(train_labels)]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels), replacement=True,
        )

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size,
            sampler=sampler, shuffle=sampler is None,
            collate_fn=collate, num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
        ),
        "dev": DataLoader(
            datasets["dev"], batch_size=max(32, args.batch_size * 2),
            collate_fn=collate, num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
        ),
        "test": DataLoader(
            datasets["test"], batch_size=max(32, args.batch_size * 2),
            collate_fn=collate, num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
        ),
    }

    model = TextEmotionModel(
        cfg["model_name"], cfg["num_labels"], cfg["proj_dim"], cfg["dropout"],
        pooling=cfg["pooling"],
    ).to(device)

    weights = class_weights_from_labels(train_labels, args.class_weighting).to(device)
    print("class weights:", {e: round(float(w), 3) for e, w in zip(EMOTIONS, weights)})
    loss_fn = FocalLoss(args.focal_gamma, weight=weights,
                        label_smoothing=args.label_smoothing)

    optimizer = build_optimizer(
        model, args.lr, args.head_lr_mult, args.weight_decay, args.layer_decay
    )
    steps_per_epoch = math.ceil(len(loaders["train"]) / max(1, args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup * total_steps), total_steps
    )
    scaler = _make_scaler(use_amp)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp else nullcontext())

    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress):
            batch = dict(batch)
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            with autocast:
                logits = model(**inputs)["logits"]
                loss = loss_fn(logits.float(), labels) / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.detach()) * args.grad_accum

            should_step = ((step + 1) % args.grad_accum == 0
                           or step + 1 == len(loaders["train"]))
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            progress.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        dev_logits, dev_labels = collect_logits(model, loaders["dev"], device)
        dev_metrics = compute_metrics(dev_logits, dev_labels, cfg)
        score = dev_metrics[args.select_metric]
        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, len(loaders["train"])), 4),
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "dev_accuracy": dev_metrics["accuracy"],
        })
        print(history[-1])

        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(model, cfg, checkpoint_path)
            print(f"  saved best checkpoint ({args.select_metric}={score:.4f})")
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs (best epoch {best_epoch})")
                break

    # ------------------------------------------------------------------ #
    # Calibrate on dev, then evaluate the untouched test split.
    # ------------------------------------------------------------------ #
    state = torch.load(checkpoint_path, map_location=device)["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    calibration = {"temperature": 1.0, "prior_strength": 0.0, "class_prior": None}
    if args.calibrate:
        dev_logits, dev_labels = collect_logits(model, loaders["dev"], device)
        temperature = fit_temperature(dev_logits, dev_labels)
        counts = np.bincount(train_labels, minlength=len(EMOTIONS)).astype(float)
        prior = ((counts + 1.0) / (counts.sum() + len(EMOTIONS))).tolist()
        strength = tune_prior_strength(dev_logits, dev_labels, prior, temperature)
        calibration = {"temperature": temperature, "prior_strength": strength,
                       "class_prior": prior}
        cfg.update(calibration)
        save_checkpoint(model, cfg, checkpoint_path)
        print("calibration:", {"temperature": round(temperature, 4),
                               "prior_strength": strength})

    dev_metrics = compute_metrics(*collect_logits(model, loaders["dev"], device),
                                  cfg=cfg)
    test_logits, test_labels = collect_logits(model, loaders["test"], device)
    uncalibrated_cfg = {**cfg, "temperature": 1.0, "prior_strength": 0.0}
    test_uncalibrated = compute_metrics(test_logits, test_labels, uncalibrated_cfg)
    test_metrics = compute_metrics(test_logits, test_labels, cfg)

    pred = torch.softmax(calibrate_logits(test_logits, cfg), dim=-1).argmax(-1).numpy()
    print(f"\nTest macro-F1: {test_metrics['macro_f1']:.4f}   "
          f"weighted-F1: {test_metrics['weighted_f1']:.4f}   "
          f"accuracy: {test_metrics['accuracy']:.4f}   ECE: {test_metrics['ece']:.4f}")
    print(classification_report(
        test_labels.numpy(), pred, labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS, digits=3, zero_division=0,
    ))

    metrics = {
        "config": cfg,
        "data": stats,
        "history": history,
        "best_epoch": best_epoch,
        "dev": dev_metrics,
        "test": test_metrics,
        "test_uncalibrated": test_uncalibrated,
        "calibration": calibration,
        "environment": {"device": device, "amp": use_amp, "seed": args.seed},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {checkpoint_path} and {out_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
