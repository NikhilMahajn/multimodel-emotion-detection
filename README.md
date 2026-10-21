# Multimodal Emotion Detection for Mental Health

## Text pipeline (current stage)

| Piece | Location |
|---|---|
| Model, encoding, calibration, inference | `backend/models/text_model.py` |
| MELD text loading (context + speaker) | `backend/services/meld_text_data.py` |
| Generic mental-health CSV loading | `backend/services/text_data.py` |
| Mental-health safety signals | `backend/services/mental_health.py` |
| Training (canonical) | `backend/training/train_text.py` |
| Notebook demo | `backend/training/text_training.ipynb` |
| Evaluation | `backend/training/evaluate_pretrained.py` |
| Word-level explanations | `backend/explainability/text_explainer.py` |
| API | `backend/api/text_routes.py` (`POST /text/predict`, `GET /text/model-info`) |
| Tests | `backend/tests/` |

### What changed and why

The original model was a RoBERTa classifier trained on MELD with weighted-F1 selection. The
improvements target the failure modes that matter when the output is used around mental health:

- **Macro-F1 selection and reporting.** MELD is ~47% `neutral`; weighted-F1 hides poor recall on
  `sadness`, `fear` and `disgust`. Training now selects on macro-F1 and reports per-class
  precision/recall, confusion matrices and ECE.
- **Focal loss + square-root-damped class weights + label smoothing** (`train_text.py`), so minority
  emotions get gradient without the model collapsing into them.
- **Recent-context encoding.** Long dialogue contexts now keep their most recent turns rather than
  losing them to right-side tokenizer truncation; the current utterance is always preserved.
- **Pooling options** (`--pooling cls|mean|attention`) for longer, less structured messages.
- **Calibration on dev** (temperature scaling + optional class-prior correction). Calibrated
  probabilities make the API's confidence and uncertainty flag more meaningful.
- **Richer inference output**: top-k emotions, entropy/uncertainty, derived sentiment and the
  256-d fusion embedding, while keeping the original `label`/`confidence`/`probabilities` keys.
- **Domain adaptation**: `--dataset csv` trains on any labelled mental-health text corpus, with
  configurable columns and a label-mapping JSON (`happy`→`joy`, `anxious`→`fear`, ...).
- **Safety layer**: transparent rule-based screening for suicidal ideation, self-harm,
  hopelessness, worthlessness, isolation, acute distress and substance use. It handles negation,
  third-person attribution, historical mentions, screening questions and safe mentions ("suicide
  prevention"), and adds a distress summary and conversation trend.
- **Bug fix**: MELD ships `train/train_sent_emo.csv` but puts `dev_sent_emo.csv` and
  `test_sent_emo.csv` at the top level; the old loader assumed the nested layout for all splits and
  crashed on dev/test.

### Run the API

```bash
pip install -r backend/requirements.txt
# uses experiments/text/best.pt if present, otherwise the pretrained HF fallback
uvicorn backend.main:app --reload
```

```bash
curl -X POST localhost:8000/text/predict -H "Content-Type: application/json" -d '{
  "text": "I feel like a burden and I do not see the point anymore.",
  "context": ["Work has been impossible lately.", "I barely sleep."],
  "explain": true,
  "analyze_context": true
}'
```

The response keeps `label`, `confidence` and `probabilities`, and adds:

- `top_k`, `sentiment`, `uncertainty` / `is_uncertain`, `calibrated`
- `risk`: `level` (`none`/`low`/`moderate`/`high`), `score`, `urgent`, matched `signals` with flags,
  a recommendation, crisis resources and a disclaimer
- `distress_index`: heuristic summary of sadness/fear/anger/disgust
- `context_trend` (when `analyze_context=true`): negative-affect trend over the supplied turns
- `explanation` (when `explain=true`): per-word occlusion importances
- `model` and `disclaimer`

`GET /text/model-info` reports the active backend, labels, device and calibration.

Environment overrides: `TEXT_MODEL_PATH`, `TEXT_PRETRAINED_MODEL`, `TEXT_DEVICE`,
`TEXT_TEMPERATURE` and `TEXT_MAX_LENGTH`. A fine-tuned checkpoint uses its training token limit
(128 by default); the pretrained fallback defaults to 256 tokens. Longer messages are truncated
from the right, so train with a larger `--max-length` for long journal-style text.

### Train

MELD layout: `datasets/MELD.Raw/train/train_sent_emo.csv`,
`datasets/MELD.Raw/dev_sent_emo.csv`, `datasets/MELD.Raw/test_sent_emo.csv`.

```bash
# GPU strongly recommended; CPU works but is slow
python backend/training/train_text.py --dataset meld --epochs 5 --batch-size 16
```

Outputs `experiments/text/best.pt` and `experiments/text/metrics.json`. Useful flags:
`--focal-gamma`, `--class-weighting sqrt|inv|none`, `--pooling`, `--context-turns`,
`--layer-decay`, `--select-metric`, `--patience`, `--no-calibrate`, `--limit-train` (smoke tests).

Adapt to a counselling/support-forum CSV:

```bash
python backend/training/train_text.py --dataset csv \
  --train-csv data/train.csv --dev-csv data/dev.csv --test-csv data/test.csv \
  --text-col message --label-col emotion --context-col previous_turns \
  --label-map data/label_map.json --output experiments/text-mh
```

Columns default to `text` and `label`. A context column can contain a JSON list or
`"turn A || turn B"`; alternatively pass `--dialogue-col` and `--utterance-col` to build context
from previous rows. `--label-map` is a JSON object mapping raw labels to MELD emotions.

### Evaluate

```bash
python backend/training/evaluate_pretrained.py --checkpoint experiments/text/best.pt
python backend/training/evaluate_pretrained.py --split dev --limit 300 --errors 10
python backend/training/evaluate_pretrained.py            # pretrained fallback baseline
```

Reports macro-F1, weighted-F1, accuracy, top-2 accuracy, ECE, per-class metrics and a confusion
matrix; `--errors N` prints misclassified minority-class examples for error analysis.

Reference run (pretrained fallback, full MELD test, CPU): macro-F1 0.358, weighted-F1 0.516,
ECE 0.161, with sadness recall 0.188 and fear recall 0.160. The report is saved at
`experiments/text/pretrained_baseline_test.json`; use it as the baseline when comparing a
fine-tuned checkpoint.

### Tests

```bash
pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -q
```

### Notes for fusion

`TextEmotionPredictor.predict_proba(..., return_embedding=True)` returns the 256-d projected
embedding from a fine-tuned checkpoint. The pretrained fallback returns a 768-d encoder state, so
check `model-info` before wiring fusion. All modality models must share the label order `EMOTIONS`
from `text_model.py`.

### Caveat

MELD is scripted TV dialogue (*Friends*), not clinical data. Predicted emotions and risk signals are
not a diagnosis or a clinical risk assessment. The safety layer is an English-only lexicon with
known false positives/negatives; it must be reviewed and validated by qualified clinicians, and
deployed with a human escalation protocol, before any mental-health use. Validate on data from your
target population and monitor per-class recall rather than accuracy.
