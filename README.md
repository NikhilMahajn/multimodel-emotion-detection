# Multimodal Emotion Detection for Mental Health

## Text pipeline (current stage)

| Piece | Location |
|---|---|
| Model + inference wrapper | `backend/models/text_model.py` |
| MELD text loading | `backend/services/meld_text_data.py` |
| Training notebook | `training/text_training.ipynb` |
| Word-level explanations | `backend/explainability/text_explainer.py` |
| API endpoint | `backend/api/text_routes.py` (`POST /text/predict`) |

### Run
```bash
pip install -r backend/requirements.txt
# 1. put MELD at datasets/MELD.Raw/{train,dev,test}/<split>_sent_emo.csv
# 2. run training/text_training.ipynb  -> experiments/text/best.pt
# 3. start the API from the project root
uvicorn backend.main:app --reload
```

```bash
curl -X POST localhost:8000/text/predict -H "Content-Type: application/json" \
  -d '{"text": "Joey: I cannot believe you asked me that!", "context": ["Ross: Did you eat my sandwich?"], "explain": true}'
```

### Notes for fusion
`TextEmotionPredictor.predict_proba(..., return_embedding=True)` returns the 256-d projected embedding.
All modality models must share the label order `EMOTIONS` from `text_model.py`.

### Caveat
MELD is scripted TV dialogue (*Friends*), not clinical data. Predicted emotions are not a diagnosis; treat
the system as a research prototype and validate on data from your target population before any mental-health use.
# multimodel-emotion-detection
