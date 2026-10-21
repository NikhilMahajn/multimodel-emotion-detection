from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.explainability.text_explainer import TextExplainer
from backend.models.text_model import EMOTIONS
from backend.services.mental_health import (CRISIS_RESOURCES, DISCLAIMER, analyze_risk,
                                            context_trend, distress_index)
from backend.services.text_service import get_model_info, get_predictor

router = APIRouter(prefix="/text", tags=["text"])

MAX_CONTEXT_TURNS = 20


class TextRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "I feel like a burden and I do not see the point anymore.",
                    "context": ["Work has been impossible lately.", "I barely sleep."],
                    "explain": False,
                    "target": None,
                    "include_risk": True,
                    "analyze_context": True,
                }
            ]
        }
    )

    text: str = Field(
        ..., min_length=1, max_length=4000,
        examples=["I feel hopeless and alone."],
    )
    context: list[str] = Field(
        default_factory=list, max_length=50,
        description="Previous utterances, oldest first (only the last 20 are used)",
        examples=[["Work has been impossible lately.", "I barely sleep."]],
    )
    explain: bool = Field(False, description="Return word-level occlusion explanations")
    target: Optional[str] = Field(
        None,
        description=(
            "Emotion to explain. Only used when explain=true; defaults to the predicted emotion. "
            f"Must be one of {EMOTIONS}."
        ),
        examples=["sadness"],
    )
    include_risk: bool = Field(
        True, description="Include rule-based mental-health safety signals and distress summary"
    )
    analyze_context: bool = Field(
        False, description="Also predict emotions for context turns to describe a trend"
    )


@router.post("/predict")
def predict_text(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must contain non-whitespace characters")

    context = [c.strip() for c in req.context if c and c.strip()][-MAX_CONTEXT_TURNS:]
    target = req.target.lower().strip() if req.target else None
    if req.explain:
        if target and target not in EMOTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown target {req.target!r}. Expected one of {EMOTIONS}.",
            )
    else:
        # `target` only affects the explanation; ignore it otherwise (Swagger sends a
        # placeholder "string" by default, which should not fail an otherwise valid request).
        target = None

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (OSError, ValueError) as exc:  # model download / config problems
        raise HTTPException(status_code=503, detail=f"Text model unavailable: {exc}")

    result = predictor.predict(text, context)
    info = predictor.info()
    result["model"] = {"backend": info.get("backend"), "model_name": info.get("model_name")}

    if req.include_risk:
        result["risk"] = analyze_risk(text, context)
        result["distress_index"] = distress_index(result["probabilities"])
        result["resources"] = CRISIS_RESOURCES

    if req.analyze_context and context:
        past = predictor.predict_proba([(turn, None) for turn in context[-8:]])
        history = [dict(zip(EMOTIONS, row)) for row in past]
        history.append(result["probabilities"])
        result["context_trend"] = context_trend(history)

    if req.explain:
        try:
            result["explanation"] = TextExplainer(predictor).explain(text, context, target)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    result["disclaimer"] = DISCLAIMER
    return result


@router.get("/model-info")
def text_model_info():
    try:
        info = get_model_info()
    except (OSError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=f"Text model unavailable: {exc}")
    return {**info, "disclaimer": DISCLAIMER}
