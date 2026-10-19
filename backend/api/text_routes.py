from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.explainability.text_explainer import TextExplainer
from backend.services.text_service import get_predictor

router = APIRouter(prefix="/text", tags=["text"])


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    context: list[str] = Field(default_factory=list, description="Previous utterances, oldest first")
    explain: bool = False
    target: Optional[str] = Field(None, description="Emotion to explain (defaults to predicted)")


@router.post("/predict")
def predict_text(req: TextRequest):
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = predictor.predict(req.text, req.context)
    if req.explain:
        result["explanation"] = TextExplainer(predictor).explain(req.text, req.context, req.target)
    return result