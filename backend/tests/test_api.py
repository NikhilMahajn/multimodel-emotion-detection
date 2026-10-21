"""API contract tests using a fake predictor (no model download/training required)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api import text_routes
from backend.main import app
from backend.models.text_model import EMOTIONS, build_prediction

PROBS = np.array([0.10, 0.10, 0.55, 0.10, 0.05, 0.05, 0.05])


class FakePredictor:
    cfg = {"context_turns": 2, "labels": list(EMOTIONS)}

    def predict(self, text, context=None):
        return build_prediction(self.cfg, PROBS)

    def predict_proba(self, items, batch_size=32, return_embedding=False, calibrated=True):
        return np.tile(PROBS, (len(items), 1))

    def info(self):
        return {"backend": "fake", "model_name": "fake-model", "labels": list(EMOTIONS),
                "temperature": 1.0}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(text_routes, "get_predictor", lambda: FakePredictor())
    monkeypatch.setattr(text_routes, "get_model_info", lambda: FakePredictor().info())
    return TestClient(app)


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_predict_returns_legacy_and_new_fields(client):
    response = client.post("/text/predict", json={"text": "I feel hopeless."})
    assert response.status_code == 200
    body = response.json()
    for key in ("label", "confidence", "probabilities", "top_k", "sentiment",
                "uncertainty", "is_uncertain", "model", "risk", "distress_index",
                "resources", "disclaimer"):
        assert key in body
    assert body["label"] == "sadness"
    assert body["risk"]["level"] == "moderate"
    assert body["model"]["backend"] == "fake"


def test_predict_risk_can_be_disabled(client):
    body = client.post("/text/predict", json={"text": "I want to die.",
                                              "include_risk": False}).json()
    assert "risk" not in body
    assert "distress_index" not in body


def test_predict_flags_high_risk_language(client):
    body = client.post("/text/predict", json={"text": "I want to die."}).json()
    assert body["risk"]["level"] == "high"
    assert body["risk"]["disclaimer"]


def test_predict_context_trend(client):
    body = client.post("/text/predict", json={
        "text": "I still feel awful.",
        "context": ["I am sad.", "It got worse."],
        "analyze_context": True,
    }).json()
    assert body["context_trend"]["turns"] == 3
    assert body["context_trend"]["trend"] in ("stable", "worsening", "improving")


def test_predict_explanation(client):
    body = client.post("/text/predict", json={
        "text": "I feel very sad", "explain": True, "target": "sadness",
    }).json()
    assert body["explanation"]["target"] == "sadness"
    assert len(body["explanation"]["tokens"]) == 4
    assert "top_tokens" in body["explanation"]


def test_predict_rejects_bad_input(client):
    assert client.post("/text/predict", json={"text": "   "}).status_code == 400
    assert client.post("/text/predict",
                       json={"text": "hi", "target": "bogus",
                             "explain": True}).status_code == 400


def test_predict_ignores_target_when_explanation_is_off(client):
    # Swagger UI sends the placeholder "string" for optional fields by default.
    response = client.post("/text/predict", json={
        "text": "hii", "context": ["string"], "explain": False,
        "target": "string", "include_risk": True, "analyze_context": False,
    })
    assert response.status_code == 200
    assert "explanation" not in response.json()


def test_model_info(client):
    body = client.get("/text/model-info").json()
    assert body["backend"] == "fake"
    assert body["labels"] == EMOTIONS
    assert "disclaimer" in body
