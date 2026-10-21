"""Tests for the rule-based mental-health screening layer."""
from backend.services.mental_health import (analyze_risk, context_trend, distress_index)


def test_clear_statement_is_high_risk():
    result = analyze_risk("I want to die.")
    assert result["level"] == "high"
    assert result["score"] >= 0.7
    assert any(s["category"] == "suicidal_ideation" for s in result["signals"])
    assert result["disclaimer"]


def test_negated_statement_does_not_score():
    result = analyze_risk("I don't want to die, I'm just exhausted.")
    assert result["score"] == 0.0
    assert result["level"] == "none"
    signal = result["signals"][0]
    assert signal["negated"] is True
    assert signal["weight"] == 0.0


def test_uncertain_negation_keeps_some_weight():
    result = analyze_risk("I don't know if I want to die.")
    assert result["level"] == "moderate"
    assert 0.0 < result["score"] < 0.8


def test_third_person_is_reported_but_discounted():
    result = analyze_risk("My friend wants to kill himself and I don't know what to do.")
    assert result["level"] == "low"
    assert result["signals"][0]["third_person"] is True
    assert result["signals"][0]["weight"] < 0.8


def test_historical_mention_is_discounted():
    result = analyze_risk("I used to want to die when I was a teenager.")
    assert result["level"] == "moderate"
    assert result["signals"][0]["historical"] is True


def test_safe_mentions_are_not_scored():
    for text in ("I'm not suicidal.",
                 "I've been reading about suicide prevention.",
                 "I lost someone to suicide last year.",
                 "I could kill myself for forgetting that."):
        result = analyze_risk(text)
        assert result["score"] == 0.0, text
        assert result["level"] == "none", text


def test_plan_or_means_marks_urgent():
    result = analyze_risk("I have a plan to kill myself tonight.")
    assert result["urgent"] is True
    assert result["level"] == "high"

    result = analyze_risk("I overdosed on pills.")
    assert result["level"] == "high"


def test_explicit_intent_is_urgent():
    result = analyze_risk("I am going to kill myself.")
    assert result["urgent"] is True


def test_third_person_risk_is_not_urgent_for_the_speaker():
    result = analyze_risk("My friend is going to kill himself tonight.")
    assert result["signals"][0]["third_person"] is True
    assert result["urgent"] is False


def test_negation_does_not_leak_across_coordinated_clauses():
    result = analyze_risk("I am not happy and I want to die.")
    assert result["level"] == "high"
    assert result["signals"][0]["negated"] is False


def test_screening_question_is_discounted():
    result = analyze_risk("Have you been feeling suicidal?")
    assert result["level"] == "low"
    assert result["signals"][0]["question"] is True
    assert result["urgent"] is False


def test_question_that_contains_a_first_person_disclosure_is_not_discounted():
    result = analyze_risk("Did you know I want to die?")
    assert result["level"] == "high"
    assert result["signals"][0]["question"] is False


def test_passive_death_wish_with_time_is_high_but_not_urgent():
    result = analyze_risk("I don't want to wake up tomorrow.")
    assert result["level"] == "high"
    assert result["urgent"] is False


def test_self_harm_is_not_double_counted_as_ideation():
    result = analyze_risk("I cut myself last night.")
    categories = {s["category"] for s in result["signals"]}
    assert categories == {"self_harm"}
    assert result["level"] == "high"
    assert result["score"] == 0.6


def test_hopelessness_and_isolation_accumulate():
    result = analyze_risk("I feel hopeless and worthless. Nobody would notice if I was gone.")
    assert result["level"] == "high"
    assert len({s["category"] for s in result["signals"]}) == 3


def test_empty_and_neutral_text():
    assert analyze_risk("")["level"] == "none"
    assert analyze_risk("I'm fine, just tired.")["level"] == "none"


def test_distress_index_orders_neutral_and_sad_text():
    sad = distress_index({"sadness": 0.7, "fear": 0.1, "neutral": 0.1, "joy": 0.05,
                          "anger": 0.05, "disgust": 0.0, "surprise": 0.0})
    neutral = distress_index({"neutral": 0.9, "joy": 0.05, "sadness": 0.05})
    assert sad["level"] == "high"
    assert neutral["level"] == "low"
    assert sad["index"] > neutral["index"]


def test_context_trend_detects_worsening():
    trend = context_trend([
        {"neutral": 0.7, "sadness": 0.2},
        {"sadness": 0.6, "neutral": 0.3},
        {"sadness": 0.8, "fear": 0.05},
    ])
    assert trend["trend"] == "worsening"
    assert trend["persistent_negative_affect"] is True


def test_context_trend_handles_no_turns():
    assert context_trend([])["trend"] == "insufficient_data"
