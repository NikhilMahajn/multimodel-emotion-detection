"""Rule-based mental-health safety signals for the text pipeline.

This module is deliberately **not** a diagnostic or risk-assessment instrument.  It is a transparent,
lexicon-based screening layer that flags language a clinician might want to follow up on (suicidal
ideation, self-harm, hopelessness, ...).  It runs next to the emotion classifier because none of
MELD's seven emotion labels capture crisis language: a message can be classified as ``sadness`` or
even ``neutral`` while containing a clear statement of intent.

Design choices
--------------
* **High recall over precision, but transparent.**  Every match is returned with the matched phrase
  and flags (``negated``, ``third_person``, ``historical``) so a human can audit it.
* **Negation / attribution aware.**  "I'm not suicidal" and "my friend wants to die" carry very
  different risk than "I want to die".  Both are reported, but only the latter contributes to the
  score.
* **No diagnosis, no prediction of behaviour.**  Output is a screening signal plus a recommendation
  to check in and connect the person with human support.

The lexicon is English-only and will miss indirect or coded language.  Treat it as a safety net that
must be validated and reviewed by qualified clinicians before any real deployment.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

DISCLAIMER = (
    "This is a research prototype, not a medical device or a clinical risk assessment. "
    "Emotion labels and risk signals are screening hints only and must not be used to diagnose, "
    "triage, or make decisions about a person's care. When in doubt, involve a qualified "
    "professional and follow your organisation's safety protocol."
)

CRISIS_RESOURCES = [
    "If there is immediate danger, contact your local emergency number now.",
    "US: call or text 988 (Suicide & Crisis Lifeline).",
    "UK & Ireland: call Samaritans on 116 123.",
    "Elsewhere: find a local helpline at https://findahelpline.com.",
]

# Base contribution of each category to the 0-1 screening score.
CATEGORY_WEIGHTS = {
    "suicidal_ideation": 0.80,
    "self_harm": 0.60,
    "hopelessness": 0.30,
    "worthlessness": 0.25,
    "isolation": 0.15,
    "acute_distress": 0.25,
    "substance_use": 0.20,
}

# Human-readable category names used in the API response.
CATEGORY_LABELS = {
    "suicidal_ideation": "suicidal ideation",
    "self_harm": "self-harm",
    "hopelessness": "hopelessness",
    "worthlessness": "worthlessness / shame",
    "isolation": "isolation",
    "acute_distress": "acute distress",
    "substance_use": "substance use",
}

# Patterns are matched against lower-cased, whitespace-normalised text.
_PATTERNS: dict[str, list[str]] = {
    "suicidal_ideation": [
        r"\b(?:i\s+am|i'?m|i\s+will|i'?ll)\s+(?:going\s+to|about\s+to|ready\s+to)\s+"
        r"(?:kill\s+myself|end\s+(?:it|my\s+life)|die)\b",
        r"\b(?:kill|killing|killed)\s+(?:myself|himself|herself|themselves|themself)\b",
        r"\b(?:end|ending|ended)\s+(?:my|his|her|their)\s+(?:own\s+)?life\b",
        r"\b(?:end|ending)\s+it\s+all\b",
        r"\b(?:take|taking|took)\s+(?:my|his|her|their)\s+own\s+life\b",
        r"\b(?:want|wants|wanted|wanna|wish|wishes|wished|wishing)\s+to\s+die\b",
        r"\b(?:want|wants|wanted|wanna|wish|wishes|wished|wishing)\s+(?:to\s+be\s+)?dead\b",
        r"\b(?:wish|wished|wishing)\s+(?:i\s+)?(?:wasn'?t|weren'?t|was\s+not|were\s+not)\s+"
        r"(?:here|alive|born)\b",
        r"\bshould\s+(?:just\s+)?die\b",
        r"\bdeserve\s+to\s+die\b",
        r"\b(?:tired|sick)\s+of\s+(?:living|life|being\s+alive)\b",
        r"\b(?:don'?t|do\s+not)\s+want\s+to\s+(?:wake\s+up|exist)\b",
        r"\b(?:want|wants|wanted|wanna|wish|wishes|wished|wishing)\s+to\s+"
        r"(?:kill\s+(?:himself|herself|themselves|themself)|end\s+(?:his|her|their)\s+life)\b",
        r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|didn'?t)\s+want\s+to\s+"
        r"(?:live|be\s+alive|be\s+here\s+anymore)\b",
        r"\b(?:no|without)\s+(?:reason|point)\s+(?:to|in)\s+"
        r"(?:live|living|go\s+on|going\s+on|being\s+alive)\b",
        r"\b(?:not|never)\s+want(?:ing|s)?\s+to\s+live\b",
        r"\bbetter\s+off\s+(?:dead|without\s+(?:me|him|her|them))\b",
        r"\bsuicid(?:e|al)\b",
        r"\b(?:hang|shoot)\s+(?:myself|himself|herself|themselves|themself)\b",
        r"\boverdos(?:e|ing|ed)\b",
        r"\b(?:wrote|writing|written|leave)\s+(?:a\s+)?(?:suicide\s+)?note\b",
        r"\b(?:plan|planning|planned)\s+to\s+"
        r"(?:kill\s+(?:myself|himself|herself|themselves)|end\s+(?:my|his|her|their)\s+life|"
        r"die\s+by\s+suicide)\b",
    ],
    "self_harm": [
        r"\b(?:hurt|hurting|harm|harming|cut|cutting|burn|burning|starve|starving|"
        r"punish|punishing)\s+(?:myself|himself|herself|themselves|themself)\b",
        r"\bself[-\s]?harm(?:ing)?\b",
        r"\b(?:cuts?|scars?)\s+(?:on\s+)?(?:my|his|her|their)\s+"
        r"(?:arms?|wrists?|legs?|body|thighs?)\b",
        r"\brelief\s+(?:from\s+)?(?:cutting|hurting|pain)\b",
    ],
    "hopelessness": [
        r"\bhopeless(?:ness)?\b",
        r"\bno\s+hope\b",
        r"\bnothing\s+(?:will\s+)?(?:ever\s+)?(?:get|gets|change|improve|be)\s+(?:better|different|okay|ok)\b",
        r"\bnever\s+(?:get|getting)\s+better\b",
        r"\bnothing\s+matters\b",
        r"\bnothing\s+(?:feels?|seems?)\s+(?:worthwhile|worth\s+it|like\s+it\s+matters)\b",
        r"\b(?:no|can'?t\s+see\s+(?:a\s+)?)way\s+out\b",
        r"\b(?:don'?t|do\s+not)\s+see\s+(?:the\s+)?point\b",
        r"\bno\s+point\s+(?:in|of)\b",
        r"\b(?:don'?t|do\s+not)\s+care\s+(?:anymore|any\s+more)\b",
        r"\b(?:given|giving)\s+up\b",
        r"\bwhat'?s\s+the\s+point\b",
        r"\bpointless\b",
        r"\b(?:can'?t|cannot)\s+(?:go\s+on|keep\s+going)\b",
    ],
    "worthlessness": [
        r"\bworthless\b",
        r"\buseless\b",
        r"\b(?:a\s+)?burden\s+(?:to|on)\b",
        r"\beveryone\s+would\s+be\s+better\s+off\s+without\s+me\b",
        r"\bhate\s+myself\b",
        r"\b(?:i'?m|i\s+am|feel\s+like\s+)\s*(?:such\s+)?(?:a\s+)?failure\b",
        r"\bnot\s+good\s+enough\b",
        r"\b(?:don'?t|do\s+not)\s+deserve\s+(?:to\s+be\s+happy|love|anything|help)\b",
    ],
    "isolation": [
        r"\b(?:no\s?one|nobody)\s+(?:cares|would\s+care|would\s+notice|would\s+miss\s+me|miss(?:es)?\s+me)\b",
        r"\b(?:have|got)\s+(?:no\s?one|nobody)\s+(?:left|to\s+turn\s+to|who\s+cares)\b",
        r"\b(?:afraid|scared)\s+(?:to\s+be\s+)?alone\b",
        r"\ball\s+alone\b",
        r"\bso\s+(?:alone|lonely)\b",
        r"\bno\s+(?:friends|one\s+to\s+talk\s+to)\b",
        r"\b(?:isolated|isolation)\b",
        r"\bfeel\s+(?:completely\s+|so\s+)?(?:alone|lonely)\b",
    ],
    "acute_distress": [
        r"\b(?:can'?t|cannot)\s+(?:take|handle)\s+(?:it|this)\s+(?:anymore|any\s+more)\b",
        r"\b(?:can'?t|cannot)\s+do\s+this\s+(?:anymore|any\s+more)\b",
        r"\b(?:can'?t|cannot)\s+stop\s+crying\b",
        r"\b(?:falling|fall)\s+apart\b",
        r"\bbreaking\s+down\b",
        r"\bpanic\s+attack\b",
        r"\b(?:can'?t|cannot)\s+breathe\b",
        r"\b(?:completely\s+)?overwhelmed\b",
        r"\blosing\s+my\s+mind\b",
        r"\bunbearable\b",
        r"\bfeel\s+(?:so\s+|completely\s+)?trapped\b",
        r"\b(?:can'?t|cannot)\s+cope\b",
    ],
    "substance_use": [
        r"\b(?:drink|drinking|drunk)\s+(?:too\s+much|every\s+day|to\s+(?:forget|numb))\b",
        r"\b(?:using|use)\s+(?:drugs|pills)\s+(?:to\s+(?:forget|numb|cope)|every\s+day)\b",
        r"\bhigh\s+(?:all\s+the\s+time|every\s+day)\b",
        r"\b(?:addicted|addiction)\b",
        r"\boverdos(?:e|ed|ing)\b",
    ],
}

_COMPILED = {
    category: [re.compile(p) for p in patterns]
    for category, patterns in _PATTERNS.items()
}

# Means / plan / farewell language: urgent even without a time word.
_PLAN_OR_MEANS = re.compile(
    r"\b(?:have\s+(?:a\s+)?plan|made\s+(?:a\s+)?plan|"
    r"(?:bought|got|have)\s+(?:a\s+)?(?:gun|rope|pills|knife|razor)|"
    r"goodbye|final\s+note|won'?t\s+be\s+here|overdos(?:ed|ing)|"
    r"(?:already\s+)?(?:took|swallowed|ingested)\s+(?:the\s+|some\s+|a\s+)?pills)\b"
)

# A time word alone is not enough: it becomes urgent only with active ideation.
_TIME_MARKER = re.compile(r"\b(?:right\s+now|tonight|today|tomorrow|this\s+week)\b")

_ACTIVE_IDEATION = re.compile(
    r"\b(?:kill(?:ing)?\s+(?:myself|himself|herself|themselves)|"
    r"end(?:ing)?\s+(?:it|my\s+life|my\s+own\s+life)|"
    r"want(?:s|ed)?\s+to\s+(?:die|kill)|wanna\s+(?:die|kill)|"
    r"going\s+to\s+(?:kill|end)|suicid(?:e|al)|overdos(?:e|ing|ed)|"
    r"(?:cut|cutting|hurt|hurting|harm|harming)\s+"
    r"(?:myself|himself|herself|themselves))\b"
)

_EXPLICIT_INTENT = re.compile(
    r"\b(?:i\s+am|i'?m|i\s+will|i'?ll)\s+(?:going\s+to|about\s+to|ready\s+to)\s+"
    r"(?:kill\s+myself|end\s+(?:it|my\s+life)|die)\b"
)

_NEGATORS = re.compile(
    r"\b(?:not|never|no\s+longer|no|none|nobody|don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|"
    r"wasn'?t|weren'?t|wouldn'?t|won'?t|cannot|can'?t|haven'?t|hasn'?t|hadn'?t|ain'?t|without)\b"
)
_UNCERTAIN_NEGATION = re.compile(r"\b(?:not\s+sure|don'?t\s+know|unsure|maybe|sometimes)\b")
_HISTORICAL = re.compile(
    r"\b(?:used\s+to|in\s+the\s+past|when\s+i\s+was|years?\s+ago|last\s+year|previously|"
    r"back\s+then|as\s+a\s+kid|childhood)\b"
)

# "Have you had thoughts of suicide?" is a screening question, not a disclosure by the speaker.
_SECOND_PERSON_QUESTION = re.compile(
    r"\b(?:have|has|do|does|did|are|were|is|was|can|could|would|will)\s+"
    r"(?:you|u|he|she|they)\b"
)

# "suicide prevention", "lost someone to suicide", ... are mentions, not ideation.
_SAFE_SUICIDE_MENTION = re.compile(
    r"\b(?:prevention|awareness|hotline|helpline|crisis\s+line|research|study|training|"
    r"policy|died\s+by|lost\s+(?:someone|a\s+\w+|my\s+\w+)\s+to)\b"
)

# "I could kill myself for forgetting" is an idiom, not intent.
_IDIOM_AFTER = re.compile(r"\s+(?:for|laughing)\b")
_MODAL_BEFORE = re.compile(r"\b(?:could|would)\b")

# Used to work out who the clause is about.
_FIRST_PERSON = re.compile(r"\b(?:i|i'?m|im|me|my|myself|i'?ve|i'?d|i'?ll|we|us|our)\b")
_THIRD_PERSON = re.compile(
    r"\b(?:he|him|his|she|her|hers|they|them|their|theirs)\b"
)
_THIRD_PERSON_NP = re.compile(
    r"\b(?:(?:my|a|the)\s+(?:friend|mom|mother|dad|father|brother|sister|partner|"
    r"boyfriend|girlfriend|husband|wife|son|daughter|child|kid|client|patient|"
    r"neighbou?r|coworker|colleague|roommate))\b"
)

_CLAUSE_BOUNDARY = re.compile(r"[.!?;\n]|\b(?:but|however|yet|although|though)\b")

# Negation scope is narrower than attribution scope: a negator in a previous coordinate clause
# ("I'm not happy and I want to die") must not cancel the current clause.
_NEGATION_BOUNDARY = re.compile(
    r"[.!?;,\n]|\b(?:but|however|yet|although|though|and)\b"
)


def _normalize(text: str) -> str:
    text = (text or "")
    for src, dst in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-")):
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _clause_prefix(text: str, start: int, window: int = 60) -> str:
    prefix = text[max(0, start - window):start]
    return _CLAUSE_BOUNDARY.split(prefix)[-1]


def _negation(text: str, start: int) -> tuple[bool, bool]:
    """Return (negated, uncertain_negation) based on the clause preceding the match."""
    prefix = text[max(0, start - 60):start]
    prefix = _NEGATION_BOUNDARY.split(prefix)[-1]
    if not _NEGATORS.search(prefix):
        return False, False
    return True, bool(_UNCERTAIN_NEGATION.search(prefix))


def _attribution(text: str, start: int) -> tuple[bool, bool]:
    """Return (third_person, historical) for a match."""
    prefix = text[max(0, start - 80):start]
    clause = _CLAUSE_BOUNDARY.split(prefix)[-1]

    # The last subject mention before the match wins, so "I'm worried my friend wants to die"
    # is attributed to the friend rather than the speaker.
    last_first = max((m.end() for m in _FIRST_PERSON.finditer(clause)), default=-1)
    last_third = max(
        max((m.end() for m in _THIRD_PERSON.finditer(clause)), default=-1),
        max((m.end() for m in _THIRD_PERSON_NP.finditer(clause)), default=-1),
    )
    third_person = last_third > last_first

    historical = bool(_HISTORICAL.search(_clause_prefix(text, start, window=45)))
    return third_person, historical


def _is_question_about_other(text: str, start: int, end: int) -> bool:
    """True for "Have you ...?" style screening questions, but not "Did you know I want to die?"."""
    prefix = _clause_prefix(text, start, window=60)
    match = _SECOND_PERSON_QUESTION.search(prefix)
    if not match or "?" not in text[end:end + 30]:
        return False
    # If the speaker reappears after the question phrase, the disclosure is about them.
    return not _FIRST_PERSON.search(prefix[match.end():])


def _iter_matches(text: str):
    for category, patterns in _COMPILED.items():
        spans: list[tuple[int, int]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(start < e and end > s for s, e in spans):
                    continue  # overlapping match from another pattern in the same category
                spans.append((start, end))
                negated, uncertain = _negation(text, start)
                third_person, historical = _attribution(text, start)
                question = _is_question_about_other(text, start, end)

                weight = CATEGORY_WEIGHTS[category]
                if negated:
                    weight *= 0.40 if uncertain else 0.0
                if third_person:
                    weight *= 0.25
                if historical:
                    weight *= 0.5
                if question:
                    weight *= 0.30

                phrase = text[start:end]
                safe_mention = False
                if category == "suicidal_ideation" and phrase in ("suicide", "suicidal"):
                    surrounding = text[max(0, start - 50):min(len(text), end + 50)]
                    if _SAFE_SUICIDE_MENTION.search(surrounding):
                        safe_mention = True
                        weight = 0.0
                if (category == "suicidal_ideation" and phrase == "kill myself"
                        and _IDIOM_AFTER.match(text[end:end + 15])
                        and _MODAL_BEFORE.search(text[max(0, start - 15):start])):
                    safe_mention = True
                    weight = 0.0

                yield {
                    "category": category,
                    "category_label": CATEGORY_LABELS[category],
                    "phrase": phrase,
                    "start": start,
                    "end": end,
                    "negated": negated,
                    "third_person": third_person,
                    "historical": historical,
                    "question": question,
                    "safe_mention": safe_mention,
                    "weight": round(weight, 4),
                }


def _level(score: float, signals: list[dict], urgent: bool) -> str:
    if urgent:
        return "high"
    if score <= 0 or not signals:
        return "none"
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "moderate"
    return "high"


def _recommendation(level: str, urgent: bool) -> str:
    if urgent:
        return (
            "Urgent: there may be immediate risk. Stay with the person if it is safe to do so, "
            "encourage them to contact local emergency services or a crisis line now, and help "
            "them reach a trusted person or clinician. Do not leave them alone."
        )
    if level == "high":
        return (
            "High-risk language detected. Gently ask directly about safety and immediate plans, "
            "encourage contact with a crisis line or clinician soon, and involve a trusted person. "
            "Follow your organisation's safeguarding protocol."
        )
    if level == "moderate":
        return (
            "Some concerning language was detected. Check in with the person, ask how they are "
            "coping, and encourage professional or social support. Re-screen if distress escalates."
        )
    if level == "low":
        return (
            "A possible risk phrase was detected, but it appears negated, historical, about "
            "someone else, or part of a question. Continue monitoring and respond supportively."
        )
    return "No crisis-related language detected. Continue routine, supportive monitoring."


def analyze_risk(text: str, context: Optional[Sequence[str]] = None) -> dict:
    """Screen one message for mental-health risk language.

    ``context`` is accepted for interface symmetry with the emotion predictor but risk is scored on
    the current message only, so prior turns are not incorrectly attributed to the speaker.  (The
    API can additionally compute an emotion trend across context turns.)
    """
    del context  # deliberately not scored; see docstring
    normalized = _normalize(text)
    if not normalized:
        return {
            "level": "none",
            "score": 0.0,
            "urgent": False,
            "signals": [],
            "recommendation": _recommendation("none", False),
            "resources": CRISIS_RESOURCES,
            "disclaimer": DISCLAIMER,
        }

    signals = list(_iter_matches(normalized))
    # Keep the strongest non-overlapping match per category.  Overlap matters: "cut myself"
    # would otherwise be counted both as suicidal ideation and self-harm.
    per_category: dict[str, float] = {}
    accepted_spans: list[tuple[int, int]] = []
    for signal in sorted(signals, key=lambda s: (-s["weight"], -(s["end"] - s["start"]))):
        if signal["weight"] <= 0:
            continue
        if any(signal["start"] < end and signal["end"] > start
               for start, end in accepted_spans):
            continue
        accepted_spans.append((signal["start"], signal["end"]))
        per_category[signal["category"]] = max(
            per_category.get(signal["category"], 0.0), signal["weight"]
        )
    score = min(1.0, round(sum(per_category.values()), 4))

    high_stakes = any(
        s["category"] in ("suicidal_ideation", "self_harm")
        and s["weight"] > 0
        and not s["third_person"]
        and not s["question"]
        for s in signals
    )
    urgent = bool(
        high_stakes
        and (
            _PLAN_OR_MEANS.search(normalized)
            or _EXPLICIT_INTENT.search(normalized)
            or (_TIME_MARKER.search(normalized) and _ACTIVE_IDEATION.search(normalized))
        )
    )
    level = _level(score, signals, urgent)
    if urgent:
        score = max(score, 0.9)

    signals.sort(key=lambda s: (-s["weight"], s["category"]))
    public_signals = [
        {key: value for key, value in signal.items() if key not in ("start", "end")}
        for signal in signals
    ]
    return {
        "level": level,
        "score": float(score),
        "urgent": urgent,
        "signals": public_signals,
        "recommendation": _recommendation(level, urgent),
        "resources": CRISIS_RESOURCES,
        "disclaimer": DISCLAIMER,
    }


def distress_index(probabilities: dict) -> dict:
    """Summarise negative affect from the seven emotion probabilities.

    The index is a transparent heuristic (not a clinical score): the summed probability of
    sadness / fear / anger / disgust, slightly discounted when joy is also high.
    """
    p = {str(k).lower(): float(v) for k, v in (probabilities or {}).items()}
    negative = sum(p.get(e, 0.0) for e in ("sadness", "fear", "anger", "disgust"))
    positive = p.get("joy", 0.0)
    index = max(0.0, min(1.0, negative * (1.0 - 0.5 * positive)))

    if index < 0.20:
        level = "low"
    elif index < 0.45:
        level = "moderate"
    else:
        level = "high"

    drivers = sorted(
        ({"emotion": e, "probability": p.get(e, 0.0)}
         for e in ("sadness", "fear", "anger", "disgust")),
        key=lambda d: -d["probability"],
    )
    return {
        "index": round(index, 4),
        "level": level,
        "negative_affect": round(negative, 4),
        "positive_affect": round(positive, 4),
        "drivers": drivers,
        "note": "Heuristic summary of the emotion classifier, not a clinical score.",
    }


def context_trend(probability_history: Sequence[dict]) -> dict:
    """Describe how negative affect changes across turns.

    ``probability_history`` is oldest-first, one ``{emotion: probability}`` dict per turn.  Returns
    a descriptive trend only; a single conversation is far too little data for a clinical claim.
    """
    scores = []
    for probs in probability_history or []:
        p = {str(k).lower(): float(v) for k, v in (probs or {}).items()}
        scores.append(sum(p.get(e, 0.0) for e in ("sadness", "fear", "anger", "disgust")))

    if not scores:
        return {
            "turns": 0,
            "trend": "insufficient_data",
            "persistent_negative_affect": False,
            "note": "No context turns were available.",
        }

    latest = scores[-1]
    mean = sum(scores) / len(scores)
    persistent = len(scores) >= 3 and sum(1 for s in scores[-3:] if s >= 0.50) >= 2
    if len(scores) >= 2:
        n = len(scores)
        x_mean = (n - 1) / 2
        denom = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
        slope = sum((i - x_mean) * (s - mean) for i, s in enumerate(scores)) / denom
        # 0.05 per turn is a deliberately small, descriptive threshold.
        trend = "worsening" if slope > 0.05 else "improving" if slope < -0.05 else "stable"
    else:
        slope = 0.0
        trend = "insufficient_data"

    return {
        "turns": len(scores),
        "negative_affect_per_turn": [round(s, 4) for s in scores],
        "mean": round(mean, 4),
        "max": round(max(scores), 4),
        "latest": round(latest, 4),
        "slope": round(slope, 4),
        "trend": trend,
        "persistent_negative_affect": persistent,
        "note": "Descriptive trend across the supplied turns, not a clinical trajectory.",
    }
