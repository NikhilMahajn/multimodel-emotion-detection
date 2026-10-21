"""Data-loading tests, including the MELD dev/test path regression and generic CSV loading."""
from pathlib import Path

from backend.models.text_model import EMOTIONS
from backend.services.meld_text_data import load_meld_text
from backend.services.text_data import dataset_stats, load_emotion_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MELD_ROOT = PROJECT_ROOT / "datasets" / "MELD.Raw"


def test_meld_splits_load_from_mixed_layout():
    """MELD ships train/ nested but dev/test CSVs at the top level."""
    expected = {"train": 9989, "dev": 1109, "test": 2610}
    for split, count in expected.items():
        samples = load_meld_text(MELD_ROOT, split)
        assert len(samples) == count, split
        assert samples[0]["context"] == []
        assert len(samples[1]["context"]) == 1
        assert all(0 <= s["label"] < len(EMOTIONS) for s in samples)
        assert samples[0]["text"].count(":") >= 1  # speaker prefix


def test_meld_speaker_prefix_can_be_disabled():
    samples = load_meld_text(MELD_ROOT, "dev", include_speaker=False)
    assert ": " not in samples[0]["text"]


def test_generic_csv_context_and_label_aliases(tmp_path):
    path = tmp_path / "mh.csv"
    path.write_text(
        "message,emotion,prev,speaker\n"
        "I am happy,happy,,Alex\n"
        "so anxious,anxious,\"[\"\"I am happy\"\"]\",Alex\n"
        "I am bored,bored,,Alex\n",
        encoding="utf-8",
    )
    samples = load_emotion_csv(
        path, text_col="message", label_col="emotion",
        context_col="prev", speaker_col="speaker", include_speaker=True,
    )
    assert [s["label"] for s in samples] == [EMOTIONS.index("joy"), EMOTIONS.index("fear")]
    assert samples[0]["text"] == "Alex: I am happy"
    # Context-column values are already-formatted turns; include speakers there if wanted.
    assert samples[1]["context"] == ["I am happy"]

    stats = dataset_stats(samples)
    assert stats["samples"] == 2
    assert stats["with_context"] == 1
    assert stats["label_counts"] == {"joy": 1, "fear": 1}


def test_generic_csv_builds_context_from_dialogue_ids(tmp_path):
    path = tmp_path / "dialogue.csv"
    rows = [
        {"dialogue_id": 1, "utterance_id": 0, "speaker": "A", "text": "Hello", "label": "neutral"},
        {"dialogue_id": 1, "utterance_id": 1, "speaker": "B", "text": "I feel awful", "label": "sad"},
        {"dialogue_id": 2, "utterance_id": 0, "speaker": "A", "text": "New chat", "label": "neutral"},
    ]
    path.write_text(
        "dialogue_id,utterance_id,speaker,text,label\n"
        + "\n".join(f"{r['dialogue_id']},{r['utterance_id']},{r['speaker']},{r['text']},{r['label']}"
                    for r in rows),
        encoding="utf-8",
    )
    samples = load_emotion_csv(
        path, text_col="text", label_col="label", speaker_col="speaker",
        dialogue_col="dialogue_id", utterance_col="utterance_id", include_speaker=True,
    )
    assert [s["text"] for s in samples] == ["A: Hello", "B: I feel awful", "A: New chat"]
    assert samples[1]["context"] == ["A: Hello"]
    assert samples[2]["context"] == []  # new dialogue resets the context


def test_generic_csv_label_map(tmp_path):
    path = tmp_path / "mapped.csv"
    path.write_text("text,label\nhello,LABEL_1\nbye,LABEL_2\n", encoding="utf-8")
    label_map = {"label_1": "joy", "label_2": "sadness"}
    samples = load_emotion_csv(path, label_map=label_map)
    assert [s["label"] for s in samples] == [EMOTIONS.index("joy"), EMOTIONS.index("sadness")]
