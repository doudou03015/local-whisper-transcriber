from __future__ import annotations

from local_whisper_transcriber.outputs import Segment
from local_whisper_transcriber.speakers import (
    UNKNOWN_SPEAKER,
    SpeakerTurn,
    assign_aligned_segments,
    assign_segments_quick,
    normalize_speaker_labels,
)


def test_labels_follow_first_appearance_and_reset_per_file() -> None:
    turns = normalize_speaker_labels(
        [SpeakerTurn(4, 8, "B"), SpeakerTurn(0, 4, "A"), SpeakerTurn(8, 10, "A")]
    )
    assert [turn.speaker for turn in turns] == ["说话人 1", "说话人 2", "说话人 1"]


def test_quick_assignment_uses_largest_overlap_and_two_second_nearest_limit() -> None:
    turns = [SpeakerTurn(0, 4, "说话人 1"), SpeakerTurn(4, 8, "说话人 2")]
    segments = [
        Segment(2, 6.5, "dominant second"),
        Segment(9, 10, "near second"),
        Segment(12, 13, "too far"),
    ]
    assigned = assign_segments_quick(segments, turns)
    assert [segment.speaker for segment in assigned] == ["说话人 2", "说话人 2", UNKNOWN_SPEAKER]


def test_precise_assignment_splits_chinese_text_on_speaker_change() -> None:
    turns = [SpeakerTurn(0, 1, "说话人 1"), SpeakerTurn(1, 2, "说话人 2")]
    aligned = [
        {
            "start": 0,
            "end": 2,
            "text": "你好谢谢",
            "words": [
                {"word": "你", "start": 0.0, "end": 0.4},
                {"word": "好", "start": 0.4, "end": 0.8},
                {"word": "谢", "start": 1.1, "end": 1.4},
                {"word": "谢", "start": 1.4, "end": 1.8},
            ],
        }
    ]
    segments = assign_aligned_segments(aligned, turns, "zh")
    assert segments == [
        Segment(0.0, 0.8, "你好", "说话人 1"),
        Segment(1.1, 1.8, "谢谢", "说话人 2"),
    ]

