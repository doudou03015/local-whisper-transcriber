from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from .outputs import Segment, collapse_text


UNKNOWN_SPEAKER = "未知说话人"
NO_SPACE_LANGUAGES = {"zh", "ja", "ko"}


@dataclass(frozen=True)
class SpeakerOptions:
    mode: str = "off"
    device: str = "auto"
    count_mode: str = "auto"
    exact_speakers: int = 2
    min_speakers: int = 2
    max_speakers: int = 5


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class WordTiming:
    start: float | None
    end: float | None
    text: str
    speaker: str | None = None


def normalize_speaker_labels(turns: Iterable[SpeakerTurn]) -> list[SpeakerTurn]:
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end))
    mapping: dict[str, str] = {}
    normalized: list[SpeakerTurn] = []
    for turn in ordered:
        if turn.speaker not in mapping:
            mapping[turn.speaker] = f"说话人 {len(mapping) + 1}"
        normalized.append(replace(turn, speaker=mapping[turn.speaker]))
    return normalized


def choose_speaker(start: float, end: float, turns: Sequence[SpeakerTurn], nearest_limit: float = 2.0) -> str:
    overlap_by_speaker: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for index, turn in enumerate(turns):
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > 0:
            overlap_by_speaker[turn.speaker] = overlap_by_speaker.get(turn.speaker, 0.0) + overlap
            first_seen.setdefault(turn.speaker, index)
    if overlap_by_speaker:
        return min(overlap_by_speaker, key=lambda name: (-overlap_by_speaker[name], first_seen[name]))

    nearest: tuple[float, int, str] | None = None
    for index, turn in enumerate(turns):
        if end <= turn.start:
            distance = turn.start - end
        elif start >= turn.end:
            distance = start - turn.end
        else:
            distance = 0.0
        candidate = (distance, index, turn.speaker)
        if nearest is None or candidate < nearest:
            nearest = candidate
    if nearest and nearest[0] <= nearest_limit:
        return nearest[2]
    return UNKNOWN_SPEAKER


def assign_segments_quick(segments: Sequence[Segment], turns: Sequence[SpeakerTurn]) -> list[Segment]:
    return [replace(segment, speaker=choose_speaker(segment.start, segment.end, turns)) for segment in segments]


def _join_tokens(tokens: list[str], language: str) -> str:
    if language in NO_SPACE_LANGUAGES:
        return collapse_text("".join(tokens))
    text = " ".join(token.strip() for token in tokens if token.strip())
    return re.sub(r"\s+([,.;:!?])", r"\1", collapse_text(text))


def assign_aligned_segments(
    aligned_segments: Sequence[Mapping[str, object]],
    turns: Sequence[SpeakerTurn],
    language: str,
) -> list[Segment]:
    output: list[Segment] = []
    for aligned in aligned_segments:
        parent_start = float(aligned.get("start", 0.0) or 0.0)
        parent_end = float(aligned.get("end", parent_start + 1.0) or parent_start + 1.0)
        parent_text = collapse_text(str(aligned.get("text", "")))
        raw_words = aligned.get("words", [])
        words = raw_words if isinstance(raw_words, list) else []
        if not words:
            if parent_text:
                output.append(
                    Segment(
                        parent_start,
                        parent_end,
                        parent_text,
                        choose_speaker(parent_start, parent_end, turns),
                    )
                )
            continue

        timed_words: list[WordTiming] = []
        parent_speaker = choose_speaker(parent_start, parent_end, turns)
        for raw_word in words:
            if not isinstance(raw_word, Mapping):
                continue
            text = str(raw_word.get("word", ""))
            start_value = raw_word.get("start")
            end_value = raw_word.get("end")
            start = float(start_value) if isinstance(start_value, (int, float)) else None
            end = float(end_value) if isinstance(end_value, (int, float)) else None
            speaker = choose_speaker(start, end, turns) if start is not None and end is not None else parent_speaker
            timed_words.append(WordTiming(start, end, text, speaker))

        if not timed_words:
            if parent_text:
                output.append(Segment(parent_start, parent_end, parent_text, parent_speaker))
            continue

        group: list[WordTiming] = []
        for word in timed_words:
            if group and word.speaker != group[-1].speaker:
                output.append(_word_group_to_segment(group, language, parent_start, parent_end))
                group = []
            group.append(word)
        if group:
            output.append(_word_group_to_segment(group, language, parent_start, parent_end))
    return output


def _word_group_to_segment(
    words: Sequence[WordTiming],
    language: str,
    fallback_start: float,
    fallback_end: float,
) -> Segment:
    starts = [word.start for word in words if word.start is not None]
    ends = [word.end for word in words if word.end is not None]
    text = _join_tokens([word.text for word in words], language)
    return Segment(
        start=min(starts) if starts else fallback_start,
        end=max(ends) if ends else fallback_end,
        text=text,
        speaker=words[0].speaker or UNKNOWN_SPEAKER,
    )

