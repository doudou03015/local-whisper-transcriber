from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INVALID_FILENAME_CHARS = '<>:"/\\|?*'
OutputFormat = str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def clean_name(value: str, fallback: str = "untitled") -> str:
    value = "".join(" " if ch in INVALID_FILENAME_CHARS else ch for ch in value)
    value = re.sub(r"\s+", " ", value).strip().rstrip(".")
    return value or fallback


def collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " ")).strip()


def normalize_segments(raw_segments: Iterable[object]) -> list[Segment]:
    normalized: list[Segment] = []
    last_end = 0.0

    for raw in raw_segments:
        text = collapse_text(str(getattr(raw, "text", "")))
        if not text:
            continue

        start = max(last_end, float(getattr(raw, "start", 0.0)))
        end = float(getattr(raw, "end", start + 1.0))
        if end <= start:
            end = start + 1.0
        normalized.append(Segment(start=start, end=end, text=text))
        last_end = end

    return normalized


def format_clock(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, second = divmod(total_seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, millisecond = divmod(milliseconds, 1000)
    minutes, second = divmod(total_seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d},{millisecond:03d}"


def escape_markdown_cell(value: str) -> str:
    return collapse_text(value).replace("\\", "\\\\").replace("|", "\\|")


def output_paths(source_path: Path, output_dir: Path, formats: set[OutputFormat]) -> dict[OutputFormat, Path]:
    base = clean_name(source_path.stem)
    paths: dict[OutputFormat, Path] = {}
    if "txt" in formats:
        paths["txt"] = output_dir / f"{base}.txt"
    if "srt" in formats:
        paths["srt"] = output_dir / f"{base}.srt"
    if "md" in formats:
        paths["md"] = output_dir / f"{base}.字幕.md"
    return paths


def selected_outputs_exist(source_path: Path, output_dir: Path, formats: set[OutputFormat]) -> bool:
    paths = output_paths(source_path, output_dir, formats)
    return bool(paths) and all(path.exists() for path in paths.values())


def write_txt(path: Path, segments: list[Segment]) -> None:
    has_speakers = any(segment.speaker for segment in segments)
    if has_speakers:
        paragraphs: list[tuple[str, list[str]]] = []
        for segment in segments:
            speaker = segment.speaker or "未知说话人"
            if paragraphs and paragraphs[-1][0] == speaker:
                paragraphs[-1][1].append(segment.text)
            else:
                paragraphs.append((speaker, [segment.text]))
        text = "\n".join(f"{speaker}：{' '.join(parts)}" for speaker, parts in paragraphs)
    else:
        text = "\n".join(segment.text for segment in segments)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_srt(path: Path, segments: list[Segment]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}",
                    f"{segment.speaker}：{segment.text}" if segment.speaker else segment.text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def write_markdown(path: Path, source_path: Path, segments: list[Segment]) -> None:
    title = source_path.stem
    has_speakers = any(segment.speaker for segment in segments)
    lines = [f"# {title} 字幕", ""]
    if has_speakers:
        lines.extend(["| 时间 | 说话人 | 原文 |", "|---|---|---|"])
    else:
        lines.extend(["| 时间 | 原文 |", "|---|---|"])
    for segment in segments:
        start = format_clock(math.floor(segment.start))
        end = format_clock(math.ceil(segment.end))
        if has_speakers:
            lines.append(
                f"| {start} - {end} | {escape_markdown_cell(segment.speaker or '未知说话人')} | "
                f"{escape_markdown_cell(segment.text)} |"
            )
        else:
            lines.append(f"| {start} - {end} | {escape_markdown_cell(segment.text)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_outputs(
    source_path: Path,
    output_dir: Path,
    segments: list[Segment],
    formats: set[OutputFormat],
    overwrite: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    paths = output_paths(source_path, output_dir, formats)
    for output_format, path in paths.items():
        if path.exists() and not overwrite:
            continue
        if output_format == "txt":
            write_txt(path, segments)
        elif output_format == "srt":
            write_srt(path, segments)
        elif output_format == "md":
            write_markdown(path, source_path, segments)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        written.append(path)
    return written
