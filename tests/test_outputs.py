from __future__ import annotations

from pathlib import Path

from local_whisper_transcriber.outputs import (
    Segment,
    clean_name,
    format_srt_timestamp,
    normalize_segments,
    output_paths,
    write_selected_outputs,
)


class RawSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


def test_clean_name_removes_windows_invalid_characters() -> None:
    assert clean_name('a<b>c:"d"/e\\f|g?h*') == "a b c d e f g h"
    assert clean_name("   ...   ") == "untitled"


def test_srt_timestamp_uses_milliseconds() -> None:
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(3723.456) == "01:02:03,456"


def test_normalize_segments_skips_empty_and_orders_times() -> None:
    segments = normalize_segments(
        [
            RawSegment(0.2, 1.4, " hello\nworld "),
            RawSegment(0.1, 0.2, "second"),
            RawSegment(9, 10, "   "),
        ]
    )

    assert segments == [
        Segment(start=0.2, end=1.4, text="hello world"),
        Segment(start=1.4, end=2.4, text="second"),
    ]


def test_write_selected_outputs_creates_txt_srt_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"")
    segments = [
        Segment(start=0, end=1.4, text="hello | world"),
        Segment(start=61.2, end=62.7, text="second line"),
    ]

    written = write_selected_outputs(
        source_path=source,
        output_dir=tmp_path,
        segments=segments,
        formats={"txt", "srt", "md"},
        overwrite=True,
    )

    assert set(written) == set(output_paths(source, tmp_path, {"txt", "srt", "md"}).values())
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "hello | world\nsecond line\n"
    assert "00:00:00,000 --> 00:00:01,400" in (tmp_path / "demo.srt").read_text(encoding="utf-8")
    md = (tmp_path / "demo.字幕.md").read_text(encoding="utf-8")
    assert "# demo 字幕" in md
    assert "hello \\| world" in md


def test_empty_segments_create_empty_plain_outputs(tmp_path: Path) -> None:
    source = tmp_path / "empty.wav"
    source.write_bytes(b"")

    write_selected_outputs(
        source_path=source,
        output_dir=tmp_path,
        segments=[],
        formats={"txt", "srt", "md"},
        overwrite=True,
    )

    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "empty.srt").read_text(encoding="utf-8") == ""
    assert "| 时间 | 原文 |" in (tmp_path / "empty.字幕.md").read_text(encoding="utf-8")


def test_speaker_outputs_include_labels_without_changing_file_names(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"")
    segments = [
        Segment(0, 1, "你好", "说话人 1"),
        Segment(1, 2, "欢迎", "说话人 1"),
        Segment(2, 3, "谢谢", "说话人 2"),
    ]

    write_selected_outputs(source, tmp_path, segments, {"txt", "srt", "md"}, overwrite=True)

    assert (tmp_path / "meeting.txt").read_text(encoding="utf-8") == (
        "说话人 1：你好 欢迎\n说话人 2：谢谢\n"
    )
    assert "说话人 2：谢谢" in (tmp_path / "meeting.srt").read_text(encoding="utf-8")
    markdown = (tmp_path / "meeting.字幕.md").read_text(encoding="utf-8")
    assert "| 时间 | 说话人 | 原文 |" in markdown
    assert "| 00:00:02 - 00:00:03 | 说话人 2 | 谢谢 |" in markdown

