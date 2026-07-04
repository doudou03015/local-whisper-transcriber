from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def is_supported_media(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS


def collect_media_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if not is_supported_media(path):
            raise ValueError(f"不支持的媒体文件：{path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"输入路径不存在：{path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(candidate for candidate in iterator if is_supported_media(candidate))


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def extract_audio(source_path: Path, audio_path: Path) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未在 PATH 中找到 ffmpeg。请先安装 ffmpeg，或把 ffmpeg 加入 PATH。")

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    subprocess.run(command, check=True)
