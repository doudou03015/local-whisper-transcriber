from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from . import __app_name__, __version__
from .media import find_ffmpeg
from .models import DEFAULT_MODEL_SIZE, app_base_dir, format_model_status, inspect_model


def _icon_path() -> Path | None:
    for candidate in (
        app_base_dir() / "logo.ico",
        Path.cwd() / "logo.ico",
        Path(__file__).resolve().parents[2] / "logo.ico",
    ):
        if candidate.exists():
            return candidate
    return None


def run_self_test() -> int:
    lines = [
        f"app={__app_name__} {__version__}",
        f"python={sys.version.split()[0]}",
        f"ffmpeg={find_ffmpeg() or 'not found'}",
    ]
    try:
        import PySide6

        lines.append(f"PySide6={PySide6.__version__}")
    except Exception as exc:  # noqa: BLE001 - self-test should report import failures.
        lines.append(f"PySide6=not available ({exc})")

    try:
        lines.append(f"faster-whisper={importlib.metadata.version('faster-whisper')}")
    except importlib.metadata.PackageNotFoundError:
        lines.append("faster-whisper=not installed")

    icon_path = _icon_path()
    lines.append(f"icon={icon_path if icon_path else 'not found'}")
    lines.append(format_model_status(inspect_model(DEFAULT_MODEL_SIZE, "")))
    lines.append("ok")

    text = "\n".join(lines) + "\n"
    if sys.stdout:
        print(text, end="")
    if getattr(sys, "frozen", False):
        try:
            (app_base_dir() / "self-test.txt").write_text(text, encoding="utf-8")
        except OSError:
            pass
    return 0
