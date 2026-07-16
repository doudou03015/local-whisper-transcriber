from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

from . import __app_name__, __version__
from .media import find_ffmpeg
from .models import DEFAULT_MODEL_SIZE, app_base_dir, format_model_status, inspect_model
from .model_manager import COMMUNITY_MODEL, alignment_model, default_model_path, default_models_root, inspect_managed_model


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
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
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

    for distribution in ("whisperx", "pyannote.audio", "torch", "torchaudio"):
        try:
            lines.append(f"{distribution}={importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{distribution}=not installed")

    try:
        import torch

        lines.append(f"torch-cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001 - self-test reports optional runtime details.
        lines.append(f"torch-runtime=not available ({exc})")

    icon_path = _icon_path()
    lines.append(f"icon={icon_path if icon_path else 'not found'}")
    lines.append(format_model_status(inspect_model(DEFAULT_MODEL_SIZE, "")))
    community = inspect_managed_model(COMMUNITY_MODEL, default_model_path(COMMUNITY_MODEL))
    chinese_alignment = alignment_model("zh")
    alignment = inspect_managed_model(chinese_alignment, default_model_path(chinese_alignment))
    model_root = default_models_root()
    writable_anchor = model_root if model_root.exists() else model_root.parent
    lines.append(f"community-1={community.state} ({community.path})")
    lines.append(f"alignment-zh={alignment.state} ({alignment.path})")
    lines.append(f"models-writable={os.access(writable_anchor, os.W_OK)} ({model_root})")
    lines.append("pyannote-telemetry=disabled")
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
