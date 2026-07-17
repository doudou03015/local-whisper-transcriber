from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

from . import __app_name__, __version__
from .media import find_ffmpeg
from .model_dialog import application_settings, configured_model_path
from .models import DEFAULT_MODEL_SIZE, app_base_dir, format_model_status, inspect_model
from .model_manager import (
    COMMUNITY_MODEL,
    alignment_model,
    default_models_root,
    inspect_managed_model,
    whisper_model,
)


def _icon_path() -> Path | None:
    for candidate in (
        app_base_dir() / "logo.ico",
        Path.cwd() / "logo.ico",
        Path(__file__).resolve().parents[2] / "logo.ico",
    ):
        if candidate.exists():
            return candidate
    return None


def _one_line_error(exc: Exception) -> str:
    return next((line.strip() for line in str(exc).splitlines() if line.strip()), exc.__class__.__name__)


def run_self_test() -> int:
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    required_runtime_ok = True
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

    try:
        import ctranslate2

        lines.append(f"ctranslate2-cuda-devices={ctranslate2.get_cuda_device_count()}")
    except Exception as exc:  # noqa: BLE001 - self-test reports optional runtime details.
        lines.append(f"ctranslate2-runtime=not available ({exc})")

    for distribution in ("whisperx", "pyannote.audio", "torch", "torchaudio"):
        try:
            lines.append(f"{distribution}={importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{distribution}=not installed")

    try:
        from pyannote.audio import Pipeline  # noqa: F401

        lines.append("pyannote-runtime=available")
    except Exception as exc:  # noqa: BLE001 - packaged imports must expose their exact failure.
        required_runtime_ok = False
        lines.append(f"pyannote-runtime=unavailable ({type(exc).__name__}: {_one_line_error(exc)})")

    try:
        import torch

        lines.append(f"torch-cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001 - self-test reports optional runtime details.
        lines.append(f"torch-runtime=not available ({exc})")

    try:
        import torchcodec

        lines.append(f"torchcodec={importlib.metadata.version('torchcodec')} (runtime available)")
    except Exception as exc:  # noqa: BLE001 - static FFmpeg installs are handled in memory.
        lines.append(f"torchcodec=runtime unavailable ({_one_line_error(exc)})")
    lines.append("pyannote-audio-input=in-memory PCM WAV (TorchCodec not required)")

    icon_path = _icon_path()
    lines.append(f"icon={icon_path if icon_path else 'not found'}")
    settings = application_settings()
    whisper_path = configured_model_path(settings, whisper_model(DEFAULT_MODEL_SIZE))
    lines.append(format_model_status(inspect_model(DEFAULT_MODEL_SIZE, str(whisper_path))))
    community = inspect_managed_model(COMMUNITY_MODEL, configured_model_path(settings, COMMUNITY_MODEL))
    chinese_alignment = alignment_model("zh")
    alignment = inspect_managed_model(chinese_alignment, configured_model_path(settings, chinese_alignment))
    model_root = default_models_root()
    writable_anchor = model_root if model_root.exists() else model_root.parent
    lines.append(f"community-1={community.state} ({community.path})")
    lines.append(f"alignment-zh={alignment.state} ({alignment.path})")
    lines.append(f"models-writable={os.access(writable_anchor, os.W_OK)} ({model_root})")
    lines.append("pyannote-telemetry=disabled")
    lines.append("ok" if required_runtime_ok else "failed")

    text = "\n".join(lines) + "\n"
    if sys.stdout:
        print(text, end="")
    if getattr(sys, "frozen", False):
        try:
            (app_base_dir() / "self-test.txt").write_text(text, encoding="utf-8")
        except OSError:
            pass
    return 0 if required_runtime_ok else 1
