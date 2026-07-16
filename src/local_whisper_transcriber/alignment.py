from __future__ import annotations

import gc
from pathlib import Path
from threading import Event
from typing import Callable, Mapping, Sequence

from .outputs import Segment
from .transcriber import TranscriptionCancelled, is_cuda_runtime_error


ProgressCallback = Callable[[float], None]
LogCallback = Callable[[str], None]


def align_segments(
    audio_path: Path,
    segments: Sequence[Segment],
    language: str,
    model_path: Path,
    device: str,
    stop_event: Event | None = None,
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
) -> list[Mapping[str, object]]:
    logger = log or (lambda message: None)
    attempts = ["cuda", "cpu"] if device == "auto" else [device]
    last_error: Exception | None = None
    transcript = [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
    ]

    for attempt_device in attempts:
        align_model = None
        try:
            import torch
            import whisperx

            logger(f"正在加载 {language} 对齐模型：{attempt_device}")
            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=language,
                    device=attempt_device,
                    model_name=str(model_path),
                    model_cache_only=True,
                )
            except TypeError:
                align_model, metadata = whisperx.load_align_model(
                    language_code=language,
                    device=attempt_device,
                    model_name=str(model_path),
                )

            def progress_callback(value: float) -> None:
                if stop_event and stop_event.is_set():
                    raise TranscriptionCancelled("词级对齐已取消")
                if progress:
                    progress(float(value))

            result = whisperx.align(
                transcript,
                align_model,
                metadata,
                str(audio_path),
                attempt_device,
                return_char_alignments=False,
                progress_callback=progress_callback,
            )
            return list(result.get("segments", []))
        except TranscriptionCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - CUDA fallback is intentional.
            last_error = exc
            logger(f"词级对齐在 {attempt_device} 上失败：{exc}")
            if attempt_device == "cpu" or not is_cuda_runtime_error(exc):
                break
        finally:
            if align_model is not None:
                del align_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    raise RuntimeError(f"词级对齐失败：{last_error}")

