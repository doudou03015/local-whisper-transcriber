from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from .media import extract_audio
from .models import DEFAULT_MODEL_SIZE, resolve_model_reference
from .outputs import Segment, clean_name, normalize_segments


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, float], None]
StageCallback = Callable[[str], None]
PostProcessCallback = Callable[[Path, "TranscriptionResult"], list[Segment]]


class TranscriptionCancelled(RuntimeError):
    pass


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: str


@dataclass
class TranscriptionOptions:
    model_size: str = DEFAULT_MODEL_SIZE
    custom_model_path: str = ""
    device: str = "auto"
    compute_type: str = ""
    language: str = "zh"
    task: str = "transcribe"
    vad_filter: bool = True
    beam_size: int = 5


class TranscriptionEngine:
    def __init__(self, options: TranscriptionOptions, log: LogCallback | None = None) -> None:
        self.options = options
        self.log = log or (lambda message: None)
        self.model = None
        self.device = ""
        self.compute_type = ""
        self.model_reference = ""

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "未安装 faster-whisper。请运行：python -m pip install -r requirements.txt"
            ) from exc

        attempts = self._device_attempts()
        self.model_reference = resolve_model_reference(
            self.options.model_size,
            self.options.custom_model_path,
        )

        last_error: Exception | None = None
        for device, compute_type in attempts:
            try:
                if device == "cpu" and last_error is not None and self.options.device == "auto":
                    self.log(f"CUDA 加载失败，正在回退 CPU int8：{last_error}")
                self.log(f"正在加载模型：{self.model_reference}，设备：{device}（{compute_type}）")
                self.model = WhisperModel(
                    self.model_reference,
                    device=device,
                    compute_type=compute_type,
                )
                self.device = device
                self.compute_type = compute_type
                if device == "cuda":
                    self.log(f"模型已加载到显卡：{_cuda_device_name()}（CUDA，{compute_type}）")
                else:
                    self.log(f"模型已加载：CPU（{compute_type}）")
                return
            except Exception as exc:  # noqa: BLE001 - backend fallback is intentional.
                last_error = exc
                self.log(f"模型在 {device} 上加载失败：{exc}")

        raise RuntimeError(f"无法加载 Whisper 模型 {self.model_reference}：{last_error}")

    def _device_attempts(self) -> list[tuple[str, str]]:
        compute_type = self.options.compute_type.strip() or None
        if self.options.device == "auto":
            return [("cuda", compute_type or "int8_float16"), ("cpu", "int8")]
        if self.options.device == "cuda":
            return [("cuda", compute_type or "int8_float16")]
        return [("cpu", compute_type or "int8")]

    def reload_cpu_int8(self) -> None:
        self.options.device = "cpu"
        self.options.compute_type = "int8"
        self.model = None
        self.load()

    def transcribe_audio(
        self,
        audio_path: Path,
        stop_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        if self.model is None:
            self.load()

        language = None if self.options.language == "auto" else self.options.language
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            task=self.options.task,
            vad_filter=self.options.vad_filter,
            beam_size=self.options.beam_size,
        )
        self.log(
            "检测到语言："
            f"{info.language} ({getattr(info, 'language_probability', 0.0):.2f})"
        )

        raw_segments = []
        for count, segment in enumerate(segments_iter, start=1):
            if stop_event and stop_event.is_set():
                raise TranscriptionCancelled("转写已取消")
            raw_segments.append(segment)
            if progress and (count == 1 or count % 5 == 0):
                progress(count, float(getattr(segment, "end", 0.0)))
        return TranscriptionResult(
            segments=normalize_segments(raw_segments),
            language=str(info.language),
        )


def is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = ("cuda", "cublas", "cudnn", "cudart", "nvrtc", "out of memory")
    return any(marker in message for marker in markers)


def _cuda_device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - logging must not break transcription.
        pass
    return "NVIDIA GPU"


def transcribe_media_file(
    engine: TranscriptionEngine,
    source_path: Path,
    work_root: Path,
    keep_audio: bool,
    stop_event: Event | None = None,
    progress: ProgressCallback | None = None,
    stage: StageCallback | None = None,
    post_process: PostProcessCallback | None = None,
) -> TranscriptionResult:
    work_dir = work_root / clean_name(source_path.stem)
    audio_path = work_dir / f"{clean_name(source_path.stem)}.wav"
    try:
        if stop_event and stop_event.is_set():
            raise TranscriptionCancelled("转写已取消")
        extract_audio(source_path, audio_path)
        if stage:
            stage("audio_extracted")
        result = engine.transcribe_audio(audio_path, stop_event=stop_event, progress=progress)
        if post_process:
            result.segments = post_process(audio_path, result)
        return result
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg 处理失败：{source_path.name}：{exc}") from exc
    finally:
        if not keep_audio:
            try:
                audio_path.unlink()
            except FileNotFoundError:
                pass
