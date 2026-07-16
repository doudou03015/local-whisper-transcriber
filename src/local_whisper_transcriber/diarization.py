from __future__ import annotations

import os
import wave
from pathlib import Path
from threading import Event
from typing import Callable

from .speakers import SpeakerOptions, SpeakerTurn, normalize_speaker_labels
from .transcriber import TranscriptionCancelled, is_cuda_runtime_error


ProgressCallback = Callable[[float], None]
LogCallback = Callable[[str], None]


def load_pcm_wav(audio_path: Path):  # noqa: ANN201 - torch is an optional runtime import.
    """Load the normalized PCM WAV without relying on TorchCodec's FFmpeg DLLs."""
    import torch

    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if channels < 1 or sample_width != 2:
        raise RuntimeError("说话人分析需要 16 位 PCM WAV 音频")

    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32)
    samples = samples.reshape(-1, channels).transpose(0, 1).contiguous() / 32768.0
    return {"waveform": samples, "sample_rate": sample_rate}


class DiarizationEngine:
    def __init__(self, model_path: Path, options: SpeakerOptions, log: LogCallback | None = None) -> None:
        self.model_path = model_path
        self.options = options
        self.log = log or (lambda message: None)
        self.pipeline = None
        self.device = ""

    def load(self) -> None:
        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        try:
            import torch
            from pyannote.audio import Pipeline
            from pyannote.audio.telemetry import set_telemetry_metrics
        except ImportError as exc:
            raise RuntimeError("未安装 pyannote.audio，无法区分说话人。") from exc

        set_telemetry_metrics(False)
        attempts = ["cuda", "cpu"] if self.options.device == "auto" else [self.options.device]
        last_error: Exception | None = None
        for device in attempts:
            try:
                self.log(f"正在加载 Community-1：{device}")
                pipeline = Pipeline.from_pretrained(str(self.model_path))
                if pipeline is None:
                    raise RuntimeError("Community-1 模型加载返回空对象")
                self.pipeline = pipeline.to(torch.device(device))
                self.device = device
                self.log(f"Community-1 已加载：{device}")
                return
            except Exception as exc:  # noqa: BLE001 - CUDA fallback is intentional.
                last_error = exc
                self.log(f"Community-1 在 {device} 上加载失败：{exc}")
                self.pipeline = None
        raise RuntimeError(f"无法加载 Community-1：{last_error}")

    def reload_cpu(self) -> None:
        self.options = SpeakerOptions(
            mode=self.options.mode,
            device="cpu",
            count_mode=self.options.count_mode,
            exact_speakers=self.options.exact_speakers,
            min_speakers=self.options.min_speakers,
            max_speakers=self.options.max_speakers,
        )
        self.pipeline = None
        self.load()

    def diarize(
        self,
        audio_path: Path,
        stop_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[SpeakerTurn]:
        if self.pipeline is None:
            self.load()

        kwargs: dict[str, int] = {}
        if self.options.count_mode == "exact":
            kwargs["num_speakers"] = self.options.exact_speakers
        elif self.options.count_mode == "range":
            kwargs["min_speakers"] = self.options.min_speakers
            kwargs["max_speakers"] = self.options.max_speakers

        last_percent = 0.0

        def hook(step_name, step_artifact, file=None, total=None, completed=None):  # noqa: ANN001
            nonlocal last_percent
            if stop_event and stop_event.is_set():
                raise TranscriptionCancelled("说话人分析已取消")
            if progress and total and completed is not None:
                percent = max(last_percent, min(99.0, float(completed) * 100.0 / float(total)))
                last_percent = percent
                progress(percent)

        try:
            if stop_event and stop_event.is_set():
                raise TranscriptionCancelled("说话人分析已取消")
            audio = load_pcm_wav(audio_path)
            self.log("说话人音频已载入内存，无需 TorchCodec 解码")
            output = self.pipeline(audio, hook=hook, **kwargs)
        except Exception as exc:
            if self.device != "cpu" and is_cuda_runtime_error(exc):
                self.log(f"说话人分析 CUDA 失败，切换 CPU 重试：{exc}")
                self.reload_cpu()
                return self.diarize(audio_path, stop_event=stop_event, progress=progress)
            raise

        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)
        turns = [
            SpeakerTurn(float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        if progress:
            progress(100.0)
        return normalize_speaker_labels(turns)
