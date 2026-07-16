from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from local_whisper_transcriber.diarization import load_pcm_wav


def _write_pcm_wav(path: Path, channels: int = 1) -> None:
    sample_rate = 16000
    values = [int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate)) for index in range(160)]
    frames = b"".join(struct.pack("<h", value) * channels for value in values)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def test_load_pcm_wav_returns_pyannote_memory_input(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_pcm_wav(audio_path)

    audio = load_pcm_wav(audio_path)

    assert audio["sample_rate"] == 16000
    assert tuple(audio["waveform"].shape) == (1, 160)
    assert audio["waveform"].dtype.is_floating_point
    assert float(audio["waveform"].abs().max()) == pytest.approx(12000 / 32768, rel=0.02)


def test_load_pcm_wav_preserves_channels(tmp_path: Path) -> None:
    audio_path = tmp_path / "stereo.wav"
    _write_pcm_wav(audio_path, channels=2)

    audio = load_pcm_wav(audio_path)

    assert tuple(audio["waveform"].shape) == (2, 160)
