from __future__ import annotations

from pathlib import Path

from local_whisper_transcriber.model_manager import (
    COMMUNITY_MODEL,
    alignment_model,
    inspect_managed_model,
    whisper_model,
)


def test_whisper_model_requires_core_ctranslate_files(tmp_path: Path) -> None:
    model = whisper_model("large-v3")
    assert inspect_managed_model(model, tmp_path).state == "invalid"
    for name in model.required_files:
        (tmp_path / name).write_bytes(b"test")
    assert inspect_managed_model(model, tmp_path).available


def test_community_and_alignment_require_weights(tmp_path: Path) -> None:
    community = tmp_path / "community"
    community.mkdir()
    (community / "config.yaml").write_text("pipeline: test", encoding="utf-8")
    assert inspect_managed_model(COMMUNITY_MODEL, community).state == "invalid"
    (community / "model.safetensors").write_bytes(b"weights")
    assert inspect_managed_model(COMMUNITY_MODEL, community).available

    alignment = tmp_path / "alignment"
    alignment.mkdir()
    (alignment / "config.json").write_text("{}", encoding="utf-8")
    (alignment / "pytorch_model.bin").write_bytes(b"weights")
    assert inspect_managed_model(alignment_model("zh"), alignment).available

