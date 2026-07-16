from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

import local_whisper_transcriber.model_manager as manager
from local_whisper_transcriber.model_manager import (
    COMMUNITY_MODEL,
    ModelDownloadCancelled,
    alignment_model,
    download_model,
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


def test_alignment_models_cover_every_explicit_gui_language() -> None:
    for language in ("zh", "en", "ja", "ko", "fr", "de", "es"):
        assert alignment_model(language).repository


def test_community_download_requires_token() -> None:
    with pytest.raises(ValueError, match="访问令牌"):
        download_model(COMMUNITY_MODEL, token="")


def test_download_uses_staging_reports_progress_and_can_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager, "app_base_dir", lambda: tmp_path)
    files = [("config.yaml", 10), ("models/model.safetensors", 90)]
    monkeypatch.setattr(manager, "_repository_files", lambda model, token: files)

    def fake_download(model, filename, staging_dir, token) -> None:  # noqa: ANN001
        path = staging_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")

    monkeypatch.setattr(manager, "_download_repository_file", fake_download)
    progress = []
    destination = tmp_path / "models" / "community"
    installed = download_model(
        COMMUNITY_MODEL,
        destination=destination,
        token="hf_read_token",
        progress=progress.append,
    )
    assert installed == destination
    assert inspect_managed_model(COMMUNITY_MODEL, destination).available
    assert progress[-1].percent == 100
    assert not (tmp_path / ".downloads" / COMMUNITY_MODEL.key).exists()


def test_cancelled_download_keeps_staging_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager, "app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(manager, "_repository_files", lambda model, token: [("config.yaml", 1)])
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ModelDownloadCancelled):
        download_model(
            COMMUNITY_MODEL,
            destination=tmp_path / "models" / "community",
            token="hf_read_token",
            cancel_event=cancelled,
        )
    assert (tmp_path / ".downloads" / COMMUNITY_MODEL.key).is_dir()
