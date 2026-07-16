from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from .models import app_base_dir


WHISPER_REPOSITORIES = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
    "tiny": "Systran/faster-whisper-tiny",
}

ALIGNMENT_REPOSITORIES = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "fr": "facebook/wav2vec2-large-xlsr-53-french",
    "de": "facebook/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
}


class ModelDownloadCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ManagedModel:
    key: str
    label: str
    repository: str
    relative_path: Path
    required_files: tuple[str, ...]
    requires_weights: bool = False
    requires_token: bool = False


@dataclass(frozen=True)
class ManagedModelStatus:
    model: ManagedModel
    path: Path
    state: str
    message: str

    @property
    def available(self) -> bool:
        return self.state == "available"


@dataclass(frozen=True)
class DownloadProgress:
    completed_files: int
    total_files: int
    completed_bytes: int
    total_bytes: int
    current_file: str

    @property
    def percent(self) -> int:
        if self.total_bytes > 0:
            return min(100, int(self.completed_bytes * 100 / self.total_bytes))
        if self.total_files > 0:
            return min(100, int(self.completed_files * 100 / self.total_files))
        return 0


ProgressCallback = Callable[[DownloadProgress], None]


def default_models_root() -> Path:
    return app_base_dir() / "models"


def whisper_model(model_size: str) -> ManagedModel:
    repository = WHISPER_REPOSITORIES.get(model_size)
    if not repository:
        raise ValueError(f"不支持自动下载 Whisper 模型：{model_size}")
    return ManagedModel(
        key=f"whisper-{model_size}",
        label=f"Whisper {model_size}",
        repository=repository,
        relative_path=Path(f"faster-whisper-{model_size}"),
        required_files=("model.bin", "config.json", "tokenizer.json"),
    )


COMMUNITY_MODEL = ManagedModel(
    key="community-1",
    label="Community-1 说话人模型",
    repository="pyannote/speaker-diarization-community-1",
    relative_path=Path("pyannote-speaker-diarization-community-1"),
    required_files=("config.yaml",),
    requires_weights=True,
    requires_token=True,
)


def alignment_model(language: str) -> ManagedModel:
    repository = ALIGNMENT_REPOSITORIES.get(language)
    if not repository:
        raise ValueError(f"精确模式暂未配置 {language} 的默认对齐模型。")
    return ManagedModel(
        key=f"alignment-{language}",
        label=f"{language} 词级对齐模型",
        repository=repository,
        relative_path=Path("whisperx-alignment") / language,
        required_files=("config.json",),
        requires_weights=True,
    )


def default_model_path(model: ManagedModel, models_root: Path | None = None) -> Path:
    return (models_root or default_models_root()) / model.relative_path


def inspect_managed_model(model: ManagedModel, path: Path | None = None) -> ManagedModelStatus:
    target = (path or default_model_path(model)).expanduser()
    if not target.exists():
        return ManagedModelStatus(model, target, "missing", "未配置")
    if not target.is_dir():
        return ManagedModelStatus(model, target, "invalid", "路径不是文件夹")

    missing = [name for name in model.required_files if not (target / name).is_file()]
    if missing:
        return ManagedModelStatus(model, target, "invalid", f"缺少文件：{', '.join(missing)}")
    if model.requires_weights and not any(
        item.is_file() and item.suffix.lower() in {".bin", ".safetensors", ".ckpt", ".pt"}
        for item in target.rglob("*")
    ):
        return ManagedModelStatus(model, target, "invalid", "未找到模型权重文件")
    return ManagedModelStatus(model, target.resolve(), "available", "可用")


def _repository_files(model: ManagedModel, token: str | None) -> list[tuple[str, int]]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("未安装 huggingface-hub，无法下载模型。") from exc

    info = HfApi().model_info(model.repository, files_metadata=True, token=token)
    files: list[tuple[str, int]] = []
    for sibling in info.siblings or []:
        name = getattr(sibling, "rfilename", "")
        if name:
            files.append((name, int(getattr(sibling, "size", 0) or 0)))
    return files


def _download_repository_file(
    model: ManagedModel,
    filename: str,
    staging_dir: Path,
    token: str | None,
) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=model.repository,
        filename=filename,
        local_dir=staging_dir,
        token=token,
    )


def download_model(
    model: ManagedModel,
    destination: Path | None = None,
    token: str | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
    replace: bool = False,
) -> Path:
    if model.requires_token and not (token or "").strip():
        raise ValueError("下载 Community-1 需要 Hugging Face 只读访问令牌。")

    target = (destination or default_model_path(model)).expanduser().resolve()
    current = inspect_managed_model(model, target)
    if current.available:
        return target
    if target.exists() and not replace:
        raise RuntimeError(f"目标目录已存在但模型不完整：{target}")

    staging_dir = app_base_dir() / ".downloads" / model.key
    staging_dir.mkdir(parents=True, exist_ok=True)
    files = _repository_files(model, token)
    if not files:
        raise RuntimeError(f"模型仓库没有可下载文件：{model.repository}")

    total_bytes = sum(size for _, size in files)
    completed_bytes = 0
    for index, (filename, size) in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            raise ModelDownloadCancelled("模型下载已取消；已下载内容会保留用于续传。")
        _download_repository_file(model, filename, staging_dir, token)
        completed_bytes += size
        if progress:
            progress(DownloadProgress(index, len(files), completed_bytes, total_bytes, filename))

    staged_status = inspect_managed_model(model, staging_dir)
    if not staged_status.available:
        raise RuntimeError(f"下载完成但模型校验失败：{staged_status.message}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(staging_dir), str(target))
    final_status = inspect_managed_model(model, target)
    if not final_status.available:
        raise RuntimeError(f"模型安装失败：{final_status.message}")
    return target

