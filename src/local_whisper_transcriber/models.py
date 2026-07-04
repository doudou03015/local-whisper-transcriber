from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_SIZE = "large-v3"
MODEL_SIZES = ("large-v3", "medium", "small", "base", "tiny")


@dataclass(frozen=True)
class ModelStatus:
    reference: str
    local_path: Path | None
    exists: bool
    model_bin_bytes: int
    directory_bytes: int

    @property
    def model_bin_gb(self) -> float:
        return self.model_bin_bytes / 1_000_000_000

    @property
    def directory_gb(self) -> float:
        return self.directory_bytes / 1_000_000_000


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _candidate_model_dirs(model_size: str) -> list[Path]:
    base = app_base_dir()
    names = [f"faster-whisper-{model_size}", model_size]
    roots: list[Path] = []
    for anchor in (base, Path.cwd()):
        roots.append(anchor / "models")
        roots.extend(parent / "models" for parent in list(anchor.parents)[:4])

    candidates: list[Path] = []
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def resolve_model_reference(model_size: str = DEFAULT_MODEL_SIZE, custom_model_path: str = "") -> str:
    custom = custom_model_path.strip()
    if custom:
        custom_path = Path(custom).expanduser()
        if custom_path.exists():
            return str(custom_path.resolve())
        return custom

    for candidate in _candidate_model_dirs(model_size):
        if (candidate / "model.bin").exists():
            return str(candidate.resolve())
    return model_size


def inspect_model(model_size: str = DEFAULT_MODEL_SIZE, custom_model_path: str = "") -> ModelStatus:
    reference = resolve_model_reference(model_size, custom_model_path)
    path = Path(reference)
    if path.exists():
        model_bin = path / "model.bin"
        model_bin_bytes = model_bin.stat().st_size if model_bin.exists() else 0
        return ModelStatus(
            reference=reference,
            local_path=path,
            exists=model_bin.exists(),
            model_bin_bytes=model_bin_bytes,
            directory_bytes=_directory_size(path),
        )

    return ModelStatus(
        reference=reference,
        local_path=None,
        exists=False,
        model_bin_bytes=0,
        directory_bytes=0,
    )


def format_model_status(status: ModelStatus) -> str:
    if status.exists and status.local_path:
        return (
            f"本地模型：{status.local_path} | "
            f"model.bin {status.model_bin_gb:.2f} GB，目录 {status.directory_gb:.2f} GB"
        )
    return (
        f"未找到本地模型。程序会尝试使用 '{status.reference}'，"
        "前提是 faster-whisper 可以下载或解析该模型。"
    )
