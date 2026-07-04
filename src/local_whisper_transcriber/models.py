from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_SIZE = "large-v3"
MODEL_SIZES = ("large-v3", "medium", "small", "base", "tiny")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    speed: str
    accuracy: str
    resource: str
    recommended_for: str
    note: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "large-v3": ModelProfile(
        name="large-v3",
        speed="最慢",
        accuracy="最高",
        resource="占用最高，适合有独显或愿意等待的正式任务",
        recommended_for="中文、长音频、噪声环境、正式字幕和最终稿",
        note="不确定选哪个时优先选它；准确率优先，速度让位。",
    ),
    "medium": ModelProfile(
        name="medium",
        speed="较慢",
        accuracy="较高",
        resource="资源占用明显低于 large-v3",
        recommended_for="中文正式转写、普通长音频、准确率和速度都要兼顾的任务",
        note="通常是质量和等待时间之间的稳妥折中。",
    ),
    "small": ModelProfile(
        name="small",
        speed="较快",
        accuracy="中等",
        resource="轻量，普通 CPU 也更容易接受",
        recommended_for="普通机器、批量预处理、内容预览和草稿",
        note="机器较慢时先用它试跑，确认流程后再换大模型。",
    ),
    "base": ModelProfile(
        name="base",
        speed="很快",
        accuracy="偏低",
        resource="占用很低",
        recommended_for="粗略预览、短音频、快速判断内容",
        note="适合快看，不适合正式字幕。",
    ),
    "tiny": ModelProfile(
        name="tiny",
        speed="最快",
        accuracy="最低",
        resource="占用最低",
        recommended_for="功能测试、极快试跑、硬件很弱的场景",
        note="只建议测试流程或快速摸底。",
    ),
}


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


def get_model_profile(model_size: str) -> ModelProfile | None:
    return MODEL_PROFILES.get(model_size)


def format_model_profile(model_size: str) -> str:
    profile = get_model_profile(model_size)
    if profile is None:
        return "自定义模型：请按模型来源确认速度、准确率和资源占用。"
    return (
        f"{profile.name}：速度 {profile.speed}，准确率 {profile.accuracy}；"
        f"{profile.resource}。适合：{profile.recommended_for}。{profile.note}"
    )


def format_model_profiles_summary() -> str:
    lines = [
        "模型区别：",
        "large-v3 准确率最高，适合正式中文转写；medium 更平衡；",
        "small/base 适合预览或普通机器；tiny 主要用于快速测试。",
        "默认建议：不确定就选 large-v3；机器慢或只是预览，先选 small/base。",
    ]
    return "\n".join(lines)
