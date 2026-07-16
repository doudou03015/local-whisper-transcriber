from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskStage:
    key: str
    label: str


TASK_STAGES = (
    TaskStage("prepare", "准备任务"),
    TaskStage("load_model", "加载模型"),
    TaskStage("check_output", "检查输出"),
    TaskStage("extract_audio", "提取音频"),
    TaskStage("transcribe", "转写识别"),
    TaskStage("align", "词级对齐"),
    TaskStage("diarize", "区分说话人"),
    TaskStage("merge_speakers", "合并说话人"),
    TaskStage("write_outputs", "写出结果"),
    TaskStage("complete", "任务完成"),
)

STAGE_STATUS_TEXT = {
    "waiting": "等待",
    "active": "进行中",
    "done": "完成",
    "skipped": "跳过",
    "failed": "失败",
}

FILE_STAGE_FRACTIONS = {
    "check_output": 0.05,
    "extract_audio": 0.25,
    "transcribe": 0.50,
    "align": 0.63,
    "diarize": 0.76,
    "merge_speakers": 0.84,
    "write_outputs": 0.90,
    "complete": 1.0,
}

GLOBAL_PROGRESS = {
    "prepare": 0,
    "load_model": 8,
}

FILE_PROGRESS_START = 10
FILE_PROGRESS_END = 99


def stage_status_text(status: str) -> str:
    return STAGE_STATUS_TEXT.get(status, status)


def progress_for_global_stage(stage_key: str) -> int:
    if stage_key == "complete":
        return 100
    return GLOBAL_PROGRESS.get(stage_key, FILE_PROGRESS_START)


def progress_for_file_stage(file_index: int, total_files: int, stage_key: str) -> int:
    if total_files <= 0:
        return progress_for_global_stage(stage_key)

    bounded_index = min(max(file_index, 1), total_files)
    completed_files = bounded_index - 1
    stage_fraction = FILE_STAGE_FRACTIONS.get(stage_key, 0.0)
    total_fraction = (completed_files + stage_fraction) / total_files
    progress_span = FILE_PROGRESS_END - FILE_PROGRESS_START
    value = FILE_PROGRESS_START + progress_span * total_fraction
    return min(FILE_PROGRESS_END, max(FILE_PROGRESS_START, int(round(value))))
