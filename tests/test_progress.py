from __future__ import annotations

from local_whisper_transcriber.progress import TASK_STAGES, progress_for_file_stage


def test_task_stage_list_matches_visible_workflow() -> None:
    assert [stage.key for stage in TASK_STAGES] == [
        "prepare",
        "load_model",
        "check_output",
        "extract_audio",
        "transcribe",
        "align",
        "diarize",
        "merge_speakers",
        "write_outputs",
        "complete",
    ]


def test_single_file_progress_advances_by_stage() -> None:
    values = [
        progress_for_file_stage(1, 1, "check_output"),
        progress_for_file_stage(1, 1, "extract_audio"),
        progress_for_file_stage(1, 1, "transcribe"),
        progress_for_file_stage(1, 1, "write_outputs"),
        progress_for_file_stage(1, 1, "complete"),
    ]

    assert values == sorted(values)
    assert values[0] > 0
    assert values[-1] == 99


def test_multi_file_progress_never_goes_backwards() -> None:
    values = []
    for file_index in (1, 2, 3):
        values.extend(
            [
                progress_for_file_stage(file_index, 3, "check_output"),
                progress_for_file_stage(file_index, 3, "extract_audio"),
                progress_for_file_stage(file_index, 3, "transcribe"),
                progress_for_file_stage(file_index, 3, "write_outputs"),
                progress_for_file_stage(file_index, 3, "complete"),
            ]
        )

    assert values == sorted(values)
    assert values[-1] == 99


def test_skipped_file_can_move_to_file_complete_stage() -> None:
    check_output = progress_for_file_stage(1, 1, "check_output")
    skipped_complete = progress_for_file_stage(1, 1, "complete")

    assert skipped_complete > check_output
    assert skipped_complete == 99
