from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from local_whisper_transcriber.gui import TranscriberWindow
from local_whisper_transcriber.model_dialog import ModelManagerDialog


def _application() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_main_window_has_disabled_by_default_speaker_controls() -> None:
    _application()
    window = TranscriberWindow()
    assert window.windowTitle() == "Local Whisper Transcriber"
    assert window.speaker_enable.text() == "启用说话人区分"
    assert not window.speaker_enable.isChecked()
    assert not window.speaker_mode_combo.isEnabled()
    assert {window.speaker_mode_combo.itemData(index) for index in range(2)} == {"quick", "precise"}
    assert {"align", "diarize", "merge_speakers"}.issubset(window.stage_status_labels)
    window.close()


def test_model_manager_shows_hugging_face_guidance() -> None:
    _application()
    dialog = ModelManagerDialog(model_size="large-v3", first_run=True)
    assert dialog.windowTitle() == "首次配置向导"
    assert dialog.token_edit.echoMode() == QtWidgets.QLineEdit.EchoMode.Password
    assert dialog.language_combo.findText("zh") >= 0
    dialog.close()
