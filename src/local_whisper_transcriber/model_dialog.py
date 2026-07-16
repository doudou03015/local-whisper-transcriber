from __future__ import annotations

import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .model_manager import (
    ALIGNMENT_REPOSITORIES,
    COMMUNITY_MODEL,
    ManagedModel,
    alignment_model,
    default_model_path,
    default_models_root,
    download_model,
    inspect_managed_model,
    whisper_model,
)


ORGANIZATION = "Local Whisper Transcriber"
APPLICATION = "Local Whisper Transcriber"


def application_settings() -> QtCore.QSettings:
    return QtCore.QSettings(ORGANIZATION, APPLICATION)


def model_setting_key(model: ManagedModel) -> str:
    return f"models/{model.key}"


def configured_model_path(settings: QtCore.QSettings, model: ManagedModel) -> Path:
    saved = str(settings.value(model_setting_key(model), "") or "").strip()
    if saved:
        return Path(saved).expanduser()
    root = Path(str(settings.value("models/root", str(default_models_root()))))
    return default_model_path(model, root)


class _DownloadSignals(QtCore.QObject):
    progress = QtCore.Signal(object)
    finished = QtCore.Signal(object, str)
    failed = QtCore.Signal(str)


class ModelManagerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        model_size: str = "large-v3",
        alignment_language: str = "zh",
        first_run: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("首次配置向导" if first_run else "模型管理")
        self.setMinimumSize(720, 620)
        self.settings = application_settings()
        self.model_size = model_size
        self.cancel_event = threading.Event()
        self.signals = _DownloadSignals(self)
        self.signals.progress.connect(self._download_progress)
        self.signals.finished.connect(self._download_finished)
        self.signals.failed.connect(self._download_failed)
        self._active_model: ManagedModel | None = None

        self.root_edit = QtWidgets.QLineEdit(
            str(self.settings.value("models/root", str(default_models_root())))
        )
        self.whisper_status = QtWidgets.QLabel()
        self.community_status = QtWidgets.QLabel()
        self.alignment_status = QtWidgets.QLabel()
        for label in (self.whisper_status, self.community_status, self.alignment_status):
            label.setWordWrap(True)

        self.token_edit = QtWidgets.QLineEdit()
        self.token_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("仅在本次 Community-1 下载中使用，不会保存")

        self.language_combo = QtWidgets.QComboBox()
        for language in ALIGNMENT_REPOSITORIES:
            self.language_combo.addItem(language)
        self.language_combo.setCurrentText(
            alignment_language if alignment_language in ALIGNMENT_REPOSITORIES else "zh"
        )
        self.language_combo.currentTextChanged.connect(self.refresh_status)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress_label = QtWidgets.QLabel("没有正在进行的下载")
        self.cancel_button = QtWidgets.QPushButton("取消下载")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_event.set)
        self.download_buttons: list[QtWidgets.QPushButton] = []

        self._build_ui(first_run)
        self.refresh_status()

    def _build_ui(self, first_run: bool) -> None:
        root = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "先准备转写模型；需要区分说话人时，再配置免费的 Community-1。"
            "精确模式还需要当前语言的对齐模型。所有模型下载后均可离线使用。"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        storage = QtWidgets.QGroupBox("模型保存位置")
        storage_layout = QtWidgets.QHBoxLayout(storage)
        storage_layout.addWidget(self.root_edit, 1)
        choose_root = QtWidgets.QPushButton("选择目录")
        choose_root.clicked.connect(self._choose_root)
        storage_layout.addWidget(choose_root)
        root.addWidget(storage)

        whisper = QtWidgets.QGroupBox(f"步骤 1：Whisper {self.model_size} 转写模型")
        whisper_layout = QtWidgets.QVBoxLayout(whisper)
        whisper_layout.addWidget(self.whisper_status)
        whisper_buttons = QtWidgets.QHBoxLayout()
        whisper_buttons.addWidget(self._select_button(whisper_model(self.model_size)))
        whisper_buttons.addWidget(self._download_button(whisper_model(self.model_size)))
        whisper_buttons.addStretch(1)
        whisper_layout.addLayout(whisper_buttons)
        root.addWidget(whisper)

        community = QtWidgets.QGroupBox("步骤 2：Community-1 说话人模型（可稍后配置）")
        community_layout = QtWidgets.QVBoxLayout(community)
        guide = QtWidgets.QLabel(
            "首次下载需要免费 Hugging Face 账号、接受 Community-1 使用条件，并创建 Read 只读令牌。"
        )
        guide.setWordWrap(True)
        community_layout.addWidget(guide)
        links = QtWidgets.QHBoxLayout()
        links.addWidget(self._link_button("1. 注册/登录", "https://huggingface.co/join"))
        links.addWidget(
            self._link_button(
                "2. 接受模型条件",
                "https://huggingface.co/pyannote/speaker-diarization-community-1",
            )
        )
        links.addWidget(self._link_button("3. 创建只读令牌", "https://huggingface.co/settings/tokens"))
        links.addStretch(1)
        community_layout.addLayout(links)
        community_layout.addWidget(self.token_edit)
        community_layout.addWidget(self.community_status)
        community_buttons = QtWidgets.QHBoxLayout()
        community_buttons.addWidget(self._select_button(COMMUNITY_MODEL))
        community_buttons.addWidget(self._download_button(COMMUNITY_MODEL, self.token_edit))
        community_buttons.addStretch(1)
        community_layout.addLayout(community_buttons)
        root.addWidget(community)

        alignment = QtWidgets.QGroupBox("步骤 3：精确模式语言对齐模型（可稍后配置）")
        alignment_layout = QtWidgets.QVBoxLayout(alignment)
        language_row = QtWidgets.QHBoxLayout()
        language_row.addWidget(QtWidgets.QLabel("语言"))
        language_row.addWidget(self.language_combo)
        language_row.addStretch(1)
        alignment_layout.addLayout(language_row)
        alignment_layout.addWidget(self.alignment_status)
        alignment_buttons = QtWidgets.QHBoxLayout()
        select_alignment = QtWidgets.QPushButton("选择已有模型")
        select_alignment.clicked.connect(self._select_current_alignment)
        download_alignment = QtWidgets.QPushButton("下载当前语言模型")
        download_alignment.clicked.connect(self._download_current_alignment)
        self.download_buttons.append(download_alignment)
        alignment_buttons.addWidget(select_alignment)
        alignment_buttons.addWidget(download_alignment)
        alignment_buttons.addStretch(1)
        alignment_layout.addLayout(alignment_buttons)
        root.addWidget(alignment)

        progress_row = QtWidgets.QHBoxLayout()
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.cancel_button)
        root.addLayout(progress_row)
        root.addWidget(self.progress_label)

        actions = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        actions.rejected.connect(self.reject)
        root.addWidget(actions)
        if first_run:
            self.progress_label.setText("Whisper 模型配置完成后即可关闭向导并开始转写。")

    def _select_button(self, model: ManagedModel) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton("选择已有模型")
        button.clicked.connect(lambda: self._select_existing(model))
        return button

    def _download_button(
        self,
        model: ManagedModel,
        token_edit: QtWidgets.QLineEdit | None = None,
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton("下载")
        button.clicked.connect(lambda: self._start_download(model, token_edit.text() if token_edit else ""))
        self.download_buttons.append(button)
        return button

    @staticmethod
    def _link_button(text: str, url: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)))
        return button

    def _choose_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择模型保存目录", self.root_edit.text())
        if selected:
            self.root_edit.setText(selected)
            self.settings.setValue("models/root", selected)
            self.refresh_status()

    def _select_current_alignment(self) -> None:
        self._select_existing(alignment_model(self.language_combo.currentText()))

    def _download_current_alignment(self) -> None:
        self._start_download(alignment_model(self.language_combo.currentText()), "")

    def _select_existing(self, model: ManagedModel) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            f"选择 {model.label} 目录",
            str(configured_model_path(self.settings, model)),
        )
        if not selected:
            return
        status = inspect_managed_model(model, Path(selected))
        if not status.available:
            QtWidgets.QMessageBox.critical(self, "模型不可用", status.message)
            return
        self.settings.setValue(model_setting_key(model), str(status.path))
        self.refresh_status()

    def _start_download(self, model: ManagedModel, token: str) -> None:
        if self._active_model is not None:
            return
        if model.requires_token and not token.strip():
            QtWidgets.QMessageBox.warning(self, "需要访问令牌", "请先输入 Hugging Face 只读访问令牌。")
            return
        models_root = Path(self.root_edit.text()).expanduser()
        try:
            models_root.mkdir(parents=True, exist_ok=True)
            probe = models_root / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "目录不可写", str(exc))
            return

        self.settings.setValue("models/root", str(models_root.resolve()))
        destination = default_model_path(model, models_root)
        self._active_model = model
        self.cancel_event.clear()
        self.progress.setValue(0)
        self.progress_label.setText(f"正在准备下载：{model.label}")
        self.cancel_button.setEnabled(True)
        for button in self.download_buttons:
            button.setEnabled(False)

        def worker() -> None:
            try:
                path = download_model(
                    model,
                    destination=destination,
                    token=token.strip() or None,
                    progress=self.signals.progress.emit,
                    cancel_event=self.cancel_event,
                    replace=True,
                )
                self.signals.finished.emit(model, str(path))
            except Exception as exc:  # noqa: BLE001 - display download failures in the dialog.
                self.signals.failed.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    @QtCore.Slot(object)
    def _download_progress(self, progress) -> None:  # noqa: ANN001
        self.progress.setValue(progress.percent)
        self.progress_label.setText(
            f"{progress.completed_files}/{progress.total_files}：{progress.current_file}"
        )

    @QtCore.Slot(object, str)
    def _download_finished(self, model: ManagedModel, path: str) -> None:
        self.settings.setValue(model_setting_key(model), path)
        self.token_edit.clear()
        self.progress.setValue(100)
        self.progress_label.setText(f"下载完成：{path}")
        self._finish_download()
        self.refresh_status()

    @QtCore.Slot(str)
    def _download_failed(self, message: str) -> None:
        self.token_edit.clear()
        self.progress_label.setText(f"下载失败：{message}")
        QtWidgets.QMessageBox.critical(self, "模型下载失败", message)
        self._finish_download()

    def _finish_download(self) -> None:
        self._active_model = None
        self.cancel_button.setEnabled(False)
        for button in self.download_buttons:
            button.setEnabled(True)

    def refresh_status(self) -> None:
        whisper = whisper_model(self.model_size)
        alignment = alignment_model(self.language_combo.currentText())
        self._set_status(self.whisper_status, inspect_managed_model(whisper, configured_model_path(self.settings, whisper)))
        self._set_status(
            self.community_status,
            inspect_managed_model(COMMUNITY_MODEL, configured_model_path(self.settings, COMMUNITY_MODEL)),
        )
        self._set_status(
            self.alignment_status,
            inspect_managed_model(alignment, configured_model_path(self.settings, alignment)),
        )

    @staticmethod
    def _set_status(label: QtWidgets.QLabel, status) -> None:  # noqa: ANN001
        color = "#18794e" if status.available else "#9a3412"
        label.setText(f"<span style='color:{color}'>{status.message}</span><br>{status.path}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._active_model is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "下载仍在进行",
                "关闭窗口会请求取消下载，确定关闭吗？",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel_event.set()
        super().closeEvent(event)

