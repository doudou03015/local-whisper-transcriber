from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from . import __app_name__
from .media import collect_media_files
from .models import (
    DEFAULT_MODEL_SIZE,
    MODEL_SIZES,
    app_base_dir,
    format_model_profile,
    format_model_profiles_summary,
    format_model_status,
    inspect_model,
)
from .outputs import selected_outputs_exist, write_selected_outputs
from .progress import TASK_STAGES, progress_for_file_stage, progress_for_global_stage, stage_status_text
from .selftest import run_self_test
from .transcriber import (
    TranscriptionCancelled,
    TranscriptionEngine,
    TranscriptionOptions,
    is_cuda_runtime_error,
    transcribe_media_file,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def app_icon_path() -> Path | None:
    candidates = [
        app_base_dir() / "logo.ico",
        Path.cwd() / "logo.ico",
        Path(__file__).resolve().parents[2] / "logo.ico",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_app_icon() -> QtGui.QIcon:
    path = app_icon_path()
    if path:
        return QtGui.QIcon(str(path))
    return QtGui.QIcon()


def startup_log_path() -> Path:
    return app_base_dir() / "startup.log"


def write_startup_log(message: str) -> None:
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        startup_log_path().write_text(f"[{timestamp}] {message}\n", encoding="utf-8")
    except OSError:
        pass


def configure_windows_app_id() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "LocalWhisperTranscriber.LocalWhisperTranscriber"
        )
    except Exception:
        pass


class TranscriberWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(__app_name__)
        self.setWindowIcon(load_app_icon())
        self.setMinimumSize(640, 480)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        default_input = Path.cwd()
        self.input_edit = QtWidgets.QLineEdit(str(default_input))
        self.output_edit = QtWidgets.QLineEdit(str(default_input / "转写结果"))
        self.work_edit = QtWidgets.QLineEdit(str(default_input / ".transcription_work" / "local_whisper"))
        self.custom_model_edit = QtWidgets.QLineEdit("")

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(MODEL_SIZES)
        self.model_combo.setCurrentText(DEFAULT_MODEL_SIZE)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])

        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItems(["zh", "auto", "en", "ja", "ko", "fr", "de", "es"])

        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.addItems(["transcribe", "translate"])

        self.compute_edit = QtWidgets.QLineEdit("")
        self.compute_edit.setPlaceholderText("可选，例如 int8")

        self.export_txt = QtWidgets.QCheckBox("TXT")
        self.export_srt = QtWidgets.QCheckBox("SRT")
        self.export_md = QtWidgets.QCheckBox("MD")
        for box in (self.export_txt, self.export_srt, self.export_md):
            box.setChecked(True)

        self.recursive_check = QtWidgets.QCheckBox("递归扫描文件夹")
        self.overwrite_check = QtWidgets.QCheckBox("覆盖已有结果")
        self.keep_audio_check = QtWidgets.QCheckBox("保留 WAV 临时音频")
        self.vad_check = QtWidgets.QCheckBox("启用 VAD 静音过滤")
        self.vad_check.setChecked(True)

        self.model_info = QtWidgets.QLabel()
        self.model_info.setWordWrap(True)
        self.model_help = QtWidgets.QLabel(format_model_profiles_summary())
        self.model_help.setWordWrap(True)
        self.status_label = QtWidgets.QLabel("就绪")
        self.current_label = QtWidgets.QLabel("")
        self.count_label = QtWidgets.QLabel("0 / 0")
        self.segment_label = QtWidgets.QLabel("")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress_note = QtWidgets.QLabel("进度按处理步骤估算，不代表剩余时间。")
        self.progress_note.setWordWrap(True)
        self.stage_status_labels: dict[str, QtWidgets.QLabel] = {}
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("Consolas", 10))

        self.start_button = QtWidgets.QPushButton("开始")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.setEnabled(False)

        self._build_ui()
        self._configure_tooltips()
        self._connect_signals()
        self._refresh_model_info()
        self._reset_stages()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._drain_events)
        self.timer.start(100)

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer.addWidget(scroll_area)

        content = QtWidgets.QWidget()
        scroll_area.setWidget(content)

        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        path_group = QtWidgets.QGroupBox("输入与目录")
        path_layout = QtWidgets.QGridLayout(path_group)
        path_layout.addWidget(QtWidgets.QLabel("输入文件或文件夹"), 0, 0)
        path_layout.addWidget(self.input_edit, 0, 1)
        file_button = QtWidgets.QPushButton("选择文件")
        folder_button = QtWidgets.QPushButton("选择文件夹")
        path_layout.addWidget(file_button, 0, 2)
        path_layout.addWidget(folder_button, 0, 3)
        file_button.clicked.connect(self._choose_input_file)
        folder_button.clicked.connect(self._choose_input_folder)

        path_layout.addWidget(QtWidgets.QLabel("输出目录"), 1, 0)
        path_layout.addWidget(self.output_edit, 1, 1)
        output_button = QtWidgets.QPushButton("选择")
        path_layout.addWidget(output_button, 1, 2, 1, 2)
        output_button.clicked.connect(self._choose_output)

        path_layout.addWidget(QtWidgets.QLabel("临时目录"), 2, 0)
        path_layout.addWidget(self.work_edit, 2, 1)
        work_button = QtWidgets.QPushButton("选择")
        path_layout.addWidget(work_button, 2, 2, 1, 2)
        work_button.clicked.connect(self._choose_work)
        path_layout.setColumnStretch(1, 1)
        root.addWidget(path_group)

        model_group = QtWidgets.QGroupBox("模型与识别")
        model_layout = QtWidgets.QGridLayout(model_group)
        model_layout.addWidget(QtWidgets.QLabel("模型"), 0, 0)
        model_layout.addWidget(self.model_combo, 0, 1)
        model_layout.addWidget(QtWidgets.QLabel("设备"), 0, 2)
        model_layout.addWidget(self.device_combo, 0, 3)
        model_layout.addWidget(QtWidgets.QLabel("语言"), 0, 4)
        model_layout.addWidget(self.language_combo, 0, 5)
        model_layout.addWidget(QtWidgets.QLabel("任务"), 0, 6)
        model_layout.addWidget(self.task_combo, 0, 7)
        model_layout.addWidget(QtWidgets.QLabel("计算类型"), 0, 8)
        model_layout.addWidget(self.compute_edit, 0, 9)

        model_layout.addWidget(QtWidgets.QLabel("自定义模型目录"), 1, 0)
        model_layout.addWidget(self.custom_model_edit, 1, 1, 1, 8)
        model_button = QtWidgets.QPushButton("选择")
        model_layout.addWidget(model_button, 1, 9)
        model_button.clicked.connect(self._choose_model)
        model_layout.addWidget(self.model_info, 2, 0, 1, 10)
        model_layout.addWidget(self.model_help, 3, 0, 1, 10)
        model_layout.setColumnStretch(9, 1)
        root.addWidget(model_group)

        output_group = QtWidgets.QGroupBox("批量与输出")
        output_layout = QtWidgets.QHBoxLayout(output_group)
        for widget in (
            self.export_txt,
            self.export_srt,
            self.export_md,
            self.recursive_check,
            self.overwrite_check,
            self.keep_audio_check,
            self.vad_check,
        ):
            output_layout.addWidget(widget)
        output_layout.addStretch(1)
        root.addWidget(output_group)

        controls = QtWidgets.QHBoxLayout()
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        open_button = QtWidgets.QPushButton("打开输出目录")
        open_button.clicked.connect(self._open_output_dir)
        check_model_button = QtWidgets.QPushButton("检查模型")
        check_model_button.clicked.connect(self._refresh_model_info)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(open_button)
        controls.addWidget(check_model_button)
        controls.addStretch(1)
        controls.addWidget(self.status_label)
        root.addLayout(controls)

        root.addWidget(self.progress)
        root.addWidget(self.progress_note)

        stats = QtWidgets.QHBoxLayout()
        self.count_label.setMinimumWidth(90)
        stats.addWidget(self.count_label)
        stats.addWidget(self.current_label, 1)
        stats.addWidget(self.segment_label)
        root.addLayout(stats)

        stage_group = QtWidgets.QGroupBox("当前步骤")
        stage_layout = QtWidgets.QGridLayout(stage_group)
        for row, stage in enumerate(TASK_STAGES):
            stage_layout.addWidget(QtWidgets.QLabel(stage.label), row, 0)
            status = QtWidgets.QLabel(stage_status_text("waiting"))
            status.setMinimumWidth(72)
            self.stage_status_labels[stage.key] = status
            stage_layout.addWidget(status, row, 1)
        stage_layout.setColumnStretch(2, 1)
        root.addWidget(stage_group)

        note = QtWidgets.QLabel(
            "本地 large-v3 模型使用 faster-whisper/CTranslate2 格式。"
            "模型不会打进 exe，也不会提交到 GitHub；请把模型放到 "
            "models/faster-whisper-large-v3。"
        )
        note.setWordWrap(True)
        root.addWidget(note)

        root.addWidget(self.log_text, 1)

    def _configure_tooltips(self) -> None:
        self.model_combo.setToolTip("模型越大通常越准，但速度更慢、资源占用更高。正式中文转写优先 large-v3。")
        self.device_combo.setToolTip("auto 会先尝试 CUDA，失败后回退 CPU；没有独显或 CUDA 环境时可直接选 CPU。")
        self.language_combo.setToolTip("zh 表示按中文识别；auto 会自动检测语言，但可能略慢或误判。")
        self.task_combo.setToolTip("transcribe 保留原语言；translate 会尽量翻译成英文。")
        self.compute_edit.setToolTip("留空使用推荐值。CPU 通常用 int8；CUDA 通常用 int8_float16 或 float16。")
        self.custom_model_edit.setToolTip("可指定本地 faster-whisper/CTranslate2 模型目录，目录内应包含 model.bin。")
        self.recursive_check.setToolTip("输入为文件夹时，同时扫描子文件夹。")
        self.overwrite_check.setToolTip("不勾选时，已存在所选输出格式的文件会被跳过。")
        self.keep_audio_check.setToolTip("保留从视频或音频转换出的 16kHz WAV 临时文件，便于排查问题但会占用空间。")
        self.vad_check.setToolTip("过滤静音片段，通常能减少无意义内容；如果漏识别，可关闭后重试。")
        self.progress.setToolTip("百分比按任务步骤估算，不代表真实剩余时间。")

    def fit_to_screen(self) -> None:
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos()) or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(960, 680)
            return

        area = screen.availableGeometry()
        width = max(640, min(1080, int(area.width() * 0.88)))
        height = max(480, min(780, int(area.height() * 0.88)))
        self.resize(width, height)
        self.move(
            area.x() + max(0, (area.width() - width) // 2),
            area.y() + max(0, (area.height() - height) // 2),
        )
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _connect_signals(self) -> None:
        self.model_combo.currentTextChanged.connect(self._refresh_model_info)
        self.custom_model_edit.editingFinished.connect(self._refresh_model_info)

    def _choose_input_file(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择媒体文件", self.input_edit.text())
        if selected:
            self.input_edit.setText(selected)
            self.output_edit.setText(str(Path(selected).parent / "转写结果"))

    def _choose_input_folder(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输入文件夹", self.input_edit.text())
        if selected:
            self.input_edit.setText(selected)
            self.output_edit.setText(str(Path(selected) / "转写结果"))

    def _choose_output(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def _choose_work(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择临时目录", self.work_edit.text())
        if selected:
            self.work_edit.setText(selected)

    def _choose_model(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择模型目录", self.custom_model_edit.text())
        if selected:
            self.custom_model_edit.setText(selected)
            self._refresh_model_info()

    def _open_output_dir(self) -> None:
        path = Path(self.output_edit.text()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)], check=False)

    def _refresh_model_info(self) -> None:
        status = inspect_model(self.model_combo.currentText(), self.custom_model_edit.text())
        self.model_info.setText(
            f"{format_model_status(status)}\n"
            f"当前选择：{format_model_profile(self.model_combo.currentText())}"
        )

    def _reset_stages(self) -> None:
        for stage in TASK_STAGES:
            self.stage_status_labels[stage.key].setText(stage_status_text("waiting"))

    def _reset_file_stages(self) -> None:
        for stage_key in ("check_output", "extract_audio", "transcribe", "write_outputs", "complete"):
            self.stage_status_labels[stage_key].setText(stage_status_text("waiting"))

    def _set_stage_status(self, stage_key: str, status: str) -> None:
        label = self.stage_status_labels.get(stage_key)
        if label is None:
            return
        label.setText(stage_status_text(status))

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _format_clock(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes, second = divmod(total_seconds, 60)
        hour, minute = divmod(minutes, 60)
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _selected_formats(self) -> set[str]:
        formats: set[str] = set()
        if self.export_txt.isChecked():
            formats.add("txt")
        if self.export_srt.isChecked():
            formats.add("srt")
        if self.export_md.isChecked():
            formats.add("md")
        return formats

    def _options(self) -> TranscriptionOptions:
        return TranscriptionOptions(
            model_size=self.model_combo.currentText(),
            custom_model_path=self.custom_model_edit.text(),
            device=self.device_combo.currentText(),
            compute_type=self.compute_edit.text(),
            language=self.language_combo.currentText(),
            task=self.task_combo.currentText(),
            vad_filter=self.vad_check.isChecked(),
        )

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        formats = self._selected_formats()
        if not formats:
            QtWidgets.QMessageBox.critical(self, "未选择输出格式", "请至少选择一种输出格式。")
            return

        input_path = Path(self.input_edit.text()).expanduser().resolve()
        output_dir = Path(self.output_edit.text()).expanduser().resolve()
        work_dir = Path(self.work_edit.text()).expanduser().resolve()
        if not input_path.exists():
            QtWidgets.QMessageBox.critical(self, "输入不存在", str(input_path))
            return

        self.stop_event.clear()
        self.progress.setValue(0)
        self.status_label.setText("准备中")
        self.current_label.setText("")
        self.segment_label.setText("")
        self.count_label.setText("0 / 0")
        self._reset_stages()
        self._set_running(True)
        self._append_log("任务开始")

        self.worker = threading.Thread(
            target=self._run_transcription,
            args=(input_path, output_dir, work_dir, formats, self._options()),
            daemon=True,
        )
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_label.setText("正在停止")
        self._append_log("已请求停止。当前片段处理结束后会停止任务。")

    def _run_transcription(
        self,
        input_path: Path,
        output_dir: Path,
        work_dir: Path,
        formats: set[str],
        options: TranscriptionOptions,
    ) -> None:
        try:
            self.events.put(("stage", ("prepare", "active", "正在扫描输入")))
            files = collect_media_files(input_path, self.recursive_check.isChecked())
            if not files:
                self.events.put(("stage", ("prepare", "failed", "未找到媒体文件")))
                self.events.put(("error", f"未找到支持的音频或视频文件：{input_path}"))
                return

            output_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            self.events.put(("log", f"找到 {len(files)} 个媒体文件"))
            self.events.put(("total", len(files)))
            self.events.put(("stage", ("prepare", "done", f"找到 {len(files)} 个媒体文件")))
            self.events.put(("stage", ("load_model", "active", "正在加载模型")))

            engine = TranscriptionEngine(options, log=lambda message: self.events.put(("log", message)))
            engine.load()
            self.events.put(("stage", ("load_model", "done", "模型已加载")))

            processed = 0
            skipped = 0
            failed = 0
            for index, source_path in enumerate(files, start=1):
                if self.stop_event.is_set():
                    self.events.put(("log", "任务已停止"))
                    break

                self.events.put(("current", (index, len(files), source_path.name)))
                self.events.put(
                    (
                        "file_stage",
                        ("check_output", "active", index, len(files), source_path.name, "正在检查输出"),
                    )
                )

                if selected_outputs_exist(source_path, output_dir, formats) and not self.overwrite_check.isChecked():
                    skipped += 1
                    self.events.put(("log", f"[{index}/{len(files)}] 跳过已有输出：{source_path.name}"))
                    self.events.put(
                        (
                            "file_stage",
                            ("check_output", "skipped", index, len(files), source_path.name, "已有输出，跳过"),
                        )
                    )
                    self.events.put(
                        (
                            "file_stage",
                            ("complete", "skipped", index, len(files), source_path.name, "当前文件已跳过"),
                        )
                    )
                    self.events.put(("summary", (processed, skipped, failed)))
                    continue

                try:
                    active_file_stage = "extract_audio"
                    self.events.put(
                        (
                            "file_stage",
                            ("check_output", "done", index, len(files), source_path.name, "需要处理"),
                        )
                    )
                    self.events.put(
                        (
                            "file_stage",
                            ("extract_audio", "active", index, len(files), source_path.name, "正在提取音频"),
                        )
                    )
                    self.events.put(("log", f"[{index}/{len(files)}] 提取音频：{source_path.name}"))

                    def progress(segment_count: int, latest_end: float) -> None:
                        self.events.put(
                            ("transcription_progress", (segment_count, latest_end, index, len(files), source_path.name))
                        )

                    def stage(stage_key: str) -> None:
                        nonlocal active_file_stage
                        if stage_key == "audio_extracted":
                            active_file_stage = "transcribe"
                            self.events.put(
                                (
                                    "file_stage",
                                    ("extract_audio", "done", index, len(files), source_path.name, "音频提取完成"),
                                )
                            )
                            self.events.put(
                                (
                                    "file_stage",
                                    ("transcribe", "active", index, len(files), source_path.name, "正在转写识别"),
                                )
                            )

                    retried_cpu = False
                    while True:
                        try:
                            segments = transcribe_media_file(
                                engine=engine,
                                source_path=source_path,
                                work_root=work_dir,
                                keep_audio=self.keep_audio_check.isChecked(),
                                stop_event=self.stop_event,
                                progress=progress,
                                stage=stage,
                            )
                            break
                        except Exception as exc:
                            if not retried_cpu and engine.device != "cpu" and is_cuda_runtime_error(exc):
                                self.events.put(("log", f"CUDA 失败，正在切换到 CPU int8 重试：{exc}"))
                                self.events.put(("status", "切换到 CPU"))
                                engine.reload_cpu_int8()
                                retried_cpu = True
                                continue
                            raise

                    self.events.put(
                        (
                            "file_stage",
                            (
                                "transcribe",
                                "done",
                                index,
                                len(files),
                                source_path.name,
                                f"转写完成，共 {len(segments)} 个片段",
                            ),
                        )
                    )
                    active_file_stage = "write_outputs"
                    self.events.put(
                        (
                            "file_stage",
                            ("write_outputs", "active", index, len(files), source_path.name, "正在写出结果"),
                        )
                    )
                    written = write_selected_outputs(
                        source_path=source_path,
                        output_dir=output_dir,
                        segments=segments,
                        formats=formats,
                        overwrite=self.overwrite_check.isChecked(),
                    )
                    processed += 1
                    self.events.put(("segments", f"已识别 {len(segments)} 个片段"))
                    self.events.put(
                        (
                            "file_stage",
                            (
                                "write_outputs",
                                "done",
                                index,
                                len(files),
                                source_path.name,
                                f"已写入 {len(written)} 个文件",
                            ),
                        )
                    )
                    active_file_stage = "complete"
                    self.events.put(
                        (
                            "file_stage",
                            ("complete", "done", index, len(files), source_path.name, "当前文件完成"),
                        )
                    )
                    self.events.put(("log", f"[{index}/{len(files)}] 已写入 {len(written)} 个文件：{source_path.name}"))
                except TranscriptionCancelled:
                    self.events.put(("log", "任务已取消"))
                    self.events.put(
                        (
                            "file_stage",
                            (active_file_stage, "skipped", index, len(files), source_path.name, "任务已取消"),
                        )
                    )
                    self.events.put(
                        (
                            "file_stage",
                            ("complete", "skipped", index, len(files), source_path.name, "任务已取消"),
                        )
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - keep batch running and report failures.
                    failed += 1
                    self.events.put(("log", f"[{index}/{len(files)}] 失败 {source_path.name}：{exc}"))
                    self.events.put(
                        (
                            "file_stage",
                            (active_file_stage, "failed", index, len(files), source_path.name, "当前阶段失败"),
                        )
                    )
                    self.events.put(
                        (
                            "file_stage",
                            ("complete", "failed", index, len(files), source_path.name, "当前文件失败"),
                        )
                    )

                self.events.put(("summary", (processed, skipped, failed)))

            self.events.put(("stage", ("complete", "done", "任务完成")))
            self.events.put(("progress", 100))
            self.events.put(("done", (processed, skipped, failed)))
        except Exception as exc:  # noqa: BLE001 - surface top-level GUI failures.
            self.events.put(("error", str(exc)))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "total":
                    self.count_label.setText(f"0 / {payload}")
                elif kind == "stage":
                    stage_key, status, message = payload  # type: ignore[misc]
                    self._set_stage_status(str(stage_key), str(status))
                    self.status_label.setText(str(message))
                    self.progress.setValue(progress_for_global_stage(str(stage_key)))
                elif kind == "file_stage":
                    stage_key, status, index, total, name, message = payload  # type: ignore[misc]
                    if stage_key == "check_output" and status == "active":
                        self._reset_file_stages()
                    self.count_label.setText(f"{index} / {total}")
                    self.current_label.setText(str(name))
                    self._set_stage_status(str(stage_key), str(status))
                    self.status_label.setText(str(message))
                    self.progress.setValue(progress_for_file_stage(int(index), int(total), str(stage_key)))
                elif kind == "status":
                    self.status_label.setText(str(payload))
                elif kind == "progress":
                    self.progress.setValue(int(float(payload)))
                elif kind == "current":
                    index, total, name = payload  # type: ignore[misc]
                    self.count_label.setText(f"{index} / {total}")
                    self.current_label.setText(str(name))
                    self.segment_label.setText("")
                elif kind == "transcription_progress":
                    segment_count, latest_end, index, total, name = payload  # type: ignore[misc]
                    self.count_label.setText(f"{index} / {total}")
                    self.current_label.setText(str(name))
                    self._set_stage_status("transcribe", "active")
                    self.status_label.setText("正在转写识别")
                    self.progress.setValue(progress_for_file_stage(int(index), int(total), "transcribe"))
                    self.segment_label.setText(
                        f"已识别 {segment_count} 个片段，最新时间 {self._format_clock(float(latest_end))}"
                    )
                elif kind == "segments":
                    self.segment_label.setText(str(payload))
                elif kind == "summary":
                    processed, skipped, failed = payload  # type: ignore[misc]
                    self.status_label.setText(f"完成 {processed}，跳过 {skipped}，失败 {failed}")
                elif kind == "done":
                    processed, skipped, failed = payload  # type: ignore[misc]
                    self.status_label.setText(f"完成 {processed}，跳过 {skipped}，失败 {failed}")
                    self._append_log(f"任务结束：完成 {processed}，跳过 {skipped}，失败 {failed}")
                    self._set_running(False)
                elif kind == "error":
                    self.status_label.setText("失败")
                    self._append_log(f"错误：{payload}")
                    self._set_running(False)
        except queue.Empty:
            pass


def main() -> int:
    if "--self-test" in sys.argv:
        return run_self_test()

    try:
        write_startup_log("启动中")
        configure_windows_app_id()
        app = QtWidgets.QApplication(sys.argv)
        app.setWindowIcon(load_app_icon())
        window = TranscriberWindow()
        window.show()
        QtCore.QTimer.singleShot(0, window.fit_to_screen)
        write_startup_log("窗口已创建")
        return int(app.exec())
    except Exception as exc:  # noqa: BLE001 - windowed exe has no console.
        write_startup_log(f"启动失败：{exc!r}")
        raise
