# Local Whisper Transcriber

## 中文说明

Local Whisper Transcriber 是一个 Windows 本地语音转文字软件，基于 `faster-whisper`、`ffmpeg` 和 `PySide6/Qt`。软件提供中文图形界面，支持音频、视频和文件夹批量转写，默认导出 `TXT`、`SRT` 和 `Markdown`。

### 功能

- 支持常见音频：`mp3`、`wav`、`m4a`、`aac`、`flac`、`ogg`、`wma`
- 支持常见视频：`mp4`、`mkv`、`mov`、`avi`、`webm`、`m4v`
- 支持文件夹批量转写和递归扫描
- 默认导出：
  - `<原文件名>.txt`
  - `<原文件名>.srt`
  - `<原文件名>.字幕.md`
- 支持 `auto`、`cuda`、`cpu` 设备选择
- 支持中文、自动检测和常见语言选项
- 支持 PyInstaller 打包成 Windows exe

### 图标

项目根目录的 `logo.ico` 是唯一图标来源。源码运行时、窗口图标、任务栏图标和 PyInstaller 打包后的 exe 图标都使用这个文件。打包时 `logo.ico` 会复制到：

```text
dist/Local Whisper Transcriber/logo.ico
```

### 模型说明

当前项目优先使用 faster-whisper/CTranslate2 格式模型。本机已有的 `faster-whisper-large-v3` 模型中，`model.bin` 约 3.09 GB，整个模型目录约 3.10 GB。这个体积是转换后的推理模型大小，不适合直接提交到 GitHub 仓库。

仓库默认不包含模型。模型查找顺序：

1. GUI 中指定的自定义模型目录
2. 当前项目的 `models/faster-whisper-<模型名>`
3. 当前项目父级目录中的 `models/faster-whisper-<模型名>`
4. `faster-whisper` 支持的模型名，例如 `large-v3`

推荐目录结构：

```text
local-whisper-transcriber/
  models/
    faster-whisper-large-v3/
      model.bin
      config.json
      tokenizer.json
      ...
```

如果不放本地模型，`faster-whisper` 可能会尝试从网络下载模型；离线环境请提前放好模型。

### 运行

先确认系统已安装并能调用 `ffmpeg`：

```powershell
ffmpeg -version
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

启动 GUI：

```powershell
.\run_gui.bat
```

也可以直接运行：

```powershell
$env:PYTHONPATH="$PWD\src"
python -m local_whisper_transcriber
```

### 自检

```powershell
.\run_gui.bat --self-test
```

自检会输出 Python、PySide6、ffmpeg、faster-whisper 和模型探测状态。打包后的 GUI exe 没有控制台窗口，运行 `Local Whisper Transcriber.exe --self-test` 时会在 exe 同目录写出 `self-test.txt`。

### 窗口显示问题

软件启动时会按当前屏幕工作区自动缩放并居中窗口，低分辨率屏幕可通过滚动区域查看完整界面。如果双击后窗口不可见，请查看 exe 同目录的 `startup.log`；正常启动会写入“窗口已创建”，启动异常会写入错误信息。

### 打包 exe

安装打包依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

打包：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

输出位置：

```text
dist/Local Whisper Transcriber/Local Whisper Transcriber.exe
```

打包默认不包含模型。发布或复制软件时，可以把模型放到：

```text
dist/Local Whisper Transcriber/models/faster-whisper-large-v3/
```

### 测试

```powershell
python -m pytest
```

### GitHub 上传建议

这个子项目已经包含 `.gitignore`，会排除模型、媒体文件、临时转写目录和打包产物。建议只上传本目录内的软件源码、README、依赖文件、测试、图标和打包脚本。

不建议提交：

- `models/`
- `.transcription_work/`
- `dist/`
- `build/`
- 大体积音视频文件

### 许可证说明

本项目暂未附加开源 LICENSE。`faster-whisper`、CTranslate2、模型权重、PySide6 和 ffmpeg 各自有独立许可证或使用条款；公开发布前请按你的用途补充许可证说明。

## English

Local Whisper Transcriber is a Windows desktop app for local speech-to-text transcription. It is built with `faster-whisper`, `ffmpeg`, and `PySide6/Qt`. The app uses a Chinese GUI, supports audio files, video files, and batch folder transcription, and exports `TXT`, `SRT`, and `Markdown` by default.

### Features

- Common audio inputs: `mp3`, `wav`, `m4a`, `aac`, `flac`, `ogg`, `wma`
- Common video inputs: `mp4`, `mkv`, `mov`, `avi`, `webm`, `m4v`
- Batch folder transcription with optional recursive scanning
- Default outputs:
  - `<source-name>.txt`
  - `<source-name>.srt`
  - `<source-name>.字幕.md`
- Device options: `auto`, `cuda`, `cpu`
- Chinese, automatic language detection, and common language presets
- Windows exe packaging through PyInstaller

### Icon

`logo.ico` in the project root is the single icon source. It is used for source-code runs, the main window icon, the taskbar icon, and the packaged exe icon. During packaging, `logo.ico` is copied to:

```text
dist/Local Whisper Transcriber/logo.ico
```

### Model Notes

This project primarily uses faster-whisper/CTranslate2 models. The local `faster-whisper-large-v3` model currently has a `model.bin` of about 3.09 GB, and the whole model directory is about 3.10 GB. This is the converted inference model size and should not be committed directly to a GitHub repository.

Models are not included in the repository by default. Model lookup order:

1. The custom model path selected in the GUI
2. `models/faster-whisper-<model-name>` in the project directory
3. `models/faster-whisper-<model-name>` in parent directories
4. A model name supported by `faster-whisper`, such as `large-v3`

Recommended layout:

```text
local-whisper-transcriber/
  models/
    faster-whisper-large-v3/
      model.bin
      config.json
      tokenizer.json
      ...
```

If no local model is available, `faster-whisper` may try to download one. For offline use, place the model directory in advance.

### Run

Make sure `ffmpeg` is installed and available:

```powershell
ffmpeg -version
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Launch the GUI:

```powershell
.\run_gui.bat
```

Or run directly:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m local_whisper_transcriber
```

### Self-Test

```powershell
.\run_gui.bat --self-test
```

The self-test reports Python, PySide6, ffmpeg, faster-whisper, and model detection status. The packaged GUI exe has no console window; running `Local Whisper Transcriber.exe --self-test` writes `self-test.txt` next to the exe.

### Window Display Troubleshooting

On startup, the app now scales and centers the window within the current screen work area. On low-resolution displays, the interface remains accessible through a scroll area. If the window is still not visible after launching the exe, check `startup.log` next to the exe; a normal launch writes “窗口已创建”, while startup failures write the error details.

### Build Exe

Install development dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Build:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

Output:

```text
dist/Local Whisper Transcriber/Local Whisper Transcriber.exe
```

Models are not bundled by default. For redistribution, place the model at:

```text
dist/Local Whisper Transcriber/models/faster-whisper-large-v3/
```

### Tests

```powershell
python -m pytest
```

### GitHub Publishing Notes

This subproject includes a `.gitignore` that excludes models, media files, temporary transcription files, and build outputs. Upload the source code, README, dependency files, tests, icon, and build script.

Do not commit:

- `models/`
- `.transcription_work/`
- `dist/`
- `build/`
- Large audio/video files

### License Notes

No project LICENSE has been added yet. `faster-whisper`, CTranslate2, model weights, PySide6, and ffmpeg each have their own licenses or terms. Add license details before public distribution.
