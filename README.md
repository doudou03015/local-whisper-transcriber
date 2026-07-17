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
- 可选 Community-1 说话人区分，输出“说话人 1/2/3”
- 快速模式按转写片段匹配；精确模式使用 WhisperX 做词或汉字级时间对齐
- 内置中文模型管理和首次配置向导
- 支持 PyInstaller 打包成 Windows exe

### 首次配置与模型管理

首次启动时，如果没有找到 Whisper 模型，软件会自动打开“首次配置向导”。主界面的“模型管理”按钮也可以随时重新打开该窗口。

模型中心提供两种配置方式：选择电脑上已有的模型目录，或者直接下载到软件旁的 `models/`。本机原有的 D 盘 `faster-whisper-large-v3` 可以直接选择，不需要复制 3GB 文件。

启用说话人区分前，需要配置免费的 [Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)：

1. 注册或登录 Hugging Face。
2. 在 Community-1 页面接受模型使用条件。
3. 创建 Read 只读访问令牌。
4. 在模型中心临时输入令牌并下载。

令牌只在当前下载任务的内存中使用，下载结束后立即清除，不会写入配置、日志或 Git。下载完成后可以完全离线运行。程序默认关闭 pyannote 匿名遥测。

默认模型结构：

```text
models/
  faster-whisper-large-v3/
  pyannote-speaker-diarization-community-1/
  whisperx-alignment/
    zh/
```

### 说话人模式

- **关闭**：保持原来的转写速度和输出格式，也是默认值。
- **快速模式**：Community-1 判断每位说话人的时间段，再按最大重叠时间给整个 Whisper 片段编号。
- **精确模式**：WhisperX 先生成词或汉字级时间轴，再与 Community-1 合并，适合说话人快速轮换的录音。

人数可以自动判断、指定固定人数，或者设置最少/最多人数。每个文件都按首次出现顺序重新编号，不会识别真实姓名。重叠说话、声音过于相似或录音噪声较大时仍可能判断错误。

启用后，TXT 和 SRT 使用 `说话人 1：文字`，Markdown 增加“说话人”列。文件名不变；已有普通转写需要勾选“覆盖已有结果”才能重新生成。`translate` 不能使用精确模式，因为翻译后的英文无法与源语言语音做可靠对齐。

### 图标

项目根目录的 `logo.ico` 是唯一图标来源。源码运行时、窗口图标、任务栏图标和 PyInstaller 打包后的 exe 图标都使用这个文件。打包时 `logo.ico` 会复制到：

```text
dist/Local Whisper Transcriber/logo.ico
```

### 模型说明

当前项目优先使用 faster-whisper/CTranslate2 格式模型。本机已有的 `faster-whisper-large-v3` 模型中，`model.bin` 约 3.09 GB，整个模型目录约 3.10 GB。这个体积是转换后的推理模型大小，不适合直接提交到 GitHub 仓库。

模型越大通常越准，但速度更慢、资源占用更高。GUI 中的模型差异可以按下面理解：

| 模型 | 速度 | 准确率 | 推荐场景 |
|---|---|---|---|
| `large-v3` | 最慢 | 最高 | 中文、长音频、噪声环境、正式字幕和最终稿 |
| `medium` | 较慢 | 较高 | 正式转写，兼顾准确率和等待时间 |
| `small` | 较快 | 中等 | 普通机器、批量预处理、内容预览和草稿 |
| `base` | 很快 | 偏低 | 粗略预览、短音频、快速判断内容 |
| `tiny` | 最快 | 最低 | 功能测试、极快试跑、硬件很弱的场景 |

默认建议：不确定就选 `large-v3`；机器慢或只是预览，先选 `small` 或 `base`。`transcribe` 会保留原语言，`translate` 会尽量翻译成英文。`auto` 设备会先尝试 CUDA，失败后回退 CPU；CPU 通常用 `int8`，CUDA 通常用 `int8_float16` 或 `float16`。

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

### 进度显示

GUI 的进度按处理步骤估算，不代表真实剩余时间。单个文件也会依次显示：

1. 准备任务
2. 加载模型
3. 检查输出
4. 提取音频
5. 转写识别
6. 写出结果
7. 任务完成

转写阶段会显示已识别片段数和最新识别到的音频时间，例如“已识别 25 个片段，最新时间 00:12:30”。批量任务会同时显示当前文件序号、文件名以及完成/跳过/失败汇总。

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

自检会输出 Python、PySide6、ffmpeg、faster-whisper、WhisperX、PyTorch/CUDA、pyannote、Community-1 和对齐模型状态。打包后的 GUI exe 没有控制台窗口，运行 `Local Whisper Transcriber.exe --self-test` 时会在 exe 同目录写出 `self-test.txt`。

### 窗口显示问题

软件启动时会按当前屏幕工作区自动缩放并居中窗口，低分辨率屏幕可通过滚动区域查看完整界面。如果双击后窗口不可见，请查看 exe 同目录的 `startup.log`；正常启动会写入“窗口已创建”，启动异常会写入错误信息。

### GPU 与音频运行库排障

选择 `auto` 时，Whisper 会先尝试 `CUDA + int8_float16`，失败才回退到 `CPU + int8`。任务日志会明确显示实际使用的显卡、设备和计算精度；自检中的 `ctranslate2-cuda-devices=1` 与 `torch-cuda=True` 分别表示 faster-whisper 和 PyTorch 已识别 CUDA。本项目已在 8GB 显存的 NVIDIA GeForce RTX 2070 SUPER 上完成 `large-v3` 实际转写验证。

FFmpeg 8 静态版可能让自检显示 `torchcodec=runtime unavailable`。这不影响本软件的基础转写或说话人区分：媒体先由 `ffmpeg.exe` 转换为 16kHz PCM WAV，说话人模块再从内存波形读取音频，不依赖 TorchCodec 的 FFmpeg 4-7 共享 DLL。只有直接调用 pyannote 的文件路径解码接口时才需要另外安装兼容的共享 DLL。

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

增强版会包含 PyTorch、WhisperX、pyannote 和 CUDA 运行库，因此 `dist` 明显大于 0.2.0。模型仍然不包含在发布包中。构建脚本还会准备 WhisperX 所需的 NLTK `punkt_tab` 数据，使词级对齐无需临时联网下载该数据。

如果旧的 `dist/Local Whisper Transcriber/models/` 已有本地模型，构建脚本会在清理发布目录前临时保留它们，并在构建成功或失败后自动恢复，避免重建 EXE 时删除已下载模型。

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

本项目暂未附加开源 LICENSE。`pyannote.audio` 使用 MIT，Community-1 使用 CC BY 4.0，WhisperX 使用 BSD-2-Clause；`faster-whisper`、CTranslate2、Whisper/对齐模型权重、PySide6 和 ffmpeg 也各自有独立许可证或使用条款。公开发布前应保留归属说明并按实际用途复核所有依赖条款。

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
- Optional Community-1 speaker diarization with quick and precise modes
- Guided local model setup and downloads
- Windows exe packaging through PyInstaller

### First-Time Setup And Model Manager

When the Whisper model is missing, the app opens a Chinese first-time setup dialog. The same model manager remains available from the main window.

It can register an existing local model directory or download models next to the application under `models/`. Community-1 requires a free Hugging Face account, acceptance of its model conditions, and a read-only access token. The token exists only in memory for the active download and is never saved to settings, logs, or Git. Once downloaded, all processing can run offline, and pyannote telemetry is disabled by default.

```text
models/
  faster-whisper-large-v3/
  pyannote-speaker-diarization-community-1/
  whisperx-alignment/
    zh/
```

### Speaker Modes

- **Off** preserves the existing behavior and is the default.
- **Quick** assigns each Whisper segment to the speaker with the greatest time overlap.
- **Precise** uses WhisperX language-specific word or character alignment before assigning speakers.

Speaker labels restart as `说话人 1/2/3` for each file and do not identify real people. Speaker-enabled TXT and SRT outputs prefix the text with the speaker label, while Markdown adds a speaker column. Precise mode is not available with `translate` because translated English text cannot be reliably aligned to source-language speech.

### Icon

`logo.ico` in the project root is the single icon source. It is used for source-code runs, the main window icon, the taskbar icon, and the packaged exe icon. During packaging, `logo.ico` is copied to:

```text
dist/Local Whisper Transcriber/logo.ico
```

### Model Notes

This project primarily uses faster-whisper/CTranslate2 models. The local `faster-whisper-large-v3` model currently has a `model.bin` of about 3.09 GB, and the whole model directory is about 3.10 GB. This is the converted inference model size and should not be committed directly to a GitHub repository.

Larger models are usually more accurate, but slower and more resource intensive. The GUI model choices follow these tradeoffs:

| Model | Speed | Accuracy | Recommended use |
|---|---|---|---|
| `large-v3` | Slowest | Highest | Chinese, long audio, noisy audio, formal subtitles, final drafts |
| `medium` | Slower | High | Formal transcription with a better speed/quality balance |
| `small` | Faster | Medium | Normal PCs, batch drafts, previews |
| `base` | Very fast | Lower | Rough previews, short audio, quick content checks |
| `tiny` | Fastest | Lowest | Workflow tests, very quick trial runs, weak hardware |

Default recommendation: choose `large-v3` when unsure; choose `small` or `base` first when the machine is slow or the task is only a preview. `transcribe` keeps the original language, while `translate` tries to translate to English. `auto` device tries CUDA first and falls back to CPU; CPU usually uses `int8`, while CUDA usually uses `int8_float16` or `float16`.

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

### Progress Display

The GUI progress percentage is step-based and does not promise a real remaining-time estimate. Even a single file now moves through:

1. Prepare task
2. Load model
3. Check outputs
4. Extract audio
5. Transcribe
6. Write outputs
7. Complete

During transcription, the GUI shows the recognized segment count and the latest recognized audio timestamp, for example “已识别 25 个片段，最新时间 00:12:30”. Batch runs also show the current file index, file name, and completed/skipped/failed summary.

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

The self-test reports Python, PySide6, ffmpeg, faster-whisper, WhisperX, PyTorch/CUDA, pyannote, Community-1, alignment models, and model-directory permissions. The packaged GUI exe has no console window; running `Local Whisper Transcriber.exe --self-test` writes `self-test.txt` next to the exe.

### Window Display Troubleshooting

On startup, the app now scales and centers the window within the current screen work area. On low-resolution displays, the interface remains accessible through a scroll area. If the window is still not visible after launching the exe, check `startup.log` next to the exe; a normal launch writes “窗口已创建”, while startup failures write the error details.

### GPU and Audio Runtime Troubleshooting

With `auto`, Whisper first tries `CUDA + int8_float16` and falls back to `CPU + int8` only after a CUDA failure. The task log reports the actual GPU, device, and compute type. In the self-test, `ctranslate2-cuda-devices=1` and `torch-cuda=True` confirm CUDA visibility for faster-whisper and PyTorch respectively. This project has completed a real `large-v3` transcription test on an 8GB NVIDIA GeForce RTX 2070 SUPER.

A static FFmpeg 8 installation may make the self-test report `torchcodec=runtime unavailable`. This does not block transcription or diarization in this app: `ffmpeg.exe` first creates a 16kHz PCM WAV, and the speaker pipeline receives the waveform from memory instead of using TorchCodec's FFmpeg 4-7 shared DLLs. Compatible shared DLLs are only required when calling pyannote's direct file-path decoder outside this application.

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

The enhanced build contains PyTorch, WhisperX, pyannote, CUDA runtime libraries, and bundled NLTK alignment data, so it is substantially larger than version 0.2.0. Model weights remain excluded.

If `dist/Local Whisper Transcriber/models/` already contains local models, the build script preserves them before cleaning the distribution directory and restores them after either a successful or failed build.

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

No project LICENSE has been added yet. `pyannote.audio` is MIT licensed, Community-1 is CC BY 4.0, and WhisperX is BSD-2-Clause. `faster-whisper`, CTranslate2, Whisper/alignment model weights, PySide6, and ffmpeg also have their own licenses or terms. Preserve attribution and review every dependency before public distribution.
