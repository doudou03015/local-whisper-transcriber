$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$IconPath = Join-Path $Root "logo.ico"
$DistAppDir = Join-Path $Root "dist\Local Whisper Transcriber"
$DistModelsPath = Join-Path $DistAppDir "models"
$PreservedModelsPath = Join-Path $Root ".build-preserved-models"
if (-not (Test-Path $IconPath)) {
    throw "logo.ico was not found: $IconPath"
}

$RootFullPath = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
$DistModelsFullPath = [System.IO.Path]::GetFullPath($DistModelsPath)
$PreservedModelsFullPath = [System.IO.Path]::GetFullPath($PreservedModelsPath)
if (-not $DistModelsFullPath.StartsWith($RootFullPath, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not $PreservedModelsFullPath.StartsWith($RootFullPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Model preservation paths must stay inside the project directory."
}

python -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Run: python -m pip install -r requirements-dev.txt"
}

$RequiredPackages = @(
    "faster-whisper",
    "whisperx",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "torchvision",
    "torchcodec"
)
foreach ($Package in $RequiredPackages) {
    python -m pip show $Package *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "$Package is not installed. Run: python -m pip install -r requirements.txt"
    }
}

$NltkDataPath = Join-Path $Root "build\nltk_data"
New-Item -ItemType Directory -Path $NltkDataPath -Force | Out-Null
python -m nltk.downloader -d $NltkDataPath punkt_tab
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare NLTK punkt_tab data."
}

$PyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name",
    "Local Whisper Transcriber",
    "--icon",
    $IconPath,
    "--add-data",
    "$IconPath;.",
    "--add-data",
    "$NltkDataPath;nltk_data",
    "--paths",
    (Join-Path $Root "src"),
    "--collect-all",
    "faster_whisper",
    "--collect-all",
    "ctranslate2",
    "--collect-all",
    "tokenizers",
    "--collect-all",
    "av",
    "--collect-all",
    "whisperx",
    "--collect-all",
    "pyannote.audio",
    "--collect-all",
    "pyannote.core",
    "--collect-all",
    "pyannote.database",
    "--collect-all",
    "pyannote.metrics",
    "--collect-all",
    "pyannote.pipeline",
    "--collect-all",
    "pyannoteai",
    "--collect-all",
    "lightning",
    "--collect-all",
    "lightning_fabric",
    "--collect-all",
    "pytorch_lightning",
    "--collect-all",
    "torchmetrics",
    "--collect-all",
    "asteroid_filterbanks",
    "--collect-all",
    "torch_audiomentations",
    "--collect-all",
    "pytorch_metric_learning",
    "--collect-all",
    "omegaconf",
    "--collect-submodules",
    "scipy._external.array_api_compat",
    "--collect-all",
    "torch",
    "--collect-all",
    "torchaudio",
    "--collect-all",
    "torchvision",
    "--collect-all",
    "torchcodec",
    "--collect-all",
    "transformers",
    "--collect-all",
    "nltk",
    "--collect-all",
    "pandas",
    "--collect-all",
    "huggingface_hub",
    (Join-Path $Root "src\local_whisper_transcriber\__main__.py")
)

$ModelsWerePreserved = $false
if (Test-Path -LiteralPath $DistModelsPath) {
    if (Test-Path -LiteralPath $PreservedModelsPath) {
        throw "Cannot preserve models because the temporary path already exists: $PreservedModelsPath"
    }
    Write-Host "Temporarily preserving existing models..."
    Move-Item -LiteralPath $DistModelsPath -Destination $PreservedModelsPath
    $ModelsWerePreserved = $true
}

try {
    python -m PyInstaller @PyInstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
}
finally {
    if ($ModelsWerePreserved -and (Test-Path -LiteralPath $PreservedModelsPath)) {
        New-Item -ItemType Directory -Path $DistAppDir -Force | Out-Null
        $RestoredModelsPath = Join-Path $DistAppDir "models"
        if (Test-Path -LiteralPath $RestoredModelsPath) {
            throw "Cannot restore preserved models because the destination already exists: $RestoredModelsPath"
        }
        Move-Item -LiteralPath $PreservedModelsPath -Destination $RestoredModelsPath
        Write-Host "Existing models restored."
    }
}

$DistIconPath = Join-Path $DistAppDir "logo.ico"
Copy-Item -LiteralPath $IconPath -Destination $DistIconPath -Force

Write-Host ""
Write-Host "Build complete:"
Write-Host "  dist\Local Whisper Transcriber\Local Whisper Transcriber.exe"
Write-Host "Icon:"
Write-Host "  logo.ico has been applied to the exe and copied into the app folder."
Write-Host ""
Write-Host "Models are intentionally not bundled. Put local models under:"
Write-Host "  dist\Local Whisper Transcriber\models\faster-whisper-large-v3"
Write-Host "  dist\Local Whisper Transcriber\models\pyannote-speaker-diarization-community-1"
Write-Host "  dist\Local Whisper Transcriber\models\whisperx-alignment\zh"
Write-Host "or keep using the app from a project folder that has models\faster-whisper-large-v3."
