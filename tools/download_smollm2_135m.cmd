@echo off
setlocal enabledelayedexpansion

REM Optional arg: destination directory, default models\llama-3b
set MODEL_DIR=%~1
if "%MODEL_DIR%"=="" set MODEL_DIR=models\llama-3b

where huggingface-cli >nul 2>&1
if errorlevel 1 (
  echo [error] huggingface-cli not found. Install with:
  echo   pip install -U "huggingface_hub[cli]"
  exit /b 1
)

mkdir "%MODEL_DIR%" 2>nul

REM Download config.json, tokenizer.json and model.safetensors (single-file variant) for SmolLM2-135M
huggingface-cli download HuggingFaceTB/SmolLM2-135M --include "config.json" "tokenizer.json" "model.safetensors" --local-dir "%MODEL_DIR%"
if errorlevel 1 (
  echo [error] Download failed.
  exit /b 1
)

echo Files downloaded to %MODEL_DIR%
exit /b 0
