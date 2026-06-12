@echo off
setlocal enabledelayedexpansion

REM Change to repo root (folder of this script is tools\)
cd /d "%~dp0.."

REM Ensure folders exist
if not exist compare mkdir compare
if not exist docs mkdir docs
if not exist docs\summaries mkdir docs\summaries

REM Copy filtered trees (exclude VCS/build folders)
robocopy candle compare\candle /E /XD .git target node_modules >nul
robocopy idea_sources\candle-vllm compare\candle-vllm /E /XD .git target node_modules >nul
robocopy idea_sources\atoma-infer compare\atoma-infer /E /XD .git target node_modules >nul

REM Generate filtered diffs
 git --no-pager diff --no-index compare\candle compare\candle-vllm > docs\diff-filtered-candle_vs_candle-vllm.patch
 git --no-pager diff --no-index compare\candle compare\atoma-infer > docs\diff-filtered-candle_vs_atoma-infer.patch

REM File change lists
 git --no-pager diff --no-index --name-status compare\candle compare\candle-vllm > docs\summaries\candle-vllm-files.txt
 git --no-pager diff --no-index --name-status compare\candle compare\atoma-infer > docs\summaries\atoma-infer-files.txt

echo Done. Filtered diffs and summaries updated under docs\
exit /b 0
