@echo off
title Chitti LLM Fine-Tuning Monitor
cd /d "%~dp0"

echo ====================================================================
echo                 CHITTI LLM FINE-TUNING PIPELINE
echo ====================================================================
echo.

if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PY_BIN=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
) else (
    set "PY_BIN=python"
)

echo [1/2] Ensuring Training Dataset is up to date...
"%PY_BIN%" scripts\generate_training_dataset.py
if errorlevel 1 (
    echo [ERROR] Dataset generation failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting Training Run with Real-Time Progress Monitoring...
echo Base model: Qwen/Qwen2.5-Coder-3B-Instruct (Optimized for RTX 5050 8GB VRAM)
echo.
"%PY_BIN%" scripts\train_chitti_lora.py --model_id Qwen/Qwen2.5-Coder-3B-Instruct --epochs 3 --batch_size 1 --grad_accum 4

echo.
echo ====================================================================
echo                      TRAINING RUN COMPLETE
echo ====================================================================
pause
