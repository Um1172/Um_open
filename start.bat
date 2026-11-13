@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [INFO] Python仮想環境 ^(venv^) を作成しています...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo [INFO] Local LLM Agent Harness を起動しています...
echo [INFO] ブラウザで http://localhost:8000 を開きます...

start "" cmd /c "timeout /t 2 /nobreak > nul && start http://localhost:8000"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
