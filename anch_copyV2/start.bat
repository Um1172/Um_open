@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment ^(venv^)...
    python -m venv venv
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
)

"venv\Scripts\python.exe" -c "print('[INFO] クリスタルLM (CrystalLM) を起動しています...'); print('[INFO] サーバー起動後、ブラウザで http://localhost:8000 を開きます...')"

start "" /b "venv\Scripts\python.exe" -c "import socket,time,webbrowser; any(((lambda s: (s.settimeout(0.5), s.connect_ex(('127.0.0.1', 8000)) == 0, s.close())[1])(socket.socket()) and (webbrowser.open('http://localhost:8000') or True)) or time.sleep(0.5) for _ in range(60))"
"venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
