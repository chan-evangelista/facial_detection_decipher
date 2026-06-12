@echo off
title Venue Emotion Tracker Launcher
cd /d "%~dp0"
echo 🚀 Initializing Isolated Venue Pipeline...

if not exist "python_env\installed.txt" (
    echo 📦 First-time deployment detected. Initializing AI libraries locally...
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    .\python_env\python.exe get-pip.py --no-warn-script-location
    .\python_env\Scripts\pip.exe install -r requirements.txt --target=.\python_env
    del get-pip.py
    echo done > "python_env\installed.txt"
    echo ✅ Local environment established!
)

echo 📷 Initializing Camera Stream and Data Telemetry...
.\python_env\python.exe emotion.py
pause