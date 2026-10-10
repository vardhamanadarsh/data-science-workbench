@echo off
cd /d "%~dp0"
echo Data Science Workbench - first start downloads about 1.2 GB, please wait...

if not exist "tools\uv.exe" (
  echo Downloading uv...
  if not exist tools mkdir tools
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile 'tools\uv.zip'; Expand-Archive 'tools\uv.zip' -DestinationPath 'tools' -Force; Remove-Item 'tools\uv.zip'"
)

if not exist "tools\uv.exe" (
  echo Could not download uv. Check your internet connection and try again.
  pause
  exit /b 1
)

"tools\uv.exe" sync --python 3.11 --no-dev --extra ml --extra explain --extra reports --extra automl
if errorlevel 1 (
  echo Installation failed. See the message above.
  pause
  exit /b 1
)

"tools\uv.exe" run --no-sync streamlit run app.py
pause