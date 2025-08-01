@echo off
setlocal enabledelayedexpansion

title Mushroom RVC Installer
cd /d "%~dp0"

echo Welcome to the Mushroom RVC Installer!
echo.

set "PRINCIPAL=%cd%"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "ENV_DIR=%PRINCIPAL%\env"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py311_25.1.1-2-Windows-x86_64.exe"
set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"

call :install_miniconda
call :create_conda_env
call :install_dependencies
call :download_ffmpeg

cls
echo Mushroom RVC has been installed successfully!
echo To start Mushroom RVC, please run 'run-Mushroom-RVC.bat'.
echo.
pause
exit /b 0

:install_miniconda
if exist "%MINICONDA_DIR%" (
    echo Miniconda already installed. Skipping installation.
    exit /b 0
)

echo Miniconda not found. Starting download and installation...
powershell -Command "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile 'miniconda.exe'"
if not exist "miniconda.exe" goto :download_error

start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
if errorlevel 1 goto :install_error

del miniconda.exe
echo Miniconda installation complete.
echo.
exit /b 0

:create_conda_env
cls
echo Creating Conda environment...
call "%MINICONDA_DIR%\_conda.exe" create --no-shortcuts -y -k --prefix "%ENV_DIR%" python=3.11
if errorlevel 1 goto :error
echo Conda environment created successfully.
echo.

if exist "%ENV_DIR%\python.exe" (
    echo Installing uv package installer...
    "%ENV_DIR%\python.exe" -m pip install uv
    if errorlevel 1 goto :error
    echo uv installation complete.
    echo.
)
exit /b 0

:install_dependencies
cls
echo Installing dependencies...
call "%MINICONDA_DIR%\condabin\conda.bat" activate "%ENV_DIR%" || goto :error
uv pip install --upgrade setuptools || goto :error
uv pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --upgrade --index-url https://download.pytorch.org/whl/cu128 || goto :error
uv pip install -r "%PRINCIPAL%\requirements.txt" || goto :error
call "%MINICONDA_DIR%\condabin\conda.bat" deactivate
echo Dependencies installation complete.
echo.
exit /b 0

:download_ffmpeg
cls
echo Checking for ffmpeg and ffprobe...
if exist "%PRINCIPAL%\ffmpeg.exe" (
    if exist "%PRINCIPAL%\ffprobe.exe" (
        echo ffmpeg and ffprobe already exist. Skipping download.
        exit /b 0
    )
)

echo Downloading ffmpeg and ffprobe...
powershell -Command "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffmpeg.exe?download=true' -OutFile 'ffmpeg.exe'"
if not exist "ffmpeg.exe" goto :download_error

powershell -Command "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffprobe.exe?download=true' -OutFile 'ffprobe.exe'"
if not exist "ffprobe.exe" goto :download_error

echo ffmpeg and ffprobe downloaded successfully.
echo.
exit /b 0

:download_error
echo.
echo Download failed. Please check your internet connection and try again.
goto :error

:install_error
echo.
echo Miniconda installation failed.
goto :error

:error
echo.
echo An error occurred during installation. Please check the output above for details.
pause
exit /b 1
