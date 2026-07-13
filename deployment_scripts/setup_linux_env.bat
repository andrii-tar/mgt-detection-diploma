@echo off
setlocal
chcp 65001 > nul

set "REMOTE_HOST=linux-box"
set "REMOTE_BASE=~/workspace"
set "REMOTE_PROJECT=%REMOTE_BASE%/text-detection"
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "LOCAL_ENV=%PROJECT_DIR%\.env"

echo ==================================================
echo Preparing Linux environment on %REMOTE_HOST%:%REMOTE_PROJECT%
echo ==================================================

ssh %REMOTE_HOST% "mkdir -p %REMOTE_PROJECT%/data"
if errorlevel 1 goto error

if exist "%LOCAL_ENV%" (
    echo Uploading local .env file.
    scp "%LOCAL_ENV%" %REMOTE_HOST%:%REMOTE_PROJECT%/.env
    if errorlevel 1 goto error
) else (
    echo Local .env not found. Creating remote .env from project defaults.
    ssh %REMOTE_HOST% "printf 'TEXT_DETECTION_DATA=%s/data\n' %REMOTE_PROJECT% > %REMOTE_PROJECT%/.env"
    if errorlevel 1 goto error
)

echo Normalizing remote .env data path for Linux.
ssh %REMOTE_HOST% "grep -q '^TEXT_DETECTION_DATA=' %REMOTE_PROJECT%/.env && sed -i 's#^TEXT_DETECTION_DATA=.*#TEXT_DETECTION_DATA=%REMOTE_PROJECT%/data#' %REMOTE_PROJECT%/.env || printf 'TEXT_DETECTION_DATA=%s/data\n' %REMOTE_PROJECT% >> %REMOTE_PROJECT%/.env"
if errorlevel 1 goto error

echo Installing uv if needed and syncing dependencies.
ssh %REMOTE_HOST% "command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh"
if errorlevel 1 goto error

ssh %REMOTE_HOST% "cd %REMOTE_PROJECT% && PATH=\"$HOME/.local/bin:$PATH\" uv sync"
if errorlevel 1 goto error

echo Linux environment is ready.
goto end

:error
echo [ERROR] Linux environment setup failed.
exit /b 1

:end
endlocal
