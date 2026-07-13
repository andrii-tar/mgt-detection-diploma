@echo off
setlocal
chcp 65001 > nul

set "REMOTE_HOST=linux-box"
set "REMOTE_BASE=~/workspace"
set "REMOTE_PROJECT=%REMOTE_BASE%/text-detection"
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "LOCAL_DATA=%PROJECT_DIR%\data"

if exist "%PROJECT_DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%PROJECT_DIR%\.env") do (
        if "%%A"=="TEXT_DETECTION_DATA" set "LOCAL_DATA=%%B"
    )
)
set "LOCAL_DATA=%LOCAL_DATA:/=\%"

echo ==================================================
echo Downloading data from %REMOTE_HOST%:%REMOTE_PROJECT%/data
echo Local data folder: %LOCAL_DATA%
echo ==================================================

if not exist "%LOCAL_DATA%\" mkdir "%LOCAL_DATA%"
scp -r %REMOTE_HOST%:%REMOTE_PROJECT%/data/* "%LOCAL_DATA%\"
if errorlevel 1 goto error

echo Data download completed.
goto end

:error
echo [ERROR] Data download failed.
exit /b 1

:end
endlocal
