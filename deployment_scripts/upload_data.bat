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

if not exist "%LOCAL_DATA%\" (
    echo [ERROR] Local data folder not found: %LOCAL_DATA%
    exit /b 1
)

echo ==================================================
echo Uploading data to %REMOTE_HOST%:%REMOTE_PROJECT%/data
echo Local data folder: %LOCAL_DATA%
echo ==================================================

ssh %REMOTE_HOST% "mkdir -p %REMOTE_PROJECT%"
if errorlevel 1 goto error

scp -r "%LOCAL_DATA%" %REMOTE_HOST%:%REMOTE_PROJECT%/
if errorlevel 1 goto error

echo Data upload completed.
goto end

:error
echo [ERROR] Data upload failed.
exit /b 1

:end
endlocal
