@echo off
setlocal
chcp 65001 > nul

set "REMOTE_HOST=linux-box"
set "REMOTE_BASE=~/workspace"
set "REMOTE_PROJECT=%REMOTE_BASE%/text-detection"
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "STAGE_DIR=%TEMP%\text-detection-upload-code"

echo ==================================================
echo Uploading project code to %REMOTE_HOST%:%REMOTE_PROJECT%
echo ==================================================

ssh %REMOTE_HOST% "mkdir -p %REMOTE_PROJECT%/scripts"
if errorlevel 1 goto error

if exist "%STAGE_DIR%\" rmdir /s /q "%STAGE_DIR%"
mkdir "%STAGE_DIR%"

robocopy "%PROJECT_DIR%\scripts" "%STAGE_DIR%\scripts" /E /XD __pycache__ /XF *.pyc
if errorlevel 8 goto error

copy "%PROJECT_DIR%\pyproject.toml" "%STAGE_DIR%\" > nul
if errorlevel 1 goto error

copy "%PROJECT_DIR%\uv.lock" "%STAGE_DIR%\" > nul
if errorlevel 1 goto error

scp -r "%STAGE_DIR%\*" %REMOTE_HOST%:%REMOTE_PROJECT%/
if errorlevel 1 goto error

if exist "%STAGE_DIR%\" rmdir /s /q "%STAGE_DIR%"

echo Code upload completed.
goto end

:error
echo [ERROR] Code upload failed.
if exist "%STAGE_DIR%\" rmdir /s /q "%STAGE_DIR%"
exit /b 1

:end
endlocal
