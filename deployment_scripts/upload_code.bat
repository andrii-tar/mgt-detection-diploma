@echo off
setlocal
chcp 65001 > nul

set "REMOTE_HOST=linux-box"
set "REMOTE_BASE=~/workspace"
set "REMOTE_PROJECT=%REMOTE_BASE%/text-detection"
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

echo ==================================================
echo Uploading project code to %REMOTE_HOST%:%REMOTE_PROJECT%
echo ==================================================

ssh %REMOTE_HOST% "mkdir -p %REMOTE_PROJECT%/scripts %REMOTE_PROJECT%/monitoring %REMOTE_PROJECT%/deployment_scripts"
if errorlevel 1 goto error

scp -r "%PROJECT_DIR%\scripts" %REMOTE_HOST%:%REMOTE_PROJECT%/
if errorlevel 1 goto error

scp -r "%PROJECT_DIR%\monitoring" %REMOTE_HOST%:%REMOTE_PROJECT%/
if errorlevel 1 goto error

scp -r "%PROJECT_DIR%\deployment_scripts" %REMOTE_HOST%:%REMOTE_PROJECT%/
if errorlevel 1 goto error

scp "%PROJECT_DIR%\pyproject.toml" "%PROJECT_DIR%\uv.lock" "%PROJECT_DIR%\docker-compose.yml" "%PROJECT_DIR%\data_description.md" "%PROJECT_DIR%\AGENTS_DATA.md" %REMOTE_HOST%:%REMOTE_PROJECT%/
if errorlevel 1 goto error

echo Code upload completed.
goto end

:error
echo [ERROR] Code upload failed.
exit /b 1

:end
endlocal
