@echo off
setlocal
title Etiquetador de granulometria

set "TOOL_DIR=%~dp0"

if /I "%CONDA_DEFAULT_ENV%"=="memoria-dexined" goto active_env

where conda >nul 2>nul
if errorlevel 1 goto missing

conda run -n memoria-dexined python "%TOOL_DIR%launcher.py" %*
goto end

:active_env
python "%TOOL_DIR%launcher.py" %*
goto end

:missing
echo No se encontro el entorno memoria-dexined.
echo Ejecuta primero: conda activate memoria-dexined
echo Luego instala las dependencias con: python -m pip install -r requirements.txt
pause

:end
endlocal
