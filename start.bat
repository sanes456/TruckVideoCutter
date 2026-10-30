@echo off
setlocal
cd /d "%~dp0"

title Truck Video Cutter

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PATH=%~dp0bin;%PATH%"

echo ============================================================
echo      Truck Video Cutter - 4K Drone and Cargo Engine
echo ============================================================
echo.

if not exist "%VENV_PY%" goto SETUP_VENV

"%VENV_PY%" -c "import sys" >nul 2>&1
if errorlevel 1 goto REBUILD_VENV

goto RUN_APP

:REBUILD_VENV
echo [ИНФО] Обнаружен перенос папки на другой компьютер или диск.
echo [ИНФО] Перенастройка окружения для этой системы...
rmdir /s /q "%VENV_DIR%" >nul 2>&1

:SETUP_VENV
echo [1/3] Поиск Python в системе...
python --version >nul 2>&1
if not errorlevel 1 (
    set "BASE_PYTHON=python"
    goto CREATE_VENV
)

py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "BASE_PYTHON=py -3"
    goto CREATE_VENV
)

echo [ВНИМАНИЕ] Python не найден на этом компьютере.
echo Попытка установки через winget...
winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
python --version >nul 2>&1
if not errorlevel 1 (
    set "BASE_PYTHON=python"
    goto CREATE_VENV
)

echo.
echo [ОШИБКА] Python не установлен!
echo Пожалуйста, установите Python с официального сайта: https://www.python.org/downloads/
echo Обязательно отметьте галочку: Add python.exe to PATH
echo.
pause
exit /b 1

:CREATE_VENV
echo [2/3] Создание виртуального окружения...
%BASE_PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ОШИБКА] Не удалось создать виртуальное окружение.
    pause
    exit /b 1
)

echo [3/3] Установка библиотек...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ОШИБКА] Ошибка при установке библиотек.
    pause
    exit /b 1
)
echo [ГОТОВО] Окружение настроено!
echo.

:RUN_APP
echo Запуск приложения Truck Video Cutter...
"%VENV_PY%" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Приложение завершилось с ошибкой.
    pause
)
