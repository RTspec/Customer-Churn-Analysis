@echo off
REM One-time environment setup for Windows.
REM Double-click this file, or run it from Command Prompt: setup.bat

REM Find a working Python launcher (some machines have "python", others only "py")
where python >nul 2>nul
if %errorlevel%==0 (
    set PYCMD=python
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set PYCMD=py
    ) else (
        echo ERROR: Python was not found on this machine.
        echo Install it from https://www.python.org/downloads/windows/
        echo During install, make sure to check "Add python.exe to PATH".
        pause
        exit /b 1
    )
)

echo Using %PYCMD% to create the virtual environment...
%PYCMD% -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing project dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Setup complete.
echo Each new terminal session, activate the environment with:
echo     venv\Scripts\activate.bat
echo.
pause
