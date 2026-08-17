@echo off
REM Launcher for the Thai Sign Language Translator project.
REM
REM NOTE: this file is intentionally written in plain ASCII only.
REM cmd.exe parses .bat files using the console code page, so UTF-8 Thai text
REM inside a batch file gets mangled and breaks the script. Thai documentation
REM lives in README.md and docs/ instead.
REM
REM Usage:
REM     run selftest     check that the installation works (no webcam needed)
REM     run camtest      check the webcam and hand detection (auto-exits)
REM     run collect      record hand gesture samples from the webcam
REM     run train        train the classifier
REM     run translate    translate signs into text in real time
REM
REM Extra arguments are passed through, e.g.  run train --model rf

setlocal
cd /d "%~dp0"

REM Switch the console to UTF-8 so Thai output from Python renders correctly.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set PYTHON=%~dp0.venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo Virtual environment not found at %PYTHON%
    echo See the installation section in README.md
    exit /b 1
)

if "%~1"=="" (
    echo Usage: run [selftest^|camtest^|collect^|train^|translate]
    exit /b 1
)

REM selftest.py and camtest.py live at the project root, not inside src/.
REM
REM These use goto labels rather than parenthesised if-blocks on purpose:
REM cmd.exe expands %1 for an entire ( ... ) block when it parses the block,
REM so a SHIFT inside such a block does not affect %1 on the following lines
REM and the subcommand name gets passed through to Python as a stray argument.
if /i "%~1"=="selftest" goto :root_script
if /i "%~1"=="camtest"  goto :root_script
goto :module

:root_script
set SCRIPT=%~1
shift
"%PYTHON%" %SCRIPT%.py %1 %2 %3 %4 %5
exit /b %errorlevel%

:module
set TARGET=%~1
shift
"%PYTHON%" -m src.%TARGET% %1 %2 %3 %4 %5
exit /b %errorlevel%
