@echo off
REM Double-click this to run an interactive smoke test against a running
REM Maya1 server. The actual logic lives in scripts/smoke_test.py (edit that
REM file to tweak the default voice/text/emotion sentences).
python "%~dp0scripts\smoke_test.py"
pause
