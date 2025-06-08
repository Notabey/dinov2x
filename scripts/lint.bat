@echo off
REM filepath: scripts\lint.bat

if not "%1"=="" (
    echo linting "%1"
)

echo running black
if not "%1"=="" (
    black "%1"
) else (
    black dinov2
)

echo running flake8
if not "%1"=="" (
    flake8 "%1"
) else (
    flake8
)

echo running pylint
if not "%1"=="" (
    pylint "%1"
) else (
    pylint dinov2
)

exit /b 0