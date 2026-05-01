#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OS="$(uname -s)"

case "$OS" in
    Darwin)
        cc -O3 -shared -fPIC -o phase3_core.so phase3_core.c
        ;;
    Linux)
        cc -O3 -shared -fPIC -o phase3_core.so phase3_core.c -lm
        ;;
    MINGW*|MSYS*|CYGWIN*)
        # Windows (Git Bash / MSYS2 / MinGW)
        cc -O3 -shared -o phase3_core.dll phase3_core.c
        ;;
    *)
        echo "Unsupported OS: $OS" >&2
        exit 1
        ;;
esac

echo "Build completed for $OS"
