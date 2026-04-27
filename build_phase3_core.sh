#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    cc -O3 -shared -fPIC -o phase3_core.so phase3_core.c
elif [ "$OS" = "Linux" ]; then
    cc -O3 -shared -fPIC -o phase3_core.so phase3_core.c -lm
else
    echo "Unsupported OS: $OS" >&2; exit 1
fi
echo "Built phase3_core.so successfully"
