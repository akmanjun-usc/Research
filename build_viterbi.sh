#!/bin/bash
# Build the Viterbi C extension as a shared library
# Usage: bash build_viterbi.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    cc -O3 -shared -fPIC -o viterbi_core.so viterbi_core.c
elif [ "$OS" = "Linux" ]; then
    cc -O3 -shared -fPIC -o viterbi_core.so viterbi_core.c -lm
else
    echo "Unsupported OS: $OS" >&2
    exit 1
fi

echo "Built viterbi_core.so successfully"
