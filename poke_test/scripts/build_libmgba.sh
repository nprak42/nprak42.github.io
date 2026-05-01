#!/usr/bin/env bash
# Build libmgba-py on macOS
# Run from the repo root: bash scripts/build_libmgba.sh

set -euo pipefail

LIBMGBA_DIR="deps/libmgba-py"

echo "=== Cloning libmgba-py ==="
mkdir -p deps
if [ ! -d "$LIBMGBA_DIR" ]; then
    git clone https://github.com/hanzi/libmgba-py.git "$LIBMGBA_DIR"
else
    echo "Already cloned, skipping."
fi

echo "=== Building libmgba-py ==="
cd "$LIBMGBA_DIR"
bash build_macos.sh

echo ""
echo "=== Build complete ==="
echo "Add the following to your shell profile (or .env):"
echo "  export LIBMGBA_PY_DIR=$(pwd)"
echo "  export PYTHONPATH=\$LIBMGBA_PY_DIR/mgba:\$PYTHONPATH"
