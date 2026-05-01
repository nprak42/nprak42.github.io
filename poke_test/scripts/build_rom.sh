#!/usr/bin/env bash
# Build Pokemon Emerald Rogue ROM from source (EX branch)
# Run from repo root: bash scripts/build_rom.sh
#
# Prerequisites:
#   - Xcode Command Line Tools: xcode-select --install
#   - libpng: brew install libpng
#   - devkitARM via devkitPro:
#       sudo dkp-pacman -Sy
#       sudo dkp-pacman -S gba-dev
#       sudo dkp-pacman -S devkitarm-rules

set -euo pipefail

ROM_DIR="deps/pokeemerald-rogue"
BRANCH="ex"  # EX version: expanded mechanics, Gen 1-9, fairy type

# devkitARM environment
export DEVKITPRO="${DEVKITPRO:-/opt/devkitpro}"
export DEVKITARM="${DEVKITARM:-$DEVKITPRO/devkitARM}"
export PATH="$DEVKITARM/bin:$DEVKITPRO/tools/bin:$PATH"

echo "=== Cloning pokeemerald-rogue (branch: $BRANCH) ==="
mkdir -p deps
if [ ! -d "$ROM_DIR" ]; then
    git clone --branch "$BRANCH" https://github.com/Pokabbie/pokeemerald-rogue.git "$ROM_DIR"
else
    echo "Already cloned, skipping."
fi

echo "=== Building ROM ==="
cd "$ROM_DIR"
NCPU=$(sysctl -n hw.logicalcpu)
make -j"$NCPU"

echo ""
echo "=== Build artifacts ==="
ls -lh pokeemerald-rogue.gba pokeemerald-rogue.map 2>/dev/null || echo "WARNING: expected artifacts not found"
echo ""
echo "Copy ROM and map file to your project:"
echo "  cp pokeemerald-rogue.gba ../../roms/"
echo "  cp pokeemerald-rogue.map ../../roms/"
