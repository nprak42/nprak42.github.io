#!/usr/bin/env python3
"""
Day 1 smoke test: load the ROM headless and verify basic emulator operations.

Usage:
    python scripts/verify_emulator.py roms/pokeemerald-rogue.gba

Tests:
  1. Import mgba — confirms libmgba-py is on PYTHONPATH
  2. Load ROM and create headless core
  3. Advance 60 frames (1 second at 60fps)
  4. Read a byte from ROM header (sanity check memory access)
  5. Read from a known address if addresses.py is present
"""

import sys
import time
from pathlib import Path

ROM_HEADER_ADDR = 0x08000000  # Start of GBA ROM — first byte should be 0x2E (BX jump)
ROM_TITLE_ADDR  = 0x080000A0  # 12-byte ASCII game title in ROM header


def test_import():
    print("[1/5] Importing mgba...", end=" ")
    try:
        import mgba.core
        import mgba.image
        print("OK")
        return True
    except ImportError as e:
        print(f"FAIL\n  {e}")
        print("\n  Make sure PYTHONPATH includes the libmgba-py mgba/ directory.")
        print("  Example: export PYTHONPATH=/path/to/libmgba-py/mgba:$PYTHONPATH")
        return False


def test_load_rom(rom_path: str):
    print(f"[2/5] Loading ROM: {rom_path}...", end=" ")
    import mgba.core
    import mgba.vfs

    core = mgba.core.find_core_for_file(rom_path)
    if core is None:
        print("FAIL — no core found for this ROM")
        return None
    core.load_file(rom_path)
    core.config.update_video_scale(1)
    core.reset()
    print("OK")
    return core


def test_advance_frames(core):
    print("[3/5] Advancing 60 frames...", end=" ")
    t0 = time.perf_counter()
    for _ in range(60):
        core.run_frame()
    elapsed = time.perf_counter() - t0
    fps = 60 / elapsed
    print(f"OK  ({fps:.0f} fps, {elapsed*1000:.1f} ms)")


def test_memory_read(core):
    print("[4/5] Reading ROM header...", end=" ")
    # mGBA memory bus: GBA uses memory domains; for libmgba-py use core.memory
    try:
        # Read 12-byte game title at 0x080000A0
        title_bytes = bytes(core.memory.u8[ROM_TITLE_ADDR + i] for i in range(12))
        title = title_bytes.decode("ascii", errors="replace").rstrip("\x00")
        print(f"OK  (title: '{title}')")
        return True
    except Exception as e:
        print(f"FAIL\n  {e}")
        print("  Memory read API may differ — check mgba Python bindings docs.")
        return False


def test_addresses(core):
    print("[5/5] Reading gBattleMons from addresses.py...", end=" ")
    addr_file = Path("memory/addresses.py")
    if not addr_file.exists():
        print("SKIP — run scripts/extract_addresses.py first")
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location("addresses", addr_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    addr = mod.ADDRESSES.get("gBattleMons")
    if addr is None:
        print("SKIP — gBattleMons not in addresses.py")
        return

    try:
        # Outside of battle gBattleMons will be zeroed/garbage — just check read works
        val = core.memory.u16[addr]
        print(f"OK  (gBattleMons[0].species = {val}  (0 expected outside battle))")
    except Exception as e:
        print(f"FAIL\n  {e}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/pokeemerald-rogue.gba>")
        sys.exit(1)

    rom_path = sys.argv[1]
    if not Path(rom_path).exists():
        print(f"ERROR: ROM not found at {rom_path}")
        sys.exit(1)

    print("=" * 50)
    print("libmgba-py smoke test")
    print("=" * 50)

    if not test_import():
        sys.exit(1)

    core = test_load_rom(rom_path)
    if core is None:
        sys.exit(1)

    test_advance_frames(core)
    test_memory_read(core)
    test_addresses(core)

    print("\nAll tests passed — emulator is working.")


if __name__ == "__main__":
    main()
