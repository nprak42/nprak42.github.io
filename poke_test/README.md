# Pokemon Emerald Rogue — RL Battle Agent

PPO agent for battles in Pokemon Emerald Rogue (EX version).

## Day-by-Day Plan

| Day | Status | Goal |
|-----|--------|------|
| 1 | 🔄 | Build libmgba-py + ROM, verify frame advance + memory reads |
| 2 | ⬜ | Extract addresses, build battle state reader, verify reads |
| 3 | ⬜ | Gym env wrapper, action sending, wait-for-input detection |
| 4 | ⬜ | Reward function, PPO training loop, first training run |
| 5 | ⬜ | Reward iteration, obs/action debugging |
| 6–7 | ⬜ | Train, evaluate, iterate |

## Setup

### 1. Build libmgba-py
```bash
bash scripts/build_libmgba.sh
# Then add to shell profile:
export PYTHONPATH=/path/to/deps/libmgba-py/mgba:$PYTHONPATH
```

### 2. Build ROM (requires devkitARM)
```bash
# Install devkitARM first: https://devkitpro.org/wiki/Getting_Started
bash scripts/build_rom.sh
mkdir -p roms
cp deps/pokeemerald-rogue/pokeemerald-rogue.gba roms/
cp deps/pokeemerald-rogue/pokeemerald-rogue.map roms/
```

### 3. Extract addresses
```bash
python scripts/extract_addresses.py roms/pokeemerald-rogue.map
# Writes memory/addresses.py
```

### 4. Verify everything works
```bash
python scripts/verify_emulator.py roms/pokeemerald-rogue.gba
```

### 5. Install Python deps
```bash
pip install -r requirements.txt
```

## Project Structure

```
poke_test/
├── scripts/
│   ├── build_libmgba.sh        # Clone + build libmgba-py
│   ├── build_rom.sh            # Clone + build Emerald Rogue ROM
│   ├── extract_addresses.py    # Parse .map → memory/addresses.py
│   └── verify_emulator.py      # Day 1 smoke test
├── memory/
│   ├── addresses.py            # RAM addresses (auto-generated from .map)
│   └── constants.py            # Struct offsets, bitmasks, enums
├── env/
│   └── emerald_env.py          # Gymnasium environment (EmeraldRogueBattleEnv)
├── agent/                      # PPO training code (Day 4)
├── saves/                      # Save states for episode resets
├── roms/                       # ROM + map file (gitignored)
├── deps/                       # External repos (gitignored)
└── requirements.txt
```

## Architecture Notes

- **Memory reading:** Direct via libmgba-py `core.memory.u8/u16/u32[addr]`
- **Addresses:** From linker `.map` file — rebuild ROM to regenerate
- **Action space:** 10 discrete (4 moves + 6 switches), with invalid action masking
- **Observation:** 107 floats — active battler stats, party summary, battle context
- **Reward:** HP delta + KO bonus + battle win/loss (Day 4)
