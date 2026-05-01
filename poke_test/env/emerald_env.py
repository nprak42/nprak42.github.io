"""
EmeraldRogueBattleEnv — Gymnasium environment stub for Day 1.

This file sets up the class skeleton and imports. Real battle state reading,
action sending, and reward computation are filled in on Days 2–4.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Will be populated by scripts/extract_addresses.py after ROM build
from memory.addresses import ADDRESSES
from memory.constants import (
    PARTY_SIZE,
    MAX_BATTLERS_COUNT,
    NUM_MOVES,
    NUM_BATTLE_STATS,
    KEY_A, KEY_B, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    B_ACTION_USE_MOVE, B_ACTION_SWITCH, B_ACTION_RUN,
    B_OUTCOME_WON, B_OUTCOME_LOST,
    BattleMon,
)


# ---------------------------------------------------------------------------
# Observation shape constants
# ---------------------------------------------------------------------------
# Per active battler (player + opponent): [hp_ratio, species, type1, type2,
#   atk, def, spa, spd, spe, 8 stat stages, status1_flags(5), status2_flags(3),
#   ability, 4*(move_id, power, type, accuracy, pp_ratio)]
# = 1 + 1 + 2 + 5 + 8 + 5 + 3 + 1 + 4*5 = 46 floats per battler
# 2 battlers (player + opponent) → 92
# Party summary: 6 * (hp_ratio, alive) = 12 (player only for now)
# Battle context: weather(1) + player_side(1) + enemy_side(1) = 3
# Total: 92 + 12 + 3 = 107
OBS_SIZE = 107

# Action space: 4 moves + 6 switches = 10 discrete actions
# Invalid actions are handled via action masking in PPO wrapper
N_ACTIONS = 10  # 0-3: moves, 4-9: switch to party slot 0-5


class EmeraldRogueBattleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rom_path: str | Path, save_state_path: str | Path | None = None):
        super().__init__()

        self.rom_path = Path(rom_path)
        self.save_state_path = Path(save_state_path) if save_state_path else None

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Emulator core — loaded in reset()
        self._core = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._load_core()
        if self.save_state_path and self.save_state_path.exists():
            self._load_save_state(self.save_state_path)
        else:
            self._core.reset()

        self._frame_count = 0
        self._wait_for_input_prompt()

        obs = self._read_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        self._send_battle_action(action)
        self._wait_for_input_prompt()

        obs = self._read_observation()
        reward = self._compute_reward()
        terminated = self._is_battle_over()
        truncated = False
        info = {"action_mask": self._get_action_mask()}

        return obs, reward, terminated, truncated, info

    def close(self):
        self._core = None

    # ------------------------------------------------------------------
    # Emulator setup
    # ------------------------------------------------------------------

    def _load_core(self):
        try:
            import mgba.core
        except ImportError:
            raise RuntimeError(
                "mgba not found. Add libmgba-py to PYTHONPATH.\n"
                "  export PYTHONPATH=/path/to/libmgba-py/mgba:$PYTHONPATH"
            )

        if self._core is not None:
            self._core = None  # let GC collect old core

        core = mgba.core.find_core_for_file(str(self.rom_path))
        if core is None:
            raise RuntimeError(f"libmgba could not find a core for {self.rom_path}")
        core.load_file(str(self.rom_path))
        core.config.update_video_scale(1)
        core.reset()
        self._core = core

    def _load_save_state(self, path: Path):
        # TODO: implement save state loading via mgba VFS
        # mgba.vfs.open(path) → core.load_state()
        raise NotImplementedError("Save state loading not yet implemented")

    # ------------------------------------------------------------------
    # Core loop helpers (stubs for Day 3)
    # ------------------------------------------------------------------

    def _wait_for_input_prompt(self, max_frames: int = 3000):
        """
        Advance frames until the game is waiting for player input.
        Detect the battle input prompt by reading a battle controller flag or
        checking gBattleMainFunc == battle_menu function address.

        TODO (Day 3): implement proper detection. For now, just advance N frames.
        """
        for _ in range(max_frames):
            self._core.run_frame()
            self._frame_count += 1
            # TODO: check if battle menu is active and return early

    def _send_battle_action(self, action: int):
        """
        Translate action index to GBA button inputs and inject them.

        Action mapping:
          0-3  → use move slot 0-3
          4-9  → switch to party slot 0-5

        TODO (Day 3): implement menu navigation. Each action requires
        navigating the cursor to the right position and pressing A.
        """
        # Placeholder: just advance a frame so the loop doesn't hang
        self._core.run_frame()
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Memory reading (stubs for Day 2)
    # ------------------------------------------------------------------

    def _mem_u8(self, addr: int) -> int:
        return self._core.memory.u8[addr]

    def _mem_u16(self, addr: int) -> int:
        return self._core.memory.u16[addr]

    def _mem_u32(self, addr: int) -> int:
        return self._core.memory.u32[addr]

    def _mem_s8(self, addr: int) -> int:
        val = self._mem_u8(addr)
        return val - 256 if val >= 128 else val

    def _read_battle_mon(self, battler_idx: int) -> dict:
        """Read BattlePokemon struct for battler 0..3 from gBattleMons."""
        base = ADDRESSES["gBattleMons"] + battler_idx * BattleMon.SIZE
        if base == battler_idx * BattleMon.SIZE:  # addresses not set yet
            return self._dummy_battle_mon()

        hp      = self._mem_u16(base + BattleMon.HP)
        max_hp  = self._mem_u16(base + BattleMon.MAX_HP)
        species = self._mem_u16(base + BattleMon.SPECIES)
        level   = self._mem_u8(base + BattleMon.LEVEL)
        moves   = [self._mem_u16(base + BattleMon.MOVES + i * 2) for i in range(4)]
        pp      = [self._mem_u8(base + BattleMon.PP + i) for i in range(4)]
        stages  = [self._mem_s8(base + BattleMon.STAT_STAGES + i) for i in range(NUM_BATTLE_STATS)]
        status1 = self._mem_u32(base + BattleMon.STATUS1)
        status2 = self._mem_u32(base + BattleMon.STATUS2)
        type1   = self._mem_u8(base + BattleMon.TYPE1)
        type2   = self._mem_u8(base + BattleMon.TYPE2)
        attack  = self._mem_u16(base + BattleMon.ATTACK)
        defense = self._mem_u16(base + BattleMon.DEFENSE)
        speed   = self._mem_u16(base + BattleMon.SPEED)
        sp_atk  = self._mem_u16(base + BattleMon.SP_ATTACK)
        sp_def  = self._mem_u16(base + BattleMon.SP_DEFENSE)
        ability = self._mem_u8(base + BattleMon.ABILITY)

        return dict(
            hp=hp, max_hp=max(max_hp, 1), species=species, level=level,
            moves=moves, pp=pp, stat_stages=stages,
            status1=status1, status2=status2,
            type1=type1, type2=type2,
            attack=attack, defense=defense, speed=speed,
            sp_atk=sp_atk, sp_def=sp_def, ability=ability,
        )

    def _dummy_battle_mon(self) -> dict:
        return dict(
            hp=1, max_hp=1, species=0, level=1,
            moves=[0, 0, 0, 0], pp=[0, 0, 0, 0],
            stat_stages=[0] * NUM_BATTLE_STATS,
            status1=0, status2=0,
            type1=0, type2=0,
            attack=1, defense=1, speed=1, sp_atk=1, sp_def=1, ability=0,
        )

    def _read_observation(self) -> np.ndarray:
        """
        Build the observation vector. Returns zeros if addresses aren't set yet.
        Full implementation in Day 2.
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # TODO (Day 2): fill in per-battler features, party summary, context
        # Rough layout:
        #   obs[0:46]   = player active mon features
        #   obs[46:92]  = opponent active mon features
        #   obs[92:104] = party summary (6 * [hp_ratio, alive])
        #   obs[104:107]= battle context (weather, player_side, enemy_side)

        return obs

    def _get_action_mask(self) -> np.ndarray:
        """
        Boolean mask over N_ACTIONS.
        Mask out: moves with 0 PP, fainted party members, current active mon.
        Full implementation in Day 3.
        """
        return np.ones(N_ACTIONS, dtype=bool)

    # ------------------------------------------------------------------
    # Reward (stub for Day 4)
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        # TODO (Day 4): implement reward shaping
        return 0.0

    def _is_battle_over(self) -> bool:
        if not ADDRESSES["gBattleOutcome"]:
            return False
        outcome = self._mem_u8(ADDRESSES["gBattleOutcome"])
        return outcome != 0
