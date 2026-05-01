# GBA / Pokemon Emerald battle constants

# --- BattlePokemon struct offsets (from battle.h) ---
class BattleMon:
    SIZE       = 0x58       # sizeof(struct BattlePokemon)
    SPECIES    = 0x00       # u16
    ATTACK     = 0x02       # u16
    DEFENSE    = 0x04       # u16
    SPEED      = 0x06       # u16
    SP_ATTACK  = 0x08       # u16
    SP_DEFENSE = 0x0A       # u16
    MOVES      = 0x0C       # u16[4]  — move IDs
    IVS        = 0x14       # u32     — packed 5-bit IVs
    STAT_STAGES= 0x18       # s8[NUM_BATTLE_STATS=8]
    ABILITY    = 0x20       # u8
    TYPE1      = 0x21       # u8
    TYPE2      = 0x22       # u8
    PP         = 0x24       # u8[4]
    HP         = 0x28       # u16
    LEVEL      = 0x2A       # u8
    FRIENDSHIP = 0x2B       # u8
    MAX_HP     = 0x2C       # u16
    ITEM       = 0x2E       # u16
    NICKNAME   = 0x30       # u8[11]
    STATUS1    = 0x4C       # u32  — burn, poison, sleep, freeze, paralysis
    STATUS2    = 0x50       # u32  — confusion, flinch, etc.


# --- Pokemon struct offsets (party mon, 104 bytes) ---
class PartyMon:
    SIZE   = 104
    HP     = 0x58   # u16  current HP (after BoxPokemon 80 bytes + status u32 + level + mail)
    MAX_HP = 0x5C   # u16
    STATUS = 0x54   # u32  (same layout as BattleMon.STATUS1)
    LEVEL  = 0x5A   # u8


# --- Battle actions ---
B_ACTION_USE_MOVE = 0
B_ACTION_USE_ITEM = 1
B_ACTION_SWITCH   = 2
B_ACTION_RUN      = 3

# --- Battle outcomes ---
B_OUTCOME_WON       = 1
B_OUTCOME_LOST      = 2
B_OUTCOME_DREW      = 3
B_OUTCOME_RAN       = 4
B_OUTCOME_PLAYER_TELEPORTED = 5
B_OUTCOME_MON_FLED  = 6
B_OUTCOME_CAUGHT    = 7
B_OUTCOME_NO_SAFARI_BALLS = 8
B_OUTCOME_FORFEITED = 9
B_OUTCOME_MON_TELEPORTED = 10

# --- Battle type flags ---
BATTLE_TYPE_DOUBLE      = (1 << 0)
BATTLE_TYPE_LINK        = (1 << 1)
BATTLE_TYPE_WILD        = (1 << 3)
BATTLE_TYPE_TRAINER     = (1 << 4)
BATTLE_TYPE_FIRST_BATTLE= (1 << 5)
BATTLE_TYPE_ROAMER      = (1 << 8)
BATTLE_TYPE_LEGENDARY   = (1 << 9)

# --- Status1 bitmasks ---
STATUS1_SLEEP_MASK  = 0x7        # bits 0-2: sleep turns
STATUS1_POISON      = (1 << 3)
STATUS1_BURN        = (1 << 4)
STATUS1_FREEZE      = (1 << 5)
STATUS1_PARALYSIS   = (1 << 6)
STATUS1_TOXIC       = (1 << 7)   # bad poison

# --- Status2 bitmasks ---
STATUS2_CONFUSION   = (0x7 << 0)
STATUS2_FLINCHED    = (1 << 3)
STATUS2_UPROAR      = (0x7 << 4)
STATUS2_BIDE        = (0x3 << 8)
STATUS2_LOCK_CONFUSE= (1 << 10)
STATUS2_MULTIPLETURNS = (1 << 11)
STATUS2_WRAPPED     = (0x7 << 12)
STATUS2_INFATUATION = (0xF << 16)
STATUS2_FOCUS_ENERGY= (1 << 20)
STATUS2_TRANSFORMED = (1 << 21)
STATUS2_RECHARGE    = (1 << 22)
STATUS2_RAGE        = (1 << 23)
STATUS2_SUBSTITUTE  = (1 << 24)
STATUS2_DEFENSE_CURL= (1 << 28)
STATUS2_ROOTING     = (1 << 29)

# --- Battle weather ---
WEATHER_NONE             = 0
WEATHER_RAIN_TEMPORARY   = (1 << 0)
WEATHER_RAIN_PERMANENT   = (1 << 1)
WEATHER_SANDSTORM_TEMPORARY = (1 << 2)
WEATHER_SANDSTORM_PERMANENT = (1 << 3)
WEATHER_SUN_TEMPORARY    = (1 << 4)
WEATHER_SUN_PERMANENT    = (1 << 5)
WEATHER_HAIL_TEMPORARY   = (1 << 6)
WEATHER_HAIL_PERMANENT   = (1 << 7)

# --- Side statuses ---
SIDE_STATUS_REFLECT      = (1 << 0)
SIDE_STATUS_LIGHTSCREEN  = (1 << 1)
SIDE_STATUS_SPIKES       = (1 << 4)
SIDE_STATUS_SAFEGUARD    = (1 << 5)
SIDE_STATUS_FUTUREATTACK = (1 << 6)
SIDE_STATUS_MIST         = (1 << 8)
SIDE_STATUS_SPIKES2      = (1 << 9)   # 2 layers
SIDE_STATUS_SPIKES3      = (1 << 10)  # 3 layers
SIDE_STATUS_STEALTH_ROCK = (1 << 11)  # EX only

# --- Party size / battler count ---
PARTY_SIZE          = 6
MAX_BATTLERS_COUNT  = 4
NUM_BATTLE_STATS    = 8   # Atk, Def, SpAtk, SpDef, Spd, Acc, Eva, (unused)
NUM_MOVES           = 4

# --- GBA buttons (for menu navigation) ---
KEY_A      = (1 << 0)
KEY_B      = (1 << 1)
KEY_SELECT = (1 << 2)
KEY_START  = (1 << 3)
KEY_RIGHT  = (1 << 4)
KEY_LEFT   = (1 << 5)
KEY_UP     = (1 << 6)
KEY_DOWN   = (1 << 7)
KEY_R      = (1 << 8)
KEY_L      = (1 << 9)
