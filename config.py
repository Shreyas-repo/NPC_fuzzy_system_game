"""
Configuration constants for the Village Simulation.
"""

# ─── Display ──────────────────────────────────────────────────────────
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
TITLE = "Village of Minds — NPC Simulation"

# ─── Tile / Map ───────────────────────────────────────────────────────
TILE_SIZE = 32
MAP_WIDTH = 80   # tiles
MAP_HEIGHT = 60  # tiles
WORLD_WIDTH = MAP_WIDTH * TILE_SIZE
WORLD_HEIGHT = MAP_HEIGHT * TILE_SIZE

# ─── Day / Night ──────────────────────────────────────────────────────
DAY_LENGTH_SECONDS = 600          # 10 minutes real time = 1 in-game day
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = DAY_LENGTH_SECONDS / HOURS_PER_DAY  # 25s per hour

# ─── NPC Counts ───────────────────────────────────────────────────────
NPC_COUNTS = {
    "Royal":      2,
    "Noble":      3,
    "Elite":      3,
    "Merchant":   4,
    "Blacksmith": 2,
    "Traveller":  3,
    "Labourer":   5,
    "Peasant":    6,
}

# ─── NPC Speeds (pixels per second) ──────────────────────────────────
NPC_SPEED = {
    "Royal":      30,
    "Noble":      35,
    "Elite":      50,
    "Merchant":   40,
    "Blacksmith": 35,
    "Traveller":  45,
    "Labourer":   38,
    "Peasant":    32,
}

PLAYER_SPEED = 150  # pixels per second

# ─── ML Parameters ───────────────────────────────────────────────────
KMEANS_N_CLUSTERS = 5
KMEANS_UPDATE_INTERVAL = 25.0     # seconds (= 1 in-game hour)
DBSCAN_EPS = 80.0                 # pixel radius for spatial clustering
DBSCAN_MIN_SAMPLES = 2
CLUSTER_INFLUENCE_RATE = 0.02     # how fast cluster members influence each other
ROUTINE_ADAPTATION_RATE = 0.05

# ─── Need Decay Rates (per real second) ──────────────────────────────
NEED_DECAY = {
    "hunger":      0.003,
    "energy":      0.002,
    "social_need": 0.004,
}

NEED_RESTORE_RATE = 0.02  # how fast needs restore at appropriate locations

# ─── Interaction ──────────────────────────────────────────────────────
INTERACTION_RADIUS = 48  # pixels — how close to NPC to interact
CHAT_MAX_HISTORY = 20
OLLAMA_CHAT_TIMEOUT = 18.0          # seconds per generation request
OLLAMA_RESPONSE_MAX_WAIT = 5.0      # seconds before chat fallback
CONVERSATION_CSV_PATH = "conversation_logs.csv"
CONVERSATION_LEARNING_ENABLED = True
CONVERSATION_MODEL_MIN_WORDS = 6
CONVERSATION_MODEL_MAX_WORDS = 22
CONVERSATION_RECENT_MEMORY = 40
CONVERSATION_FINE_TUNE_DIR = "training"
CONVERSATION_EXPORT_EVERY_N_ROWS = 40
CONVERSATION_EXPORT_MIN_ROWS = 80
CONVERSATION_TRAIN_SPLIT = 0.9
CONVERSATION_CSV_REFRESH_INTERVAL = 3.0
CONVERSATION_ADAPT_EVERY_N_ROWS = 12
CONVERSATION_REPETITION_THRESHOLD_INITIAL = 0.82
CONVERSATION_REPETITION_THRESHOLD_MIN = 0.68
CONVERSATION_REPETITION_THRESHOLD_MAX = 0.92
CONVERSATION_PATTERN_MIX_PROB = 0.72
OLLAMA_USE_LLM_EMOTION_ANALYSIS = False
OLLAMA_USE_INTERACTION_EMBEDDINGS = False
OLLAMA_USE_NPC_SNAPSHOT_EMBEDDINGS = False
OLLAMA_CHAT_NUM_PREDICT = 48
OLLAMA_MAX_HISTORY_EXCHANGES = 1
OLLAMA_ENABLE_RESPONSE_CACHE = True
OLLAMA_CACHE_MAX_ENTRIES = 256
OLLAMA_SLOW_RESPONSE_THRESHOLD = 4.0
OLLAMA_SLOW_RESPONSE_COOLDOWN = 30.0

# ─── Interior Transition ─────────────────────────────────────────────
AUTO_ENTER_DOOR_DISTANCE = 38  # pixels
AUTO_ENTER_COOLDOWN = 0.8      # seconds

# ─── Soft Computing Runtime (Research Mode) ─────────────────────────
SOFT_COMPUTING_ENABLED = True
FUZZY_UPDATE_INTERVAL = 0.6          # seconds
EVOLUTION_UPDATE_INTERVAL = 18.0     # seconds
METRICS_UPDATE_INTERVAL = 1.0        # seconds
EVOLUTION_POPULATION_SIZE = 8
EVOLUTION_ELITE_COUNT = 2
EVOLUTION_MUTATION_SIGMA = 0.12
EVOLUTION_MUTATION_PROB = 0.55
EVOLUTION_WEIGHT_MIN = 0.45
EVOLUTION_WEIGHT_MAX = 1.8
FUZZY_MIN_CONFIDENCE = 0.24
FUZZY_ACTION_HYSTERESIS = 0.08
FUZZY_SWITCH_MARGIN = 0.06
FUZZY_MEMBERSHIP = {
    # (start_rise, full_high) for high-membership ramps
    "hunger_high": (0.45, 0.82),
    "low_energy_high": (0.28, 0.7),
    "social_need_high": (0.42, 0.85),
    "threat_high": (0.25, 0.72),
    "low_trust_high": (0.35, 0.8),
    "low_mood_high": (0.32, 0.78),
    "crowd_low": (0.2, 0.55),
}

# ─── Weather & Atmosphere ───────────────────────────────────────────
WEATHER_ENABLED = True
WEATHER_CHANGE_INTERVAL_MIN = 70.0
WEATHER_CHANGE_INTERVAL_MAX = 150.0
NIGHT_FOG_ALPHA = 36
LAMP_GLOW_RADIUS = 44

# ─── Interaction Learning Performance ─────────────────────────────────
INTERACTION_SNAPSHOT_BATCH = 6      # max NPC snapshots embedded per cycle
INTERACTION_CLUSTER_INTERVAL = 24.0 # seconds between unsupervised reclustering

# ─── Social Chatter Hearing / Ambient Overhear ───────────────────────
SOCIAL_CHATTER_HEARING_RADIUS_BASE = 220.0
SOCIAL_CHATTER_HEARING_RADIUS_ALERT = 280.0
SOCIAL_CHATTER_HEAR_PROB_SAME_GROUP = 0.86
SOCIAL_CHATTER_HEAR_PROB_OUTSIDER = 0.56
SOCIAL_CHATTER_TRUST_HIT_PROB = 0.32
SOCIAL_CHATTER_MEMORY_PROB = 0.18
HUD_CHATTER_HEARING_RADIUS = 300.0

# ─── Runtime Auto-Profile Switching ──────────────────────────────────
ADAPTIVE_PROFILE_AUTO_ENABLED = True
ADAPTIVE_PROFILE_SWITCH_COOLDOWN = 24.0

# Switch to aggressive when village dynamics become unstable.
ADAPTIVE_PROFILE_AGGRESSIVE_STABILITY_MAX = 0.52
ADAPTIVE_PROFILE_AGGRESSIVE_TRUST_MAX = 0.50
ADAPTIVE_PROFILE_AGGRESSIVE_CONFLICT_MIN = 0.26

# Return to normal only after recovery (hysteresis avoids rapid flapping).
ADAPTIVE_PROFILE_NORMAL_STABILITY_MIN = 0.66
ADAPTIVE_PROFILE_NORMAL_TRUST_MIN = 0.58
ADAPTIVE_PROFILE_NORMAL_CONFLICT_MAX = 0.16

ADAPTIVE_PROFILE_PRESETS = {
    "normal": {
        "csv_refresh_interval": 6.0,
        "adapt_every_n_rows": 28,
        "repetition_threshold_initial": 0.86,
        "repetition_threshold_min": 0.74,
        "repetition_threshold_max": 0.9,
        "pattern_mix_prob": 0.5,
        "social_hearing_radius_base": 170.0,
        "social_hearing_radius_alert": 220.0,
        "social_hear_prob_same_group": 0.58,
        "social_hear_prob_outsider": 0.28,
        "social_trust_hit_prob": 0.18,
        "social_memory_prob": 0.07,
        "hud_hearing_radius": 220.0,
    },
    "aggressive": {
        "csv_refresh_interval": 3.0,
        "adapt_every_n_rows": 12,
        "repetition_threshold_initial": 0.84,
        "repetition_threshold_min": 0.7,
        "repetition_threshold_max": 0.93,
        "pattern_mix_prob": 0.84,
        "social_hearing_radius_base": 240.0,
        "social_hearing_radius_alert": 320.0,
        "social_hear_prob_same_group": 0.84,
        "social_hear_prob_outsider": 0.62,
        "social_trust_hit_prob": 0.34,
        "social_memory_prob": 0.2,
        "hud_hearing_radius": 320.0,
    },
}

# ─── Camera ───────────────────────────────────────────────────────────
CAMERA_LERP_SPEED = 0.08
SPECTATOR_SPEED = 400
SPECTATOR_ZOOM_MIN = 0.5
SPECTATOR_ZOOM_MAX = 2.0
SPECTATOR_ZOOM_STEP = 0.1

# ─── Color Palette ────────────────────────────────────────────────────
COLORS = {
    # Terrain
    "grass":           (230, 230, 230),
    "grass_dark":      (214, 214, 214),
    "dirt_path":       (170, 130, 80),
    "stone_road":      (150, 150, 150),
    "water":           (60, 120, 190),
    "water_deep":      (40, 90, 160),

    # Buildings
    "wall_stone":      (140, 130, 120),
    "wall_wood":       (139, 90, 43),
    "roof_red":        (178, 60, 50),
    "roof_blue":       (60, 80, 150),
    "roof_brown":      (120, 70, 40),
    "door":            (90, 55, 30),
    "window":          (180, 220, 255),
    "castle_stone":    (110, 105, 100),
    "castle_roof":     (70, 70, 90),

    # NPC Classes
    "npc_royal":       (200, 160, 50),    # Gold
    "npc_noble":       (130, 50, 180),    # Purple
    "npc_elite":       (180, 40, 40),     # Crimson
    "npc_merchant":    (40, 150, 80),     # Green
    "npc_blacksmith":  (80, 80, 80),      # Dark Gray
    "npc_traveller":   (50, 130, 180),    # Teal
    "npc_labourer":    (180, 130, 50),    # Tan
    "npc_peasant":     (140, 110, 80),    # Brown

    # Player
    "player":          (50, 120, 220),    # Bright Blue
    "player_outline":  (30, 80, 170),

    # UI
    "ui_bg":           (20, 20, 30),
    "ui_bg_alpha":     (20, 20, 30, 200),
    "ui_border":       (60, 60, 80),
    "ui_text":         (220, 220, 230),
    "ui_text_dim":     (140, 140, 160),
    "ui_accent":       (80, 160, 255),
    "ui_positive":     (80, 200, 120),
    "ui_negative":     (220, 80, 80),
    "ui_warning":      (220, 180, 50),
    "ui_chat_player":  (100, 180, 255),
    "ui_chat_npc":     (255, 200, 100),

    # General
    "white":           (255, 255, 255),
    "black":           (0, 0, 0),
    "shadow":          (0, 0, 0, 80),
}

# ─── Tile Types ───────────────────────────────────────────────────────
TILE_GRASS = 0
TILE_DIRT = 1
TILE_STONE = 2
TILE_WATER = 3
TILE_WALL = 4
TILE_FLOOR = 5
TILE_FARM = 6
TILE_MARKET = 7

WALKABLE_TILES = {TILE_GRASS, TILE_DIRT, TILE_STONE, TILE_FLOOR, TILE_FARM, TILE_MARKET}

# ─── Building Zone Definitions ────────────────────────────────────────
# (x, y, w, h) in tile coordinates
ZONES = {
    "castle":         (30, 0, 18, 10),
    "lake":           (52, 0, 28, 16),
    "granary_storage": (23, 27, 4, 4),
    "wheat_farm_w1":  (2, 18, 9, 5),
    "wheat_farm_w2":  (12, 18, 9, 5),
    "wheat_farm_s1":  (2, 42, 9, 5),
    "wheat_farm_s2":  (12, 42, 9, 5),
    "noble_house_w1": (30, 16, 5, 5),
    "noble_house_w2": (30, 24, 5, 5),
    "noble_house_w3": (30, 34, 5, 5),
    "noble_house_e1": (45, 16, 5, 5),
    "noble_house_e2": (45, 24, 5, 5),
    "noble_house_e3": (45, 34, 5, 5),
    "peasant_house_1": (5, 24, 5, 5),
    "peasant_house_2": (14, 24, 5, 5),
    "peasant_house_3": (5, 34, 5, 5),
    "peasant_house_4": (14, 34, 5, 5),
    "trader_house_1": (58, 24, 5, 5),
    "trader_house_2": (67, 24, 5, 5),
    "trader_house_3": (58, 34, 5, 5),
    "trader_house_4": (67, 34, 5, 5),
    "town_square":    (39, 29, 2, 2),
}

# ─── Game States ──────────────────────────────────────────────────────
STATE_PLAYING = "playing"
STATE_SPECTATOR = "spectator"
STATE_CHATTING = "chatting"
STATE_PAUSED = "paused"
