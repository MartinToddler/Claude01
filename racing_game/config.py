"""Constants, colors, and default parameters for the racing game."""

# ── Screen & Layout ─────────────────────────────────────────────────────────
SCREEN_W = 1400
SCREEN_H = 800
FPS = 60

LEFT_PANEL_W = 300
RIGHT_PANEL_W = 300
TRACK_AREA_W = SCREEN_W - LEFT_PANEL_W - RIGHT_PANEL_W  # 800
TRACK_AREA_H = SCREEN_H                                  # 800
TRACK_AREA_X = LEFT_PANEL_W

# ── Simulation ───────────────────────────────────────────────────────────────
NUM_AGENTS = 20          # cars per generation
MAX_GEN_STEPS = 3600     # max frames per generation (60 s at 60 fps)
STUCK_SPEED = 1.5        # m/s — below this for STUCK_TIME → car dies
STUCK_TIME = 3.0         # seconds
RAYCAST_COUNT = 9        # number of sensor rays per car
RAYCAST_DIST = 200       # pixels, max ray length
TRACK_HALF_W = 60        # pixels, half the driveable track width

# ── Car defaults ─────────────────────────────────────────────────────────────
CAR_DEFAULTS = {
    "power":         450,          # HP
    "gear_ratios":   [3.5, 2.3, 1.6, 1.2, 0.95, 0.78],
    "final_drive":   3.7,
    "tyre_type":     "medium",     # "soft" | "medium" | "hard"
    "tyre_pressure": 23.0,         # PSI  (optimal ~23)
    "downforce":     80,           # kg @ 200 km/h
    "driver_risk":   5,            # 1-10
    "mass":          750,          # kg
    "drag_coeff":    0.32,
    "wheel_radius":  0.33,         # m
    "wheelbase":     2.5,          # m
}

# Tyre grip multiplier by compound
TYRE_GRIP = {
    "soft":   1.45,
    "medium": 1.10,
    "hard":   0.80,
}

# Optimal tyre pressure (PSI) — grip degrades quadratically away from this
OPTIMAL_PRESSURE = 23.0
PRESSURE_GRIP_FALLOFF = 0.004   # grip loss per PSI² deviation

# Downforce: additional grip per (m/s)² unit of speed²
DOWNFORCE_GRIP_K = 0.00003      # grip_bonus = downforce_kg * K * v²

# Power → torque scale (Nm per HP at peak RPM)
HP_TO_NM = 0.72

# ── Genetic Algorithm ────────────────────────────────────────────────────────
NN_INPUTS   = RAYCAST_COUNT + 4  # rays + speed + lat_v + heading_err + gear_norm
NN_HIDDEN1  = 24
NN_HIDDEN2  = 16
NN_OUTPUTS  = 2                  # throttle_brake, steering

ELITE_COUNT    = 4               # top cars kept unchanged
MUTATION_BASE  = 0.12            # base weight mutation std-dev
CROSSOVER_RATE = 0.5             # probability of taking gene from parent A

# ── Colors ───────────────────────────────────────────────────────────────────
C = {
    # Track
    "grass":        (40,  82,  40),
    "track":        (70,  72,  82),
    "kerb_red":     (215, 40,  40),
    "kerb_white":   (240, 240, 240),
    "start_line":   (255, 255, 255),
    "racing_line":  (255, 200,  50, 80),   # RGBA, drawn with alpha

    # UI panels
    "panel_bg":     (18,  20,  30),
    "panel_border": (45,  52,  75),
    "panel_header": (28,  32,  48),

    # Text & accent
    "text":         (210, 218, 235),
    "text_dim":     (120, 130, 155),
    "accent":       (80,  160, 255),
    "good":         (80,  220, 120),
    "warn":         (255, 185,  50),
    "bad":          (255,  75,  75),

    # Background
    "bg":           (14,  15,  22),

    # Car colours (one per agent, last is gold = best)
    "cars": [
        (220,  60,  60), (60,  140, 220), (60,  200, 100),
        (220, 185,  60), (170,  60, 220), (220, 120,  60),
        (60,  210, 210), (220,  60, 160), (140, 220,  60),
        ( 90,  90, 220), (220, 140, 140), (140, 220, 160),
        (210, 210, 140), (160, 140, 200), (140, 180, 200),
        (200, 200, 100), (100, 200, 180), (180, 100, 200),
        (200, 150, 100), (255, 215,   0),   # last = gold best car
    ],
}

# ── Track scale ──────────────────────────────────────────────────────────────
# Tracks are defined in normalised [0,1]×[0,1] space, then scaled to fit the
# track render area (with padding).
TRACK_PADDING = 60   # pixels of margin around the track drawing
