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
DEFAULT_NUM_AGENTS = 20      # cars per generation (user-adjustable)
DEFAULT_MAX_LAPS   = 5       # laps before generation ends (user-adjustable)
SIM_SPEED_DEFAULT  = 1       # simulation speed multiplier
SIM_SPEED_MAX      = 100     # maximum speed multiplier

STUCK_SPEED = 1.5            # m/s — below this for STUCK_TIME → car dies
STUCK_TIME  = 4.0            # seconds

RAYCAST_COUNT = 9            # number of sensor rays per car
RAYCAST_DIST  = 200          # pixels, max ray length
TRACK_HALF_W  = 60           # pixels, half the driveable track width

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
    "num_agents":    DEFAULT_NUM_AGENTS,
    "max_laps":      DEFAULT_MAX_LAPS,
}

# Tyre grip multiplier by compound (scales Pacejka D-factor)
TYRE_GRIP = {
    "soft":   1.45,
    "medium": 1.10,
    "hard":   0.80,
}

# Optimal tyre pressure (PSI) — grip degrades quadratically away from this
OPTIMAL_PRESSURE       = 23.0
PRESSURE_GRIP_FALLOFF  = 0.004   # grip loss per PSI² deviation

# Power → peak torque conversion  (Nm per HP)
HP_TO_NM = 0.72

# ── Bicycle model vehicle parameters ─────────────────────────────────────────
# (fixed geometry, user tunes mass/power/tyres/downforce)
VEH_L_F  = 1.2      # m, CG → front axle
VEH_L_R  = 1.4      # m, CG → rear axle
VEH_I_Z  = 1200.0   # kg·m², yaw inertia
VEH_H_CG = 0.30     # m, CG height (for longitudinal weight transfer)
VEH_WHEEL_R  = 0.33 # m, wheel radius
VEH_CD       = 0.32 # drag coefficient
VEH_A_FRONT  = 1.8  # m², frontal area

# Pacejka "Magic Formula" shape parameters (B, C, E)
# D is computed per tyre from mu * Fz
PAC_B = 10.0   # stiffness factor (higher → sharper peak, less sliding)
PAC_C = 1.90   # shape factor
PAC_E = 0.97   # curvature factor

# ── Genetic Algorithm ────────────────────────────────────────────────────────
# NN inputs: 9 rays + vx_norm + vy_norm + heading_err + gear_norm + omega_norm
NN_INPUTS   = RAYCAST_COUNT + 5
NN_HIDDEN1  = 24
NN_HIDDEN2  = 16
NN_OUTPUTS  = 2                  # throttle_brake ∈[-1,1], steer ∈[-1,1]

ELITE_FRAC     = 0.20            # fraction of population kept as elites
MUTATION_BASE  = 0.12            # base weight mutation std-dev
CROSSOVER_RATE = 0.50            # probability of taking gene from parent A

# ── Colors ───────────────────────────────────────────────────────────────────
C = {
    # Track
    "grass":        (40,  82,  40),
    "track":        (70,  72,  82),
    "kerb_red":     (215,  40,  40),
    "kerb_white":   (240, 240, 240),
    "start_line":   (255, 255, 255),

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

    # Car colours — index by agent slot; last entry = gold (best of gen)
    "cars": [
        (220,  60,  60), (60,  140, 220), (60,  200, 100),
        (220, 185,  60), (170,  60, 220), (220, 120,  60),
        (60,  210, 210), (220,  60, 160), (140, 220,  60),
        ( 90,  90, 220), (220, 140, 140), (140, 220, 160),
        (210, 210, 140), (160, 140, 200), (140, 180, 200),
        (200, 200, 100), (100, 200, 180), (180, 100, 200),
        (200, 150, 100), (255, 215,   0), (180, 255, 180),
        (255, 180, 180), (180, 180, 255), (255, 255, 180),
        (180, 255, 255), (255, 180, 255), (200, 230, 200),
        (230, 200, 200), (200, 200, 230), (255, 240, 200),
        (240, 255, 200), (200, 240, 255), (230, 210, 190),
        (210, 230, 190), (190, 210, 230), (255, 200, 150),
        (200, 255, 150), (150, 200, 255), (255, 150, 200),
        (150, 255, 200), (200, 150, 255), (240, 240, 240),
        (160, 160, 160), (100, 100, 100), (255, 255,   0),
        (255,   0, 255), (  0, 255, 255), (255, 128,   0),
        (128, 255,   0), (  0, 128, 255), (255, 215,   0),  # last = gold
    ],
}

# ── Track scale ──────────────────────────────────────────────────────────────
TRACK_PADDING = 60   # pixels of margin around the track drawing
