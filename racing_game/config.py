"""Constants, colors, and default parameters for the racing game."""

# ── Screen & Layout ─────────────────────────────────────────────────────────
SCREEN_W = 1400
SCREEN_H = 800
FPS = 60

LEFT_PANEL_W = 310
RIGHT_PANEL_W = 310
TRACK_AREA_W = SCREEN_W - LEFT_PANEL_W - RIGHT_PANEL_W  # 780
TRACK_AREA_H = SCREEN_H                                  # 800
TRACK_AREA_X = LEFT_PANEL_W

# ── Simulation ───────────────────────────────────────────────────────────────
DEFAULT_NUM_AGENTS  = 20
DEFAULT_MAX_LAPS    = 5
DEFAULT_CAUTION     = 5    # 1=aggressive, 10=very cautious (throttle cap)

# Discrete sim speed choices (physics steps per rendered frame)
SIM_SPEEDS = [1, 2, 4, 10, 25, 50, 100]
SIM_SPEED_DEFAULT = 1

STUCK_SPEED = 1.5      # m/s
STUCK_TIME  = 4.0      # s

RAYCAST_COUNT = 9
RAYCAST_DIST  = 200    # pixels
TRACK_HALF_W  = 60     # pixels, half driveable width

# ── Car defaults ─────────────────────────────────────────────────────────────
CAR_DEFAULTS = {
    "power":         450,
    "gear_ratios":   [3.5, 2.3, 1.6, 1.2, 0.95, 0.78],
    "final_drive":   3.7,
    "tyre_type":     "medium",
    "tyre_pressure": 23.0,
    "downforce":     80,
    "driver_risk":   5,
    "caution":       DEFAULT_CAUTION,
    "mass":          750,
    "drive_type":    "RWD",    # FWD | RWD | AWD
    "engine_pos":    "front",  # front | mid | rear
    "num_agents":    DEFAULT_NUM_AGENTS,
    "max_laps":      DEFAULT_MAX_LAPS,
}

# Tyre grip coefficient (Pacejka D-factor scale)
TYRE_GRIP = {
    "soft":   1.65,   # racing slick
    "medium": 1.35,   # sport racing
    "hard":   1.05,   # hard slick
}

OPTIMAL_PRESSURE      = 23.0
PRESSURE_GRIP_FALLOFF = 0.004
HP_TO_NM              = 0.72

# ── Engine-position → CG geometry ────────────────────────────────────────────
# L_f = CG to front axle, L_r = CG to rear axle (metres)
# Fzf_static = m·g·L_r / (L_f+L_r)
ENGINE_POS_PARAMS = {
    "front": {"L_f": 0.90, "L_r": 1.70},  # ~65% front weight
    "mid":   {"L_f": 1.20, "L_r": 1.40},  # ~54% front weight
    "rear":  {"L_f": 1.70, "L_r": 0.90},  # ~35% front weight
}

# ── Drive type → axle torque split (front fraction, rear fraction) ────────────
DRIVE_SPLIT = {
    "FWD": (1.0, 0.0),
    "RWD": (0.0, 1.0),
    "AWD": (0.4, 0.6),
}

# ── Fixed vehicle geometry ────────────────────────────────────────────────────
VEH_I_Z      = 1200.0   # kg·m², yaw inertia
VEH_H_CG     = 0.30     # m, CG height
VEH_WHEEL_R  = 0.33     # m
VEH_CD       = 0.32
VEH_A_FRONT  = 1.8

# Pacejka shape parameters
PAC_B = 10.0
PAC_C = 1.90
PAC_E = 0.97

# ── Genetic Algorithm ────────────────────────────────────────────────────────
# NN inputs: 9 rays + vx + vy + heading_err + gear + omega
NN_INPUTS   = RAYCAST_COUNT + 5
NN_HIDDEN1  = 24
NN_HIDDEN2  = 16
NN_OUTPUTS  = 2

ELITE_FRAC     = 0.30   # increased from 0.20 — more elites survive each generation
MUTATION_BASE  = 0.12
CROSSOVER_RATE = 0.50

NN_DEFAULT_HIDDEN = [NN_HIDDEN1, NN_HIDDEN2]   # [24, 16] — used by Brain default

# ── Tooltip texts ─────────────────────────────────────────────────────────────
TOOLTIPS = {
    "power": (
        "Moc silnika w KM. Wyższa moc = większy moment obrotowy na kole\n"
        "→ lepsza akceleracja i prędkość maksymalna.\n"
        "Wzór: T_peak = P × 0.72 Nm/HP. Krzywą momentu modeluje\n"
        "parabolą z szczytem przy ~5000 RPM."
    ),
    "tyre_psi": (
        "Ciśnienie opon w PSI. Optimum: 23 PSI (maksymalna przyczepność).\n"
        "Zbyt wysokie → mniejsza powierzchnia kontaktu ze podłożem.\n"
        "Zbyt niskie → odkształcenie opony.\n"
        "Efekt: przyczepność spada ~20% przy 18 i 30 PSI (krzywa kwadratowa)."
    ),
    "downforce": (
        "Docisk aerodynamiczny w kg przy 200 km/h. Dociska opony do asfaltu.\n"
        "Skaluje się z v². Zwiększa obciążenie normalne → większy D w Pacejce.\n"
        "Podział: 45% przód / 55% tył. Minimalne działanie przy niskich v."
    ),
    "driver_risk": (
        "Agresywność mutacji AI (1=ostrożne, 10=ryzykowne).\n"
        "Skaluje szum Gaussowski dodawany do wag sieci\n"
        "neuronowej przy krzyżowaniu pokoleń.\n"
        "Wysokie ryzyko → szybkie odkrywanie nowych strategii,\n"
        "niskie → stabilna, stopniowa optymalizacja."
    ),
    "gear_ratios": (
        "Przełożenia skrzyni biegów G1–G6.\n"
        "Każde mnożone przez przełożenie mostu (Final Drive).\n"
        "Wyższe przełożenie = więcej momentu, niższa prędkość na biegu.\n"
        "Wzorzec: G1 wysoko dla startu, G6 nisko dla prędkości max."
    ),
    "final_drive": (
        "Przełożenie mostu napędowego (dyferencjał).\n"
        "Mnożnik do wszystkich przełożeń skrzyni. Zakres typowy: 2.5–5.0.\n"
        "Wyższa wartość = lepsza akceleracja, niższa prędkość max.\n"
        "Odpowiednik skali całego układu przeniesienia napędu."
    ),
    "tyre_type": (
        "Miękka (Soft): przyczepność ×1.45 — bardzo szybka, ale zużywa się.\n"
        "Średnia (Medium): ×1.10 — wyważona, domyślna dla treningu AI.\n"
        "Twarda (Hard): ×0.80 — trwała, ale najniższa przyczepność.\n"
        "Wpływa na czynnik D formuły Pacejki (szczytowa siła boczna)."
    ),
    "caution": (
        "Ostrożność kierowcy AI (1=agresywny, 10=bardzo ostrożny).\n"
        "Ogranicza maksymalny gaz: caution=1 → pełny gaz dozwolony,\n"
        "caution=10 → max ~30% gazu (bezpieczna jazda w zakrętach).\n"
        "Niskie wartości = szybkie nauki, ale częste wypadki z toru.\n"
        "Wysokie = wolniejsza nauka, ale więcej aut kończy okrążenia."
    ),
    "num_agents": (
        "Liczba samochodów AI na generację.\n"
        "Więcej agentów = większa pula genów → lepsza dywersyfikacja\n"
        "→ szybsza konwergencja. Mniej = szybsze cykle generacyjne.\n"
        "Użyj przycisków +/− obok wartości aby zmienić liczbę aut."
    ),
    "max_laps": (
        "Generacja kończy się gdy lider ukończy tyle okrążeń.\n"
        "Ranking: wygrywa najlepszy czas okrążenia (nie dystans).\n"
        "Mniej okrążeń = szybsze cykle, ale mniej różnicowania\n"
        "między szybkimi i wolnymi agentami."
    ),
    "drive_type": (
        "FWD (napęd przód): siła napędowa na przedniej osi.\n"
        "Koło Coulomba redukuje przyczepność boczną przodu\n"
        "→ tendencja do podsterowności przy pełnym gazie.\n"
        "RWD (tył): tylna oś napędowa → tendencja do nadsterowności.\n"
        "AWD: 40% przód / 60% tył — najlepsza trakcja, pośrednie cechy."
    ),
    "engine_pos": (
        "Położenie silnika zmienia statyczny rozkład masy (L_f, L_r):\n"
        "Przód (Front): ~65% na przód → dobre prowadzenie, podsterowność.\n"
        "Centralny (Mid): ~54% na przód — najlepszy balans (bolid F1/GT).\n"
        "Tył (Rear): ~35% na przód → duża trakcja tylna, nadsterowność.\n"
        "Parametry L_f i L_r wpływają na kąty poślizgu i przenoszenie masy."
    ),
    "sim_speed": (
        "Mnożnik kroków fizyki na wyrenderowaną klatkę.\n"
        "×1: tryb rzeczywisty — idealne do obserwacji zachowania.\n"
        "×10–×25: szybkie uczenie, nadal częściowo widoczne.\n"
        "×100: maksymalna prędkość treningu, brak szczegółów wizualnych."
    ),
    "track_lines": (
        "Pokazuje lub ukrywa krawężniki i linie graniczne toru.\n"
        "Wyłącz dla czystego widoku trajektorii samochodów.\n"
        "Granice toru są aktywne fizycznie niezależnie od tej opcji."
    ),
    "player_car": (
        "Włącza pojazd sterowany przez gracza (kolor cyjanowy).\n"
        "Sterowanie: ↑↓ gaz/hamulec, ←→ skręt,\n"
        "A = wyższy bieg, Z = niższy bieg, Q = auto/manual skrzynia.\n"
        "Pojazd gracza nie uczestniczy w ewolucji AI — to benchmark\n"
        "do porównania czasu okrążenia z najlepszym agentem."
    ),
}

# ── Colors ───────────────────────────────────────────────────────────────────
C = {
    "grass":        (40,  82,  40),
    "track":        (70,  72,  82),
    "kerb_red":     (215,  40,  40),
    "kerb_white":   (240, 240, 240),
    "start_line":   (255, 255, 255),

    "panel_bg":     (18,  20,  30),
    "panel_border": (45,  52,  75),
    "panel_header": (28,  32,  48),

    "text":         (210, 218, 235),
    "text_dim":     (120, 130, 155),
    "accent":       (80,  160, 255),
    "good":         (80,  220, 120),
    "warn":         (255, 185,  50),
    "bad":          (255,  75,  75),
    "player":       (0,   240, 240),   # cyan for player car
    "bg":           (14,  15,  22),

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
        (128, 255,   0), (  0, 128, 255), (255, 215,   0),
    ],
}

TRACK_PADDING = 60


def car_color(idx: int) -> tuple:
    """Return a distinct color for car index idx (supports 1000+ cars)."""
    palette = C["cars"]
    if idx < len(palette):
        return palette[idx]
    # Generate additional colors via golden-ratio hue distribution
    import math
    h = (idx * 0.618033988749895) % 1.0   # golden ratio
    # HSV → RGB (S=0.75, V=0.9)
    s, v = 0.75, 0.90
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    rgb = [(v, t, p), (q, v, p), (p, v, t),
           (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return tuple(int(c * 255) for c in rgb)
