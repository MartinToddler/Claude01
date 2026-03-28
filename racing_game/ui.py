"""
Left and right UI panels (v2).

Left panel changes:
  • Simulation speed slider  x1 – x100  (log-like: stored as int)
  • Num agents slider        5  – 50
  • Max laps slider          1  – 20
  • "Kill All → Next Gen" button

Right panel changes:
  • "Best Lap" shown prominently with gen-best and all-time-best
  • When a car is selected (click): shows that car's individual telemetry,
    lap count, and best lap time instead of the generation-best telemetry
"""

import math
import pygame
from config import (
    C, SCREEN_W, SCREEN_H, LEFT_PANEL_W, RIGHT_PANEL_W,
    TRACK_AREA_X, TRACK_AREA_W, CAR_DEFAULTS, TRACK_HALF_W,
    DEFAULT_NUM_AGENTS, DEFAULT_MAX_LAPS, SIM_SPEED_MAX,
)
from track import TRACK_NAMES

# ── Font helpers ──────────────────────────────────────────────────────────────

_fonts: dict = {}

def _f(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont("monospace", size, bold=bold)
    return _fonts[key]

def _text(surf, txt, x, y, size=13, color=None, bold=False) -> int:
    color = color or C["text"]
    s     = _f(size, bold).render(txt, True, color)
    surf.blit(s, (x, y))
    return s.get_height()


# ── Slider ────────────────────────────────────────────────────────────────────

class Slider:
    H       = 18
    TRACK_H = 4

    def __init__(self, label, vmin, vmax, value, fmt=".0f", color=None):
        self.label    = label
        self.vmin     = vmin
        self.vmax     = vmax
        self.value    = value
        self.fmt      = fmt
        self.color    = color or C["accent"]
        self.rect     = pygame.Rect(0, 0, 100, self.H)
        self._drag    = False

    def set_rect(self, x, y, w):
        self.rect = pygame.Rect(x, y, w, self.H)

    @property
    def norm(self):
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def draw(self, surf):
        r  = self.rect
        cy = r.y + self.H // 2
        # Track
        pygame.draw.rect(surf, C["panel_border"],
                         (r.x, cy - self.TRACK_H//2, r.w, self.TRACK_H), 0, 2)
        fw = max(0, int(r.w * self.norm))
        if fw:
            pygame.draw.rect(surf, self.color,
                             (r.x, cy - self.TRACK_H//2, fw, self.TRACK_H), 0, 2)
        kx = r.x + fw
        pygame.draw.circle(surf, self.color,    (kx, cy), 7)
        pygame.draw.circle(surf, (220, 225, 240), (kx, cy), 4)
        # Labels
        _text(surf, self.label, r.x, r.y - 14, size=11, color=C["text_dim"])
        vs = _f(11).render(format(self.value, self.fmt), True, C["text"])
        surf.blit(vs, (r.right - vs.get_width(), r.y - 14))

    def handle_event(self, event) -> bool:
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.inflate(0, 12).collidepoint(event.pos):
            self._drag = True
            self._update(event.pos[0])
            changed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self._drag = False
        elif event.type == pygame.MOUSEMOTION and self._drag:
            self._update(event.pos[0])
            changed = True
        return changed

    def _update(self, mx):
        t = max(0.0, min(1.0, (mx - self.rect.x) / max(self.rect.w, 1)))
        self.value = self.vmin + t * (self.vmax - self.vmin)


# ── Button ────────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, label, color=None, h=22):
        self.label  = label
        self.color  = color or C["accent"]
        self.rect   = pygame.Rect(0, 0, 80, h)
        self.active = False
        self._h     = h

    def set_rect(self, x, y, w, h=None):
        self.rect = pygame.Rect(x, y, w, h or self._h)

    def draw(self, surf):
        bg = self.color if self.active else C["panel_border"]
        pygame.draw.rect(surf, bg,         self.rect, 0, 4)
        pygame.draw.rect(surf, self.color, self.rect, 1, 4)
        ts = _f(11).render(self.label, True, C["text"])
        surf.blit(ts, (self.rect.centerx - ts.get_width()//2,
                       self.rect.centery - ts.get_height()//2))

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


# ── Gear ratio row ─────────────────────────────────────────────────────────────

class GearRatioRow:
    def __init__(self, values):
        self.sliders = [
            Slider(f"G{i+1}", 0.5, 5.0, v, ".2f", C["warn"])
            for i, v in enumerate(values)
        ]

    @property
    def values(self):
        return [s.value for s in self.sliders]

    def layout(self, x, y, total_w):
        col_w = (total_w - 8) // 2
        for i, s in enumerate(self.sliders):
            col = i % 2; row = i // 2
            s.set_rect(x + col * (col_w + 8), y + row * 36, col_w)

    def draw(self, surf):
        for s in self.sliders: s.draw(surf)

    def handle_event(self, event) -> bool:
        changed = False
        for s in self.sliders:
            if s.handle_event(event): changed = True
        return changed


# ── Left panel ────────────────────────────────────────────────────────────────

class LeftPanel:
    PAD = 14
    W   = LEFT_PANEL_W

    def __init__(self):
        d = CAR_DEFAULTS

        # Car performance sliders
        self.s_power    = Slider("Power (HP)",     150, 1000, d["power"],     ".0f")
        self.s_pressure = Slider("Tyre PSI",        15,   32, d["tyre_pressure"], ".1f", C["good"])
        self.s_downforce= Slider("Downforce (kg)",   0,  250, d["downforce"],  ".0f", C["warn"])
        self.s_risk     = Slider("Driver Risk",      1,   10, d["driver_risk"],".1f", C["bad"])
        self.gear_row   = GearRatioRow(d["gear_ratios"])

        # Simulation control sliders
        self.s_speed     = Slider("Sim Speed ×",    1, SIM_SPEED_MAX, 1, ".0f", C["accent"])
        self.s_num_agents= Slider("Num Cars",        5, 50, DEFAULT_NUM_AGENTS, ".0f", C["warn"])
        self.s_max_laps  = Slider("Max Laps / Gen",  1, 20, DEFAULT_MAX_LAPS,   ".0f", C["good"])

        # Tyre compound buttons
        self.tyre_btns  = {
            t: Button(t.capitalize(),
                      {"soft": C["bad"], "medium": C["warn"], "hard": C["good"]}[t])
            for t in ("soft", "medium", "hard")
        }
        self.tyre_type  = d["tyre_type"]
        self.tyre_btns[self.tyre_type].active = True

        # Track buttons
        self.track_btns = {n: Button(n) for n in TRACK_NAMES}
        self.track_name = TRACK_NAMES[0]
        self.track_btns[self.track_name].active = True

        # Kill-all button
        self.btn_kill = Button("⚡ Kill All → Next Gen", C["bad"], h=26)

        self._layout()

    def _layout(self):
        x  = self.PAD
        w  = self.W - 2 * self.PAD
        y  = 50

        def place(slider):
            nonlocal y
            y += 20
            slider.set_rect(x, y, w)
            y += slider.H + 4

        place(self.s_power)
        y += 2
        place(self.s_pressure)
        place(self.s_downforce)
        place(self.s_risk)

        # Tyre buttons
        y += 12
        _y_tyre = y
        bw = (w - 8) // 3
        for i, btn in enumerate(self.tyre_btns.values()):
            btn.set_rect(x + i * (bw + 4), y, bw, 24)
        y += 30

        # Gear ratios
        y += 8
        self.gear_row.layout(x, y, w)
        y += 36 * 3 + 8

        # Track buttons
        y += 10
        tw = (w - 8) // len(TRACK_NAMES)
        for i, btn in enumerate(self.track_btns.values()):
            btn.set_rect(x + i * (tw + 4), y, tw, 24)
        y += 34

        # Divider
        self._div_y1 = y
        y += 8

        # Sim control sliders
        place(self.s_speed)
        place(self.s_num_agents)
        place(self.s_max_laps)

        # Kill button
        y += 10
        self.btn_kill.set_rect(x, y, w, 28)

    def draw(self, surf):
        pygame.draw.rect(surf, C["panel_bg"], (0, 0, self.W, SCREEN_H))
        pygame.draw.line(surf, C["panel_border"],
                         (self.W - 1, 0), (self.W - 1, SCREEN_H), 1)

        _text(surf, "CAR SETUP", self.PAD, 14, size=14, color=C["accent"], bold=True)

        self.s_power.draw(surf)
        self.s_pressure.draw(surf)
        self.s_downforce.draw(surf)
        self.s_risk.draw(surf)

        tyre0 = list(self.tyre_btns.values())[0]
        _text(surf, "TYRE COMPOUND", self.PAD, tyre0.rect.y - 14,
              size=11, color=C["text_dim"])
        for btn in self.tyre_btns.values(): btn.draw(surf)

        gr0 = self.gear_row.sliders[0]
        _text(surf, "GEAR RATIOS", self.PAD, gr0.rect.y - 14,
              size=11, color=C["text_dim"])
        self.gear_row.draw(surf)

        tr0 = list(self.track_btns.values())[0]
        _text(surf, "TRACK", self.PAD, tr0.rect.y - 14,
              size=11, color=C["text_dim"])
        for btn in self.track_btns.values(): btn.draw(surf)

        # Divider
        pygame.draw.line(surf, C["panel_border"],
                         (self.PAD, self._div_y1), (self.W - self.PAD, self._div_y1), 1)
        _text(surf, "SIMULATION", self.PAD, self._div_y1 + 4,
              size=11, color=C["text_dim"], bold=True)

        self.s_speed.draw(surf)
        self.s_num_agents.draw(surf)
        self.s_max_laps.draw(surf)
        self.btn_kill.draw(surf)

    def handle_event(self, event) -> tuple[bool, bool, bool, bool]:
        """Returns (params_changed, track_changed, speed_changed, kill_all)."""
        pc = tc = sc = ka = False

        # Performance sliders
        for s in (self.s_power, self.s_pressure, self.s_downforce, self.s_risk):
            if s.handle_event(event): pc = True
        if self.gear_row.handle_event(event): pc = True

        # Sim sliders
        if self.s_num_agents.handle_event(event): pc = True
        if self.s_max_laps.handle_event(event):   pc = True
        if self.s_speed.handle_event(event):       sc = True

        # Tyre buttons
        if event.type == pygame.MOUSEBUTTONDOWN:
            for name, btn in self.tyre_btns.items():
                if btn.hit(event.pos) and name != self.tyre_type:
                    self.tyre_type = name
                    for b in self.tyre_btns.values(): b.active = False
                    btn.active = True
                    pc = True

        # Track buttons
        if event.type == pygame.MOUSEBUTTONDOWN:
            for name, btn in self.track_btns.items():
                if btn.hit(event.pos) and name != self.track_name:
                    self.track_name = name
                    for b in self.track_btns.values(): b.active = False
                    btn.active = True
                    tc = True

        # Kill all
        if event.type == pygame.MOUSEBUTTONDOWN and self.btn_kill.hit(event.pos):
            ka = True

        return pc, tc, sc, ka

    @property
    def params(self) -> dict:
        p = dict(CAR_DEFAULTS)
        p["power"]         = self.s_power.value
        p["tyre_type"]     = self.tyre_type
        p["tyre_pressure"] = self.s_pressure.value
        p["downforce"]     = self.s_downforce.value
        p["driver_risk"]   = self.s_risk.value
        p["gear_ratios"]   = self.gear_row.values
        p["num_agents"]    = int(self.s_num_agents.value)
        p["max_laps"]      = int(self.s_max_laps.value)
        return p

    @property
    def sim_speed(self) -> int:
        return max(1, int(self.s_speed.value))


# ── Right panel ───────────────────────────────────────────────────────────────

class RightPanel:
    PAD = 12

    def draw(self, surf, pop, gen_time_s: float,
             selected_idx: int = -1):
        rx = TRACK_AREA_X + TRACK_AREA_W
        rw = SCREEN_W - rx
        pygame.draw.rect(surf, C["panel_bg"], (rx, 0, rw, SCREEN_H))
        pygame.draw.line(surf, C["panel_border"], (rx, 0), (rx, SCREEN_H), 1)

        x  = rx + self.PAD
        iw = rw - 2 * self.PAD
        y  = 14

        _text(surf, "AI STATS", x, y, 14, C["accent"], bold=True)
        y += 28

        # Generation info
        _text(surf, f"Generation  {pop.generation}", x, y, 13)
        y += 18
        alive = len(pop.alive_agents)
        col   = C["good"] if alive > 5 else C["warn"] if alive > 1 else C["bad"]
        _text(surf, f"Alive       {alive}/{len(pop.agents)}", x, y, 13, col)
        y += 18
        _text(surf, f"Gen time    {gen_time_s:.1f}s", x, y, 13)
        y += 28

        # Best lap block
        pygame.draw.rect(surf, C["panel_header"],
                         (x - 4, y - 4, iw + 8, 58), 0, 4)
        _text(surf, "BEST LAP", x, y, 11, C["text_dim"])
        y += 14
        # Gen best
        gen_best_lap = _best_lap_in_gen(pop)
        gbl_s  = f"{gen_best_lap:.2f}s" if gen_best_lap else "---"
        _text(surf, f"This gen    {gbl_s}", x, y, 13, C["warn"])
        y += 16
        # All-time best
        atb_s  = f"{pop.all_time_best_lap:.2f}s" if pop.all_time_best_lap else "---"
        _text(surf, f"All time    {atb_s}", x, y, 13, C["good"])
        y += 22

        # Fitness sparkline
        y += 6
        _text(surf, "FITNESS HISTORY", x, y, 11, C["text_dim"])
        y += 14
        self._sparkline(surf, pop.fitness_history, x, y, iw, 55)
        y += 62

        # Telemetry — selected car or best alive
        if 0 <= selected_idx < len(pop.agents):
            target = pop.agents[selected_idx]
            header = f"CAR #{selected_idx}  {'(alive)' if target.car.alive else '(crashed)'}"
            hcol   = C["warn"]
        else:
            target = pop.best_agent
            header = "BEST CAR TELEMETRY"
            hcol   = C["text_dim"]

        _text(surf, header, x, y, 11, hcol)
        y += 18

        # Lap/time info for selected car
        best_lap_s = target.best_lap_s
        bl_str     = f"{best_lap_s:.2f}s" if best_lap_s else "---"
        _text(surf, f"Laps        {target.laps}", x, y, 13)
        y += 16
        _text(surf, f"Best lap    {bl_str}", x, y, 13, C["good"])
        y += 22

        tel = target.car.telemetry()
        self._gauge(surf, "Speed km/h", tel["speed_kmh"], 320, x, y, iw, C["accent"])
        y += 26
        self._gauge(surf, "RPM",        tel["rpm"],       8000, x, y, iw, C["warn"])
        y += 26
        _text(surf, f"Gear  {tel['gear']}", x, y, 13)
        y += 18

        # Throttle / brake bar
        tb = tel["throttle"]
        _text(surf, "Throttle / Brake", x, y, 11, C["text_dim"])
        y += 13
        mid = x + iw // 2
        if tb >= 0:
            fw = int(iw / 2 * tb)
            pygame.draw.rect(surf, C["good"], (mid, y, max(1, fw), 6), 0, 2)
        else:
            fw = int(iw / 2 * (-tb))
            pygame.draw.rect(surf, C["bad"],  (mid - fw, y, max(1, fw), 6), 0, 2)
        pygame.draw.line(surf, C["text_dim"], (mid, y - 2), (mid, y + 8), 1)
        y += 16

        # Steer bar
        st = tel["steer"]
        _text(surf, "Steering", x, y, 11, C["text_dim"])
        y += 13
        fw = int(iw / 2 * abs(st))
        if st >= 0:
            pygame.draw.rect(surf, C["accent"], (mid, y, max(1, fw), 6), 0, 2)
        else:
            pygame.draw.rect(surf, C["accent"], (mid - fw, y, max(1, fw), 6), 0, 2)
        pygame.draw.line(surf, C["text_dim"], (mid, y - 2), (mid, y + 8), 1)
        y += 18

        # Yaw rate / sideslip
        om = tel.get("omega_deg", 0.0)
        _text(surf, f"Yaw rate    {om:.1f}°/s", x, y, 11, C["text_dim"])
        y += 14
        vy = tel.get("vy", 0.0)
        _text(surf, f"Sideslip    {vy:.1f} m/s", x, y, 11, C["text_dim"])
        y += 20

        # Hotkeys hint
        y = SCREEN_H - 94
        pygame.draw.line(surf, C["panel_border"], (x - 4, y), (x + iw + 4, y), 1)
        y += 6
        _text(surf, "CONTROLS", x, y, 11, C["text_dim"], bold=True)
        y += 14
        for line in ["R — restart generation",
                     "T — toggle rays",
                     "K — kill all / next gen",
                     "SPACE — pause"]:
            _text(surf, line, x, y, 11, C["text_dim"])
            y += 14

    def _gauge(self, surf, label, val, vmax, x, y, w, color):
        _text(surf, label, x, y, 11, C["text_dim"])
        pygame.draw.rect(surf, C["panel_border"], (x, y + 13, w, 6), 0, 2)
        fw = int(w * min(max(val, 0) / max(vmax, 1), 1.0))
        if fw:
            pygame.draw.rect(surf, color, (x, y + 13, fw, 6), 0, 2)
        vs = _f(11).render(f"{val:.0f}", True, C["text"])
        surf.blit(vs, (x + w - vs.get_width(), y))

    def _sparkline(self, surf, history, x, y, w, h):
        pygame.draw.rect(surf, C["panel_header"], (x, y, w, h), 0, 4)
        if len(history) < 2:
            return
        mx = max(history) or 1
        win = history[-(w // 2):]
        pts = []
        for i, v in enumerate(win):
            px = x + int(i / max(len(win) - 1, 1) * (w - 2)) + 1
            py = y + h - int(v / mx * (h - 6)) - 3
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(surf, C["good"], False, pts, 2)
        vs = _f(11).render(f"{history[-1]:.2f}", True, C["good"])
        surf.blit(vs, (x + w - vs.get_width() - 2, y + 2))


def _best_lap_in_gen(pop) -> float | None:
    """Fastest lap achieved in the current generation (None if none)."""
    best = None
    for ag in pop.agents:
        if ag.best_lap_s is not None:
            if best is None or ag.best_lap_s < best:
                best = ag.best_lap_s
    return best
