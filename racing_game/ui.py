"""
Left and right UI panels.

Left panel  — car parameter sliders (power, gear ratios, tyre, pressure,
              downforce, driver_risk) + track selector.
Right panel — generation stats, best lap, fitness history sparkline,
              live telemetry of the best car.
"""

import math
import pygame
from config import (
    C, SCREEN_W, SCREEN_H, LEFT_PANEL_W, RIGHT_PANEL_W,
    TRACK_AREA_X, CAR_DEFAULTS, TRACK_HALF_W,
)
from track import TRACK_NAMES

# ── Tiny font helpers ─────────────────────────────────────────────────────────

_fonts: dict[int, pygame.font.Font] = {}

def _f(size: int) -> pygame.font.Font:
    if size not in _fonts:
        _fonts[size] = pygame.font.SysFont("monospace", size)
    return _fonts[size]

def _text(surf, txt, x, y, size=13, color=None, bold=False):
    color = color or C["text"]
    f = pygame.font.SysFont("monospace", size, bold=bold)
    s = f.render(txt, True, color)
    surf.blit(s, (x, y))
    return s.get_height()


# ── Slider widget ─────────────────────────────────────────────────────────────

class Slider:
    H = 18
    TRACK_H = 4

    def __init__(self, label: str, vmin: float, vmax: float, value: float,
                 fmt: str = ".0f", color=None):
        self.label = label
        self.vmin  = vmin
        self.vmax  = vmax
        self.value = value
        self.fmt   = fmt
        self.color = color or C["accent"]
        self.rect  = pygame.Rect(0, 0, 100, self.H)  # set in layout
        self._dragging = False

    def set_rect(self, x, y, w):
        self.rect = pygame.Rect(x, y, w, self.H)

    @property
    def norm(self) -> float:
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def draw(self, surf):
        r   = self.rect
        cx  = r.x
        cy  = r.y + self.H // 2

        # Track background
        pygame.draw.rect(surf, C["panel_border"],
                         (cx, cy - self.TRACK_H//2, r.w, self.TRACK_H), 0, 2)
        # Fill
        fill_w = int(r.w * self.norm)
        if fill_w > 0:
            pygame.draw.rect(surf, self.color,
                             (cx, cy - self.TRACK_H//2, fill_w, self.TRACK_H), 0, 2)
        # Knob
        kx = cx + fill_w
        pygame.draw.circle(surf, self.color, (kx, cy), 7)
        pygame.draw.circle(surf, (220, 225, 240), (kx, cy), 4)

        # Label left, value right
        val_str = format(self.value, self.fmt)
        _text(surf, self.label, cx, r.y - 14, size=11, color=C["text_dim"])
        ts = _f(11).render(val_str, True, C["text"])
        surf.blit(ts, (r.right - ts.get_width(), r.y - 14))

    def handle_mouse_down(self, pos):
        if self.rect.inflate(0, 12).collidepoint(pos):
            self._dragging = True
            self._update(pos[0])
            return True
        return False

    def handle_mouse_up(self):
        self._dragging = False

    def handle_mouse_move(self, pos):
        if self._dragging:
            self._update(pos[0])

    def _update(self, mx: int):
        t = (mx - self.rect.x) / max(self.rect.w, 1)
        t = max(0.0, min(1.0, t))
        self.value = self.vmin + t * (self.vmax - self.vmin)


# ── Button ────────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, label: str, color=None):
        self.label  = label
        self.color  = color or C["accent"]
        self.rect   = pygame.Rect(0, 0, 80, 22)
        self.active = False

    def set_rect(self, x, y, w, h=22):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surf):
        bg = self.color if self.active else C["panel_border"]
        pygame.draw.rect(surf, bg, self.rect, 0, 4)
        pygame.draw.rect(surf, self.color, self.rect, 1, 4)
        ts = _f(11).render(self.label, True, C["text"])
        surf.blit(ts, (self.rect.centerx - ts.get_width()//2,
                       self.rect.centery - ts.get_height()//2))

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


# ── Gear-ratio row ─────────────────────────────────────────────────────────────

class GearRatioRow:
    """6 small sliders in a 2-column grid."""

    def __init__(self, values: list[float]):
        self.sliders = [
            Slider(f"G{i+1}", 0.5, 5.0, v, ".2f", C["warn"])
            for i, v in enumerate(values)
        ]

    @property
    def values(self) -> list[float]:
        return [s.value for s in self.sliders]

    def layout(self, x, y, total_w):
        col_w = (total_w - 8) // 2
        for i, s in enumerate(self.sliders):
            col = i % 2
            row = i // 2
            sx  = x + col * (col_w + 8)
            sy  = y + row * 36
            s.set_rect(sx, sy, col_w)

    def draw(self, surf):
        for s in self.sliders:
            s.draw(surf)

    def handle_events(self, event):
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            for s in self.sliders:
                if s.handle_mouse_down(event.pos):
                    changed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            for s in self.sliders: s.handle_mouse_up()
        elif event.type == pygame.MOUSEMOTION:
            for s in self.sliders:
                s.handle_mouse_move(event.pos)
                if s._dragging: changed = True
        return changed


# ── Left panel ────────────────────────────────────────────────────────────────

class LeftPanel:
    PAD = 14
    SLIDER_W = LEFT_PANEL_W - 28

    def __init__(self):
        d = CAR_DEFAULTS
        self.s_power    = Slider("Power (HP)",    150, 1000, d["power"],    ".0f")
        self.s_pressure = Slider("Tyre PSI",       15,   32, d["tyre_pressure"], ".1f", C["good"])
        self.s_downforce= Slider("Downforce (kg)",  0,  250, d["downforce"], ".0f", C["warn"])
        self.s_risk     = Slider("Driver Risk",     1,   10, d["driver_risk"], ".1f", C["bad"])
        self.gear_row   = GearRatioRow(d["gear_ratios"])

        # Tyre type buttons
        self.tyre_btns = {
            t: Button(t.capitalize(),
                      {"soft": C["bad"], "medium": C["warn"], "hard": C["good"]}[t])
            for t in ("soft", "medium", "hard")
        }
        self.tyre_type = d["tyre_type"]
        self.tyre_btns[self.tyre_type].active = True

        # Track selector buttons
        self.track_btns = {n: Button(n) for n in TRACK_NAMES}
        self.track_name = TRACK_NAMES[0]
        self.track_btns[self.track_name].active = True

        self._params_changed = False
        self._track_changed  = False

        self._layout()

    def _layout(self):
        x = self.PAD
        w = self.SLIDER_W
        y = 50

        def place(slider, offset=0):
            nonlocal y
            y += 20 + offset
            slider.set_rect(x, y, w)
            y += slider.H + 4

        place(self.s_power)
        y += 4
        place(self.s_pressure)
        place(self.s_downforce)
        place(self.s_risk)

        # Tyre buttons
        y += 14
        bw = (w - 8) // 3
        for i, (name, btn) in enumerate(self.tyre_btns.items()):
            btn.set_rect(x + i * (bw + 4), y, bw, 24)
        y += 30

        # Gear ratio row
        y += 10
        self.gear_row.layout(x, y, w)
        y += 36 * 3 + 10

        # Track buttons
        y += 14
        tw = (w - 8) // len(TRACK_NAMES)
        for i, (name, btn) in enumerate(self.track_btns.items()):
            btn.set_rect(x + i * (tw + 4), y, tw, 24)

    def draw(self, surf):
        # Panel background
        pygame.draw.rect(surf, C["panel_bg"], (0, 0, LEFT_PANEL_W, SCREEN_H))
        pygame.draw.line(surf, C["panel_border"], (LEFT_PANEL_W-1, 0), (LEFT_PANEL_W-1, SCREEN_H), 1)

        # Title
        _text(surf, "CAR SETUP", self.PAD, 14, size=14, color=C["accent"], bold=True)

        self.s_power.draw(surf)
        self.s_pressure.draw(surf)
        self.s_downforce.draw(surf)
        self.s_risk.draw(surf)

        _text(surf, "TYRE COMPOUND", self.PAD,
              list(self.tyre_btns.values())[0].rect.y - 14, size=11, color=C["text_dim"])
        for btn in self.tyre_btns.values():
            btn.draw(surf)

        _text(surf, "GEAR RATIOS", self.PAD,
              self.gear_row.sliders[0].rect.y - 14, size=11, color=C["text_dim"])
        self.gear_row.draw(surf)

        _text(surf, "TRACK", self.PAD,
              list(self.track_btns.values())[0].rect.y - 14, size=11, color=C["text_dim"])
        for btn in self.track_btns.values():
            btn.draw(surf)

    def handle_event(self, event) -> tuple[bool, bool]:
        """Returns (params_changed, track_changed)."""
        pc = tc = False

        # Sliders
        for s in (self.s_power, self.s_pressure, self.s_downforce, self.s_risk):
            if event.type == pygame.MOUSEBUTTONDOWN and s.handle_mouse_down(event.pos): pc = True
            elif event.type == pygame.MOUSEBUTTONUP: s.handle_mouse_up()
            elif event.type == pygame.MOUSEMOTION:
                s.handle_mouse_move(event.pos)
                if s._dragging: pc = True

        if self.gear_row.handle_events(event): pc = True

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

        return pc, tc

    @property
    def params(self) -> dict:
        from config import CAR_DEFAULTS
        p = dict(CAR_DEFAULTS)
        p["power"]         = self.s_power.value
        p["tyre_type"]     = self.tyre_type
        p["tyre_pressure"] = self.s_pressure.value
        p["downforce"]     = self.s_downforce.value
        p["driver_risk"]   = self.s_risk.value
        p["gear_ratios"]   = self.gear_row.values
        return p


# ── Right panel ───────────────────────────────────────────────────────────────

class RightPanel:
    PAD = 12
    X   = TRACK_AREA_X + TRACK_AREA_X   # = LEFT_PANEL_W + TRACK_W (set in draw)

    def draw(self, surf, pop, gen_time_s: float):
        from config import TRACK_AREA_W
        rx = TRACK_AREA_X + TRACK_AREA_W
        rw = SCREEN_W - rx
        pygame.draw.rect(surf, C["panel_bg"], (rx, 0, rw, SCREEN_H))
        pygame.draw.line(surf, C["panel_border"], (rx, 0), (rx, SCREEN_H), 1)

        x  = rx + self.PAD
        y  = 14
        iw = rw - 2 * self.PAD

        _text(surf, "AI STATS", x, y, size=14, color=C["accent"], bold=True)
        y += 28

        # Generation info
        gen_str = f"Generation  {pop.generation}"
        _text(surf, gen_str, x, y, size=13, color=C["text"])
        y += 18
        alive = len(pop.alive_agents)
        col = C["good"] if alive > 5 else C["warn"] if alive > 1 else C["bad"]
        _text(surf, f"Alive       {alive}/{len(pop.agents)}", x, y, size=13, color=col)
        y += 18
        _text(surf, f"Gen time    {gen_time_s:.1f}s", x, y, size=13, color=C["text"])
        y += 28

        # Best fitness
        bf = pop.gen_best_fitness
        _text(surf, f"Best fit    {bf:.3f}", x, y, size=13, color=C["good"])
        y += 18

        # Best lap
        bl = pop.all_time_best_lap
        bl_str = f"{bl:.2f}s" if bl else "---"
        _text(surf, f"Best lap    {bl_str}", x, y, size=13, color=C["warn"])
        y += 30

        # Fitness sparkline
        _text(surf, "FITNESS HISTORY", x, y, size=11, color=C["text_dim"])
        y += 14
        self._draw_sparkline(surf, pop.fitness_history, x, y, iw, 60)
        y += 70

        # Live telemetry of best car
        best = pop.best_agent
        if best.car.alive:
            _text(surf, "BEST CAR TELEMETRY", x, y, size=11, color=C["text_dim"])
            y += 18
            tel = best.car.telemetry()

            def gauge(label, val, vmax, vy, color=C["accent"]):
                nonlocal y
                _text(surf, label, x, vy, size=11, color=C["text_dim"])
                bw = iw
                pygame.draw.rect(surf, C["panel_border"], (x, vy + 13, bw, 6), 0, 2)
                fw = int(bw * min(val / max(vmax, 1), 1.0))
                if fw > 0:
                    pygame.draw.rect(surf, color, (x, vy + 13, fw, 6), 0, 2)
                vs = _f(11).render(f"{val:.0f}", True, C["text"])
                surf.blit(vs, (x + bw - vs.get_width(), vy))

            gauge("Speed (km/h)", tel["speed_kmh"],  320,  y, C["accent"])
            y += 26
            gauge("RPM",          tel["rpm"],        8000, y, C["warn"])
            y += 26

            _text(surf, f"Gear  {tel['gear']}", x, y, size=13, color=C["text"])
            y += 18

            # Throttle / brake bar
            tb = tel["throttle"]
            _text(surf, "Throttle / Brake", x, y, size=11, color=C["text_dim"])
            y += 13
            mid = x + iw // 2
            if tb >= 0:
                fw = int(iw/2 * tb)
                pygame.draw.rect(surf, C["good"], (mid, y, fw, 6), 0, 2)
            else:
                fw = int(iw/2 * (-tb))
                pygame.draw.rect(surf, C["bad"], (mid - fw, y, fw, 6), 0, 2)
            pygame.draw.line(surf, C["panel_border"], (mid, y - 2), (mid, y + 8), 1)
            y += 16

            # Steering bar
            st = tel["steer"]
            _text(surf, "Steering", x, y, size=11, color=C["text_dim"])
            y += 13
            mid = x + iw // 2
            if st >= 0:
                fw = int(iw/2 * abs(st))
                pygame.draw.rect(surf, C["accent"], (mid, y, fw, 6), 0, 2)
            else:
                fw = int(iw/2 * abs(st))
                pygame.draw.rect(surf, C["accent"], (mid - fw, y, fw, 6), 0, 2)
            pygame.draw.line(surf, C["panel_border"], (mid, y - 2), (mid, y + 8), 1)
            y += 20

        # Hint
        y = SCREEN_H - 90
        _text(surf, "CONTROLS", x, y, size=11, color=C["text_dim"])
        y += 14
        for line in ["R  — restart generation", "T  — toggle rays", "SPACE — pause"]:
            _text(surf, line, x, y, size=11, color=C["text_dim"])
            y += 14

    def _draw_sparkline(self, surf, history: list, x, y, w, h):
        pygame.draw.rect(surf, C["panel_header"], (x, y, w, h), 0, 4)
        if len(history) < 2:
            return
        mx = max(history) or 1
        pts = []
        for i, v in enumerate(history[-w:]):
            px = x + int(i / max(len(history[-w:]) - 1, 1) * w)
            py = y + h - int(v / mx * (h - 4)) - 2
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(surf, C["good"], False, pts, 2)
        # Current value label
        ts = _f(11).render(f"{history[-1]:.2f}", True, C["good"])
        surf.blit(ts, (x + w - ts.get_width() - 2, y + 2))
