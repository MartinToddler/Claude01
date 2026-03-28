"""
UI panels v3 — left (car setup + sim controls) and right (AI stats + player).

New features:
  • Sim speed: 7 discrete buttons (×1 ×2 ×4 ×10 ×25 ×50 ×100)
  • Final drive ratio slider (exposed in gear-ratios section)
  • Drive type buttons: FWD / RWD / AWD
  • Engine position buttons: Front / Mid / Rear
  • (i) info buttons on every parameter — click to show/hide tooltip overlay
  • Player car enable/disable button + player telemetry in right panel
  • Track-lines toggle button
  • Restart simulation button
"""

import math
import pygame
from config import (
    C, SCREEN_W, SCREEN_H, LEFT_PANEL_W, RIGHT_PANEL_W,
    TRACK_AREA_X, TRACK_AREA_W, CAR_DEFAULTS,
    DEFAULT_NUM_AGENTS, DEFAULT_MAX_LAPS, SIM_SPEEDS,
    TOOLTIPS,
)
from track import TRACK_NAMES

# ── Font helpers ──────────────────────────────────────────────────────────────

_fonts: dict = {}

def _f(size: int, bold: bool = False) -> pygame.font.Font:
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = pygame.font.SysFont("monospace", size, bold=bold)
    return _fonts[k]

def _txt(surf, text, x, y, size=12, color=None, bold=False) -> int:
    color = color or C["text"]
    s = _f(size, bold).render(text, True, color)
    surf.blit(s, (x, y))
    return s.get_height()


# ── Tooltip state ─────────────────────────────────────────────────────────────

class TooltipState:
    """Singleton-ish state: which tooltip is active and where."""

    def __init__(self):
        self.active_id: str | None = None
        self._pos = (0, 0)

    def toggle(self, tip_id: str, screen_pos: tuple):
        if self.active_id == tip_id:
            self.active_id = None
        else:
            self.active_id = tip_id
            self._pos = screen_pos

    def dismiss(self):
        self.active_id = None

    def draw_overlay(self, surf):
        if self.active_id not in TOOLTIPS:
            return
        raw   = TOOLTIPS[self.active_id]
        lines = raw.split("\n")
        font  = _f(12)
        lh    = font.get_linesize() + 1
        w     = max(font.size(l)[0] for l in lines) + 20
        h     = len(lines) * lh + 16

        # Position: prefer to the right of the button, keep on screen
        bx, by = self._pos
        x = min(bx + 12, SCREEN_W - w - 4)
        y = max(4, min(by, SCREEN_H - h - 4))

        pygame.draw.rect(surf, (22, 28, 50), (x, y, w, h), 0, 6)
        pygame.draw.rect(surf, C["accent"],  (x, y, w, h), 1, 6)
        for i, line in enumerate(lines):
            ts = font.render(line, True, C["text"])
            surf.blit(ts, (x + 10, y + 8 + i * lh))


# ── Info button ───────────────────────────────────────────────────────────────

class InfoBtn:
    R = 7

    def __init__(self, tip_id: str):
        self.tip_id = tip_id
        self.cx = self.cy = 0

    def place(self, cx: int, cy: int):
        self.cx, self.cy = cx, cy

    def draw(self, surf):
        pygame.draw.circle(surf, C["accent"], (self.cx, self.cy), self.R)
        pygame.draw.circle(surf, (255, 255, 255), (self.cx, self.cy), self.R, 1)
        ts = _f(9, True).render("i", True, (255, 255, 255))
        surf.blit(ts, (self.cx - ts.get_width() // 2,
                       self.cy - ts.get_height() // 2))

    def hit(self, pos) -> bool:
        dx, dy = pos[0] - self.cx, pos[1] - self.cy
        return dx * dx + dy * dy <= (self.R + 4) ** 2


# ── Slider ────────────────────────────────────────────────────────────────────

class Slider:
    H = 16

    def __init__(self, label, vmin, vmax, value,
                 fmt=".0f", color=None, tip_id: str | None = None):
        self.label   = label
        self.vmin    = vmin
        self.vmax    = vmax
        self.value   = float(value)
        self.fmt     = fmt
        self.color   = color or C["accent"]
        self.rect    = pygame.Rect(0, 0, 100, self.H)
        self._drag   = False
        self.info    = InfoBtn(tip_id) if tip_id else None

    def set_rect(self, x, y, w):
        self.rect = pygame.Rect(x, y, w, self.H)
        if self.info:
            self.info.place(x + w + 10, y - 5)

    @property
    def norm(self):
        return (self.value - self.vmin) / max(self.vmax - self.vmin, 1e-9)

    def draw(self, surf):
        r  = self.rect
        cy = r.y + self.H // 2
        # Track
        pygame.draw.rect(surf, C["panel_border"],
                         (r.x, cy - 2, r.w, 4), 0, 2)
        fw = max(0, int(r.w * self.norm))
        if fw:
            pygame.draw.rect(surf, self.color, (r.x, cy - 2, fw, 4), 0, 2)
        kx = r.x + fw
        pygame.draw.circle(surf, self.color,     (kx, cy), 7)
        pygame.draw.circle(surf, (220, 225, 240), (kx, cy), 4)
        # Label + value
        _txt(surf, self.label, r.x, r.y - 13, 11, C["text_dim"])
        vs = _f(11).render(format(self.value, self.fmt), True, C["text"])
        surf.blit(vs, (r.right - vs.get_width(), r.y - 13))
        if self.info:
            self.info.draw(surf)

    def handle_event(self, event) -> bool:
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and \
                self.rect.inflate(0, 14).collidepoint(event.pos):
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


# ── Generic Button ────────────────────────────────────────────────────────────

class Button:
    def __init__(self, label, color=None, h=22, tip_id: str | None = None):
        self.label  = label
        self.color  = color or C["accent"]
        self.rect   = pygame.Rect(0, 0, 60, h)
        self.active = False
        self._h     = h
        self.info   = InfoBtn(tip_id) if tip_id else None

    def set_rect(self, x, y, w, h=None):
        self.rect = pygame.Rect(x, y, w, h or self._h)
        if self.info:
            self.info.place(x + (h or self._h) // 2 + 6,
                            y + (h or self._h) // 2)

    def draw(self, surf):
        bg = self.color if self.active else C["panel_border"]
        pygame.draw.rect(surf, bg,         self.rect, 0, 4)
        pygame.draw.rect(surf, self.color, self.rect, 1, 4)
        ts = _f(11).render(self.label, True, C["text"])
        surf.blit(ts, (self.rect.centerx - ts.get_width()  // 2,
                       self.rect.centery - ts.get_height() // 2))
        if self.info:
            self.info.draw(surf)

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


# ── Speed button row ──────────────────────────────────────────────────────────

class SpeedButtons:
    """Row of 7 discrete sim-speed buttons."""

    def __init__(self, tip_id: str | None = None):
        self._speeds  = SIM_SPEEDS
        self._active  = 0          # index into _speeds
        self._buttons = [Button(f"×{s}", C["accent"]) for s in self._speeds]
        self._buttons[0].active = True
        self.info = InfoBtn(tip_id) if tip_id else None

    def layout(self, x, y, w):
        n   = len(self._buttons)
        bw  = (w - (n - 1) * 3) // n
        for i, btn in enumerate(self._buttons):
            btn.set_rect(x + i * (bw + 3), y, bw, 22)
        if self.info:
            self.info.place(x + w + 10, y + 11)

    def draw(self, surf):
        for btn in self._buttons:
            btn.draw(surf)
        if self.info:
            self.info.draw(surf)

    @property
    def value(self) -> int:
        return self._speeds[self._active]

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, btn in enumerate(self._buttons):
                if btn.hit(event.pos):
                    self._active = i
                    for b in self._buttons:
                        b.active = False
                    btn.active = True
                    return True
        return False

    def _y(self):
        return self._buttons[0].rect.y if self._buttons else 0


# ── Gear ratio row ─────────────────────────────────────────────────────────────

class GearRatioRow:
    def __init__(self, values, final_drive):
        self.sliders = [
            Slider(f"G{i+1}", 0.5, 5.0, v, ".2f", C["warn"])
            for i, v in enumerate(values)
        ]
        self.s_final = Slider("Final Drive", 2.0, 6.0, final_drive,
                               ".2f", C["warn"], tip_id="final_drive")

    @property
    def gear_values(self):
        return [s.value for s in self.sliders]

    @property
    def final_drive(self):
        return self.s_final.value

    def layout(self, x, y, w):
        col_w = (w - 8) // 2
        for i, s in enumerate(self.sliders):
            col = i % 2; row = i // 2
            s.set_rect(x + col * (col_w + 8), y + row * 34, col_w)
        # Final drive below gear sliders (3 rows)
        self.s_final.set_rect(x, y + 3 * 34 + 8, w)

    def draw(self, surf):
        for s in self.sliders:
            s.draw(surf)
        self.s_final.draw(surf)

    def handle_event(self, event) -> bool:
        changed = False
        for s in self.sliders:
            if s.handle_event(event): changed = True
        if self.s_final.handle_event(event): changed = True
        return changed


# ── Left panel ────────────────────────────────────────────────────────────────

class LeftPanel:
    PAD = 12
    W   = LEFT_PANEL_W

    def __init__(self):
        d = CAR_DEFAULTS

        # Performance sliders
        self.s_power     = Slider("Power (HP)",    150, 1000, d["power"],  ".0f",
                                  tip_id="power")
        self.s_pressure  = Slider("Tyre PSI",       15,   32, d["tyre_pressure"],
                                  ".1f", C["good"], tip_id="tyre_psi")
        self.s_downforce = Slider("Downforce (kg)",  0,  250, d["downforce"],
                                  ".0f", C["warn"], tip_id="downforce")
        self.s_risk      = Slider("Driver Risk",     1,   10, d["driver_risk"],
                                  ".1f", C["bad"], tip_id="driver_risk")

        # Tyre compound
        self.tyre_btns  = {
            t: Button(t.capitalize(),
                      {"soft": C["bad"], "medium": C["warn"], "hard": C["good"]}[t])
            for t in ("soft", "medium", "hard")
        }
        self.tyre_type  = d["tyre_type"]
        self.tyre_btns[self.tyre_type].active = True
        self.tyre_info  = InfoBtn("tyre_type")

        # Gear ratios + final drive
        self.gear_row = GearRatioRow(d["gear_ratios"], d["final_drive"])

        # Drive type
        self.drive_btns = {
            t: Button(t, C["accent"]) for t in ("FWD", "RWD", "AWD")
        }
        self.drive_type = d["drive_type"]
        self.drive_btns[self.drive_type].active = True
        self.drive_info = InfoBtn("drive_type")

        # Engine position
        self.eng_btns  = {
            t: Button(t, C["warn"]) for t in ("Front", "Mid", "Rear")
        }
        self.engine_pos = d["engine_pos"]
        self.eng_btns[self.engine_pos.capitalize()].active = True
        self.eng_info  = InfoBtn("engine_pos")

        # Track selector
        self.track_btns = {n: Button(n) for n in TRACK_NAMES}
        self.track_name = TRACK_NAMES[0]
        self.track_btns[self.track_name].active = True

        # Simulation section
        self.speed_row   = SpeedButtons(tip_id="sim_speed")
        self.s_num       = Slider("Num Cars",       5, 50,
                                  DEFAULT_NUM_AGENTS, ".0f", C["warn"],
                                  tip_id="num_agents")
        self.s_laps      = Slider("Max Laps/Gen",   1, 20,
                                  DEFAULT_MAX_LAPS,  ".0f", C["good"],
                                  tip_id="max_laps")

        # Action buttons
        self.btn_player  = Button("Enable Player Car", C["player"], h=24,
                                  tip_id="player_car")
        self.btn_lines   = Button("Track Lines: ON",   C["accent"], h=22,
                                  tip_id="track_lines")
        self.btn_kill    = Button("Kill All → Next Gen", C["bad"], h=24)
        self.btn_restart = Button("Restart Simulation",  C["warn"], h=24)

        self._layout()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _layout(self):
        x  = self.PAD
        w  = self.W - 2 * self.PAD - 20   # leave 20px for (i) buttons
        y  = 46

        def sl(s):
            nonlocal y
            y += 16
            s.set_rect(x, y, w)
            y += s.H + 3

        def row3(btns, info, h=22):
            """3 buttons in a row + optional info button at right."""
            nonlocal y
            y += 4
            bw = (w - 8) // 3
            for i, btn in enumerate(btns.values()):
                btn.set_rect(x + i * (bw + 4), y, bw, h)
            if info:
                info.place(x + w + 10, y + h // 2)
            y += h + 4

        # ── POWERTRAIN ────────────────────────────────────────────────────────
        sl(self.s_power)
        row3(self.eng_btns,   self.eng_info,   22)
        row3(self.drive_btns, self.drive_info, 22)

        # Gear ratios + final drive
        y += 4
        self._gear_y = y
        self.gear_row.layout(x, y, w)
        y += 3 * 34 + 8 + 34 + 10   # 3 gear rows + final drive row

        # ── thin separator ────────────────────────────────────────────────────
        self._sep1_y = y + 4
        y += 12

        # ── TYRES ─────────────────────────────────────────────────────────────
        y += 6
        row3(self.tyre_btns, self.tyre_info, 22)
        sl(self.s_pressure)
        sl(self.s_downforce)

        # ── thin separator ────────────────────────────────────────────────────
        self._sep2_y = y + 4
        y += 12

        # ── TRACK ─────────────────────────────────────────────────────────────
        y += 4
        tw = (w - 8) // len(TRACK_NAMES)
        for i, btn in enumerate(self.track_btns.values()):
            btn.set_rect(x + i * (tw + 4), y, tw, 22)
        y += 28

        # ── main divider ──────────────────────────────────────────────────────
        self._div_y = y
        y += 14

        # ── SIMULATION ────────────────────────────────────────────────────────
        self._speed_label_y = y - 12
        self.speed_row.layout(x, y, w)
        y += 28

        sl(self.s_risk)
        sl(self.s_num)
        sl(self.s_laps)

        y += 6
        bw2 = (w - 4) // 2
        self.btn_player.set_rect(x,         y, bw2, 24)
        self.btn_lines.set_rect( x + bw2+4, y, bw2, 22)
        y += 30
        self.btn_kill.set_rect(   x,         y, bw2, 24)
        self.btn_restart.set_rect(x + bw2+4, y, bw2, 24)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surf, player_enabled: bool, track_lines: bool):
        pygame.draw.rect(surf, C["panel_bg"],     (0, 0, self.W, SCREEN_H))
        pygame.draw.line(surf, C["panel_border"], (self.W-1, 0), (self.W-1, SCREEN_H), 1)

        _txt(surf, "CAR SETUP", self.PAD, 12, 13, C["accent"], bold=True)

        # ── Powertrain ────────────────────────────────────────────────────────
        self.s_power.draw(surf)

        e0 = list(self.eng_btns.values())[0]
        _txt(surf, "ENGINE POS", self.PAD, e0.rect.y - 13, 10, C["text_dim"])
        for btn in self.eng_btns.values(): btn.draw(surf)
        self.eng_info.draw(surf)

        d0 = list(self.drive_btns.values())[0]
        _txt(surf, "DRIVE TYPE", self.PAD, d0.rect.y - 13, 10, C["text_dim"])
        for btn in self.drive_btns.values(): btn.draw(surf)
        self.drive_info.draw(surf)

        _txt(surf, "GEAR RATIOS", self.PAD, self._gear_y - 13, 10, C["text_dim"])
        self.gear_row.draw(surf)

        # ── thin separator ────────────────────────────────────────────────────
        pygame.draw.line(surf, C["panel_border"],
                         (self.PAD, self._sep1_y), (self.W - self.PAD, self._sep1_y), 1)

        # ── Tyres ─────────────────────────────────────────────────────────────
        t0 = list(self.tyre_btns.values())[0]
        _txt(surf, "COMPOUND", self.PAD, t0.rect.y - 13, 10, C["text_dim"])
        for btn in self.tyre_btns.values(): btn.draw(surf)
        self.tyre_info.draw(surf)

        self.s_pressure.draw(surf)
        self.s_downforce.draw(surf)

        # ── thin separator ────────────────────────────────────────────────────
        pygame.draw.line(surf, C["panel_border"],
                         (self.PAD, self._sep2_y), (self.W - self.PAD, self._sep2_y), 1)

        # ── Track ─────────────────────────────────────────────────────────────
        t0 = list(self.track_btns.values())[0]
        _txt(surf, "TRACK", self.PAD, t0.rect.y - 13, 10, C["text_dim"])
        for btn in self.track_btns.values(): btn.draw(surf)

        # ── main divider ──────────────────────────────────────────────────────
        pygame.draw.line(surf, C["panel_border"],
                         (self.PAD, self._div_y), (self.W - self.PAD, self._div_y), 1)
        _txt(surf, "SIMULATION", self.PAD, self._div_y + 3, 10, C["text_dim"], bold=True)

        _txt(surf, "Speed", self.PAD, self._speed_label_y, 10, C["text_dim"])
        self.speed_row.draw(surf)
        self.s_risk.draw(surf)
        self.s_num.draw(surf)
        self.s_laps.draw(surf)

        # State-dependent button labels
        self.btn_player.label = ("Disable Player Car" if player_enabled
                                  else "Enable Player Car")
        self.btn_player.color = (C["bad"] if player_enabled else C["player"])
        self.btn_player.active = player_enabled

        self.btn_lines.label = ("Track Lines: ON" if track_lines
                                 else "Track Lines: OFF")
        self.btn_lines.active = track_lines

        self.btn_player.draw(surf)
        self.btn_lines.draw(surf)
        self.btn_kill.draw(surf)
        self.btn_restart.draw(surf)

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, event, tooltip: TooltipState):
        """
        Returns dict of booleans:
          params_changed, track_changed, speed_changed,
          kill_all, restart, toggle_player, toggle_lines
        """
        result = {k: False for k in (
            "params_changed", "track_changed", "speed_changed",
            "kill_all", "restart", "toggle_player", "toggle_lines"
        )}

        # Performance sliders
        for s in (self.s_power, self.s_pressure, self.s_downforce, self.s_risk):
            if s.handle_event(event): result["params_changed"] = True

        if self.gear_row.handle_event(event): result["params_changed"] = True
        if self.speed_row.handle_event(event): result["speed_changed"] = True
        if self.s_num.handle_event(event):     result["params_changed"] = True
        if self.s_laps.handle_event(event):    result["params_changed"] = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos

            # (i) info buttons
            for obj in self._all_info_btns():
                if obj.hit(pos):
                    tooltip.toggle(obj.tip_id, pos)
                    return result

            # Dismiss tooltip on any other click in panel
            if pos[0] < self.W:
                tooltip.dismiss()

            # Tyre
            for name, btn in self.tyre_btns.items():
                if btn.hit(pos) and name != self.tyre_type:
                    self.tyre_type = name
                    for b in self.tyre_btns.values(): b.active = False
                    btn.active = True
                    result["params_changed"] = True

            # Drive type
            for name, btn in self.drive_btns.items():
                if btn.hit(pos) and name != self.drive_type:
                    self.drive_type = name
                    for b in self.drive_btns.values(): b.active = False
                    btn.active = True
                    result["params_changed"] = True

            # Engine pos
            for name, btn in self.eng_btns.items():
                if btn.hit(pos) and name.lower() != self.engine_pos:
                    self.engine_pos = name.lower()
                    for b in self.eng_btns.values(): b.active = False
                    btn.active = True
                    result["params_changed"] = True

            # Track
            for name, btn in self.track_btns.items():
                if btn.hit(pos) and name != self.track_name:
                    self.track_name = name
                    for b in self.track_btns.values(): b.active = False
                    btn.active = True
                    result["track_changed"] = True

            # Action buttons
            if self.btn_kill.hit(pos):    result["kill_all"]       = True
            if self.btn_restart.hit(pos): result["restart"]        = True
            if self.btn_player.hit(pos):  result["toggle_player"]  = True
            if self.btn_lines.hit(pos):   result["toggle_lines"]   = True

        return result

    def _all_info_btns(self):
        """Iterate over all InfoBtn instances in this panel."""
        for s in (self.s_power, self.s_pressure, self.s_downforce,
                  self.s_risk, self.s_num, self.s_laps):
            if s.info: yield s.info
        if self.tyre_info: yield self.tyre_info
        if self.drive_info: yield self.drive_info
        if self.eng_info: yield self.eng_info
        if self.speed_row.info: yield self.speed_row.info
        for s in self.gear_row.sliders:
            if s.info: yield s.info
        if self.gear_row.s_final.info: yield self.gear_row.s_final.info
        for btn in (self.btn_player, self.btn_lines):
            if btn.info: yield btn.info

    @property
    def params(self) -> dict:
        p = dict(CAR_DEFAULTS)
        p["power"]         = self.s_power.value
        p["tyre_type"]     = self.tyre_type
        p["tyre_pressure"] = self.s_pressure.value
        p["downforce"]     = self.s_downforce.value
        p["driver_risk"]   = self.s_risk.value
        p["gear_ratios"]   = self.gear_row.gear_values
        p["final_drive"]   = self.gear_row.final_drive
        p["drive_type"]    = self.drive_type
        p["engine_pos"]    = self.engine_pos
        p["num_agents"]    = int(self.s_num.value)
        p["max_laps"]      = int(self.s_laps.value)
        return p

    @property
    def sim_speed(self) -> int:
        return self.speed_row.value


# ── Right panel ───────────────────────────────────────────────────────────────

class RightPanel:
    PAD = 10

    def draw(self, surf, pop, gen_time_s: float,
             selected_idx: int = -1, selected_player: bool = False,
             player=None):
        rx = TRACK_AREA_X + TRACK_AREA_W
        rw = SCREEN_W - rx
        pygame.draw.rect(surf, C["panel_bg"], (rx, 0, rw, SCREEN_H))
        pygame.draw.line(surf, C["panel_border"], (rx, 0), (rx, SCREEN_H), 1)

        x  = rx + self.PAD
        iw = rw - 2 * self.PAD
        y  = 10

        _txt(surf, "AI STATS", x, y, 13, C["accent"], bold=True)
        y += 24

        _txt(surf, f"Generation  {pop.generation}", x, y, 12)
        y += 16
        alive = len(pop.alive_agents)
        col = C["good"] if alive > 5 else C["warn"] if alive > 1 else C["bad"]
        _txt(surf, f"Alive  {alive}/{len(pop.agents)}", x, y, 12, col)
        y += 16
        _txt(surf, f"Gen time  {gen_time_s:.1f}s", x, y, 12)
        y += 20

        # Best lap block
        pygame.draw.rect(surf, C["panel_header"], (x-2, y-2, iw+4, 50), 0, 4)
        _txt(surf, "BEST LAP", x, y, 10, C["text_dim"])
        y += 13
        gbl = _best_lap_gen(pop)
        _txt(surf, f"This gen    {gbl:.2f}s" if gbl else "This gen    ---",
             x, y, 12, C["warn"])
        y += 14
        atb = pop.all_time_best_lap
        _txt(surf, f"All time    {atb:.2f}s" if atb else "All time    ---",
             x, y, 12, C["good"])
        y += 22

        # Sparkline
        _txt(surf, "FITNESS HISTORY", x, y, 10, C["text_dim"])
        y += 12
        self._sparkline(surf, pop.fitness_history, x, y, iw, 48)
        y += 54

        # ── Telemetry section ─────────────────────────────────────────────
        if selected_player and player is not None and player.enabled:
            self._player_telemetry(surf, player, x, y, iw)
        elif 0 <= selected_idx < len(pop.agents):
            ag = pop.agents[selected_idx]
            hdr = f"CAR #{selected_idx} {'(alive)' if ag.car.alive else '(crashed)'}"
            self._car_telemetry(surf, ag.car, ag.laps, ag.best_lap_s,
                                hdr, C["warn"], x, y, iw)
        else:
            best = pop.best_agent
            self._car_telemetry(surf, best.car, best.laps, best.best_lap_s,
                                "BEST CAR", C["text_dim"], x, y, iw)

        # Player mini-panel (always visible when enabled)
        if player is not None and player.enabled and not selected_player:
            self._player_mini(surf, player, x, iw, SCREEN_H - 95)

        # Hotkeys
        y2 = SCREEN_H - 58
        pygame.draw.line(surf, C["panel_border"], (x-2, y2), (x+iw+2, y2), 1)
        y2 += 4
        for line in ["R=restart gen  T=rays  K=kill all",
                     "SPACE=pause  ESC=quit  CLICK=select"]:
            _txt(surf, line, x, y2, 10, C["text_dim"])
            y2 += 13

    def _car_telemetry(self, surf, car, laps, best_lap, header, hcol, x, y, iw):
        _txt(surf, header, x, y, 10, hcol)
        y += 14
        bl = f"{best_lap:.2f}s" if best_lap else "---"
        _txt(surf, f"Laps {laps}   Best {bl}", x, y, 12)
        y += 16
        tel = car.telemetry()
        self._gauge(surf, "Speed km/h", tel["speed_kmh"], 320, x, y, iw, C["accent"])
        y += 22
        self._gauge(surf, "RPM",        tel["rpm"],       8000, x, y, iw, C["warn"])
        y += 22
        _txt(surf, f"Gear {tel['gear']}   Yaw {tel['omega_deg']:.1f}°/s",
             x, y, 11, C["text_dim"])
        y += 15
        self._bar2(surf, "Throttle/Brake", tel["throttle"], x, y, iw)
        y += 18
        self._bar2(surf, "Steering",       tel["steer"],    x, y, iw)

    def _player_telemetry(self, surf, player, x, y, iw):
        _txt(surf, f"PLAYER  [{player.gear_mode}]", x, y, 11, C["player"])
        y += 14
        bl = f"{player.best_lap_s:.2f}s" if player.best_lap_s else "---"
        ll = f"{player.last_lap_s:.2f}s" if player.last_lap_s else "---"
        _txt(surf, f"Laps {player.laps}   Best {bl}", x, y, 12)
        y += 14
        _txt(surf, f"Last lap: {ll}", x, y, 12, C["warn"])
        y += 16
        tel = player.car.telemetry()
        self._gauge(surf, "Speed km/h", tel["speed_kmh"], 320, x, y, iw, C["player"])
        y += 22
        self._gauge(surf, "RPM",        tel["rpm"],       8000, x, y, iw, C["warn"])
        y += 22
        _txt(surf, f"Gear {tel['gear']}  {'AUTO' if player.auto_gear else 'MANUAL'}",
             x, y, 11, C["text_dim"])
        y += 15
        self._bar2(surf, "Throttle/Brake", tel["throttle"], x, y, iw)
        y += 18
        self._bar2(surf, "Steering",       tel["steer"],    x, y, iw)
        rc = player.respawn_countdown
        if rc is not None:
            y += 8
            _txt(surf, f"RESPAWN in {rc:.1f}s", x, y, 12, C["bad"])

    def _player_mini(self, surf, player, x, iw, y):
        """Compact player info always shown at bottom of right panel."""
        pygame.draw.rect(surf, C["panel_header"], (x-2, y, iw+4, 38), 0, 4)
        pygame.draw.line(surf, C["player"], (x-2, y), (x+iw+2, y), 1)
        bl = f"{player.best_lap_s:.2f}s" if player.best_lap_s else "---"
        _txt(surf, f"PLAYER  Laps:{player.laps}  Best:{bl}  "
             f"Gear:{player.car.gear}[{player.gear_mode[:1]}]  "
             f"{player.car.speed_kmh:.0f}km/h",
             x, y + 4, 10, C["player"])
        ll = f"Last: {player.last_lap_s:.2f}s" if player.last_lap_s else ""
        if ll:
            _txt(surf, ll, x, y + 20, 10, C["warn"])

    def _gauge(self, surf, label, val, vmax, x, y, w, color):
        _txt(surf, label, x, y, 10, C["text_dim"])
        pygame.draw.rect(surf, C["panel_border"], (x, y+11, w, 5), 0, 2)
        fw = int(w * min(max(val, 0) / max(vmax, 1), 1.0))
        if fw: pygame.draw.rect(surf, color, (x, y+11, fw, 5), 0, 2)
        vs = _f(10).render(f"{val:.0f}", True, C["text"])
        surf.blit(vs, (x + w - vs.get_width(), y))

    def _bar2(self, surf, label, val, x, y, w):
        _txt(surf, label, x, y, 10, C["text_dim"])
        mid = x + w // 2
        pygame.draw.line(surf, C["panel_border"], (mid, y+11), (mid, y+17))
        fw = int(w / 2 * abs(val))
        if val > 0:
            pygame.draw.rect(surf, C["good"],  (mid,      y+11, max(1, fw), 5), 0, 2)
        elif val < 0:
            pygame.draw.rect(surf, C["bad"],   (mid - fw, y+11, max(1, fw), 5), 0, 2)

    def _sparkline(self, surf, history, x, y, w, h):
        pygame.draw.rect(surf, C["panel_header"], (x, y, w, h), 0, 4)
        if len(history) < 2:
            return
        mx = max(history) or 1
        win = history[-(w // 2):]
        pts = []
        for i, v in enumerate(win):
            px = x + int(i / max(len(win)-1, 1) * (w-2)) + 1
            py = y + h - int(v / mx * (h-6)) - 3
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(surf, C["good"], False, pts, 2)
        vs = _f(10).render(f"{history[-1]:.2f}", True, C["good"])
        surf.blit(vs, (x + w - vs.get_width() - 2, y + 2))


def _best_lap_gen(pop) -> float | None:
    best = None
    for ag in pop.agents:
        if ag.best_lap_s is not None:
            if best is None or ag.best_lap_s < best:
                best = ag.best_lap_s
    return best
