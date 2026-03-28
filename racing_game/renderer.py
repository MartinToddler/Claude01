"""
Pygame rendering — track, AI cars, player car, sensor rays.

v3 changes:
  • show_track_lines flag: hides kerbs / decorations when False (track
    surface and physics remain active)
  • draw_player(): cyan car with distinct "P" label
"""

import math
import pygame
import numpy as np
from config import (
    C, TRACK_AREA_X, TRACK_AREA_W, TRACK_AREA_H,
    TRACK_HALF_W, RAYCAST_DIST, RAYCAST_COUNT,
)
from track import Track
from agent import PopulationManager, Agent


class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surface          = surface
        self._track_surf_full = None   # with kerbs
        self._track_surf_bare = None   # without kerbs
        self._track_key       = None

        self.show_rays        = False
        self.show_track_lines = True   # kerbs / decorations visible
        self.selected_idx     = -1

    # ── Track pre-render ──────────────────────────────────────────────────────

    def prerender_track(self, track: Track):
        if track.name == self._track_key:
            return
        self._track_key = track.name

        n   = len(track.pts)
        pts = track.pts
        ox  = TRACK_AREA_X

        def local(arr):
            return [(int(p[0] - ox), int(p[1])) for p in arr]

        # ── Build bare surface (track only, no kerbs) ─────────────────────
        bare = pygame.Surface((TRACK_AREA_W, TRACK_AREA_H))
        bare.fill(C["grass"])
        outer = local(track.outer)
        inner = local(track.inner)
        pygame.draw.polygon(bare, C["track"], outer + inner[::-1])
        # Start / finish line
        si  = track.start_idx
        p_i = (int(track.inner[si, 0] - ox), int(track.inner[si, 1]))
        p_o = (int(track.outer[si, 0] - ox), int(track.outer[si, 1]))
        pygame.draw.line(bare, C["start_line"], p_i, p_o, 3)
        self._track_surf_bare = bare

        # ── Build full surface (with kerbs + dashed line) ─────────────────
        full = bare.copy()
        kerb_w = max(4, TRACK_HALF_W // 5)
        for i in range(n):
            j     = (i + 1) % n
            color = C["kerb_red"] if (i // 4) % 2 == 0 else C["kerb_white"]

            op0 = (int(track.outer[i, 0] - ox), int(track.outer[i, 1]))
            op1 = (int(track.outer[j, 0] - ox), int(track.outer[j, 1]))
            ok0 = (int(track.outer[i, 0] - track.norm[i, 0] * kerb_w - ox),
                   int(track.outer[i, 1] - track.norm[i, 1] * kerb_w))
            ok1 = (int(track.outer[j, 0] - track.norm[j, 0] * kerb_w - ox),
                   int(track.outer[j, 1] - track.norm[j, 1] * kerb_w))
            pygame.draw.polygon(full, color, [op0, op1, ok1, ok0])

            ip0 = (int(track.inner[i, 0] - ox), int(track.inner[i, 1]))
            ip1 = (int(track.inner[j, 0] - ox), int(track.inner[j, 1]))
            ik0 = (int(track.inner[i, 0] + track.norm[i, 0] * kerb_w - ox),
                   int(track.inner[i, 1] + track.norm[i, 1] * kerb_w))
            ik1 = (int(track.inner[j, 0] + track.norm[j, 0] * kerb_w - ox),
                   int(track.inner[j, 1] + track.norm[j, 1] * kerb_w))
            pygame.draw.polygon(full, color, [ip0, ip1, ik1, ik0])

        for i in range(0, n, 8):
            j  = (i + 4) % n
            p0 = (int(pts[i, 0] - ox), int(pts[i, 1]))
            p1 = (int(pts[j, 0] - ox), int(pts[j, 1]))
            pygame.draw.line(full, (100, 100, 115), p0, p1, 1)

        self._track_surf_full = full

    # ── Main draw ─────────────────────────────────────────────────────────────

    def draw(self, track: Track, pop: PopulationManager, fps: float,
             player=None):
        self.prerender_track(track)
        surf = self._track_surf_full if self.show_track_lines \
               else self._track_surf_bare
        self.surface.blit(surf, (TRACK_AREA_X, 0))

        self._draw_agents(pop)

        if player is not None and player.enabled:
            self.draw_player(player)

        font = _font(14)
        fps_s = font.render(f"{fps:.0f} fps", True, C["text_dim"])
        self.surface.blit(fps_s, (TRACK_AREA_X + 8, 8))

    # ── Click / selection ─────────────────────────────────────────────────────

    def hit_test(self, mx: int, my: int, pop: PopulationManager,
                 player=None) -> tuple[int, bool]:
        """
        Returns (idx, is_player):
          idx=-1, is_player=False → nothing hit
          idx≥0,  is_player=False → AI agent at pop.agents[idx]
          idx=-1, is_player=True  → player car hit
        """
        from car import Car
        R2 = (Car.LENGTH_PX + 4) ** 2

        # Test player first (always on top)
        if player is not None and player.enabled:
            dx = mx - player.car.x
            dy = my - player.car.y
            if dx * dx + dy * dy <= R2:
                return -1, True

        for i, ag in enumerate(pop.agents):
            dx = mx - ag.car.x
            dy = my - ag.car.y
            if dx * dx + dy * dy <= R2:
                return i, False
        return -1, False

    # ── AI agents ─────────────────────────────────────────────────────────────

    def _draw_agents(self, pop: PopulationManager):
        best = pop.best_agent
        for i, agent in enumerate(pop.agents):
            car     = agent.car
            is_best = (agent is best) and car.alive
            is_sel  = (i == self.selected_idx)
            color   = C["cars"][-1] if is_best \
                      else C["cars"][i % (len(C["cars"]) - 1)]

            if not car.alive:
                self._draw_car_ghost(car, color)
            else:
                self._draw_car(car, color, is_best, is_sel)
                if self.show_rays and (is_best or is_sel):
                    self._draw_rays(car, pop.track)

    def _car_corners(self, car):
        from car import Car
        hw, hl = Car.WIDTH_PX / 2, Car.LENGTH_PX / 2
        ch, sh = math.cos(car.heading), math.sin(car.heading)
        return [
            (int(car.x + lx * ch - ly * sh),
             int(car.y + lx * sh + ly * ch))
            for lx, ly in [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        ]

    def _draw_car(self, car, color, is_best: bool, is_sel: bool):
        corners = self._car_corners(car)
        if is_sel:
            pygame.draw.circle(self.surface, (255, 255, 80),
                               (int(car.x), int(car.y)), 18, 2)
        pygame.draw.polygon(self.surface, color, corners)
        if is_best:
            pygame.draw.polygon(self.surface, (255, 255, 255), corners, 2)
        elif is_sel:
            pygame.draw.polygon(self.surface, (255, 255, 80), corners, 2)
        # Direction nub
        from car import Car
        hl    = Car.LENGTH_PX / 2
        nx    = int(car.x + hl * math.cos(car.heading))
        ny    = int(car.y + hl * math.sin(car.heading))
        pygame.draw.circle(self.surface, (255, 255, 255), (nx, ny), 2)

    def _draw_car_ghost(self, car, color):
        corners = self._car_corners(car)
        ghost   = pygame.Surface((TRACK_AREA_W, TRACK_AREA_H), pygame.SRCALPHA)
        r, g, b = color
        pygame.draw.polygon(ghost, (r, g, b, 35), corners)
        self.surface.blit(ghost, (TRACK_AREA_X, 0))

    def _draw_rays(self, car, track: Track):
        step = math.pi / (RAYCAST_COUNT - 1)
        for i in range(RAYCAST_COUNT):
            angle = car.heading - math.pi / 2 + i * step
            dist  = track.raycast(car.x, car.y, angle, RAYCAST_DIST) * RAYCAST_DIST
            ex    = int(car.x + dist * math.cos(angle))
            ey    = int(car.y + dist * math.sin(angle))
            pygame.draw.line(self.surface, (255, 240, 60),
                             (int(car.x), int(car.y)), (ex, ey), 1)

    # ── Player car ────────────────────────────────────────────────────────────

    def draw_player(self, player):
        car     = player.car
        corners = self._car_corners(car)

        # Glow
        pygame.draw.circle(self.surface, C["player"],
                           (int(car.x), int(car.y)), 20, 2)
        pygame.draw.polygon(self.surface, C["player"], corners)
        pygame.draw.polygon(self.surface, (255, 255, 255), corners, 1)

        # "P" label above car
        lbl = _font(11).render("P", True, (255, 255, 255))
        self.surface.blit(lbl, (int(car.x) - lbl.get_width() // 2,
                                int(car.y) - 22))

        # Respawn countdown
        rc = player.respawn_countdown
        if rc is not None:
            cnt = _font(14).render(f"RESPAWN {rc:.1f}s", True, C["warn"])
            self.surface.blit(cnt, (int(car.x) - cnt.get_width() // 2,
                                    int(car.y) + 14))


# ── Font cache ────────────────────────────────────────────────────────────────

_font_cache: dict = {}

def _font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        _font_cache[size] = pygame.font.SysFont("monospace", size)
    return _font_cache[size]
