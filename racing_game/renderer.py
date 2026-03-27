"""
Pygame rendering: track, cars, and overlay info.
Everything except the two side panels (handled by ui.py).
"""

import math
import pygame
import numpy as np
from config import (
    C, TRACK_AREA_X, TRACK_AREA_W, TRACK_AREA_H,
    TRACK_HALF_W, RAYCAST_DIST, NUM_AGENTS,
)
from track import Track
from agent import PopulationManager, Agent


class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self._track_surf: pygame.Surface | None = None
        self._track_key: str | None = None

        self.show_rays    = False
        self.show_ghosts  = True    # dim dead cars
        self.show_best_path = True

    # ── Track pre-render ──────────────────────────────────────────────────────

    def prerender_track(self, track: Track):
        """Build a cached surface for the static track geometry."""
        if track.name == self._track_key:
            return
        self._track_key = track.name

        surf = pygame.Surface((TRACK_AREA_W, TRACK_AREA_H))
        surf.fill(C["grass"])

        n    = len(track.pts)
        pts  = track.pts
        offs_x = TRACK_AREA_X

        def to_local(arr):
            return [(int(p[0] - offs_x), int(p[1])) for p in arr]

        # ── Grass fill (entire surface already filled) ─────────────────────

        # ── Track surface ──────────────────────────────────────────────────
        # Build a polygon: outer boundary forward, inner boundary reversed
        outer_pts = to_local(track.outer)
        inner_pts = to_local(track.inner)
        poly = outer_pts + inner_pts[::-1]
        pygame.draw.polygon(surf, C["track"], poly)

        # ── Kerbs (alternating red/white stripes along edges) ─────────────
        kerb_w = max(4, TRACK_HALF_W // 5)
        for i in range(n):
            j = (i + 1) % n
            color = C["kerb_red"] if (i // 4) % 2 == 0 else C["kerb_white"]
            # Outer kerb
            op0 = (int(track.outer[i, 0] - offs_x), int(track.outer[i, 1]))
            op1 = (int(track.outer[j, 0] - offs_x), int(track.outer[j, 1]))
            ok0 = (int((track.outer[i] - track.norm[i] * kerb_w)[0] - offs_x),
                   int((track.outer[i] - track.norm[i] * kerb_w)[1]))
            ok1 = (int((track.outer[j] - track.norm[j] * kerb_w)[0] - offs_x),
                   int((track.outer[j] - track.norm[j] * kerb_w)[1]))
            pygame.draw.polygon(surf, color, [op0, op1, ok1, ok0])

            # Inner kerb
            ip0 = (int(track.inner[i, 0] - offs_x), int(track.inner[i, 1]))
            ip1 = (int(track.inner[j, 0] - offs_x), int(track.inner[j, 1]))
            ik0 = (int((track.inner[i] + track.norm[i] * kerb_w)[0] - offs_x),
                   int((track.inner[i] + track.norm[i] * kerb_w)[1]))
            ik1 = (int((track.inner[j] + track.norm[j] * kerb_w)[0] - offs_x),
                   int((track.inner[j] + track.norm[j] * kerb_w)[1]))
            pygame.draw.polygon(surf, color, [ip0, ip1, ik1, ik0])

        # ── Centre line (dashed) ───────────────────────────────────────────
        for i in range(0, n, 8):
            j = (i + 4) % n
            p0 = (int(pts[i, 0] - offs_x), int(pts[i, 1]))
            p1 = (int(pts[j, 0] - offs_x), int(pts[j, 1]))
            pygame.draw.line(surf, (100, 100, 115), p0, p1, 1)

        # ── Start / Finish line ────────────────────────────────────────────
        si = track.start_idx
        n0 = track.norm[si]
        s_inner = track.inner[si]
        s_outer = track.outer[si]
        p_i = (int(s_inner[0] - offs_x), int(s_inner[1]))
        p_o = (int(s_outer[0] - offs_x), int(s_outer[1]))
        pygame.draw.line(surf, C["start_line"], p_i, p_o, 3)

        self._track_surf = surf

    # ── Main draw ─────────────────────────────────────────────────────────────

    def draw(self, track: Track, pop: PopulationManager, fps: float):
        self.prerender_track(track)

        # Blit cached track
        self.surface.blit(self._track_surf, (TRACK_AREA_X, 0))

        # Draw agents
        self._draw_agents(pop, track)

        # FPS counter
        font_s = _font(14)
        fps_surf = font_s.render(f"{fps:.0f} fps", True, C["text_dim"])
        self.surface.blit(fps_surf, (TRACK_AREA_X + 8, 8))

    # ── Agents ────────────────────────────────────────────────────────────────

    def _draw_agents(self, pop: PopulationManager, track: Track):
        best = pop.best_agent

        for i, agent in enumerate(pop.agents):
            car   = agent.car
            is_best = (agent is best) and car.alive
            color = C["cars"][-1] if is_best else C["cars"][i % (NUM_AGENTS - 1)]

            if not car.alive:
                if self.show_ghosts:
                    self._draw_car(car, (*color, 40), is_best=False, ghost=True)
            else:
                self._draw_car(car, color, is_best=is_best)
                if self.show_rays and is_best:
                    self._draw_rays(car, track)

    def _draw_car(self, car, color, is_best: bool, ghost: bool = False):
        from car import Car
        hw = Car.WIDTH_PX / 2
        hl = Car.LENGTH_PX / 2
        cos_h = math.cos(car.heading)
        sin_h = math.sin(car.heading)

        # Four corners in car-local space
        corners_local = [
            ( hl,  hw), ( hl, -hw),
            (-hl, -hw), (-hl,  hw),
        ]
        corners = []
        for lx, ly in corners_local:
            wx = car.x + lx * cos_h - ly * sin_h
            wy = car.y + lx * sin_h + ly * cos_h
            corners.append((int(wx), int(wy)))

        if ghost:
            ghost_surf = pygame.Surface((TRACK_AREA_W, TRACK_AREA_H), pygame.SRCALPHA)
            pygame.draw.polygon(ghost_surf, color, corners)
            self.surface.blit(ghost_surf, (TRACK_AREA_X, 0))
        else:
            pygame.draw.polygon(self.surface, color, corners)
            if is_best:
                pygame.draw.polygon(self.surface, (255, 255, 255), corners, 1)

            # Direction nub
            nx = int(car.x + hl * cos_h)
            ny = int(car.y + hl * sin_h)
            pygame.draw.circle(self.surface, (255, 255, 255), (nx, ny), 2)

    def _draw_rays(self, car, track: Track):
        import config as cfg
        angle_step = math.pi / (cfg.RAYCAST_COUNT - 1)
        for i in range(cfg.RAYCAST_COUNT):
            angle = car.heading - math.pi / 2 + i * angle_step
            dist  = track.raycast(car.x, car.y, angle, RAYCAST_DIST) * RAYCAST_DIST
            ex = int(car.x + dist * math.cos(angle))
            ey = int(car.y + dist * math.sin(angle))
            pygame.draw.line(self.surface, (255, 255, 100, 80),
                             (int(car.x), int(car.y)), (ex, ey), 1)


# ── Font cache ────────────────────────────────────────────────────────────────

_font_cache: dict[int, pygame.font.Font] = {}

def _font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        _font_cache[size] = pygame.font.SysFont("monospace", size)
    return _font_cache[size]
