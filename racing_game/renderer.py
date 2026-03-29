"""
Pygame rendering — track, AI cars, player car, sensor rays.

v4 changes:
  • Camera support: all world coords transformed through camera before drawing.
    Track surface is pre-rendered in world-space; each frame only the visible
    viewport rectangle is blitted onto the screen.
  • F1 / open-wheel car visual: narrow monocoque body, front/rear wings,
    4 separate wheels (front pair rotated by steering angle).
  • Viewport culling: cars outside the camera view are skipped entirely.
  • Tiered detail: top FULL_DETAIL cars drawn as F1 polygons; rest as dots.
  • show_track_lines flag retained.
"""

import math
import pygame
import numpy as np
from config import (
    C, car_color, TRACK_AREA_X, TRACK_AREA_W, TRACK_AREA_H,
    TRACK_HALF_W, RAYCAST_DIST, RAYCAST_COUNT,
)
from camera import Camera
from track import Track
from agent import PopulationManager, Agent

# Number of cars rendered in full F1 detail; the rest become 3-px dots
FULL_DETAIL = 50


class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surface          = surface
        self._track_surf_full = None
        self._track_surf_bare = None
        self._track_key       = None

        self.show_rays        = False
        self.show_track_lines = True
        self.selected_idx     = -1

    # ── Track pre-render (world-space surface) ────────────────────────────────

    def prerender_track(self, track: Track):
        if track.name == self._track_key:
            return
        self._track_key = track.name

        ww = int(track.world_w)
        wh = int(track.world_h)
        n  = len(track.pts)

        def lp(arr):
            """Track world-pts → int pixel tuples (no camera offset needed here)."""
            return [(int(p[0]), int(p[1])) for p in arr]

        # ── Bare surface (track + start line, no kerbs) ───────────────────
        bare = pygame.Surface((ww, wh))
        bare.fill(C["grass"])
        outer = lp(track.outer)
        inner = lp(track.inner)
        pygame.draw.polygon(bare, C["track"], outer + inner[::-1])
        si  = track.start_idx
        p_i = (int(track.inner[si, 0]), int(track.inner[si, 1]))
        p_o = (int(track.outer[si, 0]), int(track.outer[si, 1]))
        pygame.draw.line(bare, C["start_line"], p_i, p_o, 3)
        self._track_surf_bare = bare

        # ── Full surface (kerbs + dashed centreline) ──────────────────────
        full    = bare.copy()
        kerb_w  = max(4, TRACK_HALF_W // 5)
        pts     = track.pts
        for i in range(n):
            j     = (i + 1) % n
            color = C["kerb_red"] if (i // 4) % 2 == 0 else C["kerb_white"]

            op0 = (int(track.outer[i, 0]), int(track.outer[i, 1]))
            op1 = (int(track.outer[j, 0]), int(track.outer[j, 1]))
            ok0 = (int(track.outer[i, 0] - track.norm[i, 0] * kerb_w),
                   int(track.outer[i, 1] - track.norm[i, 1] * kerb_w))
            ok1 = (int(track.outer[j, 0] - track.norm[j, 0] * kerb_w),
                   int(track.outer[j, 1] - track.norm[j, 1] * kerb_w))
            pygame.draw.polygon(full, color, [op0, op1, ok1, ok0])

            ip0 = (int(track.inner[i, 0]), int(track.inner[i, 1]))
            ip1 = (int(track.inner[j, 0]), int(track.inner[j, 1]))
            ik0 = (int(track.inner[i, 0] + track.norm[i, 0] * kerb_w),
                   int(track.inner[i, 1] + track.norm[i, 1] * kerb_w))
            ik1 = (int(track.inner[j, 0] + track.norm[j, 0] * kerb_w),
                   int(track.inner[j, 1] + track.norm[j, 1] * kerb_w))
            pygame.draw.polygon(full, color, [ip0, ip1, ik1, ik0])

        for i in range(0, n, 8):
            j  = (i + 4) % n
            p0 = (int(pts[i, 0]), int(pts[i, 1]))
            p1 = (int(pts[j, 0]), int(pts[j, 1]))
            pygame.draw.line(full, (100, 100, 115), p0, p1, 1)

        self._track_surf_full = full

    # ── Main draw ─────────────────────────────────────────────────────────────

    def draw(self, track: Track, pop: PopulationManager, fps: float,
             player=None, cam: Camera | None = None):
        self.prerender_track(track)
        surf = (self._track_surf_full if self.show_track_lines
                else self._track_surf_bare)

        # Blit visible viewport portion of the (possibly large) track surface
        cam_x = int(cam.world_x) if cam else 0
        cam_y = int(cam.world_y) if cam else 0
        src_w = min(TRACK_AREA_W, surf.get_width()  - cam_x)
        src_h = min(TRACK_AREA_H, surf.get_height() - cam_y)
        if src_w > 0 and src_h > 0:
            src_rect = pygame.Rect(cam_x, cam_y, src_w, src_h)
            self.surface.blit(surf, (TRACK_AREA_X, 0), area=src_rect)

        self._draw_agents(pop, cam)

        if player is not None and player.enabled:
            self.draw_player(player, cam)

        font  = _font(14)
        fps_s = font.render(f"{fps:.0f} fps", True, C["text_dim"])
        self.surface.blit(fps_s, (TRACK_AREA_X + 8, 8))

        # Mini-map for large tracks
        if cam and (track.world_w > TRACK_AREA_W or track.world_h > TRACK_AREA_H):
            self._draw_minimap(track, pop, player, cam)

    # ── Click / selection ─────────────────────────────────────────────────────

    def hit_test(self, mx: int, my: int, pop: PopulationManager,
                 player=None, cam: Camera | None = None) -> tuple[int, bool]:
        from car import Car
        R2 = (Car.LENGTH_PX + 4) ** 2
        cam_x = cam.world_x if cam else 0
        cam_y = cam.world_y if cam else 0
        # Convert screen → world
        wx = mx - TRACK_AREA_X + cam_x
        wy = my + cam_y

        if player is not None and player.enabled:
            dx = wx - player.car.x
            dy = wy - player.car.y
            if dx * dx + dy * dy <= R2:
                return -1, True

        for i, ag in enumerate(pop.agents):
            dx = wx - ag.car.x
            dy = wy - ag.car.y
            if dx * dx + dy * dy <= R2:
                return i, False
        return -1, False

    # ── AI agents ─────────────────────────────────────────────────────────────

    def _draw_agents(self, pop: PopulationManager, cam: Camera | None):
        best    = pop.best_agent
        ranked  = sorted(
            [(i, ag) for i, ag in enumerate(pop.agents)],
            key=lambda x: -x[1].fitness
        )

        for detail_rank, (i, agent) in enumerate(ranked):
            car     = agent.car
            is_best = (agent is best) and car.alive
            is_sel  = (i == self.selected_idx)
            color   = car_color(i)

            # Viewport culling (skip off-screen cars)
            if cam and not cam.in_viewport(car.x, car.y, 30):
                continue

            sx, sy = _w2s(car.x, car.y, cam)

            if not car.alive:
                self._draw_car_f1_ghost(sx, sy, car.heading, 0.0, color)
            elif detail_rank < FULL_DETAIL:
                self._draw_car_f1(sx, sy, car.heading,
                                  getattr(car, "delta", 0.0),
                                  color, is_best, is_sel)
                if self.show_rays and (is_best or is_sel):
                    self._draw_rays(car, pop.track, cam)
            else:
                # Tiny dot for background cars
                pygame.draw.circle(self.surface, color, (sx, sy), 3)

    # ── F1 car drawing ────────────────────────────────────────────────────────

    def _draw_car_f1(self, sx: int, sy: int,
                     heading: float, delta: float,
                     color, is_best: bool, is_sel: bool):
        """Draw an open-wheel F1-style car at screen position (sx, sy)."""
        ch, sh = math.cos(heading), math.sin(heading)

        def rot(lx, ly):
            """Rotate local offset (lx=fwd, ly=left) → screen coords."""
            return (sx + int(lx * ch - ly * sh),
                    sy + int(lx * sh + ly * ch))

        # Selection circle
        if is_sel:
            pygame.draw.circle(self.surface, (255, 255, 80), (sx, sy), 18, 2)

        # Rear wing (wide bar at back)
        rw = [rot(-8, -7), rot(-6, -7), rot(-6, 7), rot(-8, 7)]
        pygame.draw.polygon(self.surface, color, rw)

        # Body (narrow monocoque)
        body = [rot(-7, -2), rot(8, -2), rot(8, 2), rot(-7, 2)]
        pygame.draw.polygon(self.surface, color, body)

        # Front wing (very wide, thin)
        fw = [rot(7, -7), rot(9, -7), rot(9, 7), rot(7, 7)]
        pygame.draw.polygon(self.surface, color, fw)

        # Cockpit tub highlight
        ct = [rot(-2, -1), rot(3, -1), rot(3, 1), rot(-2, 1)]
        pygame.draw.polygon(self.surface, (min(255, color[0]+60),
                                           min(255, color[1]+60),
                                           min(255, color[2]+60)), ct)

        # Rear wheels (aligned with car heading)
        for side in (-1, 1):
            self._draw_wheel(sx, sy, heading, 0.0, -6, side * 5, 3, 1.5, False)

        # Front wheels (rotated by steering angle delta)
        for side in (-1, 1):
            self._draw_wheel(sx, sy, heading, delta, 6, side * 5, 3, 1.5, True)

        # Direction nub
        nx = sx + int(9 * ch)
        ny = sy + int(9 * sh)
        pygame.draw.circle(self.surface, (255, 255, 255), (nx, ny), 1)

        # Outline for best / selected
        if is_best:
            pygame.draw.polygon(self.surface, (255, 255, 255), body, 1)
        elif is_sel:
            pygame.draw.polygon(self.surface, (255, 255, 80), body, 1)

    def _draw_wheel(self, sx, sy, heading, delta,
                    fwd_offset, lat_offset, half_len, half_w, steered):
        """Draw a single wheel rectangle."""
        ch, sh = math.cos(heading), math.sin(heading)
        # Wheel centre in screen coords
        cx = sx + int(fwd_offset * ch - lat_offset * sh)
        cy = sy + int(fwd_offset * sh + lat_offset * ch)

        wa = heading + delta if steered else heading
        wc, ws = math.cos(wa), math.sin(wa)
        sign = 1 if lat_offset >= 0 else -1

        corners = [
            (cx + int( half_len * wc - half_w * ws * sign),
             cy + int( half_len * ws + half_w * wc * sign)),
            (cx + int(-half_len * wc - half_w * ws * sign),
             cy + int(-half_len * ws + half_w * wc * sign)),
            (cx + int(-half_len * wc + half_w * ws * sign),
             cy + int(-half_len * ws - half_w * wc * sign)),
            (cx + int( half_len * wc + half_w * ws * sign),
             cy + int( half_len * ws - half_w * wc * sign)),
        ]
        pygame.draw.polygon(self.surface, (40, 40, 40), corners)
        pygame.draw.polygon(self.surface, (80, 80, 80), corners, 1)

    def _draw_car_f1_ghost(self, sx, sy, heading, delta, color):
        """Semi-transparent ghost for crashed cars."""
        ghost = pygame.Surface((TRACK_AREA_W, TRACK_AREA_H), pygame.SRCALPHA)
        ch, sh = math.cos(heading), math.sin(heading)

        def rot(lx, ly):
            gx = sx - TRACK_AREA_X
            return (gx + int(lx * ch - ly * sh),
                    sy + int(lx * sh + ly * ch))

        r, g, b = color
        body = [rot(-7, -2), rot(8, -2), rot(8, 2), rot(-7, 2)]
        pygame.draw.polygon(ghost, (r, g, b, 35), body)
        self.surface.blit(ghost, (TRACK_AREA_X, 0))

    def _draw_rays(self, car, track: Track, cam: Camera | None):
        step  = math.pi / (RAYCAST_COUNT - 1)
        sx, sy = _w2s(car.x, car.y, cam)
        for i in range(RAYCAST_COUNT):
            angle = car.heading - math.pi / 2 + i * step
            dist  = track.raycast_fast(car.x, car.y, angle, RAYCAST_DIST) * RAYCAST_DIST
            ex, ey = _w2s(car.x + dist * math.cos(angle),
                          car.y + dist * math.sin(angle), cam)
            pygame.draw.line(self.surface, (255, 240, 60), (sx, sy), (ex, ey), 1)

    # ── Player car ────────────────────────────────────────────────────────────

    def draw_player(self, player, cam: Camera | None = None):
        car    = player.car
        sx, sy = _w2s(car.x, car.y, cam)

        # Glow
        pygame.draw.circle(self.surface, C["player"], (sx, sy), 20, 2)
        # Draw as F1 but in cyan
        self._draw_car_f1(sx, sy, car.heading,
                          getattr(car, "delta", 0.0),
                          C["player"], False, False)
        pygame.draw.circle(self.surface, (255, 255, 255), (sx, sy - 22), 5, 1)

        lbl = _font(11).render("P", True, (255, 255, 255))
        self.surface.blit(lbl, (sx - lbl.get_width() // 2, sy - 22))

        rc = player.respawn_countdown
        if rc is not None:
            cnt = _font(14).render(f"RESPAWN {rc:.1f}s", True, C["warn"])
            self.surface.blit(cnt, (sx - cnt.get_width() // 2, sy + 14))

    # ── Mini-map (for large worlds like Daytona) ──────────────────────────────

    def _draw_minimap(self, track: Track, pop, player, cam: Camera):
        MM_W, MM_H = 160, 90
        MM_X = TRACK_AREA_X + TRACK_AREA_W - MM_W - 6
        MM_Y = TRACK_AREA_H - MM_H - 6

        scale_x = MM_W / track.world_w
        scale_y = MM_H / track.world_h

        mm = pygame.Surface((MM_W, MM_H), pygame.SRCALPHA)
        mm.fill((10, 12, 20, 200))

        # Track outline
        pts_mm = [(int(p[0] * scale_x), int(p[1] * scale_y))
                  for p in track.pts]
        if len(pts_mm) >= 2:
            pygame.draw.lines(mm, (80, 85, 100), True, pts_mm, 1)

        # Cars (dots)
        for i, ag in enumerate(pop.agents):
            if ag.car.alive:
                mx = int(ag.car.x * scale_x)
                my = int(ag.car.y * scale_y)
                pygame.draw.circle(mm, car_color(i), (mx, my), 2)

        if player and player.enabled:
            mx = int(player.car.x * scale_x)
            my = int(player.car.y * scale_y)
            pygame.draw.circle(mm, C["player"], (mx, my), 3)

        # Viewport rectangle
        vx  = int(cam.world_x * scale_x)
        vy  = int(cam.world_y * scale_y)
        vw  = int(TRACK_AREA_W * scale_x)
        vh  = int(TRACK_AREA_H * scale_y)
        pygame.draw.rect(mm, (200, 200, 255, 150), (vx, vy, vw, vh), 1)

        pygame.draw.rect(mm, (40, 50, 80, 220), (0, 0, MM_W, MM_H), 1)
        self.surface.blit(mm, (MM_X, MM_Y))


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _w2s(wx: float, wy: float, cam: Camera | None) -> tuple[int, int]:
    """World → screen pixel."""
    if cam:
        return cam.world_to_screen(wx, wy)
    return (int(wx + TRACK_AREA_X), int(wy))


# ── Font cache ────────────────────────────────────────────────────────────────

_font_cache: dict = {}

def _font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        _font_cache[size] = pygame.font.SysFont("monospace", size)
    return _font_cache[size]
