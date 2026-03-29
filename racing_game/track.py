"""
Track definitions, Catmull-Rom spline interpolation, and track-query utilities.

Each track is defined by a list of (x, y) control-points in normalised [0,1]²
space.  `build_track()` interpolates them into a smooth closed loop and returns
a Track object used by the renderer, physics, and AI.

v4 changes:
  • Coordinates are now pure WORLD space — TRACK_AREA_X is NOT baked into pts.
    The camera / renderer applies the screen offset externally.
  • Each Track stores world_w / world_h (viewport-sized for small tracks, large
    for Daytona so the camera can scroll).
  • _build_lat_field(): precomputed 2-D signed lateral-distance lookup table
    used by raycast_fast() for O(1) per-sample raycasting (eliminates Python
    loops in the sensing hot-path).
  • Daytona Speedway: 4800 × 2600 px world, tri-oval layout.
"""

import math
import numpy as np
from config import (
    TRACK_AREA_W, TRACK_AREA_H,
    TRACK_PADDING, TRACK_HALF_W,
)

# ── Catmull-Rom spline ────────────────────────────────────────────────────────

def _catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2*p0 - 5*p1 + 4*p2 - p3) * t2
        + (-p0 + 3*p1 - 3*p2 + p3) * t3
    )

def _spline_points(ctrl, samples_per_seg=40):
    """Interpolate a closed Catmull-Rom spline through ctrl points."""
    n   = len(ctrl)
    pts = []
    for i in range(n):
        p0 = np.array(ctrl[(i - 1) % n])
        p1 = np.array(ctrl[i])
        p2 = np.array(ctrl[(i + 1) % n])
        p3 = np.array(ctrl[(i + 2) % n])
        for k in range(samples_per_seg):
            t = k / samples_per_seg
            pts.append(_catmull_rom(p0, p1, p2, p3, t))
    return np.array(pts)


# ── Track raw control-point definitions (normalised [0,1]²) ──────────────────

_TRACKS = {
    "Oval": [
        (0.25, 0.20), (0.50, 0.15), (0.75, 0.20),
        (0.85, 0.35), (0.85, 0.65),
        (0.75, 0.80), (0.50, 0.85), (0.25, 0.80),
        (0.15, 0.65), (0.15, 0.35),
    ],

    "Monza": [
        (0.50, 0.12), (0.72, 0.12), (0.82, 0.20),
        (0.82, 0.35), (0.70, 0.40), (0.82, 0.50),
        (0.82, 0.65), (0.72, 0.72), (0.55, 0.75),
        (0.55, 0.88), (0.45, 0.88), (0.45, 0.75),
        (0.28, 0.72), (0.18, 0.65), (0.18, 0.50),
        (0.30, 0.40), (0.18, 0.35), (0.18, 0.20),
        (0.28, 0.12),
    ],

    "Monaco": [
        (0.50, 0.10), (0.65, 0.10), (0.78, 0.15),
        (0.85, 0.25), (0.80, 0.35), (0.68, 0.38),
        (0.75, 0.48), (0.80, 0.58), (0.75, 0.68),
        (0.62, 0.73), (0.52, 0.80), (0.52, 0.90),
        (0.42, 0.90), (0.35, 0.82), (0.28, 0.72),
        (0.22, 0.60), (0.20, 0.48), (0.25, 0.38),
        (0.20, 0.28), (0.28, 0.18), (0.40, 0.12),
    ],

    # ── Laguna Seca Raceway ───────────────────────────────────────────────────
    # 3.6 km technical circuit. Signature: Corkscrew (T8A/9) — blind crest
    # followed by sharp downhill chicane.  World: 1600 × 1200 px.
    "Laguna": [
        # Start/Finish straight → Turn 1 (hard right)
        (0.22, 0.36), (0.34, 0.28), (0.46, 0.20),
        # Turn 2 sweeping right
        (0.60, 0.18), (0.72, 0.22),
        # Turn 3 left kink
        (0.78, 0.30),
        # Turn 4 (right) → back section
        (0.82, 0.40), (0.80, 0.50),
        # Turn 5 left
        (0.74, 0.56),
        # Turn 6 right
        (0.80, 0.64),
        # Turn 7 left
        (0.74, 0.71),
        # Corkscrew T8A: blind right at crest
        (0.68, 0.78),
        # Corkscrew T9: sharp left going downhill
        (0.57, 0.84), (0.45, 0.86),
        # Turn 10: wide sweeping right
        (0.34, 0.82), (0.24, 0.76),
        # Turn 11: very slow hairpin (tightest corner)
        (0.16, 0.68), (0.13, 0.58), (0.15, 0.49),
        # Return on pit-straight
        (0.16, 0.42),
    ],

    # ── SuperOval (fictional 20 km superspeedway) ────────────────────────────
    # World: 10 000 × 3 000 px.  1 px ≈ 1 m.
    # Two long straights + wide banked turns → total length ≈ 20 km.
    "SuperOval": [
        # Start/finish on bottom straight (left to right)
        (0.14, 0.72), (0.30, 0.75), (0.50, 0.76),
        (0.70, 0.75), (0.86, 0.72),
        # Right banked turn
        (0.94, 0.63), (0.97, 0.52), (0.94, 0.41),
        # Top straight (right to left)
        (0.86, 0.32), (0.70, 0.29), (0.50, 0.28),
        (0.30, 0.29), (0.14, 0.32),
        # Left banked turn
        (0.06, 0.41), (0.03, 0.52), (0.06, 0.63),
    ],

    # ── NASCAR Daytona International Speedway (tri-oval) ─────────────────────
    # Real proportions: ~2.5 miles, tri-oval shape.
    # World size: 4800 × 2600 px.  One car-length (18 px) ≈ 5 m → 1 px ≈ 0.28 m.
    # The slight kink on the front straight is the "tri-oval" signature.
    "Daytona": [
        # Front straight — slight left kink (tri-oval dog-leg)
        (0.13, 0.46), (0.20, 0.44), (0.30, 0.43),
        (0.40, 0.44), (0.48, 0.46),
        # Turn 1 (high-bank right)
        (0.60, 0.40), (0.70, 0.30), (0.80, 0.26),
        (0.88, 0.30), (0.92, 0.38),
        # Back straight
        (0.93, 0.50), (0.92, 0.62),
        # Turn 3 (high-bank right)
        (0.88, 0.70), (0.80, 0.74), (0.70, 0.70),
        (0.60, 0.60),
        # Return to front straight
        (0.48, 0.54), (0.40, 0.56), (0.30, 0.57),
        (0.20, 0.56), (0.13, 0.54),
    ],
}

# World dimensions per track (width × height in px)
_WORLD_SIZES = {
    "Oval":      (TRACK_AREA_W, TRACK_AREA_H),
    "Monza":     (TRACK_AREA_W, TRACK_AREA_H),
    "Monaco":    (TRACK_AREA_W, TRACK_AREA_H),
    "Laguna":    (1600, 1200),
    "SuperOval": (14000, 4200),
    "Daytona":   (4800, 2600),
}

# Half-width of the driveable surface per track (px)
_TRACK_HALF_W = {
    "Oval":      60,
    "Monza":     55,
    "Monaco":    45,
    "Laguna":    45,
    "SuperOval": 80,
    "Daytona":   70,
}

TRACK_NAMES = list(_TRACKS.keys())


# ── Track object ─────────────────────────────────────────────────────────────

class Track:
    """Smooth closed track with per-point tangent, normal, and cumulative dist."""

    def __init__(self, name: str, pts: np.ndarray, half_w: int,
                 world_w: float, world_h: float):
        self.name    = name
        self.pts     = pts          # (N, 2) world-space pixel positions
        self.half_w  = half_w       # pixels
        self.world_w = world_w
        self.world_h = world_h

        n = len(pts)
        # Tangent vectors (unit)
        nxt  = np.roll(pts, -1, axis=0)
        prv  = np.roll(pts,  1, axis=0)
        tang = nxt - prv
        lens = np.linalg.norm(tang, axis=1, keepdims=True).clip(1e-9)
        self.tang = tang / lens    # (N, 2)

        # Normal vectors (perpendicular, pointing inward)
        self.norm = np.stack([-self.tang[:, 1], self.tang[:, 0]], axis=1)

        # Cumulative arc-length distances
        seg = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
        self.cum_dist   = np.cumsum(seg)
        self.total_dist = float(self.cum_dist[-1])

        # Inner / outer boundary arrays (for rendering)
        self.inner = pts - self.norm * half_w
        self.outer = pts + self.norm * half_w

        # Start-line index: point with smallest y (topmost on screen)
        self.start_idx = int(np.argmin(pts[:, 1]))

        # Precompute lateral distance field for fast raycasting
        self._build_lat_field()

    # ── Lateral distance field ────────────────────────────────────────────────

    def _build_lat_field(self, step: int = 3):
        """
        Build a 2-D lookup table where lat_field[yi, xi] = signed lateral
        distance from the track centreline (positive = left-of-heading).

        Uses a local-neighbourhood flood: for each track point we update only
        the nearby cells, keeping the nearest-point assignment.  This runs in
        O(N × margin²) which is fast enough at startup.
        """
        self._lat_step = step
        w = int(math.ceil(self.world_w / step)) + 2
        h = int(math.ceil(self.world_h / step)) + 2

        best_d2   = np.full((h, w), 1e18, dtype=np.float64)
        lat_field = np.zeros((h, w), dtype=np.float32)

        margin = int(math.ceil((self.half_w + 40) / step)) + 1

        for i in range(len(self.pts)):
            px, py = float(self.pts[i, 0]), float(self.pts[i, 1])
            nx, ny = float(self.norm[i, 0]), float(self.norm[i, 1])

            ix_c  = int(px / step)
            iy_c  = int(py / step)
            ix_lo = max(0, ix_c - margin)
            ix_hi = min(w, ix_c + margin + 1)
            iy_lo = max(0, iy_c - margin)
            iy_hi = min(h, iy_c + margin + 1)

            gx = (np.arange(ix_lo, ix_hi) * step).astype(np.float32)
            gy = (np.arange(iy_lo, iy_hi) * step).astype(np.float32)
            xx, yy = np.meshgrid(gx, gy)   # (ny_rng, nx_rng)

            dx  = xx - px
            dy  = yy - py
            d2  = dx * dx + dy * dy
            lat = (dx * nx + dy * ny).astype(np.float32)

            cur_d2 = best_d2[iy_lo:iy_hi, ix_lo:ix_hi]
            mask   = d2 < cur_d2

            best_d2[iy_lo:iy_hi, ix_lo:ix_hi]   = np.where(mask, d2, cur_d2)
            lat_field[iy_lo:iy_hi, ix_lo:ix_hi] = np.where(
                mask, lat, lat_field[iy_lo:iy_hi, ix_lo:ix_hi])

        self.lat_field = lat_field

    # ── Query helpers ─────────────────────────────────────────────────────────

    def closest_idx(self, x: float, y: float) -> int:
        d2 = (self.pts[:, 0] - x)**2 + (self.pts[:, 1] - y)**2
        return int(np.argmin(d2))

    def project(self, x: float, y: float):
        """
        Returns (progress_norm, lateral_offset_px, track_heading_rad).
        progress_norm ∈ [0, 1), measured from start_idx.
        lateral_offset_px > 0 → left of centreline.
        """
        idx = self.closest_idx(x, y)
        pt  = self.pts[idx]
        t   = self.tang[idx]
        nr  = self.norm[idx]

        dx  = x - pt[0]
        dy  = y - pt[1]
        lat = dx * nr[0] + dy * nr[1]

        raw_dist   = self.cum_dist[idx]
        start_dist = self.cum_dist[self.start_idx]
        rel        = (raw_dist - start_dist) % self.total_dist
        progress   = rel / self.total_dist

        heading = math.atan2(t[1], t[0])
        return progress, lat, heading

    def raycast_fast(self, ox: float, oy: float,
                     angle_rad: float, max_dist: float) -> float:
        """
        Fast ray-to-track-edge cast using the precomputed lat_field.
        Returns normalised distance ∈ [0, 1]  (0 = wall, 1 = max_dist).
        Pure numpy — no Python loop over samples.
        """
        step_px = 6.0
        n_steps = max(1, int(max_dist / step_px))
        t   = (np.arange(1, n_steps + 1) * step_px).astype(np.float32)
        rxs = (ox + t * math.cos(angle_rad)).astype(np.float32)
        rys = (oy + t * math.sin(angle_rad)).astype(np.float32)

        s   = self._lat_step
        fh, fw = self.lat_field.shape
        xi  = np.clip((rxs / s).astype(np.int32), 0, fw - 1)
        yi  = np.clip((rys / s).astype(np.int32), 0, fh - 1)

        lats = np.abs(self.lat_field[yi, xi])
        hits = np.nonzero(lats > self.half_w)[0]
        if len(hits):
            return float(max(0.0, (hits[0] * step_px - step_px) / max_dist))
        return 1.0

    # Keep old raycast as fallback (used by renderer ray-visualisation)
    def raycast(self, x: float, y: float, angle_rad: float,
                max_dist: float) -> float:
        return self.raycast_fast(x, y, angle_rad, max_dist)

    def is_off_track(self, x: float, y: float) -> bool:
        _, lat, _ = self.project(x, y)
        return abs(lat) > self.half_w * 1.1


# ── Build a track from name ───────────────────────────────────────────────────

def build_track(name: str, samples_per_seg: int = 50) -> Track:
    """
    Interpolate control points → smooth closed loop → Track object.

    World coordinates:
      • For standard tracks (Oval / Monza / Monaco): world = viewport size
        (TRACK_AREA_W × TRACK_AREA_H).  No TRACK_AREA_X offset is baked in;
        the camera/renderer adds that offset externally.
      • For Laguna: world = 1600 × 1200 px.
      • For Daytona: world = 4800 × 2600 px.
      • For SuperOval: world = 10 000 × 3 000 px (camera must scroll).
    """
    ctrl = _TRACKS[name]
    raw  = _spline_points(ctrl, samples_per_seg)   # (N, 2) in [0,1]²

    world_w, world_h = _WORLD_SIZES.get(name, (TRACK_AREA_W, TRACK_AREA_H))

    mn   = raw.min(axis=0)
    mx   = raw.max(axis=0)
    span = (mx - mn).clip(1e-9)

    avail_w = world_w - 2 * TRACK_PADDING
    avail_h = world_h - 2 * TRACK_PADDING
    scale   = min(avail_w / span[0], avail_h / span[1])

    scaled   = (raw - mn) * scale
    sb       = scaled.max(axis=0)
    offset_x = TRACK_PADDING + (avail_w - sb[0]) / 2
    offset_y = TRACK_PADDING + (avail_h - sb[1]) / 2

    pts = scaled + np.array([offset_x, offset_y])

    half_w = _TRACK_HALF_W.get(name, TRACK_HALF_W)
    return Track(name, pts, half_w, world_w, world_h)


def get_start_pose(track: Track):
    """Return (x, y, heading_rad) for the start/finish line."""
    idx     = track.start_idx
    x, y    = track.pts[idx]
    heading = math.atan2(track.tang[idx, 1], track.tang[idx, 0])
    return float(x), float(y), heading
