"""
Track definitions, Catmull-Rom spline interpolation, and track-query utilities.

Each track is defined by a list of (x, y) control-points in normalised [0,1]²
space.  `build_track()` interpolates them into a smooth closed loop and returns
a Track object used by the renderer, physics, and AI.
"""

import math
import numpy as np
from config import (
    TRACK_AREA_W, TRACK_AREA_H, TRACK_AREA_X,
    TRACK_PADDING, TRACK_HALF_W,
)

# ── Catmull-Rom spline ────────────────────────────────────────────────────────

def _catmull_rom(p0, p1, p2, p3, t):
    """Single Catmull-Rom point at parameter t ∈ [0,1]."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2*p1
        + (-p0 + p2) * t
        + (2*p0 - 5*p1 + 4*p2 - p3) * t2
        + (-p0 + 3*p1 - 3*p2 + p3) * t3
    )

def _spline_points(ctrl, samples_per_seg=40):
    """Interpolate a closed Catmull-Rom spline through ctrl points."""
    n = len(ctrl)
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


# ── Track raw control-point definitions (normalised) ─────────────────────────

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
}

TRACK_NAMES = list(_TRACKS.keys())


# ── Track object ─────────────────────────────────────────────────────────────

class Track:
    """Smooth closed track with per-point tangent, normal, and cumulative dist."""

    def __init__(self, name: str, pts: np.ndarray, half_w: int):
        self.name   = name
        self.pts    = pts          # (N, 2) pixel positions
        self.half_w = half_w       # pixels

        n = len(pts)
        # Tangent vectors (unit)
        nxt = np.roll(pts, -1, axis=0)
        prv = np.roll(pts,  1, axis=0)
        tang = nxt - prv
        lens = np.linalg.norm(tang, axis=1, keepdims=True).clip(1e-9)
        self.tang = tang / lens    # (N, 2)

        # Normal vectors (perpendicular, pointing inward-ish)
        self.norm = np.stack([-self.tang[:, 1], self.tang[:, 0]], axis=1)

        # Cumulative arc-length distances
        seg = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
        self.cum_dist = np.cumsum(seg)
        self.total_dist = float(self.cum_dist[-1])

        # Inner / outer boundary arrays (for rendering)
        self.inner = pts - self.norm * half_w
        self.outer = pts + self.norm * half_w

        # Start-line index: point with smallest y (topmost on screen)
        self.start_idx = int(np.argmin(pts[:, 1]))

    # ── Query helpers ────────────────────────────────────────────────────────

    def closest_idx(self, x: float, y: float) -> int:
        """Return index of the closest track centreline point."""
        d2 = (self.pts[:, 0] - x)**2 + (self.pts[:, 1] - y)**2
        return int(np.argmin(d2))

    def project(self, x: float, y: float):
        """
        Returns (progress_norm, lateral_offset_px, track_heading_rad,
                 speed_limit_hint).
        progress_norm ∈ [0, 1), measured from start_idx.
        lateral_offset_px > 0 means left of centreline.
        """
        idx = self.closest_idx(x, y)
        pt  = self.pts[idx]
        t   = self.tang[idx]
        nr  = self.norm[idx]

        dx = x - pt[0]
        dy = y - pt[1]

        lat = dx * nr[0] + dy * nr[1]   # signed lateral offset

        # Normalised progress: wrap so start_idx = 0.0
        raw_dist = self.cum_dist[idx]
        start_dist = self.cum_dist[self.start_idx]
        rel = (raw_dist - start_dist) % self.total_dist
        progress = rel / self.total_dist

        heading = math.atan2(t[1], t[0])
        return progress, lat, heading

    def raycast(self, x: float, y: float, angle_rad: float,
                max_dist: float) -> float:
        """
        Cast a ray from (x,y) at angle_rad, return distance to track edge
        (normalised 0-1, 0=wall, 1=max_dist).
        Approximated by stepping along the ray and checking lateral offset.
        """
        step = 8.0
        steps = int(max_dist / step)
        for i in range(1, steps + 1):
            d = i * step
            rx = x + d * math.cos(angle_rad)
            ry = y + d * math.sin(angle_rad)
            _, lat, _ = self.project(rx, ry)
            if abs(lat) > self.half_w:
                return (i - 1) * step / max_dist
        return 1.0

    def is_off_track(self, x: float, y: float) -> bool:
        _, lat, _ = self.project(x, y)
        return abs(lat) > self.half_w * 1.1


# ── Build a track from name, scaled to the render area ───────────────────────

def build_track(name: str, samples_per_seg: int = 50) -> Track:
    ctrl = _TRACKS[name]
    raw  = _spline_points(ctrl, samples_per_seg)   # (N, 2) in [0,1]²

    # Compute bounding box of raw points
    mn = raw.min(axis=0)
    mx = raw.max(axis=0)
    span = (mx - mn).clip(1e-9)

    # Available pixel space inside padding
    avail_w = TRACK_AREA_W - 2 * TRACK_PADDING
    avail_h = TRACK_AREA_H - 2 * TRACK_PADDING

    scale = min(avail_w / span[0], avail_h / span[1])

    # Centre inside the track render area
    scaled = (raw - mn) * scale
    sb = scaled.max(axis=0)
    offset_x = TRACK_AREA_X + TRACK_PADDING + (avail_w - sb[0]) / 2
    offset_y =               TRACK_PADDING + (avail_h - sb[1]) / 2

    pts = scaled + np.array([offset_x, offset_y])

    return Track(name, pts, TRACK_HALF_W)


def get_start_pose(track: Track):
    """Return (x, y, heading_rad) for the start/finish line."""
    idx = track.start_idx
    x, y = track.pts[idx]
    heading = math.atan2(track.tang[idx, 1], track.tang[idx, 0])
    return float(x), float(y), heading
