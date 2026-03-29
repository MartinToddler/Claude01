"""
Camera — viewport controller for the racing game.

Supports smooth auto-follow of a target car and optional manual pan.
World coordinates are independent of screen layout; the camera provides the
transform:
    screen_x = world_x - cam.world_x + TRACK_AREA_X
    screen_y = world_y - cam.world_y

For small tracks (world ≤ viewport) cam stays near (0, 0) and effectively
acts as a passthrough.  For Daytona (4800 × 2600) the camera scrolls to keep
the target car centred.
"""

import math
from config import TRACK_AREA_W, TRACK_AREA_H, TRACK_AREA_X


class Camera:
    """Smooth-follow viewport camera."""

    # Follow speed: fraction of gap closed per second
    FOLLOW_SPEED = 6.0
    # Pan speed in px/s when using keyboard pan
    PAN_SPEED    = 400.0

    def __init__(self, world_w: float, world_h: float):
        self.world_w  = world_w   # total world width
        self.world_h  = world_h   # total world height
        self.world_x  = 0.0       # left edge of viewport in world coords
        self.world_y  = 0.0       # top  edge of viewport in world coords
        self.target   = None      # object with .car.x / .car.y  (or None)
        self._drag    = False
        self._drag_start_screen = (0, 0)
        self._drag_start_world  = (0.0, 0.0)

    # ── World ↔ screen helpers ────────────────────────────────────────────────

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates to screen pixel position."""
        return (int(wx - self.world_x + TRACK_AREA_X),
                int(wy - self.world_y))

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        """Convert screen pixel position to world coordinates."""
        return (float(sx - TRACK_AREA_X + self.world_x),
                float(sy + self.world_y))

    def in_viewport(self, wx: float, wy: float,
                    margin: float = 30.0) -> bool:
        """True if world point is within the visible viewport (+ margin)."""
        return (self.world_x - margin <= wx <= self.world_x + TRACK_AREA_W + margin and
                self.world_y - margin <= wy <= self.world_y + TRACK_AREA_H + margin)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float):
        """Advance camera toward target (call once per rendered frame)."""
        if self.target is None:
            return
        # Resolve car attribute — target may be Agent or PlayerCar
        car = getattr(self.target, "car", self.target)
        tx  = car.x - TRACK_AREA_W / 2
        ty  = car.y - TRACK_AREA_H / 2
        alpha = min(1.0, self.FOLLOW_SPEED * dt)
        self.world_x += (tx - self.world_x) * alpha
        self.world_y += (ty - self.world_y) * alpha
        self._clamp()

    def snap_to(self, wx: float, wy: float):
        """Immediately centre viewport on (wx, wy) without smoothing."""
        self.world_x = wx - TRACK_AREA_W / 2
        self.world_y = wy - TRACK_AREA_H / 2
        self._clamp()

    def snap_to_track_start(self, track):
        """Jump camera to show the track start/finish line."""
        si = track.start_idx
        self.snap_to(float(track.pts[si, 0]), float(track.pts[si, 1]))

    # ── Mouse drag pan ────────────────────────────────────────────────────────

    def begin_drag(self, screen_pos: tuple):
        """Start manual pan (middle-mouse button pressed)."""
        if self.target is not None:
            return   # don't pan while following a car
        self._drag = True
        self._drag_start_screen = screen_pos
        self._drag_start_world  = (self.world_x, self.world_y)

    def update_drag(self, screen_pos: tuple):
        if not self._drag:
            return
        dx = screen_pos[0] - self._drag_start_screen[0]
        dy = screen_pos[1] - self._drag_start_screen[1]
        self.world_x = self._drag_start_world[0] - dx
        self.world_y = self._drag_start_world[1] - dy
        self._clamp()

    def end_drag(self):
        self._drag = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clamp(self):
        """Keep camera within world bounds."""
        max_x = max(0.0, self.world_w - TRACK_AREA_W)
        max_y = max(0.0, self.world_h - TRACK_AREA_H)
        self.world_x = max(0.0, min(self.world_x, max_x))
        self.world_y = max(0.0, min(self.world_y, max_y))
