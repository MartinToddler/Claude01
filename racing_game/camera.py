"""
Camera — viewport controller for the racing game.

Transform (with zoom):
    screen_x = (world_x - cam.world_x) * zoom + TRACK_AREA_X
    screen_y = (world_y - cam.world_y) * zoom

Controls:
  WASD            — pan (works whenever follow is OFF, or overrides follow)
  = / + key       — zoom in  (up to ×4)
  - key           — zoom out (down to ×0.20)
  "Follow" button — toggle auto-follow target car
"""

from config import TRACK_AREA_W, TRACK_AREA_H, TRACK_AREA_X

ZOOM_MIN   = 0.20
ZOOM_MAX   = 4.0
ZOOM_STEP  = 1.30   # multiplicative step per key-press


class Camera:
    """Smooth-follow viewport camera with zoom."""

    FOLLOW_SPEED = 6.0     # fraction of gap closed per second (smooth follow)
    PAN_SPEED    = 500.0   # world px/s for WASD pan

    def __init__(self, world_w: float, world_h: float):
        self.world_w      = world_w
        self.world_h      = world_h
        self.world_x      = 0.0      # world coord at left viewport edge
        self.world_y      = 0.0      # world coord at top  viewport edge
        self.zoom         = 1.0      # >1 = zoomed in, <1 = zoomed out
        self.follow       = True     # auto-follow enabled
        self.target       = None     # Agent / PlayerCar or None

    # ── Viewport size in world coords ─────────────────────────────────────────

    @property
    def vp_w(self) -> float:
        """Visible world width at current zoom."""
        return TRACK_AREA_W / self.zoom

    @property
    def vp_h(self) -> float:
        """Visible world height at current zoom."""
        return TRACK_AREA_H / self.zoom

    # ── World ↔ screen helpers ────────────────────────────────────────────────

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        return (int((wx - self.world_x) * self.zoom + TRACK_AREA_X),
                int((wy - self.world_y) * self.zoom))

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        return ((sx - TRACK_AREA_X) / self.zoom + self.world_x,
                sy / self.zoom + self.world_y)

    def in_viewport(self, wx: float, wy: float, margin: float = 30.0) -> bool:
        """True if world point is within visible viewport (+ margin)."""
        return (self.world_x - margin / self.zoom <= wx <=
                self.world_x + self.vp_w + margin / self.zoom and
                self.world_y - margin / self.zoom <= wy <=
                self.world_y + self.vp_h + margin / self.zoom)

    # ── Update (called once per rendered frame) ───────────────────────────────

    def update(self, dt: float, keys=None):
        """
        1. If follow=True and target set → smooth-follow target car.
        2. WASD pan always applied (overrides follow when held).
        """
        moved_manually = False
        if keys is not None:
            import pygame
            dx = dy = 0
            spd = self.PAN_SPEED / self.zoom * dt   # consistent screen speed
            if keys[pygame.K_a]: dx -= spd
            if keys[pygame.K_d]: dx += spd
            if keys[pygame.K_w]: dy -= spd
            if keys[pygame.K_s]: dy += spd
            if dx or dy:
                self.world_x += dx
                self.world_y += dy
                moved_manually = True

        if self.follow and self.target is not None and not moved_manually:
            car   = getattr(self.target, "car", self.target)
            tx    = car.x - self.vp_w / 2
            ty    = car.y - self.vp_h / 2
            alpha = min(1.0, self.FOLLOW_SPEED * dt)
            self.world_x += (tx - self.world_x) * alpha
            self.world_y += (ty - self.world_y) * alpha

        self._clamp()

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def zoom_in(self):
        cx, cy = self._centre_world()
        self.zoom = min(ZOOM_MAX, self.zoom * ZOOM_STEP)
        self._recenter(cx, cy)

    def zoom_out(self):
        cx, cy = self._centre_world()
        self.zoom = max(ZOOM_MIN, self.zoom / ZOOM_STEP)
        self._recenter(cx, cy)

    def _centre_world(self):
        return self.world_x + self.vp_w / 2, self.world_y + self.vp_h / 2

    def _recenter(self, cx, cy):
        self.world_x = cx - self.vp_w / 2
        self.world_y = cy - self.vp_h / 2
        self._clamp()

    # ── Snap helpers ──────────────────────────────────────────────────────────

    def snap_to(self, wx: float, wy: float):
        self.world_x = wx - self.vp_w / 2
        self.world_y = wy - self.vp_h / 2
        self._clamp()

    def snap_to_track_start(self, track):
        si = track.start_idx
        self.snap_to(float(track.pts[si, 0]), float(track.pts[si, 1]))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clamp(self):
        max_x = max(0.0, self.world_w - self.vp_w)
        max_y = max(0.0, self.world_h - self.vp_h)
        self.world_x = max(0.0, min(self.world_x, max_x))
        self.world_y = max(0.0, min(self.world_y, max_y))
