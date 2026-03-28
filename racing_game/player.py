"""
Human-controlled player car.

Uses the same bicycle model as AI cars but receives input from keyboard.
Does NOT participate in genetic evolution — serves as a lap-time benchmark.

Controls:
    ↑ / ↓      throttle / brake
    ← / →      steer left / right
    A           shift up (manual mode)
    Z           shift down (manual mode)
    Q           toggle auto / manual gearbox
"""

import math
import pygame
from car import Car
from config import CAR_DEFAULTS


class PlayerCar:
    """Wraps Car with human input handling and lap timing."""

    COLOR       = (0, 240, 240)   # cyan
    RESPAWN_TIME = 2.0             # seconds off-track before respawn

    def __init__(self, params: dict):
        self.car       = Car(0, 0, 0, params)
        self.auto_gear = True
        self.enabled   = False

        # Lap tracking (same delta-accumulation as Agent)
        self.laps         = 0
        self.best_lap_s   = None    # seconds
        self.last_lap_s   = None
        self.lap_times: list[float] = []
        self._lap_start   = 0.0
        self._best_prog   = 0.0
        self._prev_prog   = None

        # Respawn state
        self._off_track_timer = 0.0
        self._start_x = 0.0
        self._start_y = 0.0
        self._start_h = 0.0

    def reset(self, x: float, y: float, heading: float, params: dict):
        self._start_x, self._start_y, self._start_h = x, y, heading
        self.car.reset(x, y, heading)
        self.car.update_params(params)
        self._prev_prog       = None
        self._best_prog       = 0.0
        self.laps             = 0
        self.best_lap_s       = None
        self.last_lap_s       = None
        self.lap_times        = []
        self._lap_start       = 0.0
        self._off_track_timer = 0.0

    def update_params(self, params: dict):
        self.car.update_params(params)

    # ── Per-frame update ──────────────────────────────────────────────────────

    def step(self, keys, track, time_s: float, dt: float = 1 / 60):
        """Called once per physics frame with current key state."""
        if not self.enabled:
            return

        throttle = 1.0 if keys[pygame.K_UP]    else 0.0
        brake    = 1.0 if keys[pygame.K_DOWN]   else 0.0
        steer    = 0.0
        if keys[pygame.K_LEFT]:  steer -= 1.0
        if keys[pygame.K_RIGHT]: steer += 1.0

        tb = throttle - brake
        self.car.step(tb, steer, dt, auto_gear=self.auto_gear)

        # Off-track respawn
        if track.is_off_track(self.car.x, self.car.y):
            self._off_track_timer += dt
            if self._off_track_timer >= self.RESPAWN_TIME:
                self._respawn()
        else:
            self._off_track_timer = 0.0

        self._update_lap(track, time_s)

    def _respawn(self):
        """Return to start position."""
        self.car.reset(self._start_x, self._start_y, self._start_h)
        self._off_track_timer = 0.0

    # ── Event handling (called from main event loop) ──────────────────────────

    def handle_keydown(self, key):
        """Call from main event loop for KEYDOWN events."""
        if not self.enabled:
            return
        if key == pygame.K_a:
            self.shift_up()
        elif key == pygame.K_z:
            self.shift_down()
        elif key == pygame.K_q:
            self.auto_gear = not self.auto_gear

    def shift_up(self):
        if not self.auto_gear and self.car.gear < self.car.num_gears:
            self.car.gear += 1

    def shift_down(self):
        if not self.auto_gear and self.car.gear > 1:
            self.car.gear -= 1

    # ── Lap tracking ─────────────────────────────────────────────────────────

    def _update_lap(self, track, time_s: float):
        progress, _, _ = track.project(self.car.x, self.car.y)

        if self._prev_prog is None:
            self._prev_prog = progress
            return

        delta = progress - self._prev_prog
        self._prev_prog = progress

        if 0.0 < delta < 0.10:
            self._best_prog += delta

        if self._best_prog >= 1.0:
            lap_t = time_s - self._lap_start
            if lap_t > 2.0:
                self.laps += 1
                self.last_lap_s = lap_t
                self.lap_times.append(lap_t)
                if self.best_lap_s is None or lap_t < self.best_lap_s:
                    self.best_lap_s = lap_t
            self._lap_start  = time_s
            self._best_prog -= 1.0

    # ── Telemetry ─────────────────────────────────────────────────────────────

    @property
    def gear_mode(self) -> str:
        return "AUTO" if self.auto_gear else "MANUAL"

    @property
    def respawn_countdown(self) -> float | None:
        """Seconds remaining until respawn, or None if on track."""
        if self._off_track_timer > 0:
            return max(0.0, self.RESPAWN_TIME - self._off_track_timer)
        return None
