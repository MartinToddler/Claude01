"""
Car physics model — simplified but realistic enough to respond meaningfully
to the tunable parameters (power, gear ratios, tyre compound/pressure,
downforce, driver_risk).

Coordinate system: pygame pixels, x→right, y→down.
Internal physics uses SI units (m/s, N, kg) then converts to pixels/frame.
"""

import math
from config import (
    CAR_DEFAULTS, TYRE_GRIP, OPTIMAL_PRESSURE,
    PRESSURE_GRIP_FALLOFF, DOWNFORCE_GRIP_K, HP_TO_NM,
)

# 1 pixel = PX_PER_M metres  (track half-width 60 px ≈ 7 m real → ~8.5 px/m)
PX_PER_M = 8.5
M_PER_PX = 1.0 / PX_PER_M


class Car:
    """
    State: position (px), heading (rad), speed (m/s), lat_speed (m/s), gear.
    Action: throttle_brake ∈ [-1, 1], steer ∈ [-1, 1].
    """

    # Visual size
    LENGTH_PX = 18
    WIDTH_PX  = 10

    def __init__(self, x: float, y: float, heading: float, params: dict | None = None):
        p = {**CAR_DEFAULTS, **(params or {})}
        self.reset(x, y, heading)
        self._apply_params(p)

    # ── Parameter application ─────────────────────────────────────────────────

    def _apply_params(self, p: dict):
        self.power        = float(p["power"])
        self.gear_ratios  = list(p["gear_ratios"])
        self.final_drive  = float(p["final_drive"])
        self.tyre_type    = p["tyre_type"]
        self.tyre_pressure= float(p["tyre_pressure"])
        self.downforce    = float(p["downforce"])
        self.driver_risk  = float(p["driver_risk"])
        self.mass         = float(p["mass"])
        self.drag_coeff   = float(p["drag_coeff"])
        self.wheel_radius = float(p["wheel_radius"])
        self.wheelbase    = float(p["wheelbase"])   # metres (used for steering geometry)

        # Derived
        self.peak_torque  = self.power * HP_TO_NM   # Nm
        self.num_gears    = len(self.gear_ratios)

    def update_params(self, p: dict):
        self._apply_params(p)

    # ── State reset ───────────────────────────────────────────────────────────

    def reset(self, x: float, y: float, heading: float):
        self.x       = x
        self.y       = y
        self.heading = heading       # radians, 0 = right, π/2 = down
        self.speed   = 0.0          # m/s  (longitudinal)
        self.lat_v   = 0.0          # m/s  (lateral, positive = left drift)
        self.gear    = 1
        self.rpm     = 1000.0
        self.alive   = True
        self.stuck_timer  = 0.0
        self.lap_progress = 0.0     # normalised, increases monotonically
        self.laps_done    = 0
        self.distance_px  = 0.0     # total pixel-distance for fitness

        # telemetry
        self.throttle  = 0.0
        self.steer_val = 0.0

    # ── Physics step ─────────────────────────────────────────────────────────

    def step(self, throttle_brake: float, steer: float, dt: float = 1/60):
        """
        throttle_brake: +1 = full throttle, -1 = full brake
        steer:          +1 = full right,   -1 = full left
        dt:             seconds per frame
        """
        self.throttle  = throttle_brake
        self.steer_val = steer

        # ── Gear shifting ─────────────────────────────────────────────────
        self._auto_shift()

        # ── Engine force ──────────────────────────────────────────────────
        # Torque curve: peaks at 60% of rev range, drops 30% at extremes
        rpm_norm = min(max(self.rpm / 8000, 0), 1)
        torque_factor = 1.0 - 0.3 * (2 * rpm_norm - 1) ** 2   # parabola peak=1 at 50%
        torque = self.peak_torque * torque_factor

        ratio = self.gear_ratios[self.gear - 1] * self.final_drive
        drive_force = torque * ratio / self.wheel_radius   # N (at wheel)

        if throttle_brake >= 0:
            F_long = drive_force * throttle_brake
        else:
            # Braking: brake force = 0.85 g worth
            F_long = self.mass * 9.81 * 0.85 * throttle_brake

        # ── Aerodynamic drag ─────────────────────────────────────────────
        v2 = self.speed ** 2
        F_drag = -self.drag_coeff * 0.5 * 1.2 * 1.8 * v2 * math.copysign(1, self.speed)

        # ── Tyre grip budget ─────────────────────────────────────────────
        base_grip  = TYRE_GRIP[self.tyre_type]
        dp         = self.tyre_pressure - OPTIMAL_PRESSURE
        pres_factor= max(0.5, 1.0 - PRESSURE_GRIP_FALLOFF * dp**2)
        df_bonus   = self.downforce * DOWNFORCE_GRIP_K * v2
        grip_coeff = base_grip * pres_factor + df_bonus

        max_F      = grip_coeff * self.mass * 9.81   # N total tyre budget

        # ── Steering / lateral force ──────────────────────────────────────
        steer_angle = steer * math.radians(25)       # max 25° front steer
        # Lateral acceleration from Ackermann geometry at current speed
        if abs(self.speed) > 0.5:
            turn_radius = self.wheelbase / max(abs(math.tan(steer_angle)), 1e-4)
            F_lat_wanted = self.mass * self.speed**2 / turn_radius * math.copysign(1, steer)
        else:
            F_lat_wanted = 0.0

        # Friction circle: share grip between longitudinal and lateral
        F_long_clamped = min(abs(F_long), max_F) * math.copysign(1, F_long)
        remaining_lat  = math.sqrt(max(0, max_F**2 - F_long_clamped**2))
        F_lat = max(-remaining_lat, min(remaining_lat, F_lat_wanted))

        # ── Integration ───────────────────────────────────────────────────
        a_long = (F_long_clamped + F_drag) / self.mass
        a_lat  = F_lat / self.mass

        self.speed += a_long * dt
        self.speed  = max(-5.0, self.speed)   # small reverse allowed

        # Lateral speed decays (tyre self-aligns)
        self.lat_v  = (self.lat_v + a_lat * dt) * 0.85

        # ── Heading update ────────────────────────────────────────────────
        if abs(self.speed) > 0.3:
            # Yaw rate from lateral force
            yaw_rate = F_lat / (self.mass * max(abs(self.speed), 0.3))
            self.heading += yaw_rate * dt

        # ── Position update (px) ──────────────────────────────────────────
        spx = self.speed  * PX_PER_M * dt
        lpx = self.lat_v  * PX_PER_M * dt
        self.x += spx * math.cos(self.heading) - lpx * math.sin(self.heading)
        self.y += spx * math.sin(self.heading) + lpx * math.cos(self.heading)

        self.distance_px += abs(spx)

        # ── RPM update ────────────────────────────────────────────────────
        wheel_rads = self.speed / self.wheel_radius
        ratio = self.gear_ratios[self.gear - 1] * self.final_drive
        self.rpm = max(800, wheel_rads * ratio * 60 / (2 * math.pi))

    # ── Auto gearbox ─────────────────────────────────────────────────────────

    def _auto_shift(self):
        if self.rpm > 7200 and self.gear < self.num_gears:
            self.gear += 1
        elif self.rpm < 2500 and self.gear > 1:
            self.gear -= 1

    # ── Stuck detection ──────────────────────────────────────────────────────

    def update_stuck(self, dt: float, stuck_speed: float, stuck_time: float):
        if abs(self.speed) < stuck_speed:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0
        if self.stuck_timer > stuck_time:
            self.alive = False

    # ── Speed helpers ─────────────────────────────────────────────────────────

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6

    @property
    def speed_norm(self) -> float:
        """Normalised speed, 0-1 where 1 ≈ 300 km/h."""
        return min(abs(self.speed) / 83.3, 1.0)

    # ── Telemetry dict ────────────────────────────────────────────────────────

    def telemetry(self) -> dict:
        return {
            "speed_kmh": self.speed_kmh,
            "gear":      self.gear,
            "rpm":       self.rpm,
            "throttle":  self.throttle,
            "steer":     self.steer_val,
        }
