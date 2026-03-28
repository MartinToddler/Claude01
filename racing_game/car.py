"""
Vehicle dynamics using the single-track (bicycle) model with Pacejka
"Magic Formula" tyre model.

This is the standard model used in vehicle dynamics research and professional
racing simulators — it correctly captures:
  • Tyre slip angles (front / rear)
  • Load-dependent lateral grip (Pacejka D = mu * Fz)
  • Longitudinal weight transfer under braking/acceleration
  • Speed-dependent downforce (increased normal load → more grip)
  • Oversteer / understeer characteristics
  • Tyre saturation at high slip angles

State (vehicle body frame, x=forward, y=rightward):
  x, y       — world position (pixels)
  heading    — yaw angle (radians; 0=right, increases CW in pygame y-down)
  vx         — longitudinal speed (m/s)
  vy         — lateral speed, positive = rightward in car frame (m/s)
  omega      — yaw rate (rad/s, positive = CW = turning right)

Control inputs (normalised):
  throttle_brake ∈ [-1, 1]   (-1=full brake, +1=full throttle)
  steer          ∈ [-1, 1]   (-1=full left,   +1=full right)
"""

import math
from config import (
    CAR_DEFAULTS, TYRE_GRIP, OPTIMAL_PRESSURE, PRESSURE_GRIP_FALLOFF,
    HP_TO_NM,
    VEH_L_F, VEH_L_R, VEH_I_Z, VEH_H_CG,
    VEH_WHEEL_R, VEH_CD, VEH_A_FRONT,
    PAC_B, PAC_C, PAC_E,
)

PX_PER_M = 8.5      # 1 metre = 8.5 pixels
G = 9.81            # m/s²


class Car:
    # Visual dimensions
    LENGTH_PX = 18
    WIDTH_PX  = 10

    def __init__(self, x: float, y: float, heading: float,
                 params: dict | None = None):
        p = {**CAR_DEFAULTS, **(params or {})}
        self._apply_params(p)
        self.reset(x, y, heading)

    # ── Parameter application ─────────────────────────────────────────────────

    def _apply_params(self, p: dict):
        self.power         = float(p["power"])
        self.gear_ratios   = list(p["gear_ratios"])
        self.final_drive   = float(p.get("final_drive", 3.7))
        self.tyre_type     = p["tyre_type"]
        self.tyre_pressure = float(p["tyre_pressure"])
        self.downforce_kg  = float(p["downforce"])
        self.driver_risk   = float(p["driver_risk"])
        self.mass          = float(p["mass"])

        # Derived
        self.peak_torque = self.power * HP_TO_NM
        self.num_gears   = len(self.gear_ratios)
        self._L          = VEH_L_F + VEH_L_R   # wheelbase (m)

    def update_params(self, p: dict):
        self._apply_params(p)

    # ── State reset ───────────────────────────────────────────────────────────

    def reset(self, x: float, y: float, heading: float):
        self.x       = x
        self.y       = y
        self.heading = heading
        self.vx      = 5.0    # initial push-start speed (m/s forward)
        self.vy      = 0.0    # lateral velocity
        self.omega   = 0.0    # yaw rate
        self.gear    = 1
        self.rpm     = 2000.0

        self.alive        = True
        self.stuck_timer  = 0.0
        self.distance_px  = 0.0

        # Telemetry outputs
        self.throttle  = 0.0
        self.steer_val = 0.0
        self._ax_prev  = 0.0   # previous longitudinal accel for weight transfer

    # ── Tyre helpers ─────────────────────────────────────────────────────────

    def _mu(self) -> float:
        """Effective friction coefficient (compound × pressure)."""
        base = TYRE_GRIP[self.tyre_type]
        dp   = self.tyre_pressure - OPTIMAL_PRESSURE
        return base * max(0.5, 1.0 - PRESSURE_GRIP_FALLOFF * dp ** 2)

    def _pacejka(self, alpha: float, D: float) -> float:
        """
        Pacejka Magic Formula — lateral force for slip angle alpha.
        F_y = D·sin(C·atan(B·α − E·(B·α − atan(B·α))))
        Returns positive force for positive alpha (slides right → pushes left
        correction: this returns unsigned; caller applies sign convention).
        """
        # Downforce stiffens the tyre (increases B slightly)
        v2  = self.vx ** 2
        df_ratio = self.downforce_kg / 100.0
        B   = PAC_B * (1.0 + 0.3 * df_ratio)
        C   = PAC_C
        E   = PAC_E
        Ba  = B * alpha
        return D * math.sin(C * math.atan(Ba - E * (Ba - math.atan(Ba))))

    # ── Gear auto-shift ───────────────────────────────────────────────────────

    def _auto_shift(self):
        if self.rpm > 7200 and self.gear < self.num_gears:
            self.gear += 1
        elif self.rpm < 2500 and self.gear > 1:
            self.gear -= 1

    def _torque_factor(self) -> float:
        """Normalised torque curve (1.0 at peak RPM ≈ 5 000)."""
        rn = min(max(self.rpm / 8000.0, 0.0), 1.0)
        return 1.0 - 0.35 * (2.0 * rn - 1.0) ** 2

    # ── Physics step ─────────────────────────────────────────────────────────

    def step(self, throttle_brake: float, steer: float, dt: float = 1 / 60):
        self.throttle  = throttle_brake
        self.steer_val = steer

        # Decompose input
        if throttle_brake >= 0:
            throttle, brake = throttle_brake, 0.0
        else:
            throttle, brake = 0.0, -throttle_brake

        # Speed-dependent steering limit: 25° at rest → ~4° at 200 km/h
        v_kmh   = abs(self.vx) * 3.6
        dmax_deg = max(4.0, 25.0 - 0.105 * v_kmh)
        delta    = steer * math.radians(dmax_deg)   # front wheel angle (rad)

        # Gear shift
        self._auto_shift()

        # ── Normal loads with longitudinal weight transfer ─────────────────
        Fz_total  = self.mass * G
        Fzf0      = Fz_total * VEH_L_R / self._L   # static front
        Fzr0      = Fz_total * VEH_L_F / self._L   # static rear
        dFz       = self.mass * self._ax_prev * VEH_H_CG / self._L
        Fzf       = max(100.0, Fzf0 - dFz)
        Fzr       = max(100.0, Fzr0 + dFz)

        # Downforce adds to normal load (proportional to v²)
        df_peak   = self.downforce_kg * G           # force at 200 km/h
        v2        = self.vx ** 2
        df_force  = df_peak * v2 / (55.56 ** 2)    # 55.56 m/s ≈ 200 km/h
        Fzf      += df_force * 0.45
        Fzr      += df_force * 0.55

        # ── Pacejka lateral forces ─────────────────────────────────────────
        mu  = self._mu()
        Df  = mu * Fzf
        Dr  = mu * Fzr

        # Slip angles (vehicle body frame, y = rightward)
        vx_safe   = max(abs(self.vx), 1.0)          # avoid div/0 at standstill
        alpha_f   = math.atan2(self.vy + VEH_L_F * self.omega, vx_safe) - delta
        alpha_r   = math.atan2(self.vy - VEH_L_R * self.omega, vx_safe)
        if self.vx < 0:                              # reverse: flip slip angles
            alpha_f, alpha_r = -alpha_f, -alpha_r

        # Lateral forces: positive slip → car slides right → tyre pushes left
        # Convention: Fy > 0 = rightward force
        Fyf = -self._pacejka(alpha_f, Df)
        Fyr = -self._pacejka(alpha_r, Dr)

        # ── Longitudinal forces ────────────────────────────────────────────
        ratio    = self.gear_ratios[self.gear - 1] * self.final_drive
        torque   = self.peak_torque * self._torque_factor()
        F_drive  = torque * ratio / VEH_WHEEL_R * throttle
        F_brake  = self.mass * G * 0.90 * brake * (1.0 if self.vx > 0 else 0.0)
        F_aero   = -0.5 * VEH_CD * VEH_A_FRONT * 1.225 * self.vx * abs(self.vx)
        Fx       = F_drive - F_brake + F_aero

        # ── Equations of motion (vehicle body frame, rotating with omega) ──
        # m·(dvx/dt) = Fx  + m·vy·ω   (Coriolis)
        # m·(dvy/dt) = Fyf+Fyr − m·vx·ω
        # Iz·(dω/dt) = Lf·Fyf − Lr·Fyr
        ax    = Fx / self.mass + self.vy * self.omega
        ay    = (Fyf + Fyr) / self.mass - self.vx * self.omega
        alpha = (VEH_L_F * Fyf - VEH_L_R * Fyr) / VEH_I_Z

        self._ax_prev = ax

        # ── Integration ───────────────────────────────────────────────────
        self.vx    += ax    * dt
        self.vy    += ay    * dt
        self.omega += alpha * dt

        # Clamp unrealistic values
        self.vx    = max(-15.0, self.vx)
        self.omega = max(-6.0,  min(6.0, self.omega))
        # Gentle tyre self-alignment damping on lateral velocity
        self.vy   *= 0.97

        # ── Heading and position update ────────────────────────────────────
        self.heading += self.omega * dt

        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        dx    = (self.vx * cos_h - self.vy * sin_h) * PX_PER_M * dt
        dy    = (self.vx * sin_h + self.vy * cos_h) * PX_PER_M * dt
        self.x += dx
        self.y += dy

        self.distance_px += abs(self.vx) * PX_PER_M * dt

        # ── RPM update ────────────────────────────────────────────────────
        wheel_rads = abs(self.vx) / VEH_WHEEL_R
        ratio      = self.gear_ratios[self.gear - 1] * self.final_drive
        self.rpm   = max(800.0, wheel_rads * ratio * 60.0 / (2.0 * math.pi))

    # ── Stuck detection ───────────────────────────────────────────────────────

    def update_stuck(self, dt: float, stuck_speed: float, stuck_time: float):
        if abs(self.vx) < stuck_speed:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0
        if self.stuck_timer > stuck_time:
            self.alive = False

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def speed(self) -> float:
        """Forward speed (m/s)."""
        return self.vx

    @property
    def speed_kmh(self) -> float:
        return self.vx * 3.6

    @property
    def speed_norm(self) -> float:
        """Normalised speed: 0–1 where 1 ≈ 300 km/h."""
        return min(abs(self.vx) / 83.3, 1.0)

    @property
    def lat_v(self) -> float:
        return self.vy

    def telemetry(self) -> dict:
        return {
            "speed_kmh": self.speed_kmh,
            "gear":      self.gear,
            "rpm":       self.rpm,
            "throttle":  self.throttle,
            "steer":     self.steer_val,
            "vy":        self.vy,
            "omega_deg": math.degrees(self.omega),
        }
