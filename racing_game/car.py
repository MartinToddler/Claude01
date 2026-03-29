"""
Vehicle dynamics — single-track (bicycle) model with Pacejka tyre model.

v3 additions:
  • drive_type (FWD / RWD / AWD): splits drive torque between axles and
    applies the friction-circle reduction to the driven axle's lateral grip.
  • engine_pos (front / mid / rear): sets L_f / L_r (CG geometry), changing
    the static weight distribution and thus each axle's peak lateral force.
  • final_drive exposed as a slider-settable parameter.
  • All VEH_L_F / VEH_L_R references replaced by self.L_f / self.L_r.
"""

import math
from config import (
    CAR_DEFAULTS, TYRE_GRIP, OPTIMAL_PRESSURE, PRESSURE_GRIP_FALLOFF,
    HP_TO_NM, ENGINE_POS_PARAMS, DRIVE_SPLIT,
    VEH_I_Z, VEH_H_CG, VEH_WHEEL_R, VEH_CD, VEH_A_FRONT,
    PAC_B, PAC_C, PAC_E,
)

PX_PER_M = 8.5
G = 9.81


class Car:
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
        self.drive_type    = p.get("drive_type",  "RWD")
        self.engine_pos    = p.get("engine_pos",  "front")

        # CG geometry from engine position
        ep         = ENGINE_POS_PARAMS.get(self.engine_pos, ENGINE_POS_PARAMS["front"])
        self.L_f   = ep["L_f"]
        self.L_r   = ep["L_r"]
        self._L    = self.L_f + self.L_r

        # Derived
        self.peak_torque = self.power * HP_TO_NM
        self.num_gears   = len(self.gear_ratios)

    def update_params(self, p: dict):
        self._apply_params(p)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, x: float, y: float, heading: float):
        self.x        = x
        self.y        = y
        self.heading  = heading
        self.vx       = 5.0    # initial push-start (m/s)
        self.vy       = 0.0
        self.omega    = 0.0
        self.gear     = 1
        self.rpm      = 2000.0
        self.delta    = 0.0    # current front-wheel steering angle (rad)
        self.alive    = True
        self.stuck_timer  = 0.0
        self.distance_px  = 0.0
        self.throttle     = 0.0
        self.steer_val    = 0.0
        self._ax_prev     = 0.0

    # ── Tyre ─────────────────────────────────────────────────────────────────

    def _mu(self) -> float:
        base = TYRE_GRIP[self.tyre_type]
        dp   = self.tyre_pressure - OPTIMAL_PRESSURE
        return base * max(0.5, 1.0 - PRESSURE_GRIP_FALLOFF * dp ** 2)

    def _pacejka(self, alpha: float, D: float) -> float:
        """Pacejka lateral force. Positive alpha → positive return value."""
        B   = PAC_B * (1.0 + 0.3 * self.downforce_kg / 100.0)
        Ba  = B * alpha
        return D * math.sin(PAC_C * math.atan(Ba - PAC_E * (Ba - math.atan(Ba))))

    # ── Gearbox ───────────────────────────────────────────────────────────────

    def _auto_shift(self):
        if self.rpm > 7200 and self.gear < self.num_gears:
            self.gear += 1
        elif self.rpm < 2500 and self.gear > 1:
            self.gear -= 1

    def _torque_factor(self) -> float:
        rn = min(max(self.rpm / 8000.0, 0.0), 1.0)
        return 1.0 - 0.35 * (2.0 * rn - 1.0) ** 2

    # ── Physics step ─────────────────────────────────────────────────────────

    def step(self, throttle_brake: float, steer: float, dt: float = 1 / 60,
             auto_gear: bool = True):
        self.throttle  = throttle_brake
        self.steer_val = steer

        if throttle_brake >= 0:
            throttle, brake = throttle_brake, 0.0
        else:
            throttle, brake = 0.0, -throttle_brake

        # Speed-dependent steering limit
        v_kmh     = abs(self.vx) * 3.6
        dmax_deg  = max(4.0, 25.0 - 0.105 * v_kmh)
        delta     = steer * math.radians(dmax_deg)
        self.delta = delta   # store for renderer (wheel steer angle)

        if auto_gear:
            self._auto_shift()

        # ── Normal loads (static + weight-transfer + downforce) ───────────
        Fz_total = self.mass * G
        Fzf0 = Fz_total * self.L_r / self._L
        Fzr0 = Fz_total * self.L_f / self._L
        dFz  = self.mass * self._ax_prev * VEH_H_CG / self._L
        Fzf  = max(100.0, Fzf0 - dFz)
        Fzr  = max(100.0, Fzr0 + dFz)

        v2       = self.vx ** 2
        df_force = self.downforce_kg * G * v2 / (55.56 ** 2)
        Fzf += df_force * 0.45
        Fzr += df_force * 0.55

        mu  = self._mu()
        Df  = mu * Fzf   # peak lateral force front (D in Pacejka)
        Dr  = mu * Fzr

        # ── Slip angles ───────────────────────────────────────────────────
        vx_safe = max(abs(self.vx), 1.0)
        alpha_f = math.atan2(self.vy + self.L_f * self.omega, vx_safe) - delta
        alpha_r = math.atan2(self.vy - self.L_r * self.omega, vx_safe)
        if self.vx < 0:
            alpha_f, alpha_r = -alpha_f, -alpha_r

        # ── Drive/brake force split to axles ──────────────────────────────
        ratio     = self.gear_ratios[self.gear - 1] * self.final_drive
        torque    = self.peak_torque * self._torque_factor()
        F_drive   = torque * ratio / VEH_WHEEL_R * throttle
        F_brake   = self.mass * G * 0.90 * brake * (1.0 if self.vx > 0 else 0.0)
        F_aero    = -0.5 * VEH_CD * VEH_A_FRONT * 1.225 * self.vx * abs(self.vx)

        # Rolling resistance: ~1.5% of total weight, opposes motion
        F_roll = (-math.copysign(1.0, self.vx) * 0.015 * Fz_total
                  if abs(self.vx) > 0.3 else 0.0)

        ff, rf    = DRIVE_SPLIT.get(self.drive_type, (0.0, 1.0))
        F_xf_drv  = F_drive * ff
        F_xr_drv  = F_drive * rf
        F_xf_brk  = F_brake * 0.70   # 70 % front brake bias
        F_xr_brk  = F_brake * 0.30
        F_xf_aero = F_aero  * 0.50
        F_xr_aero = F_aero  * 0.50

        F_xf = F_xf_drv - F_xf_brk + F_xf_aero
        F_xr = F_xr_drv - F_xr_brk + F_xr_aero
        Fx   = F_xf + F_xr + F_roll   # total longitudinal force at CG

        # ── Friction circle: reduce lateral grip on driven axle ───────────
        # Available lateral = sqrt(D² - Fx_axle²)
        Df_avail = math.sqrt(max(0.0, Df ** 2 - F_xf ** 2))
        Dr_avail = math.sqrt(max(0.0, Dr ** 2 - F_xr ** 2))

        Fyf = -self._pacejka(alpha_f, Df_avail)
        Fyr = -self._pacejka(alpha_r, Dr_avail)

        # ── Low-speed lateral force scaling ──────────────────────────────
        # Tyre lateral forces must go to zero as vehicle speed → 0.
        # Without this, Pacejka generates full force at standstill due to
        # vx_safe clamp, causing unrealistic perpetual spinning.
        speed_factor = min(1.0, abs(self.vx) / 4.0)
        Fyf *= speed_factor
        Fyr *= speed_factor

        # ── Equations of motion (vehicle body frame) ──────────────────────
        ax    = Fx / self.mass + self.vy * self.omega
        ay    = (Fyf + Fyr) / self.mass - self.vx * self.omega
        alpha = (self.L_f * Fyf - self.L_r * Fyr) / VEH_I_Z

        self._ax_prev = ax

        self.vx    += ax    * dt
        self.vy    += ay    * dt
        self.omega += alpha * dt

        self.vx    = max(-15.0, self.vx)
        self.omega = max(-6.0,  min(6.0,  self.omega))

        # Yaw damping: tyres self-align; 0.98/frame = ~30% retention/s at 60fps
        # (was 0.93 = 2.5%/s — too aggressive, killed cornering yaw rate)
        self.omega *= 0.98

        # Lateral velocity damping: less aggressive than before (was 0.92-0.99)
        # Rolling resistance now handles low-speed stabilisation
        vy_damp = 0.97 + 0.02 * min(1.0, abs(self.vx) / 20.0)
        self.vy *= vy_damp

        # ── Heading + position ────────────────────────────────────────────
        self.heading += self.omega * dt
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        dx = (self.vx * cos_h - self.vy * sin_h) * PX_PER_M * dt
        dy = (self.vx * sin_h + self.vy * cos_h) * PX_PER_M * dt
        self.x += dx
        self.y += dy
        self.distance_px += abs(self.vx) * PX_PER_M * dt

        # ── RPM ───────────────────────────────────────────────────────────
        wheel_rads = abs(self.vx) / VEH_WHEEL_R
        r          = self.gear_ratios[self.gear - 1] * self.final_drive
        self.rpm   = max(800.0, wheel_rads * r * 60.0 / (2.0 * math.pi))

    # ── Stuck ─────────────────────────────────────────────────────────────────

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
        return self.vx

    @property
    def speed_kmh(self) -> float:
        return self.vx * 3.6

    @property
    def speed_norm(self) -> float:
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
