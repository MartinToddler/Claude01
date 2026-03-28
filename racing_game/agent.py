"""
Neural-network brain + neuro-evolution (genetic algorithm) for the racing AI.

Changes vs v1:
  • Starting grid: cars placed in 2-column grid perpendicular to track,
    ALL facing the same (forward) direction.
  • Fitness = 1/best_lap_time if lap completed, else small progress score.
    Selection favours fastest laps, not longest survivors.
  • max_laps per generation: generation ends when the leading car finishes
    max_laps, not just when all are dead.
  • Dynamic population size: num_agents can be changed between generations.
  • kill_all(): immediately marks all cars dead → triggers breed() next frame.
  • NN inputs extended to 14 (adds omega_norm, vy_norm).
"""

import math
import numpy as np
from config import (
    NN_INPUTS, NN_OUTPUTS,
    NN_DEFAULT_HIDDEN,
    ELITE_FRAC, MUTATION_BASE, CROSSOVER_RATE,
    RAYCAST_COUNT, RAYCAST_DIST,
    STUCK_SPEED, STUCK_TIME,
    DEFAULT_NUM_AGENTS, DEFAULT_MAX_LAPS,
)
from car import Car
from track import Track, get_start_pose


# ── Neural-network brain ──────────────────────────────────────────────────────

class Brain:
    """
    Configurable MLP with tanh activations, weights stored as flat array.

    hidden_sizes controls the number and width of hidden layers, e.g.:
      [24, 16]       → 14→24→16→2  (default, 2 hidden layers)
      [32, 32, 16]   → 14→32→32→16→2  (3 hidden layers)
    """

    def __init__(self,
                 hidden_sizes: list | None = None,
                 weights: np.ndarray | None = None):
        if hidden_sizes is None:
            hidden_sizes = list(NN_DEFAULT_HIDDEN)
        all_sizes = [NN_INPUTS] + list(hidden_sizes) + [NN_OUTPUTS]
        self._shapes = [(all_sizes[i], all_sizes[i + 1])
                        for i in range(len(all_sizes) - 1)]
        total = sum(r * c + c for r, c in self._shapes)
        if weights is not None:
            self.weights = weights.copy().astype(np.float32)
        else:
            self.weights = np.random.randn(total).astype(np.float32) * 0.4

    @property
    def size(self) -> int:
        return len(self.weights)

    @property
    def hidden_sizes(self) -> list:
        return [c for _, c in self._shapes[:-1]]

    def _unpack(self):
        idx, Ws, Bs = 0, [], []
        for r, c in self._shapes:
            w_size = r * c
            Ws.append(self.weights[idx: idx + w_size].reshape(r, c))
            idx += w_size
            Bs.append(self.weights[idx: idx + c])
            idx += c
        return Ws, Bs

    def forward(self, x: np.ndarray) -> np.ndarray:
        Ws, Bs = self._unpack()
        h = x
        for W, b in zip(Ws, Bs):
            h = np.tanh(h @ W + b)
        return h


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, car: Car, brain: Brain):
        self.car   = car
        self.brain = brain
        self._init_tracking()

    def _init_tracking(self):
        self.fitness      = 0.0
        self.laps         = 0
        self.best_lap_s   = None   # seconds (None = no lap completed)
        self._lap_start   = 0.0
        self._lap_active  = True   # start timing immediately
        self._best_prog   = 0.0    # cumulative forward progress (delta-based)
        self._prev_prog   = None   # None until first project() call
        # Ancestry (set by breed(), None for initial population)
        self.parent_gen:  int | None = None
        self.parent_rank: int | None = None

    def reset(self, x: float, y: float, heading: float, params: dict):
        self.car.reset(x, y, heading)
        self.car.update_params(params)
        self._init_tracking()

    # ── Sensing ───────────────────────────────────────────────────────────────

    def sense(self, track: Track) -> np.ndarray:
        car = self.car
        rays = np.zeros(RAYCAST_COUNT, dtype=np.float32)
        angle_step = math.pi / (RAYCAST_COUNT - 1)
        for i in range(RAYCAST_COUNT):
            angle  = car.heading - math.pi / 2 + i * angle_step
            rays[i] = track.raycast(car.x, car.y, angle, RAYCAST_DIST)

        _, _, track_heading = track.project(car.x, car.y)
        heading_err = math.atan2(
            math.sin(car.heading - track_heading),
            math.cos(car.heading - track_heading),
        ) / math.pi

        gear_norm  = (car.gear - 1) / max(car.num_gears - 1, 1)
        omega_norm = max(-1.0, min(1.0, car.omega / 5.0))
        vy_norm    = max(-1.0, min(1.0, car.vy   / 20.0))

        return np.array(
            [*rays,
             car.speed_norm,
             vy_norm,
             heading_err,
             gear_norm,
             omega_norm],
            dtype=np.float32,
        )

    def act(self, track: Track) -> tuple[float, float]:
        inp = self.sense(track)
        out = self.brain.forward(inp)
        return float(out[0]), float(out[1])  # throttle_brake, steer

    # ── Fitness / lap tracking ────────────────────────────────────────────────

    def update_fitness(self, track: Track, time_s: float):
        """
        Delta-accumulation fitness (immune to backward exploit).
        Ranking metric: 1/best_lap_s (fastest lap wins).
        """
        progress, _, _ = track.project(self.car.x, self.car.y)

        # Initialise prev_prog on first call (no jump at frame 0)
        if self._prev_prog is None:
            self._prev_prog = progress
            return

        # Forward-only delta in [0, 0.10] — ignores backward / wrap jumps
        delta = progress - self._prev_prog
        self._prev_prog = progress

        if 0.0 < delta < 0.10:
            self._best_prog += delta

        # Lap completion: cumulative progress crosses 1.0
        if self._best_prog >= 1.0:
            lap_t = time_s - self._lap_start
            self.laps += 1
            if lap_t > 1.0:   # ignore impossibly short laps
                if self.best_lap_s is None or lap_t < self.best_lap_s:
                    self.best_lap_s = lap_t
            self._lap_start  = time_s
            self._best_prog -= 1.0

        # fitness: fastest-lap metric (higher = better) + progress fallback
        if self.best_lap_s is not None:
            self.fitness = 1000.0 / self.best_lap_s + self.laps * 0.5
        else:
            self.fitness = self._best_prog   # progress within first lap

    @property
    def sort_key(self):
        """Primary sort key: number of laps, then best lap time (asc)."""
        lap_t = self.best_lap_s if self.best_lap_s is not None else 1e9
        return (self.laps, -lap_t, self._best_prog)


# ── Population manager ────────────────────────────────────────────────────────

class PopulationManager:
    def __init__(self, track: Track, params: dict):
        self.track        = track
        self.params       = params
        self.num_agents   = int(params.get("num_agents", DEFAULT_NUM_AGENTS))
        self.max_laps     = int(params.get("max_laps",   DEFAULT_MAX_LAPS))
        # NN architecture — only updated on full restart
        self._hidden_sizes: list = list(
            params.get("nn_hidden_sizes", NN_DEFAULT_HIDDEN))

        self.generation         = 0
        self.time_s             = 0.0
        self.gen_best_fitness   = 0.0
        self.all_time_best_lap  = None     # seconds
        self.fitness_history    = []       # best fitness per generation

        self.agents: list[Agent] = []
        self._build_initial_population()

    def _make_brain(self, weights: np.ndarray | None = None) -> Brain:
        """Create a Brain with the current hidden-layer configuration."""
        return Brain(hidden_sizes=self._hidden_sizes, weights=weights)

    # ── Population construction ───────────────────────────────────────────────

    def _build_initial_population(self):
        self.agents = []
        positions   = self._start_grid(self.num_agents)
        for x, y, h in positions:
            car   = Car(x, y, h, self.params)
            brain = self._make_brain()
            self.agents.append(Agent(car, brain))

    def _start_grid(self, n: int) -> list[tuple[float, float, float]]:
        """
        Place n cars in a 2-column starting grid, all facing FORWARD along
        the track.  Each row is 3 track-points behind the previous; columns
        are separated by 22 px across the track (left / right of centre).
        """
        track  = self.track
        pts    = track.pts
        npts   = len(pts)
        si     = track.start_idx
        norm   = track.norm     # unit perpendicular (left-of-forward)
        tang   = track.tang

        positions = []
        for i in range(n):
            col     = i % 2
            row     = i // 2
            idx     = (si + row * 3) % npts   # step backward 3 pts each row
            # Lateral offset: col 0 = slightly left, col 1 = slightly right
            across  = (col - 0.5) * 22.0      # ±11 px from centreline
            x       = float(pts[idx, 0] + norm[idx, 0] * across)
            y       = float(pts[idx, 1] + norm[idx, 1] * across)
            heading = math.atan2(float(tang[idx, 1]), float(tang[idx, 0]))
            positions.append((x, y, heading))
        return positions

    # ── Per-frame step ────────────────────────────────────────────────────────

    def step(self, dt: float = 1 / 60) -> int:
        """Advance simulation by dt.  Returns number of still-alive cars."""
        self.time_s += dt
        alive_count  = 0
        best_laps    = 0

        for agent in self.agents:
            if not agent.car.alive:
                continue
            alive_count += 1

            tb, st = agent.act(self.track)
            agent.car.step(tb, st, dt)
            agent.car.update_stuck(dt, STUCK_SPEED, STUCK_TIME)

            if self.track.is_off_track(agent.car.x, agent.car.y):
                agent.car.alive = False
                continue

            agent.update_fitness(self.track, self.time_s)

            if agent.laps > best_laps:
                best_laps = agent.laps

        # End generation when leader finishes max_laps
        if best_laps >= self.max_laps:
            return 0

        # Identify best agent (by sort_key — laps then lap time)
        if self.agents:
            best       = max(self.agents, key=lambda a: a.sort_key)
            self._best = best

        return alive_count

    # ── Breed next generation ─────────────────────────────────────────────────

    def breed(self):
        ranked = sorted(self.agents, key=lambda a: a.sort_key, reverse=True)

        self.gen_best_fitness = ranked[0].fitness
        self.fitness_history.append(self.gen_best_fitness)

        # Update all-time best lap
        for ag in ranked:
            if ag.best_lap_s is not None:
                if (self.all_time_best_lap is None or
                        ag.best_lap_s < self.all_time_best_lap):
                    self.all_time_best_lap = ag.best_lap_s

        self.generation += 1

        # Dynamic population resize
        num = int(self.params.get("num_agents", DEFAULT_NUM_AGENTS))
        self.max_laps = int(self.params.get("max_laps", DEFAULT_MAX_LAPS))

        # Mutation rate driven by driver_risk (1→0.6× base, 10→1.8× base)
        risk      = float(self.params.get("driver_risk", 5))
        mut_scale = 0.6 + (risk - 1) / 9.0 * 1.2
        mut_std   = MUTATION_BASE * mut_scale

        # ── Elites (top ELITE_FRAC = 30%) — preserved unchanged ──────────
        n_elite = max(2, int(len(ranked) * ELITE_FRAC))
        new_entries: list[tuple[np.ndarray, int, int]] = []   # (weights, parent_gen, parent_rank)
        for rank_i, ag in enumerate(ranked[:n_elite]):
            new_entries.append((ag.brain.weights.copy(), self.generation - 1, rank_i))

        # Parent pool: top 30% only — higher quality than old top-50%
        pool_size = max(2, int(len(ranked) * 0.30))
        pool = ranked[:pool_size]

        n_remain  = num - n_elite
        n_clone   = n_remain // 2   # 50% of non-elites: clones with soft mutation
        n_cross   = n_remain - n_clone  # 50% of non-elites: crossover (best × pool)

        # ── Clones: round-robin from elites + half-strength mutation ──────
        for i in range(n_clone):
            parent_rank = i % n_elite
            parent_w    = ranked[parent_rank].brain.weights
            child = parent_w.copy() + (
                np.random.randn(parent_w.size).astype(np.float32) * mut_std * 0.5)
            new_entries.append((child, self.generation - 1, parent_rank))

        # ── Crossover: ranked[0] always parent A, random pool member as B ─
        best_w = ranked[0].brain.weights
        for _ in range(n_cross):
            pb_agent = pool[np.random.randint(len(pool))]
            pb       = pb_agent.brain.weights
            mask     = np.random.rand(best_w.size) < CROSSOVER_RATE
            child    = np.where(mask, best_w, pb).astype(np.float32)
            child   += np.random.randn(child.size).astype(np.float32) * mut_std
            pb_rank  = ranked.index(pb_agent)
            new_entries.append((child, self.generation - 1, pb_rank))

        # ── Rebuild agents list ───────────────────────────────────────────
        positions  = self._start_grid(num)
        new_agents = []
        for i in range(num):
            x, y, h = positions[i]
            w, p_gen, p_rank = new_entries[i]
            if i < len(self.agents):
                ag = self.agents[i]
                ag.reset(x, y, h, self.params)
                ag.brain = self._make_brain(w)
            else:
                car = Car(x, y, h, self.params)
                ag  = Agent(car, self._make_brain(w))
            ag.parent_gen  = p_gen
            ag.parent_rank = p_rank
            new_agents.append(ag)

        self.agents     = new_agents
        self.num_agents = num
        self.time_s     = 0.0
        self._best      = self.agents[0] if self.agents else None

    # ── Utilities ─────────────────────────────────────────────────────────────

    def kill_all(self):
        """Immediately kill every car (triggers breed() in the main loop)."""
        for ag in self.agents:
            ag.car.alive = False

    def update_params(self, params: dict):
        """Apply new car parameters and restart generation (keep brains)."""
        self.params    = params
        self.max_laps  = int(params.get("max_laps",   DEFAULT_MAX_LAPS))
        positions      = self._start_grid(len(self.agents))
        for i, (ag, (x, y, h)) in enumerate(zip(self.agents, positions)):
            ag.reset(x, y, h, params)
        self.time_s = 0.0

    def change_track(self, track: Track):
        self.track = track
        positions  = self._start_grid(len(self.agents))
        for ag, (x, y, h) in zip(self.agents, positions):
            brain_bk = ag.brain
            ag.reset(x, y, h, self.params)
            ag.brain = brain_bk
        self.time_s = 0.0

    def restart(self, params: dict):
        """Full restart: re-read NN architecture and rebuild population from scratch."""
        self.params        = params
        self._hidden_sizes = list(params.get("nn_hidden_sizes", NN_DEFAULT_HIDDEN))
        self.num_agents    = int(params.get("num_agents", DEFAULT_NUM_AGENTS))
        self.max_laps      = int(params.get("max_laps",   DEFAULT_MAX_LAPS))
        self.generation    = 0
        self.time_s        = 0.0
        self.gen_best_fitness  = 0.0
        self.all_time_best_lap = None
        self.fitness_history   = []
        self._build_initial_population()

    @property
    def alive_agents(self) -> list[Agent]:
        return [a for a in self.agents if a.car.alive]

    @property
    def best_agent(self) -> Agent:
        if not hasattr(self, "_best") or self._best is None:
            self._best = max(self.agents, key=lambda a: a.sort_key)
        return self._best
