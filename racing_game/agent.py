"""
Neural-network brain + neuro-evolution (genetic algorithm) for the racing AI.

Each car has a Brain — a small fully-connected network (tanh activations).
The PopulationManager runs a generation: all cars drive simultaneously until
dead, computes fitness, then breeds the next generation via elitism +
crossover + mutation.

No external ML libraries required — pure NumPy.
"""

import math
import numpy as np
from config import (
    NN_INPUTS, NN_HIDDEN1, NN_HIDDEN2, NN_OUTPUTS,
    ELITE_COUNT, MUTATION_BASE, CROSSOVER_RATE,
    NUM_AGENTS, RAYCAST_COUNT, RAYCAST_DIST,
)
from car import Car
from track import Track, get_start_pose


# ── Neural-network brain ──────────────────────────────────────────────────────

class Brain:
    """Two hidden-layer MLP, all tanh, weights stored as flat array."""

    # Layer sizes
    _SHAPES = [
        (NN_INPUTS,  NN_HIDDEN1),
        (NN_HIDDEN1, NN_HIDDEN2),
        (NN_HIDDEN2, NN_OUTPUTS),
    ]

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = weights if weights is not None else self._random_weights()

    @classmethod
    def _random_weights(cls) -> np.ndarray:
        total = sum(r * c + c for r, c in cls._SHAPES)   # W + bias per layer
        return np.random.randn(total).astype(np.float32) * 0.5

    @property
    def size(self) -> int:
        return len(self.weights)

    # ── Unpack weights into layer matrices ───────────────────────────────────

    def _unpack(self):
        idx = 0
        Ws, Bs = [], []
        for r, c in self._SHAPES:
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
        return h   # shape (NN_OUTPUTS,)


# ── Agent: Brain + Car + sensors ─────────────────────────────────────────────

class Agent:
    def __init__(self, car: Car, brain: Brain):
        self.car   = car
        self.brain = brain
        self.fitness      = 0.0
        self.laps         = 0
        self.best_lap_s   = None    # seconds
        self._lap_start   = 0.0
        self._prev_prog   = 0.0
        self._best_prog   = 0.0
        self._lap_active  = False

    def reset(self, x, y, heading, params):
        self.car.reset(x, y, heading)
        self.car.update_params(params)
        self.fitness       = 0.0
        self.laps          = 0
        self.best_lap_s    = None
        self._prev_prog    = 0.0
        self._best_prog    = 0.0   # max forward progress in current lap
        self._lap_active   = False

    def sense(self, track: Track) -> np.ndarray:
        """Build input vector for the neural net."""
        car = self.car
        rays = np.zeros(RAYCAST_COUNT, dtype=np.float32)
        angle_step = math.pi / (RAYCAST_COUNT - 1)   # 0..180° spread
        for i in range(RAYCAST_COUNT):
            angle = car.heading - math.pi / 2 + i * angle_step
            rays[i] = track.raycast(car.x, car.y, angle, RAYCAST_DIST)

        _, lat, track_heading = track.project(car.x, car.y)
        heading_err = math.atan2(
            math.sin(car.heading - track_heading),
            math.cos(car.heading - track_heading),
        ) / math.pi   # normalised -1..1

        gear_norm = (car.gear - 1) / max(car.num_gears - 1, 1)

        return np.array(
            [*rays,
             car.speed_norm,
             car.lat_v / 10.0,
             heading_err,
             gear_norm],
            dtype=np.float32,
        )

    def act(self, track: Track) -> tuple[float, float]:
        inp = self.sense(track)
        out = self.brain.forward(inp)
        throttle_brake = float(out[0])   # tanh → [-1, 1]
        steer          = float(out[1])   # tanh → [-1, 1]
        return throttle_brake, steer

    def update_fitness(self, track: Track, dt: float, time_s: float):
        """Called after car.step(); accumulates fitness and lap detection.

        Uses a delta-accumulation approach: only small *forward* increments
        in progress are counted.  Large jumps (wraps, backward movement) are
        ignored, making the metric exploit-proof.
        """
        car = self.car
        progress, _, _ = track.project(car.x, car.y)

        # Forward delta: raw difference clamped to [0, 0.1]
        # — negative → backward (ignored)
        # — >0.1    → spurious wrap jump (ignored)
        delta = progress - self._prev_prog
        self._prev_prog = progress

        if 0.0 < delta < 0.10:
            self._best_prog += delta

        # Lap detection: accumulated forward progress crosses 1.0
        if self._best_prog >= 1.0:
            if self._lap_active:
                lap_t = time_s - self._lap_start
                self.laps += 1
                if self.best_lap_s is None or lap_t < self.best_lap_s:
                    self.best_lap_s = lap_t
            self._lap_start  = time_s
            self._lap_active = True
            self._best_prog -= 1.0   # keep remainder

        # Fitness = completed laps + fractional progress in current lap
        self.fitness = self.laps + self._best_prog


# ── Population manager ────────────────────────────────────────────────────────

class PopulationManager:
    """
    Manages a generation of Agents.  Call step() each frame; when all agents
    are dead, breed() and reset.
    """

    def __init__(self, track: Track, params: dict):
        self.track       = track
        self.params      = params
        self.generation  = 0
        self.time_s      = 0.0
        self.gen_best_fitness   = 0.0
        self.all_time_best_lap  = None
        self.fitness_history    = []   # list of best-fitness per generation

        sx, sy, sh = get_start_pose(track)
        self.start = (sx, sy, sh)

        # Create initial population
        self.agents: list[Agent] = []
        for _ in range(NUM_AGENTS):
            car   = Car(sx, sy, sh, params)
            brain = Brain()
            self.agents.append(Agent(car, brain))

        # Index of current best agent (for highlighting)
        self.best_idx = 0

    # ── Frame step ────────────────────────────────────────────────────────────

    def step(self, dt: float = 1/60):
        from config import STUCK_SPEED, STUCK_TIME

        self.time_s += dt
        alive_count = 0

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

            agent.update_fitness(self.track, dt, self.time_s)

        # Identify best live agent by fitness
        best = max(self.agents, key=lambda a: a.fitness)
        self.best_idx = self.agents.index(best)

        return alive_count  # 0 → time to breed

    # ── Breed next generation ─────────────────────────────────────────────────

    def breed(self):
        # Sort by fitness descending
        ranked = sorted(self.agents, key=lambda a: a.fitness, reverse=True)

        self.gen_best_fitness = ranked[0].fitness
        self.fitness_history.append(self.gen_best_fitness)

        if ranked[0].best_lap_s is not None:
            if (self.all_time_best_lap is None or
                    ranked[0].best_lap_s < self.all_time_best_lap):
                self.all_time_best_lap = ranked[0].best_lap_s

        self.generation += 1

        # Mutation rate scales with driver_risk (1→10 maps to 0.6x→1.8x base)
        risk     = self.params.get("driver_risk", 5)
        mut_scale= 0.6 + (risk - 1) / 9 * 1.2    # 0.6 .. 1.8
        mut_std  = MUTATION_BASE * mut_scale

        elites = [a.brain.weights.copy() for a in ranked[:ELITE_COUNT]]
        new_weights = list(elites)   # elites survive unchanged

        while len(new_weights) < NUM_AGENTS:
            # Tournament selection from top half
            pool = ranked[: max(2, len(ranked) // 2)]
            pa   = np.random.choice(pool).brain.weights
            pb   = np.random.choice(pool).brain.weights

            # Crossover
            mask = np.random.rand(pa.size) < CROSSOVER_RATE
            child = np.where(mask, pa, pb).astype(np.float32)

            # Mutation
            noise = np.random.randn(child.size).astype(np.float32) * mut_std
            child += noise

            new_weights.append(child)

        # Reset all agents with new brains
        sx, sy, sh = self.start
        for i, agent in enumerate(self.agents):
            agent.reset(sx, sy, sh, self.params)
            agent.brain = Brain(new_weights[i])

        self.time_s = 0.0
        self.best_idx = 0

    # ── Public helpers ────────────────────────────────────────────────────────

    def update_params(self, params: dict):
        """Called when user changes car settings in the UI."""
        self.params = params
        sx, sy, sh  = get_start_pose(self.track)
        self.start  = (sx, sy, sh)
        # Restart generation with new params (keep brains)
        for agent in self.agents:
            agent.car.reset(sx, sy, sh)
            agent.car.update_params(params)
            agent.fitness = 0.0
            agent.laps    = 0
        self.time_s = 0.0

    def change_track(self, track: Track):
        self.track = track
        sx, sy, sh = get_start_pose(track)
        self.start = (sx, sy, sh)
        for agent in self.agents:
            brain_backup = agent.brain
            agent.reset(sx, sy, sh, self.params)
            agent.brain = brain_backup
        self.time_s = 0.0

    @property
    def alive_agents(self) -> list[Agent]:
        return [a for a in self.agents if a.car.alive]

    @property
    def best_agent(self) -> Agent:
        return self.agents[self.best_idx]
