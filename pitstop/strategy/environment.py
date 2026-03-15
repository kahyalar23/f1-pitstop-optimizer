"""
pitstop/strategy/environment.py

OpenAI Gymnasium race environment for pit stop strategy RL agent.

State space (8-dim continuous):
    [lap_norm, position_norm, tyre_age_norm, compound_idx,
     gap_ahead_norm, gap_behind_norm, safety_car, tyre_temp_norm]

Action space (discrete, 4 actions):
    0: continue   — stay out
    1: pit_soft   — pit for soft tyres
    2: pit_medium — pit for medium tyres
    3: pit_hard   — pit for hard tyres

Reward:
    r_t = -delta_laptime_penalty + position_gain_reward - pit_loss_cost + finish_bonus
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional

from pitstop.simulation.tire_model import (
    TyreState, simulate_tyre_lap, COMPOUNDS, undercut_delta
)
from pitstop.simulation.monte_carlo import (
    run_monte_carlo, MonteCarloConfig, CREW_PROFILES
)


# ---------------------------------------------------------------------------
# Race configuration
# ---------------------------------------------------------------------------

RACE_CONFIGS = {
    "monaco": {"n_laps": 78, "track_temp": 45, "ambient": 22, "base_laptime": 74.0,
               "overtake_difficulty": 0.95, "pit_entry_loss": 20.0, "pit_exit_loss": 17.0},
    "monza":  {"n_laps": 53, "track_temp": 38, "ambient": 26, "base_laptime": 80.0,
               "overtake_difficulty": 0.30, "pit_entry_loss": 19.5, "pit_exit_loss": 16.0},
    "spa":    {"n_laps": 44, "track_temp": 28, "ambient": 18, "base_laptime": 105.0,
               "overtake_difficulty": 0.50, "pit_entry_loss": 21.0, "pit_exit_loss": 17.5},
    "generic":{"n_laps": 60, "track_temp": 40, "ambient": 25, "base_laptime": 90.0,
               "overtake_difficulty": 0.50, "pit_entry_loss": 20.5, "pit_exit_loss": 17.0},
}


@dataclass
class Competitor:
    """Simplified competitor model for gap tracking."""
    position: int
    tyre_compound: str
    tyre_age: int
    gap_to_leader: float  # seconds
    strategy: str = "one_stop"  # "one_stop" | "two_stop" | "reactive"

    def update(self, race_cfg: dict, lap: int):
        """Simple update: lap time increases with tyre age."""
        compound = COMPOUNDS[self.tyre_compound]
        deg = compound.delta_per_lap * self.tyre_age ** 0.6
        lap_noise = np.random.normal(0, 0.1)
        self.gap_to_leader += (deg + lap_noise)
        self.tyre_age += 1


# ---------------------------------------------------------------------------
# Main Gym environment
# ---------------------------------------------------------------------------

class F1RaceEnv(gym.Env):
    """
    F1 race strategy environment.

    Observation (normalised to [0,1] or [-1,1]):
        0: lap / n_laps
        1: 1 - position / n_cars   (higher = better)
        2: tyre_age / 50
        3: compound_idx / 3        (0=soft, 1=med, 2=hard)
        4: gap_ahead / 30          (clipped at 30s)
        5: gap_behind / 30
        6: safety_car              (0 or 1)
        7: (tyre_temp - 70) / 50

    Reward shaping:
        Each lap: -lap_delta (tyre deg cost)
        Pit stop: -pit_loss (stationary time loss, ~22s equivalent in laps)
        Position gain: +0.5 per position gained
        Finish: +bonus based on final position
    """

    metadata = {"render_modes": ["human", "ansi"]}
    COMPOUND_MAP = {0: "soft", 1: "medium", 2: "hard"}
    COMPOUND_IDX = {"soft": 0, "medium": 1, "hard": 2}
    N_CARS = 20

    def __init__(
        self,
        race: str = "generic",
        crew_name: str = "elite",
        render_mode: Optional[str] = None,
        seed: int = None,
    ):
        super().__init__()
        self.race_cfg = RACE_CONFIGS[race]
        self.crew_name = crew_name
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, -1, 0, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1,  1,  1, 1,  1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)  # 0=stay, 1=soft, 2=med, 3=hard

        # State (initialised in reset)
        self.lap = 0
        self.position = 0
        self.tyre_state: TyreState = None
        self.gap_ahead = 0.0
        self.gap_behind = 0.0
        self.safety_car = False
        self.pit_count = 0
        self.total_time = 0.0
        self.competitors: list[Competitor] = []
        self.lap_times: list[float] = []

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = self.race_cfg
        self.lap = 0
        self.position = self._rng.integers(1, self.N_CARS + 1)
        compound_start = self._rng.choice(["soft", "medium"])
        self.tyre_state = TyreState(
            compound=COMPOUNDS[compound_start],
            temperature=COMPOUNDS[compound_start].theta_optimal * 0.85,
        )
        self.gap_ahead  = float(self._rng.uniform(0.5, 8.0))
        self.gap_behind = float(self._rng.uniform(0.5, 8.0))
        self.safety_car = False
        self.pit_count = 0
        self.total_time = 0.0
        self.lap_times = []
        self.competitors = self._init_competitors()

        return self._get_obs(), {}

    def step(self, action: int):
        cfg = self.race_cfg
        self.lap += 1

        # ---- Safety car (random, ~5% per lap) ----
        self.safety_car = self._rng.random() < 0.05

        # ---- Pit decision ----
        pitted = False
        pit_time_loss = 0.0

        if action > 0:  # pit action
            new_compound = self.COMPOUND_MAP[action - 1]
            pit_result = self._simulate_pitstop()
            pit_time_loss = pit_result["pit_loss"]

            self.tyre_state = TyreState(
                compound=COMPOUNDS[new_compound],
                temperature=COMPOUNDS[new_compound].theta_optimal * 0.75,
            )
            self.pit_count += 1
            pitted = True

        # ---- Tyre deg this lap ----
        slip = 1.0 if not self.safety_car else 0.4
        self.tyre_state, lap_delta = simulate_tyre_lap(
            self.tyre_state, slip,
            cfg["ambient"], cfg["track_temp"]
        )
        laptime = cfg["base_laptime"] + lap_delta + pit_time_loss

        # ---- Update competitors and gaps ----
        pos_change = self._update_competitors(laptime)
        self.total_time += laptime
        self.lap_times.append(laptime)

        # ---- Reward ----
        reward = self._compute_reward(lap_delta, pit_time_loss, pos_change, pitted)

        # ---- Termination ----
        terminated = self.lap >= cfg["n_laps"]
        truncated = False

        if terminated:
            reward += self._finish_bonus()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        cfg = self.race_cfg
        n = cfg["n_laps"]
        obs = np.array([
            self.lap / n,
            1 - self.position / self.N_CARS,
            min(self.tyre_state.age_laps / 50, 1.0),
            self.COMPOUND_IDX.get(self.tyre_state.compound.name, 1) / 3,
            np.clip(self.gap_ahead / 30, -1, 1),
            np.clip(self.gap_behind / 30, -1, 1),
            float(self.safety_car),
            np.clip((self.tyre_state.temperature - 70) / 50, -1, 1),
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "lap": self.lap,
            "position": self.position,
            "tyre_age": self.tyre_state.age_laps,
            "compound": self.tyre_state.compound.name,
            "pit_count": self.pit_count,
            "total_time": round(self.total_time, 3),
            "gap_ahead": round(self.gap_ahead, 3),
        }

    def _simulate_pitstop(self) -> dict:
        """Sample a real pit stop duration using Monte Carlo."""
        cfg_mc = MonteCarloConfig(n_iterations=1, crew_name=self.crew_name, seed=None)
        crew = CREW_PROFILES[self.crew_name]
        from pitstop.simulation.monte_carlo import simulate_one
        result = simulate_one(crew, rng=self._rng)
        race = self.race_cfg
        # Pit entry + stationary + exit
        pit_loss = race["pit_entry_loss"] + result.pit_time + race["pit_exit_loss"]
        return {"pit_loss": pit_loss, "stationary": result.pit_time}

    def _init_competitors(self) -> list[Competitor]:
        comps = []
        compounds = ["soft", "medium", "medium", "hard", "soft"]
        for i in range(self.N_CARS - 1):
            c = Competitor(
                position=i + 1,
                tyre_compound=compounds[i % len(compounds)],
                tyre_age=self._rng.integers(0, 5),
                gap_to_leader=float(i * 2.0 + self._rng.uniform(-0.5, 0.5)),
            )
            comps.append(c)
        return comps

    def _update_competitors(self, our_laptime: float) -> int:
        """Update competitor states, return our position change."""
        cfg = self.race_cfg
        old_pos = self.position

        # Competitors evolve
        for comp in self.competitors:
            comp.update(cfg, self.lap)

        # Simple gap update
        noise = float(self._rng.normal(0, 0.15))
        self.gap_ahead  = max(0.1, self.gap_ahead  + noise - 0.05)
        self.gap_behind = max(0.1, self.gap_behind + noise)

        # Probabilistic overtake (difficulty affects P)
        overtake_p = (1 - cfg["overtake_difficulty"]) * max(0, 3 - self.gap_ahead) / 3
        if self._rng.random() < overtake_p and self.position > 1:
            self.position -= 1
            self.gap_ahead = self.gap_ahead + self.gap_behind
            self.gap_behind = float(self._rng.uniform(0.5, 2.0))

        return old_pos - self.position  # positive = gained positions

    def _compute_reward(
        self, lap_delta: float, pit_loss: float, pos_change: int, pitted: bool
    ) -> float:
        # Lap time cost (normalised)
        r_deg = -lap_delta * 0.5

        # Pit time cost (normalised by base laptime)
        r_pit = -(pit_loss / self.race_cfg["base_laptime"]) * 3.0 if pitted else 0.0

        # Position gain
        r_pos = pos_change * 0.5

        # Safety car bonus (pitting under SC is much cheaper)
        r_sc = 1.5 if (pitted and self.safety_car) else 0.0

        # Tyre cliff penalty
        r_cliff = -2.0 if self.tyre_state.past_cliff else 0.0

        return r_deg + r_pit + r_pos + r_sc + r_cliff

    def _finish_bonus(self) -> float:
        """Reward based on final race position."""
        position_rewards = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                            6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        return float(position_rewards.get(self.position, 0)) * 0.5

    def render(self):
        if self.render_mode == "ansi":
            return (f"Lap {self.lap}/{self.race_cfg['n_laps']} | "
                    f"P{self.position} | "
                    f"{self.tyre_state.compound.name.upper()} "
                    f"({self.tyre_state.age_laps} laps) | "
                    f"Gap: +{self.gap_ahead:.1f}s")


# Register with gymnasium
gym.register(
    id="F1Race-v0",
    entry_point="pitstop.strategy.environment:F1RaceEnv",
)
