"""
pitstop/simulation/monte_carlo.py

Stochastic pit stop duration model using:
- Log-Normal task sampling (always positive, right-skewed, realistic)
- Critical Path Method (CPM) to find bottleneck wheel station
- Crew skill + fatigue modifiers on distribution parameters

Key equation:
    T_pit = T_reaction + max_i(T_i) + T_jack_down
    T_i   = sum of 4 sub-tasks per wheel station
    t_task ~ LogNormal(mu_ln, sigma_ln)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from scipy.stats import lognorm, ks_2samp


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TaskParams:
    """
    Log-Normal parameters for a single pit stop sub-task.
    We store (mean, std) in real-time space and convert to (mu_ln, sigma_ln).

    Conversion:
        sigma_ln = sqrt(ln(1 + (std/mean)^2))
        mu_ln    = ln(mean) - 0.5 * sigma_ln^2
    """
    mean: float  # seconds
    std: float   # seconds

    @property
    def mu_ln(self) -> float:
        cv2 = (self.std / self.mean) ** 2
        return np.log(self.mean) - 0.5 * np.log(1 + cv2)

    @property
    def sigma_ln(self) -> float:
        cv2 = (self.std / self.mean) ** 2
        return np.sqrt(np.log(1 + cv2))

    def sample(self, n: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.lognormal(self.mu_ln, self.sigma_ln, size=n)


@dataclass
class CrewProfile:
    """
    Crew skill level affects task mean and variance.
    Based on empirical F1 pit stop literature (Collings et al., 2019).
    """
    name: str
    # Multiplicative factor on base task mean
    mean_factor: float
    # Multiplicative factor on base task std
    std_factor: float
    # Base clash/collision probability (tool/space interference)
    base_clash_prob: float


CREW_PROFILES: dict[str, CrewProfile] = {
    "rookie":  CrewProfile("rookie",  mean_factor=1.25, std_factor=1.60, base_clash_prob=0.12),
    "mid":     CrewProfile("mid",     mean_factor=1.05, std_factor=1.20, base_clash_prob=0.05),
    "elite":   CrewProfile("elite",   mean_factor=1.00, std_factor=1.00, base_clash_prob=0.02),
    "mercedes":CrewProfile("mercedes",mean_factor=0.93, std_factor=0.85, base_clash_prob=0.01),
}


# ---------------------------------------------------------------------------
# Base task timing database (elite crew, dry conditions)
# Source: aggregate of FastF1 2021-2023, adjusted for observable station times
# ---------------------------------------------------------------------------

BASE_TASKS: dict[str, TaskParams] = {
    "loosen":     TaskParams(mean=0.42, std=0.055),
    "tyre_off":   TaskParams(mean=0.25, std=0.040),
    "tyre_on":    TaskParams(mean=0.28, std=0.045),
    "tighten":    TaskParams(mean=0.44, std=0.060),
    "jack_front": TaskParams(mean=0.30, std=0.035),  # front jack up+down
    "jack_rear":  TaskParams(mean=0.32, std=0.038),
    "reaction":   TaskParams(mean=0.11, std=0.025),  # driver reaction to lollipop
}

TASK_SEQUENCE = ["loosen", "tyre_off", "tyre_on", "tighten"]
CORNERS = ["FL", "FR", "RL", "RR"]


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

@dataclass
class PitStopResult:
    """Result of a single simulated pit stop."""
    pit_time: float          # total pit stop duration (s)
    corner_times: dict       # {corner: total_time}
    task_breakdown: dict     # {corner: {task: duration}}
    critical_corner: str     # which corner was the bottleneck
    critical_path_time: float
    had_clash: bool
    clash_penalty: float
    jack_time: float

    @property
    def station_slack(self) -> dict:
        """Slack time per corner relative to critical path (CPM concept)."""
        return {c: self.critical_path_time - t for c, t in self.corner_times.items()}


def simulate_one(
    crew: CrewProfile,
    fatigue_factor: float = 0.0,  # [0, 1] — increases mean/std
    weather_factor: float = 0.0,  # [0, 1] — e.g., 0.1 for damp conditions
    rng: np.random.Generator = None,
) -> PitStopResult:
    """
    Simulate one pit stop.

    Fatigue increases both mean and variance:
        mean_eff = mean_base * crew.mean_factor * (1 + 0.15 * fatigue)
        std_eff  = std_base  * crew.std_factor  * (1 + 0.25 * fatigue)

    Weather increases variance:
        std_eff *= (1 + 0.20 * weather)
    """
    rng = rng or np.random.default_rng()

    task_breakdown = {}
    corner_times = {}

    for corner in CORNERS:
        task_breakdown[corner] = {}
        total = 0.0
        for task in TASK_SEQUENCE:
            base = BASE_TASKS[task]
            mean_eff = base.mean * crew.mean_factor * (1 + 0.15 * fatigue_factor)
            std_eff  = base.std  * crew.std_factor  * (1 + 0.25 * fatigue_factor) * (1 + 0.20 * weather_factor)
            t = TaskParams(mean_eff, std_eff).sample(1, rng)[0]
            task_breakdown[corner][task] = round(t, 5)
            total += t
        corner_times[corner] = round(total, 5)

    # Critical path: bottleneck corner determines when jacks can come down
    critical_corner = max(corner_times, key=corner_times.get)
    critical_path_time = corner_times[critical_corner]

    # Jack time (front and rear in parallel, both must finish)
    j_front = BASE_TASKS["jack_front"].sample(1, rng)[0] * crew.mean_factor
    j_rear  = BASE_TASKS["jack_rear"].sample(1, rng)[0]  * crew.mean_factor
    jack_time = max(j_front, j_rear)

    # Reaction time (driver leaves once jack down + lollipop)
    reaction = BASE_TASKS["reaction"].sample(1, rng)[0]

    # Clash: independent probability, causes +0.3-0.8s penalty
    clash_prob = crew.base_clash_prob * (1 + 0.4 * fatigue_factor)
    had_clash = rng.random() < clash_prob
    clash_penalty = rng.uniform(0.3, 0.8) if had_clash else 0.0

    pit_time = reaction + critical_path_time + jack_time + clash_penalty

    return PitStopResult(
        pit_time=round(pit_time, 5),
        corner_times=corner_times,
        task_breakdown=task_breakdown,
        critical_corner=critical_corner,
        critical_path_time=critical_path_time,
        had_clash=had_clash,
        clash_penalty=clash_penalty,
        jack_time=round(jack_time, 5),
    )


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloConfig:
    n_iterations: int = 10_000
    crew_name: str = "elite"
    fatigue_factor: float = 0.0
    weather_factor: float = 0.0
    seed: int = 42


@dataclass
class MonteCarloResults:
    config: MonteCarloConfig
    pit_times: np.ndarray
    results: list[PitStopResult]

    # Summary statistics (computed on init)
    mean: float = field(init=False)
    std: float = field(init=False)
    p05: float = field(init=False)
    p50: float = field(init=False)
    p95: float = field(init=False)
    clash_rate: float = field(init=False)
    critical_corner_dist: dict = field(init=False)

    def __post_init__(self):
        t = self.pit_times
        self.mean  = float(np.mean(t))
        self.std   = float(np.std(t))
        self.p05   = float(np.percentile(t, 5))
        self.p50   = float(np.percentile(t, 50))
        self.p95   = float(np.percentile(t, 95))
        self.clash_rate = float(np.mean([r.had_clash for r in self.results]))
        corners = [r.critical_corner for r in self.results]
        n = len(corners)
        self.critical_corner_dist = {c: corners.count(c)/n for c in CORNERS}

    def sub_threshold_probability(self, threshold: float) -> float:
        """P(pit_time < threshold)"""
        return float(np.mean(self.pit_times < threshold))

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"pit_time": r.pit_time, "critical_corner": r.critical_corner,
                   "had_clash": r.had_clash, "clash_penalty": r.clash_penalty,
                   "jack_time": r.jack_time}
            for corner in CORNERS:
                row[f"{corner}_time"] = r.corner_times[corner]
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        return (
            f"Monte Carlo Results ({self.config.n_iterations:,} iterations)\n"
            f"Crew: {self.config.crew_name}  |  "
            f"Fatigue: {self.config.fatigue_factor:.2f}  |  "
            f"Weather: {self.config.weather_factor:.2f}\n"
            f"{'─'*50}\n"
            f"Mean pit time : {self.mean:.3f}s\n"
            f"Std dev       : {self.std:.3f}s\n"
            f"P05 (best)    : {self.p05:.3f}s\n"
            f"P50 (median)  : {self.p50:.3f}s\n"
            f"P95 (worst)   : {self.p95:.3f}s\n"
            f"Clash rate    : {self.clash_rate*100:.1f}%\n"
            f"Sub-2.5s prob : {self.sub_threshold_probability(2.5)*100:.1f}%\n"
            f"Critical corner: {max(self.critical_corner_dist, key=self.critical_corner_dist.get)}\n"
        )


def run_monte_carlo(config: MonteCarloConfig) -> MonteCarloResults:
    """Run full Monte Carlo simulation."""
    rng = np.random.default_rng(config.seed)
    crew = CREW_PROFILES[config.crew_name]

    results = [
        simulate_one(crew, config.fatigue_factor, config.weather_factor, rng)
        for _ in range(config.n_iterations)
    ]
    pit_times = np.array([r.pit_time for r in results])
    return MonteCarloResults(config=config, pit_times=pit_times, results=results)


def compare_crews(
    n_iterations: int = 5000,
    fatigue_factor: float = 0.0,
) -> pd.DataFrame:
    """Compare all crew profiles head-to-head."""
    rows = []
    for crew_name in CREW_PROFILES:
        cfg = MonteCarloConfig(n_iterations=n_iterations, crew_name=crew_name,
                               fatigue_factor=fatigue_factor)
        res = run_monte_carlo(cfg)
        rows.append({
            "crew": crew_name,
            "mean": round(res.mean, 3),
            "std": round(res.std, 3),
            "p05": round(res.p05, 3),
            "p95": round(res.p95, 3),
            "clash_rate_%": round(res.clash_rate * 100, 1),
            "sub_2.5s_%": round(res.sub_threshold_probability(2.5) * 100, 1),
        })
    return pd.DataFrame(rows).set_index("crew")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Pit Stop Monte Carlo Simulation")
    parser.add_argument("--crew", default="elite", choices=list(CREW_PROFILES.keys()))
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--fatigue", type=float, default=0.0)
    parser.add_argument("--weather", type=float, default=0.0)
    parser.add_argument("--compare", action="store_true", help="Compare all crew profiles")
    args = parser.parse_args()

    if args.compare:
        print("\n=== Crew Comparison ===")
        print(compare_crews(args.iterations, args.fatigue).to_string())
    else:
        cfg = MonteCarloConfig(
            n_iterations=args.iterations,
            crew_name=args.crew,
            fatigue_factor=args.fatigue,
            weather_factor=args.weather,
        )
        res = run_monte_carlo(cfg)
        print(res.summary())
