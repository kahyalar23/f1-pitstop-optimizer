"""
pitstop/simulation/human_factors.py

Human fatigue and cognitive error model for F1 pit crew.

Based on:
- Osman & Sheridan (1984) Cumulative Fatigue Model
- THERP (Technique for Human Error Rate Prediction) — adapted for motor sports
- Yerkes-Dodson arousal-performance curve

Key equations:
    F(t) = F_max * (1 - exp(-beta * W_cumulative))      [fatigue accumulation]
    P_error(F) = P_base * exp(gamma * F)                 [error probability]
    arousal_mod = 1 + k * exp(-0.5 * ((A - A_opt)/sigma_A)^2)  [Yerkes-Dodson]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Crew member model
# ---------------------------------------------------------------------------

@dataclass
class CrewMember:
    """
    Individual crew member with fatigue state and skill profile.

    role: e.g. "gunman_FL", "tyre_carrier_FR", "jack_front"
    """
    name: str
    role: str
    base_task_time: float     # seconds for primary task (fresh, no fatigue)
    task_time_std: float      # base standard deviation
    base_error_prob: float    # probability of minor task error when fresh (THERP baseline)
    fitness_level: float      # [0, 1] — higher = fatigue accumulates slower
    experience: float         # [0, 1] — reduces base error prob

    # State (mutable)
    fatigue: float = 0.0          # [0, 1]
    cumulative_workload: float = 0.0
    arousal: float = 0.5          # [0, 1]
    errors_this_session: int = 0

    # Constants
    F_MAX: float = 1.0
    BETA: float = 0.03          # workload → fatigue rate
    GAMMA: float = 2.5          # fatigue → error prob amplifier
    A_OPT: float = 0.65         # optimal arousal level (Yerkes-Dodson)
    K_AROUSAL: float = 0.20     # max arousal performance boost

    def update_fatigue(self, workload_unit: float = 1.0):
        """
        Advance fatigue by one pit stop event.

        F(n+1) = F_max * (1 - exp(-beta * W_n+1))
        Fitness slows accumulation: effective_beta = beta / (1 + fitness)
        """
        effective_beta = self.BETA / (1 + self.fitness_level)
        self.cumulative_workload += workload_unit
        self.fatigue = self.F_MAX * (
            1 - np.exp(-effective_beta * self.cumulative_workload)
        )
        self.fatigue = float(np.clip(self.fatigue, 0.0, 1.0))

    def update_arousal(self, race_hour: float, crowd_noise_db: float = 95.0):
        """
        Arousal evolves with race duration and environmental stimulation.
        High arousal early, declines with fatigue.
        """
        time_decay = np.exp(-0.15 * race_hour)
        noise_boost = 0.1 * np.tanh((crowd_noise_db - 90) / 10)
        fatigue_suppression = 0.3 * self.fatigue
        self.arousal = float(np.clip(
            0.7 * time_decay + noise_boost - fatigue_suppression + np.random.normal(0, 0.05),
            0.0, 1.0
        ))

    @property
    def error_probability(self) -> float:
        """
        P_error(F) = P_base * exp(gamma * F) * experience_correction * arousal_mod

        Experience reduces base error rate (learned motor programs).
        Arousal modifies via inverted-U (Yerkes-Dodson).
        """
        # Fatigue amplification
        p_fatigue = self.base_error_prob * np.exp(self.GAMMA * self.fatigue)
        # Experience correction
        p_exp = p_fatigue * (1 - 0.5 * self.experience)
        # Arousal: Yerkes-Dodson inverted U
        arousal_mod = 1 + self.K_AROUSAL * np.exp(
            -0.5 * ((self.arousal - self.A_OPT) / 0.25) ** 2
        )
        return float(np.clip(p_exp / arousal_mod, 0.0, 0.95))

    @property
    def effective_task_time(self) -> tuple[float, float]:
        """
        Returns (mean, std) of task time accounting for fatigue.

        mean_eff = base_time * (1 + 0.12 * fatigue)
        std_eff  = base_std  * (1 + 0.30 * fatigue)
        """
        mean = self.base_task_time * (1 + 0.12 * self.fatigue)
        std  = self.task_time_std  * (1 + 0.30 * self.fatigue)
        return mean, std

    def sample_task_time(self, rng: np.random.Generator = None) -> float:
        """Sample task duration from log-normal with fatigue-adjusted parameters."""
        rng = rng or np.random.default_rng()
        mean, std = self.effective_task_time
        cv2 = (std / mean) ** 2
        mu_ln = np.log(mean) - 0.5 * np.log(1 + cv2)
        sigma_ln = np.sqrt(np.log(1 + cv2))
        return float(rng.lognormal(mu_ln, sigma_ln))

    def attempt_task(self, rng: np.random.Generator = None) -> dict:
        """
        Perform one task. Returns dict with duration and outcome.

        Error types (THERP-inspired):
            0: No error (nominal)
            1: Fumble — partial drop/retry → +0.15-0.40s
            2: Miss — wrong motion, correct → +0.40-0.80s
            3: Cross-coupling — adjacent crew collision → +0.60-1.20s
        """
        rng = rng or np.random.default_rng()
        duration = self.sample_task_time(rng)
        p_err = self.error_probability

        error_type = 0
        penalty = 0.0

        if rng.random() < p_err:
            # Sample error severity
            severity = rng.random()
            if severity < 0.60:
                error_type = 1  # fumble
                penalty = rng.uniform(0.15, 0.40)
            elif severity < 0.90:
                error_type = 2  # miss
                penalty = rng.uniform(0.40, 0.80)
            else:
                error_type = 3  # cross-coupling (rare, catastrophic)
                penalty = rng.uniform(0.60, 1.20)
            self.errors_this_session += 1

        return {
            "nominal_duration": round(duration, 4),
            "error_type": error_type,
            "error_penalty": round(penalty, 4),
            "total_duration": round(duration + penalty, 4),
            "fatigue_level": round(self.fatigue, 3),
            "error_probability": round(p_err, 4),
        }


# ---------------------------------------------------------------------------
# Full crew simulation
# ---------------------------------------------------------------------------

@dataclass
class PitCrew:
    """Full 18-member pit crew with individual fatigue tracking."""
    members: list[CrewMember]

    @classmethod
    def create_standard(cls, skill_level: float = 0.8) -> "PitCrew":
        """
        Factory: create a standard 18-man F1 pit crew.
        skill_level: [0, 1] — scales fitness and experience
        """
        roles = [
            # Corner crew: 3 per corner = 12 members
            ("gunman_FL", 0.48, 0.06, 0.018),
            ("outer_carrier_FL", 0.28, 0.04, 0.022),
            ("inner_carrier_FL", 0.26, 0.04, 0.022),
            ("gunman_FR", 0.48, 0.06, 0.018),
            ("outer_carrier_FR", 0.28, 0.04, 0.022),
            ("inner_carrier_FR", 0.26, 0.04, 0.022),
            ("gunman_RL", 0.48, 0.06, 0.018),
            ("outer_carrier_RL", 0.28, 0.04, 0.022),
            ("inner_carrier_RL", 0.26, 0.04, 0.022),
            ("gunman_RR", 0.48, 0.06, 0.018),
            ("outer_carrier_RR", 0.28, 0.04, 0.022),
            ("inner_carrier_RR", 0.26, 0.04, 0.022),
            # Support: 6 members
            ("jack_front", 0.30, 0.04, 0.015),
            ("jack_rear", 0.32, 0.04, 0.015),
            ("lollipop", 0.15, 0.03, 0.010),
            ("front_wing_1", 0.60, 0.08, 0.025),
            ("front_wing_2", 0.60, 0.08, 0.025),
            ("fire_marshal", 0.10, 0.02, 0.005),
        ]
        members = []
        for i, (role, t_mean, t_std, p_err) in enumerate(roles):
            members.append(CrewMember(
                name=f"crew_{i+1:02d}",
                role=role,
                base_task_time=t_mean,
                task_time_std=t_std,
                base_error_prob=p_err,
                fitness_level=skill_level * np.random.uniform(0.85, 1.0),
                experience=skill_level * np.random.uniform(0.80, 1.0),
            ))
        return cls(members=members)

    def simulate_pit_stop(self, rng: np.random.Generator = None) -> dict:
        """
        Simulate one pit stop with full crew, update fatigue state.
        Returns detailed breakdown.
        """
        rng = rng or np.random.default_rng()
        outcomes = {}
        for member in self.members:
            member.update_fatigue(workload_unit=1.0)
            outcomes[member.role] = member.attempt_task(rng)

        # Compute critical path from corner tasks
        corner_times = {}
        for corner in ["FL", "FR", "RL", "RR"]:
            # gunman + carrier tasks in sequence (simplified)
            gunman = outcomes.get(f"gunman_{corner}", {}).get("total_duration", 0.48)
            t_corner = gunman * 2  # loosen + tighten (dominant)
            corner_times[corner] = t_corner

        jack_f = outcomes.get("jack_front", {}).get("total_duration", 0.30)
        jack_r = outcomes.get("jack_rear",  {}).get("total_duration", 0.32)
        reaction = outcomes.get("lollipop", {}).get("total_duration", 0.15)

        critical = max(corner_times.values())
        pit_time = critical + max(jack_f, jack_r) + reaction

        total_errors = sum(o.get("error_type", 0) > 0 for o in outcomes.values())

        return {
            "pit_time": round(pit_time, 4),
            "critical_corner": max(corner_times, key=corner_times.get),
            "corner_times": corner_times,
            "total_errors": total_errors,
            "avg_fatigue": round(np.mean([m.fatigue for m in self.members]), 3),
            "max_fatigue": round(max(m.fatigue for m in self.members), 3),
            "outcomes": outcomes,
        }

    def fatigue_report(self) -> pd.DataFrame:
        """Return current fatigue state for all crew members."""
        rows = [{
            "role": m.role,
            "fatigue": round(m.fatigue, 3),
            "error_prob_%": round(m.error_probability * 100, 2),
            "workload": round(m.cumulative_workload, 1),
            "errors": m.errors_this_session,
            "fitness": round(m.fitness_level, 2),
        } for m in self.members]
        return pd.DataFrame(rows)


def simulate_race_pitstops(
    n_stops: int = 3,
    skill_level: float = 0.8,
    stops_per_hour: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate N pit stops across a race, tracking fatigue accumulation.
    Returns per-stop summary.
    """
    rng = np.random.default_rng(seed)
    crew = PitCrew.create_standard(skill_level=skill_level)
    rows = []

    for stop_n in range(1, n_stops + 1):
        race_hour = (stop_n - 1) / stops_per_hour
        for m in crew.members:
            m.update_arousal(race_hour)

        result = crew.simulate_pit_stop(rng)
        rows.append({
            "stop_number": stop_n,
            "race_hour": round(race_hour, 2),
            "pit_time": result["pit_time"],
            "critical_corner": result["critical_corner"],
            "total_errors": result["total_errors"],
            "avg_fatigue": result["avg_fatigue"],
            "max_fatigue": result["max_fatigue"],
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Race Pit Stop Fatigue Simulation ===")
    df = simulate_race_pitstops(n_stops=5, skill_level=0.85)
    print(df.to_string(index=False))

    print("\n=== Crew Fatigue Report (after 5 stops) ===")
    crew = PitCrew.create_standard(0.85)
    rng = np.random.default_rng(42)
    for _ in range(5):
        crew.simulate_pit_stop(rng)
    report = crew.fatigue_report()
    print(report.to_string(index=False))
