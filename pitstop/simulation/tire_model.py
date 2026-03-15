"""
pitstop/simulation/tire_model.py

Tyre degradation model combining:
1. Pacejka Magic Formula — grip as a function of slip energy
2. Thermal dynamics     — 1st-order ODE for tyre temperature
3. Wear accumulation    — irreversible degradation over laps
4. Compound database    — Soft / Medium / Hard / Inter / Wet

Key equations:
    mu(E, theta) = D * sin(C * arctan(B*E - E*(B*E - arctan(B*E)))) * exp(-lambda * E_cum)
    d_theta/dt   = (1/tau) * (theta_target - theta) + alpha * P_slip
    delta_laptime = k_deg * (1 - mu_current / mu_new)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Tyre compound database
# ---------------------------------------------------------------------------

CompoundName = Literal["soft", "medium", "hard", "inter", "wet"]


@dataclass
class TyreCompound:
    name: CompoundName

    # Pacejka Magic Formula coefficients
    B: float   # stiffness factor
    C: float   # shape factor
    D: float   # peak grip coefficient
    E: float   # curvature factor (in [-1, 0])

    # Thermal parameters
    theta_optimal: float   # optimal operating temperature (°C)
    theta_min: float       # below this: glazing grip penalty
    theta_max: float       # above this: graining/blistering onset
    tau_thermal: float     # time constant for thermal response (laps)
    alpha_heat: float      # slip power → heat conversion

    # Wear model
    lambda_deg: float      # exponential wear rate per kJ/m slip energy
    wear_cliff: float      # cumulative energy threshold for cliff degradation
    cliff_severity: float  # additional grip loss rate after cliff (multiplier)

    # Lap delta per lap of age (s), relative to new tyre
    delta_per_lap: float
    optimal_window: tuple = field(default_factory=lambda: (0, 20))  # (min, max) lap age


COMPOUNDS: dict[CompoundName, TyreCompound] = {
    "soft": TyreCompound(
        name="soft",
        B=10.0, C=1.9, D=1.55, E=-0.5,
        theta_optimal=95.0, theta_min=70.0, theta_max=115.0,
        tau_thermal=0.8, alpha_heat=0.006,
        lambda_deg=0.0045, wear_cliff=280.0, cliff_severity=2.2,
        delta_per_lap=0.08, optimal_window=(0, 18),
    ),
    "medium": TyreCompound(
        name="medium",
        B=9.0, C=1.85, D=1.42, E=-0.4,
        theta_optimal=90.0, theta_min=65.0, theta_max=110.0,
        tau_thermal=1.1, alpha_heat=0.005,
        lambda_deg=0.0028, wear_cliff=400.0, cliff_severity=1.8,
        delta_per_lap=0.05, optimal_window=(0, 30),
    ),
    "hard": TyreCompound(
        name="hard",
        B=8.2, C=1.80, D=1.32, E=-0.3,
        theta_optimal=88.0, theta_min=62.0, theta_max=108.0,
        tau_thermal=1.4, alpha_heat=0.004,
        lambda_deg=0.0017, wear_cliff=550.0, cliff_severity=1.5,
        delta_per_lap=0.03, optimal_window=(0, 45),
    ),
    "inter": TyreCompound(
        name="inter",
        B=6.0, C=1.70, D=1.10, E=-0.2,
        theta_optimal=55.0, theta_min=35.0, theta_max=75.0,
        tau_thermal=1.0, alpha_heat=0.003,
        lambda_deg=0.0040, wear_cliff=200.0, cliff_severity=2.5,
        delta_per_lap=0.15, optimal_window=(0, 15),
    ),
    "wet": TyreCompound(
        name="wet",
        B=4.5, C=1.60, D=0.90, E=-0.1,
        theta_optimal=40.0, theta_min=25.0, theta_max=60.0,
        tau_thermal=0.9, alpha_heat=0.002,
        lambda_deg=0.0060, wear_cliff=120.0, cliff_severity=3.0,
        delta_per_lap=0.20, optimal_window=(0, 10),
    ),
}


# ---------------------------------------------------------------------------
# Tyre state
# ---------------------------------------------------------------------------

@dataclass
class TyreState:
    compound: TyreCompound
    age_laps: int = 0
    temperature: float = 0.0      # °C, starts cold
    cumulative_energy: float = 0.0  # kJ/m — tracks total wear energy
    is_new: bool = True

    def __post_init__(self):
        if self.temperature == 0.0:
            self.temperature = self.compound.theta_optimal * 0.7  # cold start

    @property
    def past_cliff(self) -> bool:
        return self.cumulative_energy > self.compound.wear_cliff


def pacejka_grip(compound: TyreCompound, slip_energy: float) -> float:
    """
    Pacejka Magic Formula for grip coefficient.

    mu = D * sin(C * arctan(B*x - E*(B*x - arctan(B*x))))

    slip_energy: normalised slip energy [0..1] proxy for lateral+longitudinal slip
    """
    x = slip_energy
    B, C, D, E = compound.B, compound.C, compound.D, compound.E
    phi = B * x - E * (B * x - np.arctan(B * x))
    return D * np.sin(C * np.arctan(phi))


def thermal_penalty(compound: TyreCompound, temperature: float) -> float:
    """
    Grip multiplier from temperature mismatch.
    1.0 = optimal, <1.0 = cold or overheating.
    """
    theta = temperature
    opt = compound.theta_optimal
    if theta < compound.theta_min:
        # Cold tyre: linear penalty
        return 0.75 + 0.25 * (theta - compound.theta_min) / (opt - compound.theta_min)
    elif theta > compound.theta_max:
        # Overheating: faster penalty
        excess = theta - compound.theta_max
        return max(0.70, 1.0 - 0.015 * excess)
    else:
        # Parabolic peak around optimal
        delta = abs(theta - opt)
        width = (compound.theta_max - compound.theta_min) / 2.0
        return 1.0 - 0.15 * (delta / width) ** 2


def wear_multiplier(compound: TyreCompound, cumulative_energy: float) -> float:
    """
    Exponential wear degradation, with cliff after threshold.

        mu_wear = exp(-lambda * E_cum)   [before cliff]
        mu_wear = exp(-lambda * cliff) * exp(-lambda * cliff_severity * (E - cliff))  [after cliff]
    """
    lam = compound.lambda_deg
    cliff = compound.wear_cliff
    if cumulative_energy <= cliff:
        return np.exp(-lam * cumulative_energy)
    else:
        beyond = cumulative_energy - cliff
        return np.exp(-lam * cliff) * np.exp(-lam * compound.cliff_severity * beyond)


def simulate_tyre_lap(
    state: TyreState,
    slip_energy_per_lap: float = 1.0,  # relative [0..2], 1.0 = normal lap
    ambient_temp: float = 25.0,
    track_temp: float = 40.0,
) -> tuple[TyreState, float]:
    """
    Advance tyre state by one lap.

    Returns:
        new_state: updated TyreState
        lap_time_delta: additional seconds vs new-tyre reference
    """
    c = state.compound

    # ---- Thermal update (Euler step over ~90 seconds / lap) ----
    theta_track_effect = track_temp * 0.4 + ambient_temp * 0.2
    theta_target = c.theta_optimal * 0.9 + theta_track_effect * 0.1
    P_slip = slip_energy_per_lap * 15.0  # W/kg proxy

    dt = 1.0  # 1 lap
    d_theta = (1.0 / c.tau_thermal) * (theta_target - state.temperature) + c.alpha_heat * P_slip
    new_temp = state.temperature + d_theta * dt
    new_temp = float(np.clip(new_temp, 20.0, 140.0))

    # ---- Wear energy accumulation ----
    energy_this_lap = slip_energy_per_lap * 35.0  # kJ/m per lap (typical ~35 on normal lap)
    new_energy = state.cumulative_energy + energy_this_lap

    # ---- Grip calculation ----
    mu_base      = pacejka_grip(c, min(slip_energy_per_lap, 1.5))
    mu_thermal   = thermal_penalty(c, new_temp)
    mu_wear      = wear_multiplier(c, new_energy)
    mu_effective = mu_base * mu_thermal * mu_wear

    # Reference grip for new tyre at optimal temp
    mu_new = pacejka_grip(c, 1.0) * 1.0 * wear_multiplier(c, 0.0)

    # Lap time delta (s): grip loss maps to ~0.03s per 1% grip loss (empirical F1 coefficient)
    grip_ratio = mu_effective / mu_new
    lap_delta = c.delta_per_lap * (state.age_laps + 1) ** 0.7  # Sublinear aging
    if state.past_cliff or new_energy > c.wear_cliff:
        lap_delta *= c.cliff_severity * 0.6  # Cliff makes it much worse

    new_state = TyreState(
        compound=c,
        age_laps=state.age_laps + 1,
        temperature=new_temp,
        cumulative_energy=new_energy,
        is_new=False,
    )

    return new_state, round(lap_delta, 4)


def simulate_stint(
    compound_name: CompoundName,
    n_laps: int,
    starting_energy: float = 0.0,
    ambient_temp: float = 25.0,
    track_temp: float = 40.0,
    slip_profile: list[float] | None = None,
) -> pd.DataFrame:
    """Simulate a full tyre stint. Returns lap-by-lap DataFrame."""
    import pandas as pd

    compound = COMPOUNDS[compound_name]
    state = TyreState(compound=compound, cumulative_energy=starting_energy)
    slip_profile = slip_profile or [1.0] * n_laps

    rows = []
    for lap in range(n_laps):
        slip = slip_profile[lap % len(slip_profile)]
        state, delta = simulate_tyre_lap(state, slip, ambient_temp, track_temp)
        rows.append({
            "lap": lap + 1,
            "compound": compound_name,
            "age": state.age_laps,
            "temperature": round(state.temperature, 1),
            "cumulative_energy": round(state.cumulative_energy, 1),
            "lap_delta": delta,
            "past_cliff": state.past_cliff,
            "grip_mu": round(
                pacejka_grip(compound, slip)
                * thermal_penalty(compound, state.temperature)
                * wear_multiplier(compound, state.cumulative_energy), 4
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Undercut / Overcut delta calculator
# ---------------------------------------------------------------------------

def undercut_delta(
    gap_ahead: float,       # seconds to car ahead on track
    pit_loss: float,        # estimated pit stop time loss (stationary + in/out)
    fresh_tyre_gain: float, # lap time gain with new vs opponent's worn tyres (s/lap)
    laps_to_respond: int = 1,  # laps opponent takes to react and pit themselves
) -> dict:
    """
    Estimate undercut opportunity.

    Undercut delta over N laps:
        delta_uc = gap_ahead - (pit_loss - fresh_tyre_gain * laps_to_respond)

    If delta_uc > 0: undercut is viable (you emerge ahead).

    Returns dict with analysis breakdown.
    """
    # One-lap lookahead
    net_time_cost = pit_loss - fresh_tyre_gain * laps_to_respond
    delta_uc = gap_ahead - net_time_cost

    # Overcut: stay out, opponent pits — does gap work in our favour?
    # Viable if our worn tyres are only slightly slower AND gap is large enough
    overcut_viable = gap_ahead > pit_loss * 0.8 and fresh_tyre_gain < 0.5

    return {
        "gap_ahead": gap_ahead,
        "pit_loss": pit_loss,
        "fresh_tyre_gain_per_lap": fresh_tyre_gain,
        "net_time_cost": round(net_time_cost, 3),
        "undercut_delta": round(delta_uc, 3),
        "undercut_viable": delta_uc > 0,
        "overcut_viable": overcut_viable,
        "recommendation": (
            "UNDERCUT — pit now" if delta_uc > 0.3 else
            "MARGINAL — undercut risky" if delta_uc > 0 else
            "OVERCUT — stay out" if overcut_viable else
            "NEUTRAL — wait for VSC/SC"
        ),
    }


if __name__ == "__main__":
    import pandas as pd

    # Demo: simulate a 25-lap soft stint
    df = simulate_stint("soft", n_laps=25, track_temp=45.0)
    print("=== Soft Tyre Stint ===")
    print(df[["lap", "temperature", "lap_delta", "grip_mu", "past_cliff"]].to_string())

    print("\n=== Undercut Analysis ===")
    result = undercut_delta(gap_ahead=1.8, pit_loss=22.5, fresh_tyre_gain=0.8)
    for k, v in result.items():
        print(f"  {k}: {v}")
