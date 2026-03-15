"""
api/main.py — FastAPI REST backend for F1 Pit Stop Optimizer

Endpoints:
    POST /simulate/pitstop       — Run Monte Carlo simulation
    POST /simulate/stint         — Run tyre stint simulation
    POST /strategy/undercut      — Undercut/overcut analysis
    POST /strategy/fatigue       — Crew fatigue impact analysis
    GET  /validate               — Run simulation vs real data validation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pitstop.simulation.monte_carlo import (
    run_monte_carlo, MonteCarloConfig, compare_crews
)
from pitstop.simulation.tire_model import (
    simulate_stint, undercut_delta, COMPOUNDS
)
from pitstop.simulation.human_factors import simulate_race_pitstops


app = FastAPI(
    title="F1 Pit Stop Optimizer API",
    description="Stochastic simulation and RL-based strategy optimization for F1 pit stops",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class PitstopSimRequest(BaseModel):
    crew_name: Literal["rookie", "mid", "elite", "mercedes"] = "elite"
    n_iterations: int = Field(default=5000, ge=100, le=50000)
    fatigue_factor: float = Field(default=0.0, ge=0.0, le=1.0)
    weather_factor: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = 42


class StintSimRequest(BaseModel):
    compound: Literal["soft", "medium", "hard", "inter", "wet"] = "medium"
    n_laps: int = Field(default=30, ge=1, le=70)
    track_temp: float = Field(default=40.0, ge=15.0, le=65.0)
    ambient_temp: float = Field(default=25.0, ge=5.0, le=45.0)


class UndercutRequest(BaseModel):
    gap_ahead: float = Field(description="Gap to car ahead (seconds)", ge=0.0)
    pit_loss: float = Field(description="Expected pit stop time loss (seconds)", ge=15.0)
    fresh_tyre_gain: float = Field(description="Lap time gain on fresh vs worn tyre (s/lap)")
    laps_to_respond: int = Field(default=1, ge=1, le=5)


class FatigueRequest(BaseModel):
    n_stops: int = Field(default=3, ge=1, le=10)
    skill_level: float = Field(default=0.8, ge=0.0, le=1.0)
    seed: int = 42


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "F1 Pit Stop Optimizer",
        "version": "1.0.0",
        "endpoints": ["/simulate/pitstop", "/simulate/stint",
                      "/strategy/undercut", "/strategy/fatigue",
                      "/simulate/compare_crews", "/validate"],
    }


@app.post("/simulate/pitstop")
def simulate_pitstop(req: PitstopSimRequest):
    """
    Run Monte Carlo pit stop simulation.
    Returns distribution statistics and per-corner analysis.
    """
    cfg = MonteCarloConfig(
        n_iterations=req.n_iterations,
        crew_name=req.crew_name,
        fatigue_factor=req.fatigue_factor,
        weather_factor=req.weather_factor,
        seed=req.seed,
    )
    result = run_monte_carlo(cfg)

    # Build histogram for frontend
    times = result.pit_times
    hist_counts, hist_edges = np.histogram(times, bins=25, range=(1.5, 4.5))
    histogram = {
        "counts": hist_counts.tolist(),
        "edges": [round(e, 3) for e in hist_edges.tolist()],
        "bin_centers": [round((hist_edges[i] + hist_edges[i+1])/2, 3)
                        for i in range(len(hist_edges)-1)],
    }

    return {
        "config": {
            "crew": req.crew_name,
            "iterations": req.n_iterations,
            "fatigue_factor": req.fatigue_factor,
            "weather_factor": req.weather_factor,
        },
        "statistics": {
            "mean": round(result.mean, 4),
            "std": round(result.std, 4),
            "p05": round(result.p05, 4),
            "p50": round(result.p50, 4),
            "p95": round(result.p95, 4),
            "clash_rate": round(result.clash_rate, 4),
            "sub_2_5s_probability": round(result.sub_threshold_probability(2.5), 4),
        },
        "critical_corner_distribution": result.critical_corner_dist,
        "histogram": histogram,
    }


@app.post("/simulate/stint")
def simulate_stint_endpoint(req: StintSimRequest):
    """Simulate a full tyre stint and return lap-by-lap degradation."""
    df = simulate_stint(
        req.compound, req.n_laps,
        ambient_temp=req.ambient_temp, track_temp=req.track_temp
    )
    return {
        "compound": req.compound,
        "laps": df["lap"].tolist(),
        "lap_deltas": df["lap_delta"].tolist(),
        "temperatures": df["temperature"].tolist(),
        "grip_mu": df["grip_mu"].tolist(),
        "past_cliff": df["past_cliff"].tolist(),
        "total_delta": round(df["lap_delta"].sum(), 3),
        "cliff_lap": int(df[df["past_cliff"]]["lap"].min()) if df["past_cliff"].any() else None,
    }


@app.post("/strategy/undercut")
def undercut_analysis(req: UndercutRequest):
    """Calculate undercut/overcut opportunity."""
    result = undercut_delta(
        gap_ahead=req.gap_ahead,
        pit_loss=req.pit_loss,
        fresh_tyre_gain=req.fresh_tyre_gain,
        laps_to_respond=req.laps_to_respond,
    )
    return result


@app.post("/strategy/fatigue")
def fatigue_analysis(req: FatigueRequest):
    """Simulate crew fatigue accumulation across multiple pit stops."""
    df = simulate_race_pitstops(
        n_stops=req.n_stops,
        skill_level=req.skill_level,
        seed=req.seed,
    )
    return {
        "stops": df.to_dict(orient="records"),
        "fatigue_trend": df["avg_fatigue"].tolist(),
        "pit_time_trend": df["pit_time"].tolist(),
        "total_errors": int(df["total_errors"].sum()),
        "pit_time_increase": round(
            df["pit_time"].iloc[-1] - df["pit_time"].iloc[0], 4
        ) if len(df) > 1 else 0.0,
    }


@app.get("/simulate/compare_crews")
def compare_crews_endpoint(n_iterations: int = 2000, fatigue: float = 0.0):
    """Compare all crew profiles head-to-head."""
    df = compare_crews(n_iterations=n_iterations, fatigue_factor=fatigue)
    return df.reset_index().to_dict(orient="records")


@app.get("/validate")
def run_validation(race: str = "Monaco", year: int = 2023, n_sim: int = 2000):
    """Run full validation pipeline against real FastF1 data."""
    from pitstop.data.fastf1_loader import (
        load_race_pitstops, validate_pitstop_simulation
    )
    real_df = load_race_pitstops(year, race)
    real_times = real_df["stationary_approx"].dropna().values
    real_times = real_times[(real_times > 1.5) & (real_times < 10)]

    cfg = MonteCarloConfig(n_iterations=n_sim, crew_name="elite")
    mc = run_monte_carlo(cfg)
    val = validate_pitstop_simulation(mc.pit_times, real_times)

    return {
        "ks_statistic": val.ks_statistic,
        "ks_pvalue": val.ks_pvalue,
        "distribution_match": val.distribution_match,
        "mae": val.mae,
        "mean_sim": val.mean_sim,
        "mean_real": val.mean_real,
        "coverage_90": val.coverage_90,
        "n_real": val.n_real,
        "n_sim": val.n_sim,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
