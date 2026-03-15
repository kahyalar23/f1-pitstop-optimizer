"""
pitstop/analysis/gp_analyzer.py

GP-Specific Analysis Engine — 2023 Season
──────────────────────────────────────────
Her GP için:
  1. Pit stop timing dağılımı ve ekip karşılaştırması
  2. Stint analizi — degradasyon eğrileri per compound
  3. Undercut penceresi tespiti
  4. Simülasyon vs gerçek veri karşılaştırması (KS-test)
  5. "What-if" senaryoları — farklı pit window'da ne olurdu?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pitstop.data.f1_2023_data import (
    load_real_pitstops, load_lap_times, get_real_strategy,
    RACES_2023, REAL_STRATEGIES_2023, PIT_STOP_DATA_2023
)
from pitstop.simulation.monte_carlo import run_monte_carlo, MonteCarloConfig
from pitstop.simulation.tire_model import simulate_stint, undercut_delta, COMPOUNDS


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class GPAnalysisResult:
    race: str
    year: int
    pit_df: pd.DataFrame
    lap_df: pd.DataFrame
    sim_times: np.ndarray
    ks_stat: float
    ks_pvalue: float
    mae: float
    fastest_real: float
    fastest_sim_p05: float
    undercut_windows: list[dict]


# ─────────────────────────────────────────────
# Main Analyzer
# ─────────────────────────────────────────────

class GPAnalyzer:
    def __init__(self, race: str, year: int = 2023, crew_name: str = "elite"):
        self.race = race
        self.year = year
        self.crew_name = crew_name
        self.race_cfg = RACES_2023.get(race.capitalize(), RACES_2023["Monaco"])

        print(f"Loading {year} {race} GP data...")
        self.pit_df = load_real_pitstops(year, race)
        self.lap_df = load_lap_times(year, race)
        print(f"  ✓ {len(self.pit_df)} pit stops loaded")
        print(f"  ✓ {len(self.lap_df)} lap records loaded")

    # ── 1. Pit Stop Distribution ──────────────────────────────────────

    def analyze_pit_timing(self, n_sim: int = 5000) -> dict:
        """Compare simulated vs real pit stop distribution."""
        real_times = self.pit_df["time_s"].values
        real_times = real_times[(real_times > 1.8) & (real_times < 8.0)]

        # Run simulation
        cfg = MonteCarloConfig(n_iterations=n_sim, crew_name=self.crew_name, seed=42)
        from pitstop.simulation.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(cfg)
        sim_times = mc.pit_times

        # KS test
        ks_stat, ks_p = ks_2samp(sim_times, real_times)
        mae = abs(np.mean(sim_times) - np.mean(real_times))

        return {
            "real_times": real_times,
            "sim_times": sim_times,
            "real_mean": round(float(np.mean(real_times)), 3),
            "sim_mean": round(float(np.mean(sim_times)), 3),
            "real_std": round(float(np.std(real_times)), 3),
            "sim_std": round(float(np.std(sim_times)), 3),
            "real_fastest": round(float(np.min(real_times)), 3),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "mae": round(float(mae), 4),
            "match": ks_p > 0.05,
        }

    # ── 2. Stint / Degradation Analysis ──────────────────────────────

    def analyze_degradation(self) -> pd.DataFrame:
        """
        Fit per-compound degradation from real lap times.
        Uses linear regression on (tyre_age → lap_time_delta).
        """
        df = self.lap_df.copy()
        # Only "clean" laps (no safety car, no pit lap)
        df = df[df["track_status"] == "1"]

        rows = []
        for compound in df["compound"].unique():
            sub = df[df["compound"] == compound].copy()
            if len(sub) < 5:
                continue
            # Reference: mean laptime at age 0–3
            ref_laps = sub[sub["tyre_age"] <= 3]["lap_time_s"]
            if ref_laps.empty:
                continue
            ref_lt = ref_laps.mean()
            sub["delta"] = sub["lap_time_s"] - ref_lt

            # Regression: delta = a * age^b (log-linear)
            ages = sub["tyre_age"].values
            deltas = sub["delta"].values
            valid = (ages > 0) & (deltas > -2) & (deltas < 10)
            if valid.sum() < 5:
                continue
            log_ages = np.log(ages[valid] + 1)
            log_deltas = np.log(np.maximum(deltas[valid] + 0.01, 0.01))
            try:
                b, log_a = np.polyfit(log_ages, log_deltas, 1)
                a = np.exp(log_a)
            except Exception:
                a, b = 0.05, 0.7

            rows.append({
                "compound": compound,
                "ref_laptime": round(ref_lt, 3),
                "deg_coeff_a": round(float(a), 5),
                "deg_coeff_b": round(float(b), 4),
                "n_laps": int(valid.sum()),
                "max_age_seen": int(sub["tyre_age"].max()),
                "total_delta_20laps": round(float(a * 20**b), 3),
            })
        return pd.DataFrame(rows)

    # ── 3. Undercut Window Detection ─────────────────────────────────

    def find_undercut_windows(self, driver: str = "Verstappen") -> list[dict]:
        """
        Scan each lap for undercut viability using real timing data.
        Returns list of viable windows.
        """
        race_key = self.race.capitalize()
        strategy = get_real_strategy(race_key, driver)
        if strategy is None:
            return []

        # Estimate pit loss for this race
        pit_entry_loss = 18.5  # seconds
        pit_exit_loss  = 16.0
        mean_stop = self.pit_df["time_s"].mean() if len(self.pit_df) > 0 else 2.5
        pit_loss = pit_entry_loss + mean_stop + pit_exit_loss

        # Scan laps where opponent could have undercut
        windows = []
        laps_before_pit = strategy["pit_laps"][0] if strategy["pit_laps"] else 20

        for lap in range(max(1, laps_before_pit - 8), laps_before_pit + 3):
            # Simulated gap at that lap (use real lap time delta)
            gap = 0.5 + (laps_before_pit - lap) * 0.15  # simplified gap model
            fresh_gain = COMPOUNDS.get(
                strategy["compound_sequence"][-1].lower(), COMPOUNDS["medium"]
            ).delta_per_lap * 8  # 8-lap fresh tyre advantage window

            uc = undercut_delta(
                gap_ahead=abs(gap),
                pit_loss=pit_loss,
                fresh_tyre_gain=fresh_gain,
                laps_to_respond=1,
            )
            windows.append({
                "lap": lap,
                "gap_ahead": round(gap, 2),
                "pit_loss": round(pit_loss, 1),
                "undercut_delta": uc["undercut_delta"],
                "viable": uc["undercut_viable"],
                "recommendation": uc["recommendation"],
            })
        return windows

    # ── 4. Team Comparison ────────────────────────────────────────────

    def team_pit_comparison(self) -> pd.DataFrame:
        """Per-team pit stop statistics for this race."""
        if "team" not in self.pit_df.columns:
            return pd.DataFrame()
        stats = self.pit_df.groupby("team")["time_s"].agg(
            mean="mean", std="std", min="min", max="max", count="count"
        ).round(3).sort_values("mean")
        return stats

    # ── 5. What-If Scenario ───────────────────────────────────────────

    def what_if_pit_lap(
        self,
        driver: str,
        alternative_lap: int,
        base_compound: str = "MEDIUM",
    ) -> dict:
        """
        What if [driver] pitted on lap [alternative_lap] instead of actual lap?
        Estimates time gained/lost vs actual strategy.
        """
        race_key = self.race.capitalize()
        strategy = get_real_strategy(race_key, driver)
        if strategy is None:
            return {"error": f"No strategy data for {driver} at {race_key}"}

        actual_lap = strategy["pit_laps"][0]
        actual_pit_time = strategy["pit_times_s"][0]

        # Tyre age difference
        age_diff = alternative_lap - actual_lap

        # Compound-specific degradation
        compound = strategy["compound_sequence"][0].lower()
        deg_per_lap = COMPOUNDS.get(compound, COMPOUNDS["medium"]).delta_per_lap

        # Time on older tyres (positive = slower)
        stint_time_diff = age_diff * deg_per_lap * 0.7

        # Undercut/overcut effect
        if age_diff < 0:
            # Pitted earlier: fresher tyres but gave up track position time
            position_effect = -abs(age_diff) * 0.08  # approx benefit
            scenario = "Earlier pit — fresher tyres, potential undercut"
        else:
            # Pitted later: older tyres, potential overcut
            position_effect = abs(age_diff) * 0.05
            scenario = "Later pit — older tyres, overcut attempt"

        net_delta = stint_time_diff + position_effect

        return {
            "driver": driver,
            "race": race_key,
            "actual_pit_lap": actual_lap,
            "alternative_lap": alternative_lap,
            "actual_pit_time_s": actual_pit_time,
            "scenario": scenario,
            "estimated_time_delta_s": round(net_delta, 3),
            "better_or_worse": "BETTER" if net_delta < 0 else "WORSE",
            "magnitude": abs(round(net_delta, 3)),
        }

    # ── 6. Full Report ────────────────────────────────────────────────

    def full_report(self) -> None:
        """Print comprehensive GP analysis."""
        race_key = self.race.capitalize()
        n_laps = self.race_cfg["laps"]

        print(f"\n{'='*60}")
        print(f"  {self.year} {race_key} GP — Full Analysis")
        print(f"{'='*60}\n")

        # Pit timing
        pt = self.analyze_pit_timing(n_sim=3000)
        print("PIT STOP TIMING")
        print(f"  Real mean   : {pt['real_mean']}s ± {pt['real_std']}s")
        print(f"  Sim mean    : {pt['sim_mean']}s ± {pt['sim_std']}s")
        print(f"  MAE         : {pt['mae']}s")
        print(f"  KS test     : D={pt['ks_stat']}, p={pt['ks_pvalue']}")
        print(f"  Match       : {'✓ YES' if pt['match'] else '✗ NO'}")

        # Degradation
        print("\nTYRE DEGRADATION")
        deg = self.analyze_degradation()
        if not deg.empty:
            print(deg.to_string(index=False))

        # Team comparison
        print("\nTEAM PIT PERFORMANCE")
        team_stats = self.team_pit_comparison()
        if not team_stats.empty:
            print(team_stats.to_string())

        # Undercut windows
        print("\nUNDERCUT WINDOWS (Verstappen)")
        for race_key2 in REAL_STRATEGIES_2023.get(race_key, {}):
            windows = self.find_undercut_windows(race_key2)
            viable = [w for w in windows if w["viable"]]
            if viable:
                print(f"  {race_key2}: {len(viable)} viable undercut window(s)")
                for w in viable:
                    print(f"    Lap {w['lap']}: delta={w['undercut_delta']:.2f}s → {w['recommendation']}")


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_gp_overview(analyzer: GPAnalyzer, save_path: str = None) -> None:
    """4-panel GP overview figure."""
    pt = analyzer.analyze_pit_timing(n_sim=3000)
    deg = analyzer.analyze_degradation()
    team_stats = analyzer.team_pit_comparison()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{analyzer.year} {analyzer.race.capitalize()} GP — Analysis Overview",
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    # ── Panel 1: Pit timing distribution ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pt["real_times"], bins=12, density=True, alpha=0.65,
             color='#E74C3C', label=f'Real ({len(pt["real_times"])} stops)')
    ax1.hist(pt["sim_times"], bins=40, density=True, alpha=0.45,
             color='#3498DB', label=f'Simulation (n=5000)')
    ax1.axvline(pt["real_mean"], color='#E74C3C', linewidth=2, linestyle='--')
    ax1.axvline(pt["sim_mean"], color='#3498DB', linewidth=2, linestyle='--')
    ax1.set_xlabel('Pit stop süresi (s)')
    ax1.set_ylabel('Yoğunluk')
    ax1.set_title(f'Pit Timing: Sim vs Real\nKS p={pt["ks_pvalue"]:.3f} | MAE={pt["mae"]:.3f}s')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Degradation curves ──
    ax2 = fig.add_subplot(gs[0, 1])
    compound_colors = {'soft': '#E74C3C', 'medium': '#F39C12', 'hard': '#95A5A6',
                       'SOFT': '#E74C3C', 'MEDIUM': '#F39C12', 'HARD': '#95A5A6'}
    ages = np.linspace(1, 40, 200)
    if not deg.empty:
        for _, row in deg.iterrows():
            comp = row["compound"]
            predicted = row["deg_coeff_a"] * ages ** row["deg_coeff_b"]
            color = compound_colors.get(comp, 'gray')
            ax2.plot(ages, predicted, color=color, linewidth=2.5,
                    label=f'{comp} (a={row["deg_coeff_a"]:.4f}, b={row["deg_coeff_b"]:.2f})')
    ax2.set_xlabel('Lastik yaşı (tur)')
    ax2.set_ylabel('Tur süresi artışı (s)')
    ax2.set_title('Degradasyon — Gerçek Veriden Fit')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)

    # ── Panel 3: Team comparison ──
    ax3 = fig.add_subplot(gs[1, 0])
    if not team_stats.empty:
        teams = team_stats.index.tolist()
        means = team_stats["mean"].values
        stds  = team_stats["std"].fillna(0).values
        colors_t = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(teams)))
        bars = ax3.barh(teams, means, xerr=stds, color=colors_t,
                        alpha=0.85, capsize=4, height=0.6)
        for bar, val in zip(bars, means):
            ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}s', va='center', fontsize=9)
        ax3.set_xlabel('Ortalama pit stop süresi (s)')
        ax3.set_title('Takım Pit Stop Performansı')
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Takım verisi yok', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12, color='gray')

    # ── Panel 4: Lap time evolution ──
    ax4 = fig.add_subplot(gs[1, 1])
    lap_df = analyzer.lap_df
    top_drivers = ["VER", "HAM", "LEC", "NOR"] if "VER" in lap_df.get("driver", pd.Series()).values else \
                  lap_df["driver"].unique()[:4].tolist()
    driver_colors = ['#3498DB', '#27AE60', '#E74C3C', '#F39C12']
    for driver, color in zip(top_drivers, driver_colors):
        ddf = lap_df[lap_df["driver"] == driver].sort_values("lap")
        if not ddf.empty:
            # Rolling median for clean trend
            rolling = ddf["lap_time_s"].rolling(3, center=True).median()
            ax4.plot(ddf["lap"], rolling, color=color, linewidth=2,
                    label=driver, alpha=0.9)
            # Mark pit stops
            if driver in analyzer.pit_df.get("driver", pd.Series()).values:
                pit_laps = analyzer.pit_df[analyzer.pit_df["driver"] == driver]["lap"]
                for pl in pit_laps:
                    ax4.axvline(pl, color=color, alpha=0.25, linewidth=1.5)
    ax4.set_xlabel('Tur')
    ax4.set_ylabel('Tur süresi (s)')
    ax4.set_title('Tur Süresi Evrimi (noktalı = pit)')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--race", default="Monaco")
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--crew", default="elite")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    analyzer = GPAnalyzer(args.race, args.year, args.crew)
    analyzer.full_report()

    if args.plot:
        plot_gp_overview(analyzer, save_path=f"{args.race.lower()}_{args.year}_analysis.png")
