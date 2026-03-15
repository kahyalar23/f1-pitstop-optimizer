"""
pitstop/analysis/strategy_comparison.py

RL Agent vs Real 2023 F1 Strategy Comparison
─────────────────────────────────────────────
Soru: "Pekiştirmeli öğrenme ajanı, gerçek F1 takımlarının aldığı
      strateji kararlarını ne kadar iyi taklit edebiliyor?"

Metodoloji:
  1. Gerçek strateji kararlarını (pit lap, compound) "doğru cevap" olarak kullan
  2. Rule-based baseline ile aynı yarışı simüle et
  3. Kural tabanlı stratejinin pit kararlarını gerçekle karşılaştır
  4. Metrikler: pit lap MAE, compound uyum, pozisyon delta
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pitstop.data.f1_2023_data import (
    REAL_STRATEGIES_2023, RACES_2023, PIT_STOP_DATA_2023
)
from pitstop.strategy.environment import F1RaceEnv
from pitstop.strategy.rl_agent import RuleBasedStrategy


# ─────────────────────────────────────────────
# Strategy Comparison Data Classes
# ─────────────────────────────────────────────

@dataclass
class StrategyDecision:
    lap: int
    action: str          # "stay_out" | "pit_soft" | "pit_medium" | "pit_hard"
    compound_before: str
    tyre_age: int
    gap_ahead: float
    safety_car: bool

@dataclass
class StrategyEpisode:
    driver: str
    race: str
    source: str           # "real" | "rule_based" | "rl_agent"
    decisions: list[StrategyDecision] = field(default_factory=list)
    pit_laps: list[int]   = field(default_factory=list)
    compounds: list[str]  = field(default_factory=list)
    finish_position: Optional[int] = None
    total_time: Optional[float] = None
    total_reward: Optional[float] = None

    @property
    def n_stops(self) -> int:
        return len(self.pit_laps)

    @property
    def first_stop_lap(self) -> Optional[int]:
        return self.pit_laps[0] if self.pit_laps else None


# ─────────────────────────────────────────────
# Real Strategy Extractor
# ─────────────────────────────────────────────

def extract_real_strategies(race: str) -> dict[str, StrategyEpisode]:
    """Convert REAL_STRATEGIES_2023 into StrategyEpisode objects."""
    race_key = race.capitalize()
    strategies = REAL_STRATEGIES_2023.get(race_key, {})
    episodes = {}

    for driver, strat in strategies.items():
        ep = StrategyEpisode(
            driver=driver,
            race=race_key,
            source="real",
            pit_laps=strat["pit_laps"],
            compounds=strat["compound_sequence"],
            finish_position=strat["finish_position"],
        )
        episodes[driver] = ep
    return episodes


# ─────────────────────────────────────────────
# Rule-Based Strategy Simulator
# ─────────────────────────────────────────────

def simulate_rule_based(
    race: str,
    n_episodes: int = 50,
    seed: int = 42,
) -> list[StrategyEpisode]:
    """Run rule-based agent on race environment, collect strategy decisions."""
    race_lower = race.lower()
    agent = RuleBasedStrategy()
    episodes = []

    for ep_i in range(n_episodes):
        env = F1RaceEnv(race=race_lower if race_lower in ["monaco", "monza", "spa"] else "generic",
                        crew_name="elite", seed=seed + ep_i)
        obs, _ = env.reset(seed=seed + ep_i)

        episode = StrategyEpisode(
            driver=f"RuleAgent_{ep_i:03d}",
            race=race,
            source="rule_based",
        )

        ACTION_MAP = {0: "stay_out", 1: "pit_soft", 2: "pit_medium", 3: "pit_hard"}
        done = False
        total_reward = 0.0

        while not done:
            action, _ = agent.predict(obs)
            info_before = env._get_info()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if action > 0:  # pitted
                episode.pit_laps.append(info_before["lap"])
                compound_name = ["soft", "medium", "hard"][action - 1]
                episode.compounds.append(compound_name.upper())

            episode.decisions.append(StrategyDecision(
                lap=info_before["lap"],
                action=ACTION_MAP[action],
                compound_before=info_before["compound"],
                tyre_age=info_before["tyre_age"],
                gap_ahead=info_before["gap_ahead"],
                safety_car=False,
            ))

        episode.finish_position = info["position"]
        episode.total_time = info["total_time"]
        episode.total_reward = total_reward
        episodes.append(episode)

    return episodes


# ─────────────────────────────────────────────
# Comparison Metrics
# ─────────────────────────────────────────────

@dataclass
class ComparisonMetrics:
    race: str
    real_strategies: dict[str, StrategyEpisode]
    agent_episodes: list[StrategyEpisode]

    def pit_lap_mae(self) -> float:
        """Mean absolute error of first pit stop lap vs real."""
        real_laps = [ep.first_stop_lap for ep in self.real_strategies.values()
                     if ep.first_stop_lap is not None]
        agent_laps = [ep.first_stop_lap for ep in self.agent_episodes
                      if ep.first_stop_lap is not None]
        if not real_laps or not agent_laps:
            return float('nan')
        real_mean = np.mean(real_laps)
        return float(np.mean([abs(l - real_mean) for l in agent_laps]))

    def n_stops_match_rate(self) -> float:
        """% of agent episodes with same number of stops as most common real strategy."""
        real_n_stops = [ep.n_stops for ep in self.real_strategies.values()]
        most_common = max(set(real_n_stops), key=real_n_stops.count) if real_n_stops else 1
        agent_match = sum(1 for ep in self.agent_episodes if ep.n_stops == most_common)
        return agent_match / len(self.agent_episodes) if self.agent_episodes else 0.0

    def first_stop_lap_distribution(self) -> dict:
        """Distribution of first stop laps for real vs agent."""
        real_laps = sorted([ep.first_stop_lap for ep in self.real_strategies.values()
                            if ep.first_stop_lap])
        agent_laps = sorted([ep.first_stop_lap for ep in self.agent_episodes
                             if ep.first_stop_lap])
        return {
            "real_laps": real_laps,
            "real_mean": round(float(np.mean(real_laps)), 1) if real_laps else None,
            "agent_laps": agent_laps,
            "agent_mean": round(float(np.mean(agent_laps)), 1) if agent_laps else None,
            "agent_std": round(float(np.std(agent_laps)), 1) if agent_laps else None,
            "mae": round(self.pit_lap_mae(), 2),
        }

    def summary_table(self) -> pd.DataFrame:
        rows = []
        # Real strategies
        for driver, ep in self.real_strategies.items():
            rows.append({
                "source": f"Real — {driver}",
                "n_stops": ep.n_stops,
                "first_stop_lap": ep.first_stop_lap,
                "compounds": " → ".join(ep.compounds),
                "finish_pos": ep.finish_position,
                "total_reward": "—",
            })
        # Agent (aggregate)
        agent_laps = [ep.first_stop_lap for ep in self.agent_episodes if ep.first_stop_lap]
        agent_stops = [ep.n_stops for ep in self.agent_episodes]
        agent_rewards = [ep.total_reward for ep in self.agent_episodes if ep.total_reward]
        rows.append({
            "source": "Rule Agent (mean)",
            "n_stops": round(float(np.mean(agent_stops)), 1) if agent_stops else "—",
            "first_stop_lap": round(float(np.mean(agent_laps)), 1) if agent_laps else "—",
            "compounds": "Variable",
            "finish_pos": round(float(np.mean([ep.finish_position for ep in self.agent_episodes
                                               if ep.finish_position])), 1),
            "total_reward": round(float(np.mean(agent_rewards)), 2) if agent_rewards else "—",
        })
        return pd.DataFrame(rows)

    def report(self) -> str:
        dist = self.first_stop_lap_distribution()
        return (
            f"\n{'='*55}\n"
            f"  Strategy Comparison — {self.race} GP 2023\n"
            f"{'='*55}\n"
            f"Real drivers:\n"
            f"  First stop laps: {dist['real_laps']}\n"
            f"  Mean:            lap {dist['real_mean']}\n\n"
            f"Rule-based agent ({len(self.agent_episodes)} episodes):\n"
            f"  Mean first stop: lap {dist['agent_mean']} ± {dist['agent_std']}\n"
            f"  MAE vs real:     {dist['mae']} laps\n"
            f"  Stop-count match: {self.n_stops_match_rate()*100:.1f}%\n\n"
            f"{self.summary_table().to_string(index=False)}\n"
        )


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_strategy_comparison(
    race: str,
    real_strategies: dict,
    agent_episodes: list,
    save_path: str = None,
) -> None:
    metrics = ComparisonMetrics(race, real_strategies, agent_episodes)
    dist = metrics.first_stop_lap_distribution()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Strategy Comparison — {race} GP 2023\nRL Agent vs Real F1 Teams",
                 fontsize=13, fontweight='bold')

    # Panel 1: First stop lap distribution
    ax = axes[0]
    agent_laps = dist["agent_laps"]
    real_mean  = dist["real_mean"]
    ax.hist(agent_laps, bins=20, color='#3498DB', alpha=0.7, label='Rule Agent')
    for driver, ep in real_strategies.items():
        if ep.first_stop_lap:
            ax.axvline(ep.first_stop_lap, linewidth=2.5, linestyle='--',
                      label=f'{driver} (lap {ep.first_stop_lap})')
    ax.set_xlabel('İlk pit stop turu')
    ax.set_ylabel('Frekans')
    ax.set_title(f'İlk Pit Stop Turu Dağılımı\nMAE = {dist["mae"]} tur')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Strategy timeline
    ax2 = axes[1]
    y_pos = 0
    all_sources = list(real_strategies.items()) + [
        (f"Agent (ep{i+1})", ep) for i, ep in enumerate(agent_episodes[:5])
    ]
    compound_colors = {"SOFT":"#E74C3C", "MEDIUM":"#F39C12", "HARD":"#95A5A6",
                       "soft":"#E74C3C", "medium":"#F39C12", "hard":"#95A5A6"}
    max_laps = RACES_2023.get(race.capitalize(), {}).get("laps", 60)

    for label, ep in all_sources:
        stints = []
        prev_lap = 0
        source_color = "#2C3E50" if "Agent" not in label else "#7F8C8D"
        for i, pl in enumerate(ep.pit_laps or []):
            compound = (ep.compounds[i] if i < len(ep.compounds) else "MEDIUM")
            stint_len = pl - prev_lap
            stints.append((prev_lap, stint_len, compound))
            prev_lap = pl
        stints.append((prev_lap, max_laps - prev_lap,
                       ep.compounds[-1] if ep.compounds else "HARD"))

        for start, length, compound in stints:
            color = compound_colors.get(compound, "gray")
            ax2.barh(y_pos, length, left=start, height=0.6, color=color, alpha=0.85,
                    edgecolor='white', linewidth=0.5)
        ax2.text(-1, y_pos, label[:18], ha='right', va='center', fontsize=8)
        y_pos += 1

    ax2.set_xlabel('Tur')
    ax2.set_title('Strateji Zaman Çizelgesi')
    ax2.set_yticks([])
    ax2.set_xlim(-2, max_laps + 2)
    ax2.grid(axis='x', alpha=0.3)
    legend_patches = [mpatches.Patch(color=c, label=k)
                      for k, c in [("SOFT","#E74C3C"),("MEDIUM","#F39C12"),("HARD","#95A5A6")]]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=8)

    # Panel 3: Agent performance distribution
    ax3 = axes[2]
    positions = [ep.finish_position for ep in agent_episodes if ep.finish_position]
    rewards = [ep.total_reward for ep in agent_episodes if ep.total_reward]
    ax3_twin = ax3.twinx()
    ax3.hist(positions, bins=range(1, 22), color='#3498DB', alpha=0.7, label='Bitiş pozisyonu')
    ax3_twin.plot([], [], color='#E74C3C', linewidth=2, label='Reward')  # dummy for legend
    ax3.set_xlabel('Bitiş pozisyonu')
    ax3.set_ylabel('Frekans', color='#3498DB')
    ax3.set_title(f'Agent Performans Dağılımı\n({len(agent_episodes)} episode)')
    # Real positions
    for driver, ep in real_strategies.items():
        if ep.finish_position:
            ax3.axvline(ep.finish_position, linewidth=2.5, linestyle='--',
                       label=f'{driver}: P{ep.finish_position}')
    lines1, labels1 = ax3.get_legend_handles_labels()
    ax3.legend(lines1, labels1, fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# Main comparison runner
# ─────────────────────────────────────────────

def run_comparison(race: str = "Monaco", n_episodes: int = 100) -> ComparisonMetrics:
    print(f"\nRunning strategy comparison: {race} GP 2023")
    print(f"  Extracting real strategies...")
    real = extract_real_strategies(race)
    print(f"  Simulating rule-based agent ({n_episodes} episodes)...")
    agent = simulate_rule_based(race, n_episodes=n_episodes)
    metrics = ComparisonMetrics(race, real, agent)
    print(metrics.report())
    return metrics


if __name__ == "__main__":
    for race in ["Monaco", "Monza"]:
        metrics = run_comparison(race, n_episodes=50)
        plot_strategy_comparison(
            race,
            metrics.real_strategies,
            metrics.agent_episodes,
            save_path=f"{race.lower()}_strategy_comparison.png",
        )
