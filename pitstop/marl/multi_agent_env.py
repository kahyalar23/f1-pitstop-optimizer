"""
pitstop/marl/multi_agent_env.py

Multi-Agent Race Strategy Environment
=======================================
Her araç bağımsız bir ajan — hepsi aynı anda gözlemler, karar alır.
PettingZoo AEC (Agent-Environment-Cycle) API'sini taklit eder.

Neden MARL F1 için doğal seçim?
  • Undercut kararı rakibin davranışına bağlı → oyun teorisi
  • Nash dengesi: "herkes pit yaparsa kimse kazanmaz"
  • Dominant strateji dinamiği: SC döneminde tüm ajanlar aynı anda karar

Gözlem uzayı (her ajan için, 16 boyutlu):
  [0]  lap_norm
  [1]  own_position_norm
  [2]  own_tyre_age_norm
  [3]  own_compound_idx
  [4]  own_gap_ahead_norm
  [5]  own_gap_behind_norm
  [6]  safety_car
  [7]  own_tyre_temp_norm
  [8]  leader_gap_norm           ← YENİ (tek ajan env'de yoktu)
  [9]  nearest_rival_compound    ← YENİ
  [10] nearest_rival_tyre_age    ← YENİ
  [11] n_cars_same_compound_norm ← YENİ (kaç araç aynı compound?)
  [12] n_cars_pitted_this_lap    ← YENİ
  [13] tyre_on_edge              ← YENİ
  [14] race_progress_norm        ← YENİ
  [15] own_pace_offset_norm      ← YENİ (takım hız farkı)

Aksiyon uzayı: Discrete(4) — stay | pit_soft | pit_medium | pit_hard

Ödül yapısı:
  r_t = -Δlaptime - pit_cost + pos_gain + SC_bonus + cliff_penalty
  Ek olarak: relative_reward = own_reward - mean(other_rewards)
  Bu "zero-sum" bileşeni ajanları birbirini geçmeye yönlendirir.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from pitstop.race_engine.track_state import (
    RaceEngine, RaceState, CarState, DRIVER_GRID_2023,
    DEG_RATES, PIT_LOSS_BY_CIRCUIT
)
from pitstop.simulation.monte_carlo import simulate_one, CREW_PROFILES
from pitstop.simulation.tire_model import simulate_tyre_lap, COMPOUNDS as TYRE_COMPOUNDS


# ─────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────

ACTION_MAP = {0: "stay", 1: "pit_soft", 2: "pit_medium", 3: "pit_hard"}
COMPOUND_MAP = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTER": 1, "WET": 0}
N_AGENTS = 20
OBS_DIM = 16

POSITION_POINTS = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}


# ─────────────────────────────────────────────────────────────
# Multi-Agent Environment
# ─────────────────────────────────────────────────────────────

class F1MultiAgentEnv:
    """
    20-car multi-agent F1 race environment.

    PettingZoo-compatible API (basitleştirilmiş):
        env.reset() -> dict[agent_id, obs]
        env.step(actions: dict[agent_id, action]) -> (obs, rewards, dones, infos)

    Gymnasium yoksa temel Python ile çalışır.
    """

    def __init__(
        self,
        circuit: str = "generic",
        total_laps: int = 60,
        base_laptime: float = 90.0,
        n_agents: int = N_AGENTS,
        cooperative_weight: float = 0.0,   # 0=tam rekabetçi, 1=tam kooperatif
        seed: int = 42,
    ):
        self.circuit = circuit
        self.total_laps = total_laps
        self.base_laptime = base_laptime
        self.n_agents = min(n_agents, len(DRIVER_GRID_2023))
        self.cooperative_weight = cooperative_weight
        self.seed = seed

        self.agents = [DRIVER_GRID_2023[i][0] for i in range(self.n_agents)]
        self.rng = np.random.default_rng(seed)

        self.engine: Optional[RaceEngine] = None
        self.state: Optional[RaceState] = None
        self._step_count = 0

        # Gymnasium spaces (opsiyonel)
        if GYM_AVAILABLE:
            self.observation_spaces = {
                agent: spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(OBS_DIM,), dtype=np.float32
                )
                for agent in self.agents
            }
            self.action_spaces = {
                agent: spaces.Discrete(4)
                for agent in self.agents
            }

    # ── API ──────────────────────────────────────────────────────────

    def reset(self, seed: int = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed

        self.engine = RaceEngine(
            circuit=self.circuit,
            total_laps=self.total_laps,
            base_laptime=self.base_laptime,
            seed=self.seed,
        )
        self.state = self.engine.state
        self._step_count = 0
        self._pitted_this_lap: dict[str, bool] = {a: False for a in self.agents}

        return {agent: self._obs(agent) for agent in self.agents}

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict, dict, dict, dict, dict]:
        """
        actions: {driver: action_int}
        Returns: (observations, rewards, terminateds, truncateds, infos)
        """
        assert self.state is not None, "Call reset() first"

        # Pit kararlarını derle
        pit_decisions = {}
        for agent, action in actions.items():
            if action > 0 and not self.state.cars[agent].dnf:
                pit_decisions[agent] = COMPOUND_MAP[action - 1]

        self._pitted_this_lap = {a: a in pit_decisions for a in self.agents}

        # Motoru bir tur ilerlet
        prev_positions = {a: self.state.cars[a].position
                         for a in self.agents if not self.state.cars[a].dnf}
        prev_times = {a: self.state.cars[a].total_race_time for a in self.agents}

        self.state = self.engine.step(pit_decisions)
        self._step_count += 1

        terminated = self._step_count >= self.total_laps
        obs   = {a: self._obs(a) for a in self.agents}
        rews  = self._compute_rewards(prev_positions, prev_times, pit_decisions)
        terms = {a: terminated for a in self.agents}
        truns = {a: False for a in self.agents}
        infos = {a: self._info(a) for a in self.agents}

        if terminated:
            for a in self.agents:
                rews[a] += POSITION_POINTS.get(self.state.cars[a].position, 0) * 0.5

        return obs, rews, terms, truns, infos

    # ── Gözlem ───────────────────────────────────────────────────────

    def _obs(self, agent: str) -> np.ndarray:
        car = self.state.cars[agent]
        state = self.state
        n = self.total_laps

        # En yakın rakip (önündeki araç)
        sorted_cars = state.sorted_cars
        my_idx = next((i for i, c in enumerate(sorted_cars) if c.driver == agent), 0)
        rival_ahead = sorted_cars[my_idx - 1] if my_idx > 0 else car
        rival_behind = sorted_cars[my_idx + 1] if my_idx < len(sorted_cars)-1 else car

        # Aynı compound'daki araç sayısı
        same_compound = sum(
            1 for c in state.cars.values()
            if c.tyre.compound == car.tyre.compound and not c.dnf
        )

        # Bu tur pit yapan araç sayısı
        n_pitted = sum(1 for p in self._pitted_this_lap.values() if p)

        obs = np.array([
            state.current_lap / n,                                          # [0]
            1.0 - car.position / self.n_agents,                            # [1]
            min(car.tyre.age / 50, 1.0),                                   # [2]
            COMPOUND_IDX.get(car.tyre.compound, 1) / 3,                    # [3]
            np.clip(car.gap_ahead / 30, -1, 1),                            # [4]
            np.clip(car.gap_behind / 30, -1, 1),                           # [5]
            float(state.safety_car),                                        # [6]
            np.clip((car.tyre.temperature - 70) / 50, -1, 1),              # [7]
            np.clip(car.gap_to_leader / 60, 0, 1),                         # [8] YENİ
            COMPOUND_IDX.get(rival_ahead.tyre.compound, 1) / 3,            # [9] YENİ
            min(rival_ahead.tyre.age / 50, 1.0),                          # [10] YENİ
            same_compound / self.n_agents,                                  # [11] YENİ
            n_pitted / self.n_agents,                                       # [12] YENİ
            float(car.tyre.laps_on_edge),                                  # [13] YENİ
            state.current_lap / n,                                          # [14] YENİ
            np.clip(car.base_pace_offset / 2.0, -1, 1),                   # [15] YENİ
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── Ödül ─────────────────────────────────────────────────────────

    def _compute_rewards(
        self,
        prev_positions: dict,
        prev_times: dict,
        pit_decisions: dict,
    ) -> dict[str, float]:
        rewards = {}
        all_rewards_raw = {}

        for agent in self.agents:
            car = self.state.cars[agent]
            if car.dnf:
                all_rewards_raw[agent] = -5.0
                continue

            prev_pos = prev_positions.get(agent, car.position)
            pos_change = prev_pos - car.position   # pozitif = kazanım

            # Tur süresi maliyeti
            dt = car.total_race_time - prev_times.get(agent, car.total_race_time)
            r_laptime = -min(dt / self.base_laptime, 0.5)

            # Pit maliyeti
            r_pit = -0.8 if agent in pit_decisions else 0.0

            # SC bonus — SC'de pit yapmak çok değerli
            r_sc = 1.5 if (agent in pit_decisions and self.state.safety_car) else 0.0

            # Pozisyon kazanımı
            r_pos = pos_change * 0.4

            # Cliff cezası
            r_cliff = -1.5 if car.tyre.laps_on_edge else 0.0

            all_rewards_raw[agent] = r_laptime + r_pit + r_sc + r_pos + r_cliff

        # Relative reward bileşeni (zero-sum kısmı)
        mean_raw = np.mean(list(all_rewards_raw.values()))
        for agent in self.agents:
            raw = all_rewards_raw.get(agent, 0.0)
            relative = raw - mean_raw
            # Mix: (1-α)*absolute + α*relative
            rewards[agent] = (
                (1 - self.cooperative_weight) * raw +
                self.cooperative_weight * relative
            )

        return rewards

    # ── Info ─────────────────────────────────────────────────────────

    def _info(self, agent: str) -> dict:
        car = self.state.cars[agent]
        return {
            "lap": self.state.current_lap,
            "position": car.position,
            "tyre_age": car.tyre.age,
            "compound": car.tyre.compound,
            "gap_ahead": round(car.gap_ahead, 3),
            "n_stops": car.n_stops,
            "dnf": car.dnf,
        }

    # ── Yardımcı ─────────────────────────────────────────────────────

    def render_grid(self) -> str:
        """ASCII pist tabelası."""
        lines = [
            f"{'─'*55}",
            f"  Lap {self.state.current_lap}/{self.total_laps} | "
            f"{'SC 🟡' if self.state.safety_car else 'GREEN 🟢'}",
            f"{'─'*55}",
            f"  {'Pos':>3} {'Driver':<6} {'Team':<14} {'Gap':>8} {'Comp':>6} {'Age':>4} {'S':>2}",
            f"{'─'*55}",
        ]
        for car in self.state.sorted_cars[:10]:
            gap = f"+{car.gap_to_leader:.2f}" if car.gap_to_leader > 0 else "LEAD"
            edge = "⚠" if car.tyre.laps_on_edge else " "
            lines.append(
                f"  {car.position:>3} {car.driver:<6} {car.team[:13]:<14} "
                f"{gap:>8} {car.tyre.compound:>6} {car.tyre.age:>4} "
                f"{car.n_stops:>2}{edge}"
            )
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Basit MARL eğitim döngüsü (Independent PPO)
# ─────────────────────────────────────────────────────────────

class IndependentAgents:
    """
    Her araç için bağımsız kural-tabanlı politika.
    Gerçek PPO eğitimine başlamadan önce baseline olarak kullanılır.

    Politika:
      - Lastik kritik yaşta ise → pit
      - SC varsa → pit
      - Yakın rakip önde ve o lastik yaşlıysa → undercut dene
      - Değilse → stay
    """

    def __init__(self, drivers: list[str]):
        self.drivers = drivers

    def act(
        self,
        observations: dict[str, np.ndarray],
        state: RaceState,
    ) -> dict[str, int]:
        actions = {}
        for driver in self.drivers:
            obs = observations[driver]
            car = state.cars.get(driver)
            if car is None or car.dnf:
                actions[driver] = 0
                continue

            tyre_age_norm = float(obs[2])
            sc = bool(obs[6])
            on_edge = bool(obs[13])
            rival_compound_idx = int(obs[9] * 3)
            rival_age_norm = float(obs[10])
            laps_remaining_norm = 1.0 - float(obs[14])

            # SC → hemen pit (medium tercih)
            if sc and car.n_stops < 2:
                actions[driver] = 2  # medium

            # Kritik lastik
            elif on_edge:
                laps_rem = int(laps_remaining_norm * 60)
                if laps_rem > 25:
                    actions[driver] = 3   # hard
                elif laps_rem > 12:
                    actions[driver] = 2   # medium
                else:
                    actions[driver] = 1   # soft

            # Undercut fırsatı: rakip yaşlı lastik, biz daha gençiz
            elif (rival_age_norm > tyre_age_norm + 0.15 and
                  rival_compound_idx >= 1 and
                  float(obs[4]) < 0.1):  # gap_ahead < 3s
                actions[driver] = 2  # medium — undercut

            # Normal devam
            else:
                actions[driver] = 0  # stay out

        return actions


def run_demo(n_laps: int = 20, seed: int = 42) -> dict:
    """Kısa bir multi-agent demo çalıştır, sonuçları döndür."""
    env = F1MultiAgentEnv(circuit="generic", total_laps=n_laps, seed=seed)
    obs = env.reset(seed=seed)
    agent_policy = IndependentAgents(env.agents)

    total_rewards = {a: 0.0 for a in env.agents}
    done = False

    while not done:
        actions = agent_policy.act(obs, env.state)
        obs, rewards, terms, truns, infos = env.step(actions)
        for a in env.agents:
            total_rewards[a] += rewards[a]
        done = all(terms.values())

    final_positions = {a: env.state.cars[a].position for a in env.agents}
    return {
        "final_positions": final_positions,
        "total_rewards": total_rewards,
        "winner": min(final_positions, key=final_positions.get),
        "lap_history_len": len(env.state.lap_history),
    }


if __name__ == "__main__":
    print("=== Multi-Agent F1 Environment Demo ===\n")
    result = run_demo(n_laps=30)
    print(f"Winner: {result['winner']}")
    print("\nFinal standings:")
    for driver, pos in sorted(result["final_positions"].items(), key=lambda x: x[1]):
        r = result["total_rewards"][driver]
        print(f"  P{pos:>2} {driver}  (reward: {r:+.2f})")
