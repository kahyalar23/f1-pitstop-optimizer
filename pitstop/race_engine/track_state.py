"""
pitstop/race_engine/track_state.py

Full Race State Engine — 20 Cars on Track
==========================================
Her turda tüm araçların tam durumunu takip eder:
  • Pist pozisyonu (gap to leader saniye cinsinden)
  • Lastik durumu (compound, yaş, sıcaklık, degradasyon)
  • Pit stop geçmişi
  • Safety car / Virtual Safety Car durumu

Bu engine Yol 1'in kalbi:
  "Ben şu an pit yaparsam kaçıncı çıkarım?" sorusunu
  tüm 20 aracın konumunu bilerek cevaplar.

Veri yapısı seçimi:
  Her araç için CarState dataclass → RaceState içinde dict
  Her turda snapshot alınır → history list
  Bu sayede "5 tur öncesinde ne düşünseydik?" analizi yapılabilir
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

# ─────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTER", "WET"]

# Degradasyon katsayıları (s/tur, lastik yaşı^0.7 ile çarpılır)
DEG_RATES = {"SOFT": 0.078, "MEDIUM": 0.047, "HARD": 0.028,
             "INTER": 0.130, "WET": 0.180}

# Her compound için pit stop sonrası avantaj penceresi (tur)
FRESH_ADVANTAGE_WINDOW = {"SOFT": 8, "MEDIUM": 12, "HARD": 18}

# Pit entry/exit kayıpları (saniye) — pist bazında
PIT_LOSS_BY_CIRCUIT = {
    "monaco":     {"entry": 18.5, "exit": 16.2},
    "monza":      {"entry": 19.8, "exit": 16.5},
    "spa":        {"entry": 21.0, "exit": 17.8},
    "silverstone":{"entry": 20.2, "exit": 17.1},
    "generic":    {"entry": 20.0, "exit": 17.0},
}

# 2023 F1 grid — başlangıç için referans takım hızları
TEAM_PACE_2023 = {
    "Red Bull":      0.00,   # referans (en hızlı)
    "Mercedes":      0.35,
    "Ferrari":       0.28,
    "Aston Martin":  0.62,
    "McLaren":       0.71,
    "Alpine":        0.95,
    "Williams":      1.18,
    "AlphaTauri":    1.05,
    "Alfa Romeo":    1.12,
    "Haas":          1.24,
}

DRIVER_GRID_2023 = [
    ("VER", "Red Bull",     1),  ("PER", "Red Bull",     2),
    ("HAM", "Mercedes",     3),  ("RUS", "Mercedes",     4),
    ("LEC", "Ferrari",      5),  ("SAI", "Ferrari",      6),
    ("ALO", "Aston Martin", 7),  ("STR", "Aston Martin", 8),
    ("NOR", "McLaren",      9),  ("PIA", "McLaren",     10),
    ("GAS", "Alpine",      11),  ("OCO", "Alpine",      12),
    ("TSU", "AlphaTauri",  13),  ("DEV", "AlphaTauri",  14),
    ("ALB", "Williams",    15),  ("SAR", "Williams",    16),
    ("BOT", "Alfa Romeo",  17),  ("ZHO", "Alfa Romeo",  18),
    ("MAG", "Haas",        19),  ("HUL", "Haas",        20),
]


# ─────────────────────────────────────────────────────────────
# Veri yapıları
# ─────────────────────────────────────────────────────────────

@dataclass
class TyreState:
    compound: str = "MEDIUM"
    age: int = 0                    # tur
    temperature: float = 85.0      # °C
    cumulative_energy: float = 0.0
    is_new: bool = True

    @property
    def deg_penalty(self) -> float:
        """Mevcut turda bu lastikten kaynaklanan süre kaybı (s)."""
        rate = DEG_RATES.get(self.compound, 0.047)
        cliff = 1.0
        if self.compound == "SOFT" and self.age > 18:
            cliff = 2.2
        elif self.compound == "MEDIUM" and self.age > 30:
            cliff = 1.6
        return rate * (self.age ** 0.7) * cliff

    @property
    def laps_on_edge(self) -> bool:
        """Lastik kritik yaşa yakın mı?"""
        limits = {"SOFT": 20, "MEDIUM": 32, "HARD": 48}
        return self.age >= limits.get(self.compound, 35) * 0.85


@dataclass
class CarState:
    driver: str
    team: str
    grid_position: int

    # Pist konumu
    position: int = 0               # anlık sıralama (1=lider)
    gap_to_leader: float = 0.0      # saniye
    gap_ahead: float = 999.0        # öndeki araca gap
    gap_behind: float = 999.0       # arkadaki araca gap

    # Lastik
    tyre: TyreState = field(default_factory=TyreState)

    # Yarış durumu
    lap: int = 0
    total_race_time: float = 0.0
    pit_stops: list[dict] = field(default_factory=list)
    is_in_pits: bool = False
    dnf: bool = False
    lapped: bool = False

    # Pace karakteristiği
    base_pace_offset: float = 0.0   # takım bazı hız farkı (s)
    driver_skill: float = 1.0       # [0.9, 1.1] — 1.0 = ortalama

    @property
    def n_stops(self) -> int:
        return len(self.pit_stops)

    @property
    def last_pit_lap(self) -> Optional[int]:
        return self.pit_stops[-1]["lap"] if self.pit_stops else None

    def laptime(self, base_laptime: float, sc_active: bool = False) -> float:
        """
        Mevcut tur süresi tahmini.
        base_laptime: pistin referans tur süresi
        """
        if sc_active:
            return base_laptime * 1.35 + np.random.normal(0, 0.05)
        pace = base_laptime + self.base_pace_offset + self.tyre.deg_penalty
        noise = np.random.normal(0, 0.12)
        return max(pace * 0.98, pace + noise) * (2.0 - self.driver_skill)

    def pit(self, new_compound: str, stationary_time: float, lap: int) -> None:
        """Pit stop gerçekleştir."""
        self.pit_stops.append({
            "lap": lap,
            "old_compound": self.tyre.compound,
            "old_age": self.tyre.age,
            "new_compound": new_compound,
            "stationary_time": stationary_time,
        })
        self.tyre = TyreState(compound=new_compound, age=0, is_new=True,
                              temperature=new_compound == "SOFT" and 75.0 or 80.0)
        self.is_in_pits = False


@dataclass
class RaceState:
    """
    Tüm yarışın anlık snapshot'ı.
    Her turda güncellenir, history'e kaydedilir.
    """
    circuit: str
    total_laps: int
    current_lap: int = 0
    base_laptime: float = 90.0

    # Tüm araçlar
    cars: dict[str, CarState] = field(default_factory=dict)

    # Yarış koşulları
    safety_car: bool = False
    vsc: bool = False               # Virtual Safety Car
    drs_enabled: bool = True
    weather: str = "dry"            # "dry" | "damp" | "wet"

    # Geçmiş
    lap_history: list[dict] = field(default_factory=list)

    @property
    def pit_loss(self) -> dict:
        cfg = PIT_LOSS_BY_CIRCUIT.get(self.circuit.lower(),
                                      PIT_LOSS_BY_CIRCUIT["generic"])
        return cfg

    @property
    def sorted_cars(self) -> list[CarState]:
        """Pozisyona göre sıralanmış araç listesi."""
        return sorted(
            [c for c in self.cars.values() if not c.dnf],
            key=lambda c: c.total_race_time
        )

    def update_positions(self) -> None:
        """Toplam süreye göre pozisyonları yeniden hesapla."""
        sorted_cars = self.sorted_cars
        leader_time = sorted_cars[0].total_race_time if sorted_cars else 0

        for i, car in enumerate(sorted_cars):
            car.position = i + 1
            car.gap_to_leader = car.total_race_time - leader_time
            if i > 0:
                car.gap_ahead = car.total_race_time - sorted_cars[i-1].total_race_time
            else:
                car.gap_ahead = 999.0
            if i < len(sorted_cars) - 1:
                car.gap_behind = sorted_cars[i+1].total_race_time - car.total_race_time
            else:
                car.gap_behind = 999.0

    def snapshot(self) -> dict:
        """Mevcut tur için tam snapshot döndür."""
        return {
            "lap": self.current_lap,
            "sc": self.safety_car,
            "vsc": self.vsc,
            "cars": {
                drv: {
                    "pos": c.position,
                    "gap_leader": round(c.gap_to_leader, 3),
                    "gap_ahead": round(c.gap_ahead, 3),
                    "gap_behind": round(c.gap_behind, 3),
                    "compound": c.tyre.compound,
                    "tyre_age": c.tyre.age,
                    "n_stops": c.n_stops,
                    "total_time": round(c.total_race_time, 3),
                }
                for drv, c in self.cars.items()
            }
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Mevcut durumu DataFrame olarak döndür."""
        rows = []
        for car in self.sorted_cars:
            rows.append({
                "Pos": car.position,
                "Driver": car.driver,
                "Team": car.team,
                "Gap": f"+{car.gap_to_leader:.3f}s" if car.gap_to_leader > 0 else "LEADER",
                "Gap Ahead": f"{car.gap_ahead:.2f}s" if car.gap_ahead < 100 else "—",
                "Compound": car.tyre.compound,
                "Tyre Age": car.tyre.age,
                "Stops": car.n_stops,
                "On Edge": "⚠️" if car.tyre.laps_on_edge else "",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Race Engine
# ─────────────────────────────────────────────────────────────

class RaceEngine:
    """
    Deterministic + stochastic race simulator.
    20 araçı tur tur simüle eder, her tur tam state snapshot alır.
    """

    def __init__(
        self,
        circuit: str = "generic",
        total_laps: int = 60,
        base_laptime: float = 90.0,
        seed: int = 42,
    ):
        self.circuit = circuit
        self.total_laps = total_laps
        self.base_laptime = base_laptime
        self.rng = np.random.default_rng(seed)

        self.state = self._init_race()

    def _init_race(self) -> RaceState:
        state = RaceState(
            circuit=self.circuit,
            total_laps=self.total_laps,
            base_laptime=self.base_laptime,
        )
        # Tüm araçları başlat
        for driver, team, grid in DRIVER_GRID_2023:
            start_gap = (grid - 1) * 0.6 + self.rng.uniform(-0.1, 0.1)
            skill = self.rng.uniform(0.97, 1.03)
            # İlk tur compound seçimi
            start_compound = self.rng.choice(
                ["SOFT", "SOFT", "MEDIUM"], p=[0.5, 0.2, 0.3]
            )
            car = CarState(
                driver=driver,
                team=team,
                grid_position=grid,
                position=grid,
                gap_to_leader=start_gap,
                total_race_time=start_gap,
                base_pace_offset=TEAM_PACE_2023.get(team, 1.0),
                driver_skill=skill,
                tyre=TyreState(compound=start_compound, age=0),
            )
            state.cars[driver] = car
        state.update_positions()
        return state

    def step(self, pit_decisions: dict[str, str] | None = None) -> RaceState:
        """
        Bir turu simüle et.

        pit_decisions: {driver: new_compound} — bu turda pit yapacak araçlar
                       None veya eksik driver = pit yok

        Returns: güncellenmiş RaceState
        """
        pit_decisions = pit_decisions or {}
        self.state.current_lap += 1
        lap = self.state.current_lap

        # Safety car olasılığı (%4/tur)
        self.state.safety_car = self.rng.random() < 0.04
        self.state.vsc = (not self.state.safety_car) and self.rng.random() < 0.03

        # Monte Carlo pit stop süresi
        from pitstop.simulation.monte_carlo import simulate_one, CREW_PROFILES
        elite_crew = CREW_PROFILES["elite"]

        for driver, car in self.state.cars.items():
            if car.dnf:
                continue

            # DNF olasılığı (%0.3/tur)
            if self.rng.random() < 0.003:
                car.dnf = True
                continue

            # Pit stop gerçekleştirme
            if driver in pit_decisions:
                new_compound = pit_decisions[driver]
                # Gerçekçi pit stop süresi
                pit_result = simulate_one(elite_crew, rng=self.rng)
                stationary = pit_result.pit_time
                pit_cfg = self.state.pit_loss
                pit_total = pit_cfg["entry"] + stationary + pit_cfg["exit"]
                car.pit(new_compound, stationary, lap)
                car.total_race_time += pit_total
                car.tyre.age += 1
            else:
                # Normal tur
                lt = car.laptime(self.base_laptime, self.state.safety_car)
                car.total_race_time += lt
                car.tyre.age += 1
                car.tyre.temperature = min(
                    car.tyre.temperature + self.rng.normal(0.3, 0.1), 115
                )

            car.lap = lap

        self.state.update_positions()
        self.state.lap_history.append(self.state.snapshot())
        return self.state

    def simulate_full_race(
        self,
        strategy_fn=None,
    ) -> list[dict]:
        """
        Tüm yarışı simüle et.
        strategy_fn(state) -> {driver: new_compound} pit kararları döndürür.
        """
        self.state = self._init_race()
        for _ in range(self.total_laps):
            pit_decisions = strategy_fn(self.state) if strategy_fn else {}
            self.step(pit_decisions)
        return self.state.lap_history

    def get_position_history(self) -> pd.DataFrame:
        """Tüm araçlar için tur-pozisyon matrisi."""
        rows = []
        for snap in self.state.lap_history:
            for driver, data in snap["cars"].items():
                rows.append({
                    "lap": snap["lap"],
                    "driver": driver,
                    "position": data["pos"],
                    "gap_leader": data["gap_leader"],
                    "compound": data["compound"],
                    "tyre_age": data["tyre_age"],
                })
        return pd.DataFrame(rows)
