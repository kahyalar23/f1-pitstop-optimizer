"""
pitstop/race_engine/strategy_optimizer.py

Real-Time Strategy Optimizer
==============================
Tüm 20 aracın anlık konumunu görür ve her araç için:
  "Şu an pit yapsam, pit sonrası kaçıncı çıkarım?"
  "Hangi tur pit yapmak en fazla pozisyon kazandırır?"
  "Undercut/overcut penceresi açık mı?"

Çözüm yöntemi: Exhaustive Look-Ahead + Scoring
  • Her araç için [şimdiki tur, şimdiki+8 tur] aralığını tara
  • Her olası pit turu için monte carlo ile pit sonrası pozisyon simüle et
  • En yüksek skoru veren turu öner

Matematiksel temel:
  PitScore(t_pit) = Σᵢ [P_emerge_ahead(i)] - PitLossCost(t_pit)

  P_emerge_ahead(i) = P(gap_i > pit_loss + deg_advantage * Δage_i)

Bu yaklaşım Mercedes ve Red Bull'un gerçekte kullandığı
"Decision Tree with Probabilistic Outcomes" metodolojisine benzer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from pitstop.race_engine.track_state import (
    RaceState, CarState, TyreState, DEG_RATES,
    FRESH_ADVANTAGE_WINDOW, PIT_LOSS_BY_CIRCUIT
)
from pitstop.simulation.monte_carlo import (
    run_monte_carlo, MonteCarloConfig
)


# ─────────────────────────────────────────────────────────────
# Strateji Kararı
# ─────────────────────────────────────────────────────────────

@dataclass
class PitRecommendation:
    driver: str
    current_lap: int
    current_position: int
    current_compound: str
    current_tyre_age: int

    # Öneri
    action: str                  # "PIT_NOW" | "PIT_LAP_N" | "STAY_OUT" | "OVERCUT"
    recommended_lap: int
    recommended_compound: str
    urgency: str                 # "CRITICAL" | "OPTIMAL" | "EARLY" | "NONE"

    # Beklenen çıktı
    expected_position_after: float
    position_delta: float        # pozitif = kazanım
    pit_loss_s: float
    net_gain_s: float

    # Pist trafiği analizi
    cars_ahead_in_window: list[str]   # pit sonrası önümde kalacak araçlar
    cars_to_undercut: list[str]       # undercut edebileceğimiz araçlar
    threat_from_behind: list[str]     # arkadan undercut tehdidi

    # Güven skoru [0, 1]
    confidence: float
    reasoning: str


@dataclass
class RaceStrategyReport:
    """Bir tur için tüm araçların strateji analizi."""
    lap: int
    circuit: str
    sc_active: bool

    recommendations: dict[str, PitRecommendation]

    @property
    def pit_now_drivers(self) -> list[str]:
        return [d for d, r in self.recommendations.items()
                if r.action == "PIT_NOW"]

    @property
    def critical_drivers(self) -> list[str]:
        return [d for d, r in self.recommendations.items()
                if r.urgency == "CRITICAL"]

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for driver, rec in sorted(
            self.recommendations.items(),
            key=lambda x: x[1].current_position
        ):
            rows.append({
                "Driver": driver,
                "Pos": rec.current_position,
                "Compound": f"{rec.current_compound} ({rec.current_tyre_age}L)",
                "Action": rec.action,
                "Rec. Lap": rec.recommended_lap,
                "Rec. Compound": rec.recommended_compound,
                "Exp. Pos.": f"P{rec.expected_position_after:.0f}",
                "Pos Δ": f"{rec.position_delta:+.1f}",
                "Net Gain": f"{rec.net_gain_s:+.2f}s",
                "Urgency": rec.urgency,
                "Confidence": f"{rec.confidence*100:.0f}%",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Core Optimizer
# ─────────────────────────────────────────────────────────────

class StrategyOptimizer:
    """
    20-car track state görerek her araç için strateji önerisi üretir.

    Algoritma:
    1. Her araç için mevcut lastik durumunu değerlendir
    2. Olası pit pencerelerini tara (şu an + 8 tur)
    3. Her pencere için "pit sonrası pozisyon" simüle et
       → Pit öncesi tüm araçların konumunu bil
       → Pit sonrası hangi araçlar pit kazandı/kaybetti?
    4. En iyi pencereyi seç, öneri üret
    """

    def __init__(
        self,
        circuit: str = "generic",
        n_mc_samples: int = 500,    # pit süre tahmininde monte carlo örnekleri
        lookahead_laps: int = 8,    # kaç tur ileri bak
        seed: int = 42,
    ):
        self.circuit = circuit
        self.n_mc_samples = n_mc_samples
        self.lookahead_laps = lookahead_laps
        self.rng = np.random.default_rng(seed)

        # Monte carlo pit süresi dağılımını önceden hesapla
        cfg = MonteCarloConfig(n_iterations=n_mc_samples, crew_name="elite", seed=seed)
        mc = run_monte_carlo(cfg)
        self.pit_time_dist = mc.pit_times  # numpy array — hızlı sampling için

        pit_cfg = PIT_LOSS_BY_CIRCUIT.get(circuit.lower(), PIT_LOSS_BY_CIRCUIT["generic"])
        self.pit_entry = pit_cfg["entry"]
        self.pit_exit  = pit_cfg["exit"]

    def _sample_pit_loss(self) -> float:
        """Monte Carlo'dan pit süre örnekle."""
        return float(self.rng.choice(self.pit_time_dist)) + self.pit_entry + self.pit_exit

    def _deg_advantage(
        self,
        current_compound: str,
        current_age: int,
        new_compound: str,
        n_laps_fresh: int = 5,
    ) -> float:
        """
        Yeni lastiğin n_laps_fresh tur boyunca kümülatif avantajı (saniye).
        Kıyaslama: mevcut lastik (current_compound, current_age)
                   vs yeni lastik (new_compound, age=0)
        """
        adv = 0.0
        for l in range(1, n_laps_fresh + 1):
            old_deg = DEG_RATES[current_compound] * (current_age + l) ** 0.7
            new_deg = DEG_RATES[new_compound] * l ** 0.7
            adv += (old_deg - new_deg)
        return adv

    def _estimate_position_after_pit(
        self,
        target: CarState,
        all_cars: list[CarState],
        pit_lap: int,
        current_lap: int,
        new_compound: str,
    ) -> tuple[float, list[str], list[str], list[str]]:
        """
        target araç pit_lap turunda pit yaparsa beklenen pozisyon tahmini.

        Yaklaşım:
          • Pit öncesi: target ile aynı bölgedeki araçların
            pit_lap - current_lap tur sonraki konumunu tahmin et
          • Pit süresi: monte carlo sample (değil tek değer, stokastik)
          • Pit sonrası: tüm araçlara göre pozisyon hesapla
        """
        laps_until_pit = max(0, pit_lap - current_lap)

        # Pit anındaki toplam süre (sabit tempo varsayımı — basit)
        base_lt = target.total_race_time / max(current_lap, 1)

        # Target'ın pit anındaki tahmini süresi
        target_at_pit = target.total_race_time
        for l in range(laps_until_pit):
            deg = DEG_RATES[target.tyre.compound] * (target.tyre.age + l) ** 0.7
            target_at_pit += base_lt + deg * 0.3

        # Pit süresi (stokastik — 200 örnek ortalaması)
        pit_loss_samples = [self._sample_pit_loss() for _ in range(50)]
        pit_loss_mean = float(np.mean(pit_loss_samples))
        target_after_pit = target_at_pit + pit_loss_mean

        # Fresh lastik avantajı (ilk 5 tur)
        fresh_adv = self._deg_advantage(
            target.tyre.compound, target.tyre.age,
            new_compound, n_laps_fresh=5
        )
        target_after_pit -= fresh_adv

        # Her rakibe göre pozisyon hesapla
        ahead_after = []
        to_undercut = []
        threat_behind = []

        for car in all_cars:
            if car.driver == target.driver or car.dnf:
                continue

            # Rakibin pit anındaki tahmini süresi
            rival_base = car.total_race_time / max(current_lap, 1)
            rival_at_pit = car.total_race_time
            for l in range(laps_until_pit):
                rival_deg = DEG_RATES[car.tyre.compound] * (car.tyre.age + l) ** 0.7
                rival_at_pit += rival_base + rival_deg * 0.3

            # Pit yapmayanlar → stayout
            if rival_at_pit < target_after_pit:
                ahead_after.append(car.driver)
            else:
                # Biz onların önünde çıkabiliriz
                if car.position < target.position:
                    to_undercut.append(car.driver)

            # Arkadan undercut tehdidi
            if (car.position > target.position and
                car.tyre.laps_on_edge and
                car.gap_behind < pit_loss_mean + 2.0):
                threat_behind.append(car.driver)

        expected_pos = len(ahead_after) + 1
        return expected_pos, ahead_after, to_undercut, threat_behind

    def _best_compound_choice(
        self,
        car: CarState,
        laps_remaining: int,
        n_stops_done: int,
    ) -> str:
        """
        Kalan tura ve stop sayısına göre optimal compound seç.
        Basit kural: lastik yarış bitirilecek kadar dayanmalı.
        """
        if laps_remaining > 35:
            return "HARD"
        elif laps_remaining > 18:
            return "MEDIUM"
        else:
            return "SOFT"

    def analyze_car(
        self,
        target_driver: str,
        state: RaceState,
    ) -> PitRecommendation:
        """
        Tek bir araç için tam strateji analizi.
        """
        car = state.cars[target_driver]
        all_cars = list(state.cars.values())
        current_lap = state.current_lap
        total_laps = state.total_laps
        laps_remaining = total_laps - current_lap

        # Aciliyet değerlendirmesi
        tyre = car.tyre
        urgency = "NONE"
        if tyre.laps_on_edge:
            urgency = "CRITICAL"
        elif tyre.age > {
            "SOFT": 14, "MEDIUM": 24, "HARD": 38
        }.get(tyre.compound, 25) * 0.8:
            urgency = "OPTIMAL"

        # Optimal compound seçimi
        best_compound = self._best_compound_choice(car, laps_remaining, car.n_stops)

        # Pit pencerelerini tara: şimdiden lookahead_laps sonrasına
        best_score = float("-inf")
        best_lap = current_lap
        best_pos = car.position
        best_undercuts = []
        best_ahead = []
        best_threats = []

        for pit_lap in range(current_lap, min(current_lap + self.lookahead_laps + 1,
                                              total_laps - 3)):
            exp_pos, ahead, undercuts, threats = self._estimate_position_after_pit(
                car, all_cars, pit_lap, current_lap, best_compound
            )

            # Skor = pozisyon kazanımı - beklemek için ödenen deg maliyeti
            laps_waited = pit_lap - current_lap
            deg_cost = DEG_RATES[tyre.compound] * (tyre.age + laps_waited) ** 0.7 * laps_waited
            pos_gain = car.position - exp_pos
            score = pos_gain * 3.0 - deg_cost + (1.5 if state.safety_car else 0)

            if score > best_score:
                best_score = score
                best_lap = pit_lap
                best_pos = exp_pos
                best_undercuts = undercuts
                best_ahead = ahead
                best_threats = threats

        pos_delta = car.position - best_pos
        pit_loss = self.pit_entry + float(np.mean(self.pit_time_dist)) + self.pit_exit
        fresh_adv = self._deg_advantage(tyre.compound, tyre.age, best_compound, 10)
        net_gain = fresh_adv - (pit_loss * 0 if best_lap == current_lap else 0)

        # Aksiyon kararı
        if urgency == "CRITICAL" or state.safety_car:
            action = "PIT_NOW"
        elif best_lap == current_lap and pos_delta >= 0:
            action = "PIT_NOW"
        elif best_lap > current_lap:
            action = f"PIT_LAP_{best_lap}"
        elif pos_delta < -1 and not tyre.laps_on_edge:
            action = "OVERCUT"
        else:
            action = "STAY_OUT"

        # Güven skoru
        confidence = min(0.95, max(0.30,
            0.5 + pos_delta * 0.1 + (0.2 if urgency == "CRITICAL" else 0)
            - (0.1 if len(best_threats) > 1 else 0)
        ))

        # Gerekçe
        parts = []
        if state.safety_car:
            parts.append("Safety car — pit loss minimize")
        if urgency == "CRITICAL":
            parts.append(f"{tyre.compound} on edge ({tyre.age} laps)")
        if best_undercuts:
            parts.append(f"Can undercut: {', '.join(best_undercuts[:2])}")
        if best_threats:
            parts.append(f"Threat from: {', '.join(best_threats[:2])}")
        if not parts:
            parts.append("No strategic advantage identified")
        reasoning = " | ".join(parts)

        return PitRecommendation(
            driver=target_driver,
            current_lap=current_lap,
            current_position=car.position,
            current_compound=tyre.compound,
            current_tyre_age=tyre.age,
            action=action,
            recommended_lap=best_lap,
            recommended_compound=best_compound,
            urgency=urgency,
            expected_position_after=round(best_pos, 1),
            position_delta=round(pos_delta, 1),
            pit_loss_s=round(pit_loss, 2),
            net_gain_s=round(net_gain, 2),
            cars_ahead_in_window=best_ahead,
            cars_to_undercut=best_undercuts,
            threat_from_behind=best_threats,
            confidence=round(confidence, 2),
            reasoning=reasoning,
        )

    def analyze_full_grid(self, state: RaceState) -> RaceStrategyReport:
        """Tüm grid için strateji analizi — tek tur snapshot."""
        recs = {}
        for driver in state.cars:
            if not state.cars[driver].dnf:
                recs[driver] = self.analyze_car(driver, state)

        return RaceStrategyReport(
            lap=state.current_lap,
            circuit=self.circuit,
            sc_active=state.safety_car,
            recommendations=recs,
        )

    def pit_window_heatmap_data(
        self,
        target_driver: str,
        state: RaceState,
        window: int = 15,
    ) -> pd.DataFrame:
        """
        Pit penceresi ısı haritası için veri üret.
        Her (pit_lap, compound) kombinasyonu için beklenen pozisyon.
        """
        car = state.cars[target_driver]
        rows = []
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            for pit_lap in range(
                state.current_lap,
                min(state.current_lap + window, state.total_laps - 3)
            ):
                exp_pos, _, undercuts, _ = self._estimate_position_after_pit(
                    car, list(state.cars.values()),
                    pit_lap, state.current_lap, compound
                )
                rows.append({
                    "pit_lap": pit_lap,
                    "compound": compound,
                    "expected_position": round(exp_pos, 1),
                    "n_undercuts": len(undercuts),
                    "gain": round(car.position - exp_pos, 1),
                })
        return pd.DataFrame(rows)
