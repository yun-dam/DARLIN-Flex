"""
RBC v4 — adaptive net-load-flattening controller for CityLearn v2.

Flattens each building's net-load curve: discharge to clip peaks, charge to fill
valleys and soak solar. Baseline = slow EMA of net load. Because charging targets
valleys (far from the district peak), it avoids the coincident-charging spike.

v4.1 tuning: the capacity check showed a LARGE battery (6.4 kWh, 5 kW nominal vs
~1.2 kW typical load) — plenty to shave the peak — so discharge is aggressive:
act as soon as net rises above baseline, at a high rate. Charge stays gentle.

Stateful per building (keyed by building_id) to hold the EMA baseline.

------------------------------------------------------------------------------
YOU OWN THESE — tune and defend them:
  ema_alpha, hi_band/mid_band/lo_band, charge/discharge rates, soc bounds
------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RBCv4Params:
    ema_alpha: float = 0.02      # ~2-day baseline; small = smoother/stabler

    hi_band: float = 1.15        # net > baseline*hi_band -> discharge hard
    mid_band: float = 1.02       # net barely above baseline -> discharge soft
    lo_band: float = 0.80        # net < baseline*lo_band  -> charge (valley/solar)

    soc_min: float = 0.15
    soc_max: float = 0.95

    # AGGRESSIVE discharge (large battery); gentle charge (avoid new peak).
    charge_rate: float = 0.15
    discharge_soft_rate: float = 0.18
    discharge_hard_rate: float = 0.38

    warmup_steps: int = 24


VALID_MODES = ("charge", "hold", "discharge_soft", "discharge_hard")


def _read(o: Dict[str, float], *candidates, default=None):
    for k in candidates:
        if k in o:
            return o[k]
    return default


class RBCv4:
    def __init__(self, params: RBCv4Params = None):
        self.p = params or RBCv4Params()
        self._ema: Dict[int, float] = {}
        self._seen: Dict[int, int] = {}

    def _apply_soc_guard(self, action: float, soc: float) -> float:
        p = self.p
        if action > 0:
            return min(action, max(0.0, p.soc_max - soc))
        if action < 0:
            return -min(-action, max(0.0, soc - p.soc_min))
        return 0.0

    def mode_to_action(self, mode: str, soc: float) -> float:
        p = self.p
        if mode == "charge":
            desired = +p.charge_rate
        elif mode == "discharge_soft":
            desired = -p.discharge_soft_rate
        elif mode == "discharge_hard":
            desired = -p.discharge_hard_rate
        else:
            desired = 0.0
        return self._apply_soc_guard(desired, soc)

    def decide_mode(self, o: Dict[str, float]) -> str:
        p = self.p
        bid = int(_read(o, "building_id", default=0))
        load = float(_read(o, "non_shiftable_load", default=0.0))
        solar = float(_read(o, "solar_generation", default=0.0))
        soc = float(_read(o, "electrical_storage_soc", default=0.5))
        net = load - solar

        if bid not in self._ema:
            self._ema[bid] = net
            self._seen[bid] = 0
        self._ema[bid] = p.ema_alpha * net + (1 - p.ema_alpha) * self._ema[bid]
        self._seen[bid] += 1
        base = self._ema[bid]

        if self._seen[bid] < p.warmup_steps:
            return "hold"

        if net > base * p.hi_band and soc > p.soc_min:
            return "discharge_hard"
        if net > base * p.mid_band and soc > p.soc_min:
            return "discharge_soft"
        if (net < base * p.lo_band or solar > load) and soc < p.soc_max:
            return "charge"
        return "hold"

    def act(self, o: Dict[str, float]) -> float:
        soc = float(_read(o, "electrical_storage_soc", default=0.5))
        return self.mode_to_action(self.decide_mode(o), soc)


# --------------------------------------------------------------------------- #
# Self-test: synthetic daily profile. python rbc_v4.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math
    ctrl = RBCv4()
    soc = 0.5
    print(f"{'hour':>4} {'load':>6} {'solar':>6} {'net':>7} {'mode':>15} {'action':>8} {'soc':>6}")
    for day in range(3):
        for hr in range(24):
            load = 2.0 + 1.5 * math.exp(-((hr - 19) ** 2) / 8)
            solar = max(0.0, 3.0 * math.sin(math.pi * (hr - 6) / 12))
            o = {"building_id": 0, "non_shiftable_load": load,
                 "solar_generation": solar, "electrical_storage_soc": soc}
            mode = ctrl.decide_mode(o)
            act = ctrl.mode_to_action(mode, soc)
            soc = min(0.95, max(0.15, soc + act))
            if day == 2:
                print(f"{hr:>4} {load:6.2f} {solar:6.2f} {load-solar:7.2f} "
                      f"{mode:>15} {act:8.3f} {soc:6.2f}")
    print("\nExpected: charge midday, discharge evening ~17-21, hold otherwise.")
