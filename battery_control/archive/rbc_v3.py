"""
RBC v3 — parameterized executor for CityLearn v2 battery control.

    obs --> [ brain ] --> mode --> [ executor ] --> battery action
            rule_policy()          mode_to_action()  (SOC-guarded)

v3.3 — SOLAR-ONLY CHARGING experiment.
Trajectory so far on all_time_peak_average: 1.621 (naive) -> 1.248 (gentle rate)
-> 1.050 (staggered coordination). Remaining overshoot is grid charging landing
on a bad instant. Here grid_charge_soc_ceiling = 0.0 turns OFF off-peak grid
charging entirely: the battery banks ONLY solar surplus, which reduces net grid
draw and can never create a new peak. Discharge still shaves the evening peak.

To re-enable staggered grid charging, set grid_charge_soc_ceiling back to ~0.40.

obs_dict is PER BUILDING and carries building_id + n_buildings for coordination.

------------------------------------------------------------------------------
YOU OWN THESE — set and defend them:
  peak_hours / offpeak_hours          -> time-of-use windows (hour is 1-24)
  soc_min / soc_max / soc_target      -> safety + reserve bands
  charge_rate / discharge_*_rate      -> aggressiveness
  grid_charge_soc_ceiling             -> 0.0 = solar-only; >0 = grid-charge when low
  stagger_charging                    -> multi-building coordination switch
------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class RBCParams:
    peak_hours: tuple = (16, 17, 18, 19, 20, 21)
    offpeak_hours: tuple = (1, 2, 3, 4, 5, 6, 23, 24)

    soc_min: float = 0.15
    soc_max: float = 0.95
    soc_target: float = 0.75

    charge_rate: float = 0.12
    discharge_soft_rate: float = 0.15
    discharge_hard_rate: float = 0.45

    grid_charge_soc_ceiling: float = 0.0     # 0.0 = SOLAR-ONLY charging (this test)

    pv_surplus_ratio: float = 1.0
    stagger_charging: bool = True
    use_pricing_if_available: bool = False


VALID_MODES = ("charge", "hold", "discharge_soft", "discharge_hard")


def obs_to_dict(values: Sequence[float], names: Sequence[str]) -> Dict[str, float]:
    return {n: v for n, v in zip(names, values)}


def _read(o: Dict[str, float], *candidates, default=None):
    for key in candidates:
        if key in o:
            return o[key]
    return default


class RBCv3:
    def __init__(self, params: RBCParams = None):
        self.p = params or RBCParams()

    # ---- EXECUTOR -------------------------------------------------------- #
    def mode_to_action(self, mode: str, soc: float) -> float:
        p = self.p
        if mode not in VALID_MODES:
            mode = "hold"
        if mode == "charge":
            desired = +p.charge_rate
        elif mode == "discharge_soft":
            desired = -p.discharge_soft_rate
        elif mode == "discharge_hard":
            desired = -p.discharge_hard_rate
        else:
            desired = 0.0
        return self._apply_soc_guard(desired, soc)

    def _apply_soc_guard(self, action: float, soc: float) -> float:
        p = self.p
        if action > 0:
            return min(action, max(0.0, p.soc_max - soc))
        if action < 0:
            return -min(-action, max(0.0, soc - p.soc_min))
        return 0.0

    # ---- BRAIN ----------------------------------------------------------- #
    def rule_policy(self, o: Dict[str, float]) -> str:
        p = self.p
        hour = int(_read(o, "hour", default=0))
        soc = float(_read(o, "electrical_storage_soc", default=0.5))
        pv_surplus = self._pv_surplus(o)

        # 1. Evening peak: shave by discharging.
        if hour in p.peak_hours and soc > p.soc_min:
            return "discharge_hard" if soc >= p.soc_target else "discharge_soft"
        # 2. Solar surplus: always store it (reduces export, adds no grid draw).
        if pv_surplus and soc < p.soc_max:
            return "charge"
        # 3. Off-peak grid charge, staggered. Disabled when ceiling = 0.0.
        if soc < p.grid_charge_soc_ceiling and self._my_charge_turn(o, hour):
            return "charge"
        return "hold"

    def _my_charge_turn(self, o: Dict[str, float], hour: int) -> bool:
        p = self.p
        if hour not in p.offpeak_hours:
            return False
        if not p.stagger_charging:
            return True
        bid = int(_read(o, "building_id", default=0))
        nb = max(1, int(_read(o, "n_buildings", default=1)))
        slot = p.offpeak_hours.index(hour)
        return slot % nb == bid % nb

    # ---- helpers --------------------------------------------------------- #
    def _pv_surplus(self, o: Dict[str, float]) -> bool:
        solar = _read(o, "solar_generation")
        load = _read(o, "non_shiftable_load")
        if solar is None or load is None:
            return False
        return solar >= self.p.pv_surplus_ratio * load

    # ---- public API ------------------------------------------------------ #
    def execute(self, mode: str, o: Dict[str, float]) -> float:
        soc = float(_read(o, "electrical_storage_soc", default=0.5))
        return self.mode_to_action(mode, soc)

    def act(self, o: Dict[str, float]) -> float:
        return self.execute(self.rule_policy(o), o)


# --------------------------------------------------------------------------- #
# Self-test. python rbc_v3.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ctrl = RBCv3()
    traps = [
        ("scen3  discharge_hard @ empty (soc=0.10)", "discharge_hard", 0.10, "zero"),
        ("scen5  charge @ full        (soc=0.96)", "charge",         0.96, "zero"),
        ("scen9  charge @ full        (soc=0.96)", "charge",         0.96, "zero"),
        ("ok     discharge_hard @ 0.80",            "discharge_hard", 0.80, "neg"),
        ("ok     charge @ 0.40",                    "charge",         0.40, "pos"),
        ("ok     hold  @ 0.50",                     "hold",           0.50, "zero"),
    ]
    print(f"{'case':<42} {'action':>8}  {'result':>6}")
    print("-" * 60)
    all_ok = True
    for label, mode, soc, expect in traps:
        a = ctrl.mode_to_action(mode, soc)
        sign = "zero" if abs(a) < 1e-9 else ("pos" if a > 0 else "neg")
        ok = (sign == expect)
        all_ok &= ok
        print(f"{label:<42} {a:>8.3f}  {'PASS' if ok else 'FAIL':>6}")
    print("-" * 60)
    print("GUARD OK — traps clipped to hold" if all_ok else "GUARD BROKEN")
