"""
RBC v5 — fixes two real bugs found in v4, and is the shared executor for the
LLM controller so the only difference between policies is the BRAIN.

FIX 1 — warmup starvation.
  v4 used warmup_steps=24: in a 24 h pilot the RBC was inert for 23 of 24 steps
  (verified), so "LLM beats RBC" was really "LLM beats a switched-off RBC".
  v5 defaults warmup_steps=6 and the SAME value is applied to the LLM policy,
  so both controllers are handicapped identically.

FIX 2 — ratio thresholds break on negative baselines.
  v4 tested `net > base * 1.15`. Because net = load - solar, a solar-heavy
  building's EMA baseline can go NEGATIVE, and multiplying a negative baseline
  by 1.15 makes the threshold *more* negative -> almost any positive load fired
  discharge_hard. v5 uses SCALE-FREE ABSOLUTE OFFSETS:

      threshold = base + k * dev      where dev = EMA of |net - base|

  This is correct for negative, zero, or positive baselines.

Optional: proportional discharge (smoother => less ramping). Default OFF so the
LLM and RBC share an identical executor and the comparison stays clean.
"""

from dataclasses import dataclass
from typing import Dict

VALID_MODES = ("charge", "hold", "discharge_soft", "discharge_hard")


@dataclass
class RBCv5Params:
    # --- baseline tracking ---
    ema_alpha: float = 0.05     # net-load baseline (faster than v4's 0.02)
    dev_alpha: float = 0.05     # scale (mean abs deviation) tracker
    min_dev: float = 0.05       # kW floor so a flat load isn't hair-trigger

    # --- thresholds in units of dev (absolute offsets, not ratios) ---
    hi_k: float = 1.20          # net > base + hi_k*dev  -> discharge_hard
    mid_k: float = 0.35         # net > base + mid_k*dev -> discharge_soft
    lo_k: float = 0.50          # net < base - lo_k*dev  -> charge (valley)

    # --- SOC guard ---
    soc_min: float = 0.15
    soc_max: float = 0.95

    # --- rates (fraction of capacity per hour) ---
    charge_rate: float = 0.15
    discharge_soft_rate: float = 0.18
    discharge_hard_rate: float = 0.38

    # --- fairness / behaviour ---
    warmup_steps: int = 6       # applied to LLM policy too
    solar_only_charge: bool = True   # never grid-charge (kills coincident peak)
    proportional: bool = False       # True = graded discharge magnitude


def _read(o: Dict[str, float], *keys, default=None):
    for k in keys:
        if k in o:
            return o[k]
    return default


class BaselineTracker:
    """Per-building EMA of net load plus an EMA of |deviation| for scale."""

    def __init__(self, p: RBCv5Params):
        self.p = p
        self.base: Dict[int, float] = {}
        self.dev: Dict[int, float] = {}
        self.seen: Dict[int, int] = {}

    def update(self, bid: int, net: float):
        p = self.p
        if bid not in self.base:
            self.base[bid] = net
            self.dev[bid] = p.min_dev
            self.seen[bid] = 0
        d = abs(net - self.base[bid])
        self.base[bid] = p.ema_alpha * net + (1 - p.ema_alpha) * self.base[bid]
        self.dev[bid] = max(p.min_dev,
                            p.dev_alpha * d + (1 - p.dev_alpha) * self.dev[bid])
        self.seen[bid] += 1
        return self.base[bid], self.dev[bid], self.seen[bid]


class RBCv5:
    def __init__(self, params: RBCv5Params = None):
        self.p = params or RBCv5Params()
        self.tracker = BaselineTracker(self.p)
        self.mode_counts = {m: 0 for m in VALID_MODES}
        self.grid_charge_events = 0     # charge decisions taken while net > 0

    # ---------------- executor (shared with the LLM) ---------------- #
    def _soc_guard(self, action: float, soc: float) -> float:
        p = self.p
        if action > 0:
            return min(action, max(0.0, p.soc_max - soc))
        if action < 0:
            return -min(-action, max(0.0, soc - p.soc_min))
        return 0.0

    def mode_to_action(self, mode: str, soc: float, net: float = None) -> float:
        """Discrete mode -> SOC-guarded action. Optionally blocks grid-charging."""
        p = self.p
        if mode not in VALID_MODES:
            mode = "hold"
        if mode == "charge":
            # hard safety rule: only store FREE solar surplus (net < 0)
            if p.solar_only_charge and net is not None and net >= 0:
                return 0.0
            desired = +p.charge_rate
        elif mode == "discharge_soft":
            desired = -p.discharge_soft_rate
        elif mode == "discharge_hard":
            desired = -p.discharge_hard_rate
        else:
            desired = 0.0
        return self._soc_guard(desired, soc)

    # ---------------- brain ---------------- #
    def decide_mode(self, o: Dict[str, float]) -> str:
        p = self.p
        bid = int(_read(o, "building_id", default=0))
        load = float(_read(o, "non_shiftable_load", default=0.0))
        solar = float(_read(o, "solar_generation", default=0.0))
        soc = float(_read(o, "electrical_storage_soc", default=0.5))
        net = load - solar

        base, dev, seen = self.tracker.update(bid, net)
        if seen <= p.warmup_steps:
            return "hold"

        # absolute offsets — correct even when base is negative
        if net > base + p.hi_k * dev and soc > p.soc_min:
            return "discharge_hard"
        if net > base + p.mid_k * dev and soc > p.soc_min:
            return "discharge_soft"
        if (solar > load or net < base - p.lo_k * dev) and soc < p.soc_max:
            if p.solar_only_charge and net >= 0:
                # the executor's guard would zero this out anyway (grid-charging
                # is disallowed) -- decide "hold" so the brain agrees with it
                return "hold"
            return "charge"
        return "hold"

    def act(self, o: Dict[str, float]) -> float:
        soc = float(_read(o, "electrical_storage_soc", default=0.5))
        net = float(_read(o, "non_shiftable_load", default=0.0)) - \
              float(_read(o, "solar_generation", default=0.0))
        mode = self.decide_mode(o)
        self.mode_counts[mode] += 1
        if mode == "charge" and net >= 0:
            self.grid_charge_events += 1
        if self.p.proportional and mode.startswith("discharge"):
            return self._proportional(o, soc, net)
        return self.mode_to_action(mode, soc, net)

    def _proportional(self, o, soc, net) -> float:
        """Graded discharge: scale between soft and hard by how far past base."""
        p = self.p
        bid = int(_read(o, "building_id", default=0))
        base, dev = self.tracker.base[bid], self.tracker.dev[bid]
        lo = base + p.mid_k * dev
        hi = base + p.hi_k * dev
        frac = 0.0 if hi <= lo else (net - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        rate = p.discharge_soft_rate + frac * (p.discharge_hard_rate - p.discharge_soft_rate)
        return self._soc_guard(-rate, soc)


# --------------------------------------------------------------------------- #
# Self-test — proves both bug fixes.  python rbc_v5.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math

    print("=== TEST 1: SOC guard clips the validation-gate traps ===")
    c = RBCv5()
    traps = [("discharge_hard @ empty", "discharge_hard", 0.10, 0.0),
             ("charge @ full", "charge", 0.96, 0.0),
             ("discharge_hard @ 0.80", "discharge_hard", 0.80, -0.38),
             ("charge @ 0.40 (solar surplus)", "charge", 0.40, +0.15),
             ("hold @ 0.50", "hold", 0.50, 0.0)]
    ok = True
    for label, mode, soc, want in traps:
        got = c.mode_to_action(mode, soc, net=-1.0)   # net<0 => charging allowed
        good = abs(got - want) < 1e-9
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {label:32s} -> {got:+.3f} (want {want:+.3f})")

    print("\n=== TEST 2: FIX 1 — warmup no longer starves short windows ===")
    def count_active(hours, warm):
        ctrl = RBCv5(RBCv5Params(warmup_steps=warm))
        active = 0
        for t in range(hours):
            hr = t % 24
            load = 2.0 + 1.5 * math.exp(-((hr - 19) ** 2) / 8)
            solar = max(0.0, 3.0 * math.sin(math.pi * (hr - 6) / 12))
            m = ctrl.decide_mode({"building_id": 0, "non_shiftable_load": load,
                                  "solar_generation": solar,
                                  "electrical_storage_soc": 0.5})
            active += (m != "hold")
        return active
    a24_old, a24_new = count_active(24, 24), count_active(24, 6)
    print(f"  24 h window, warmup=24 (v4): {a24_old:2d}/24 active steps  <- the bug")
    print(f"  24 h window, warmup=6  (v5): {a24_new:2d}/24 active steps")
    print(f"  [{'PASS' if a24_new > a24_old else 'FAIL'}] v5 actually controls in a short window")

    print("\n=== TEST 3: FIX 2 — negative baseline no longer inverts logic ===")
    print("  Solar-heavy building (net goes negative midday). Counting the")
    print("  PHYSICALLY WRONG event: discharging while the building EXPORTS solar.")
    p = RBCv5Params()

    def solar_heavy(hours=72, pv=10.0):
        """NET-EXPORTING building (annual mean net < 0) — the only condition under
        which the v4 ratio bug actually fires. Verified: PV>=8 makes mean net<0."""
        out = []
        for t in range(hours):
            hr = t % 24
            load = 1.5 + 1.2 * math.exp(-((hr - 19) ** 2) / 8)
            solar = max(0.0, pv * math.sin(math.pi * (hr - 6) / 12))
            out.append((load, solar))
        return out

    # v4-style: EMA baseline + RATIO thresholds
    v4_bad = v5_bad = 0
    ema = None
    for load, solar in solar_heavy():
        net = load - solar
        ema = net if ema is None else 0.02 * net + 0.98 * ema
        if net > ema * 1.15:          # v4 rule
            if net < 0:               # discharging while exporting -> wrong
                v4_bad += 1
    # v5-style: EMA baseline + ABSOLUTE OFFSET thresholds scaled by deviation
    tr = BaselineTracker(p)
    for load, solar in solar_heavy():
        net = load - solar
        base, dev, seen = tr.update(0, net)
        if seen <= p.warmup_steps:
            continue
        if net > base + p.hi_k * dev:  # v5 rule
            if net < 0:
                v5_bad += 1
    print(f"  v4 ratio rule : {v4_bad:3d} wrong discharges while exporting solar")
    print(f"  v5 offset rule: {v5_bad:3d} wrong discharges while exporting solar")
    print(f"  [{'PASS' if v5_bad < v4_bad else 'FAIL'}] v5 removes the negative-baseline pathology")

    print("\n=== TEST 4: solar_only_charge blocks grid charging ===")
    c2 = RBCv5()
    a_grid = c2.mode_to_action("charge", 0.5, net=+2.0)   # positive net => blocked
    a_solar = c2.mode_to_action("charge", 0.5, net=-2.0)  # surplus => allowed
    good = (abs(a_grid) < 1e-9) and (a_solar > 0)
    print(f"  charge @ net=+2.0 -> {a_grid:+.3f} (blocked)   "
          f"charge @ net=-2.0 -> {a_solar:+.3f} (allowed)")
    print(f"  [{'PASS' if good else 'FAIL'}] grid-charge leak closed at the executor")
    print("\nAll executor-level checks done." if ok and good else "\nREVIEW FAILURES ABOVE.")
