"""
job6_rate_sweep.py -- RBC rate sweep experiment.

Tests whether L3's phase_1 advantage (cost 0.825/carbon 0.839/ramping 0.881
at 48h, vs RBC v5's 0.878/0.881/0.755) is a genuine conditional strategy, or
just a hyperparameter (lower cycling rate) the rule already has access to via
a single global rate multiplier k applied to charge_rate, discharge_soft_rate,
and discharge_hard_rate together. Nothing else changes -- same thresholds
(hi_k/mid_k/lo_k), same warmup_steps, same executor, same solar_only_charge.

k in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2], run at both the 48h ladder
window and the full year, on phase_1. Do-nothing baselines for normalisation
are all REUSED from prior validated runs (not recomputed):
  phase_1 48h    -- results_ladder_reference.json['v0']
  phase_1/2/3 year -- results_transfer.json raw_do_nothing per schema
  phase_2/3 48h  -- results_llm_transfer_phase_{2,3}_summary.json

GATE: k=1.0 must reproduce known reference values before anything else is
trusted (48h: cost 0.878/carbon 0.881/ramping 0.755; full year: cost
0.807/carbon 0.878/ramping 0.780), tolerance 0.005. Exits loudly if not.

CRITICAL GUARD: any k that beats k=1.0 on phase_1 on ANY of cost/carbon/
ramping/D, at either window, is flagged and immediately re-run on phase_2 AND
phase_3 at the SAME window (against those schemas' own k=1.0), checking
whether the improvement survives out-of-sample. In-sample-only wins are
reported as such, never as improvements.
"""
import json
import time

from rbc_v5 import RBCv5, RBCv5Params
from run_transfer import (make_env, flat_names, build_index, reset_env,
                          step_env, unwrap, set_start_soc, true_soc, kpis,
                          normalise, challenge_kpis, fmt, START_SOC)

K_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
BASE_RATES = {"charge_rate": 0.15, "discharge_soft_rate": 0.18, "discharge_hard_rate": 0.38}
GATE_TOL = 0.005
GATE_48H = {"cost_total": 0.878, "carbon_emissions_total": 0.881, "ramping_average": 0.755}
GATE_YEAR = {"cost_total": 0.807, "carbon_emissions_total": 0.878, "ramping_average": 0.780}
L3_PHASE1_48H = {"cost": 0.825, "carbon": 0.839, "ramping": 0.881}
OUT = "results_rate_sweep.json"

SCHEMA_MAP = {"phase_1": "citylearn_challenge_2022_phase_1",
             "phase_2": "citylearn_challenge_2022_phase_2",
             "phase_3": "citylearn_challenge_2022_phase_3"}


def load_baseline(schema_tag, window):
    if window == "48h":
        if schema_tag == "phase_1":
            return json.load(open("results_ladder_reference.json"))["v0"]["kpi"]
        return json.load(open(f"results_llm_transfer_{schema_tag}_summary.json"))["do_nothing_kpi"]
    d = json.load(open("results_transfer.json"))
    return d["results"][SCHEMA_MAP[schema_tag]]["raw_do_nothing"]


def run_rbc(schema_tag, k, hours):
    schema = SCHEMA_MAP[schema_tag]
    params = RBCv5Params(
        charge_rate=BASE_RATES["charge_rate"] * k,
        discharge_soft_rate=BASE_RATES["discharge_soft_rate"] * k,
        discharge_hard_rate=BASE_RATES["discharge_hard_rate"] * k,
    )
    env = make_env(schema)
    names = flat_names(env)
    idx = build_index(names)
    n = len(env.buildings)
    obs = reset_env(env)
    set_start_soc(env, START_SOC)
    ctrl = RBCv5(params)
    limit = hours if hours else (env.time_steps - 1)
    t, done = 0, False
    while t < limit and not done:
        flat = unwrap(obs)
        hour = flat[idx["hour"]] if idx["hour"] is not None else 0
        av = []
        for b in range(n):
            o = {"hour": hour, "electrical_storage_soc": true_soc(env, b, t),
                 "solar_generation": flat[idx["solar"][b]],
                 "non_shiftable_load": flat[idx["load"][b]],
                 "building_id": b, "n_buildings": n}
            av.append(float(ctrl.act(o)))
        obs, done = step_env(env, [av])
        t += 1
    return kpis(env), t, dict(ctrl.mode_counts)


def evaluate(schema_tag, k, window):
    hours = 48 if window == "48h" else 0
    base_kpi = load_baseline(schema_tag, window)
    raw_kpi, steps, modes = run_rbc(schema_tag, k, hours)
    norm = normalise(raw_kpi, base_kpi)
    c, g, d = challenge_kpis(norm)
    return {
        "schema": schema_tag, "k": k, "window": window, "steps": steps,
        "cost_total": norm.get("cost_total"),
        "carbon_emissions_total": norm.get("carbon_emissions_total"),
        "ramping_average": norm.get("ramping_average"),
        "C": c, "G": g, "D": d, "mode_counts": modes,
    }


def interpolate_cost_at_ramping(sweep, target_ramping):
    """Linear interpolation of cost_total at target_ramping, between the two
    sweep points bracketing it. Returns (interpolated_cost, bracket) or
    (None, None) if target_ramping is outside the sweep's observed range."""
    pts = sorted(sweep, key=lambda r: r["ramping_average"])
    ramps = [p["ramping_average"] for p in pts]
    if target_ramping < ramps[0] or target_ramping > ramps[-1]:
        return None, None
    for i in range(len(pts) - 1):
        r0, r1 = ramps[i], ramps[i + 1]
        if r0 <= target_ramping <= r1:
            if r1 == r0:
                return pts[i]["cost_total"], (pts[i], pts[i + 1])
            frac = (target_ramping - r0) / (r1 - r0)
            c0, c1 = pts[i]["cost_total"], pts[i + 1]["cost_total"]
            return c0 + frac * (c1 - c0), (pts[i], pts[i + 1])
    return None, None


def main():
    t0 = time.time()
    results = {"phase_1": {"48h": [], "year": []}}

    print("=== phase_1 sweep ===")
    for k in K_VALUES:
        for window in ("48h", "year"):
            r = evaluate("phase_1", k, window)
            results["phase_1"][window].append(r)
            print(f"  k={k}  {window:<4}  cost={fmt(r['cost_total'])}  "
                  f"carbon={fmt(r['carbon_emissions_total'])}  "
                  f"ramping={fmt(r['ramping_average'])}  D={fmt(r['D'])}")

    # ---- gate check on k=1.0 ----
    gate_problems = []
    for window, target in (("48h", GATE_48H), ("year", GATE_YEAR)):
        r1 = next(r for r in results["phase_1"][window] if r["k"] == 1.0)
        for metric, expect in target.items():
            got = r1[metric]
            if got is None or abs(got - expect) > GATE_TOL:
                gate_problems.append(f"{window} {metric}: got {got}, expected {expect} (tol {GATE_TOL})")
    results["gate"] = {"passed": not gate_problems, "problems": gate_problems}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    if gate_problems:
        print("\n" + "!" * 72)
        print("!! GATE FAILED on k=1.0 phase_1 control -- STOPPING")
        for pr in gate_problems:
            print("!!  ", pr)
        print("!" * 72)
        return
    print("\n[GATE PASSED] k=1.0 reproduces the known phase_1 reference at both windows.")

    # ---- frontier analysis, 48h (matches L3's window) ----
    sweep_48h = results["phase_1"]["48h"]
    l3_cost, l3_ramp = L3_PHASE1_48H["cost"], L3_PHASE1_48H["ramping"]

    nearest = min(sweep_48h, key=lambda r: abs(r["ramping_average"] - l3_ramp))
    dominated_by = [r for r in sweep_48h
                    if r["ramping_average"] <= l3_ramp and r["cost_total"] <= l3_cost
                    and (r["ramping_average"] < l3_ramp or r["cost_total"] < l3_cost)]
    interp_cost, bracket = interpolate_cost_at_ramping(sweep_48h, l3_ramp)

    if dominated_by:
        classification = "DOMINATED"
    elif interp_cost is not None:
        classification = "ON_OR_INSIDE" if interp_cost <= l3_cost else "OUTSIDE"
    else:
        # L3's ramping is outside the sweep's observed range -- fall back to nearest-neighbor
        classification = "ON_OR_INSIDE" if nearest["cost_total"] <= l3_cost else "OUTSIDE"

    results["l3_frontier_analysis"] = {
        "l3_point": L3_PHASE1_48H,
        "nearest_k_by_ramping": nearest,
        "interpolated_cost_at_l3_ramping": interp_cost,
        "interpolation_bracket": bracket,
        "dominating_k_values": dominated_by,
        "classification": classification,
    }
    print(f"\nL3 point: cost={l3_cost} ramping={l3_ramp}")
    print(f"Nearest k by ramping: k={nearest['k']}  cost={fmt(nearest['cost_total'])}  "
          f"ramping={fmt(nearest['ramping_average'])}")
    print(f"Interpolated frontier cost at ramping={l3_ramp}: "
          f"{fmt(interp_cost) if interp_cost is not None else 'N/A (outside sweep range)'}")
    print(f"k values dominating L3 (both <=, at least one strictly <): "
          f"{[r['k'] for r in dominated_by]}")
    print(f"CLASSIFICATION: {classification}")

    # ---- critical guard: any k beating k=1.0 on phase_1, either window ----
    flagged = []
    for window in ("48h", "year"):
        r1 = next(r for r in results["phase_1"][window] if r["k"] == 1.0)
        for r in results["phase_1"][window]:
            if r["k"] == 1.0:
                continue
            for metric in ("cost_total", "carbon_emissions_total", "ramping_average", "D"):
                if r[metric] is not None and r1[metric] is not None and r[metric] < r1[metric]:
                    flagged.append((window, r["k"], metric, r[metric], r1[metric]))

    print(f"\n=== critical guard: {len(flagged)} (window, k, metric) combos beat k=1.0 on phase_1 ===")
    for window, k, metric, val, base in flagged:
        print(f"  {window} k={k} {metric}: {val:.4f} < k=1.0's {base:.4f}")

    validation = []
    checked = set()
    for window, k, metric, val, base in flagged:
        key = (window, k)
        if key in checked:
            continue
        checked.add(key)
        print(f"\n--- out-of-sample validation: k={k}, window={window} ---")
        for schema_tag in ("phase_2", "phase_3"):
            r_k = evaluate(schema_tag, k, window)
            r_1 = evaluate(schema_tag, 1.0, window)
            survives = {}
            for metric2 in ("cost_total", "carbon_emissions_total", "ramping_average", "D"):
                if r_k[metric2] is not None and r_1[metric2] is not None:
                    survives[metric2] = bool(r_k[metric2] < r_1[metric2])
            entry = {"window": window, "k": k, "schema": schema_tag,
                    "k_result": r_k, "k1_result": r_1, "beats_k1": survives}
            validation.append(entry)
            print(f"  {schema_tag}: k={k} vs k=1.0 -> beats_k1={survives}")

    results["critical_guard"] = {
        "flagged": [{"window": w, "k": k, "metric": m, "k_value": v, "k1_value": b}
                   for w, k, m, v, b in flagged],
        "out_of_sample_validation": validation,
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ntotal seconds: {time.time() - t0:.0f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
