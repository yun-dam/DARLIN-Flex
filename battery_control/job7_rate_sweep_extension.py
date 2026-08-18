"""
job7_rate_sweep_extension.py -- extends job6's rate sweep downward to
bracket L3's phase_1 48h ramping point (0.8810), which fell outside the
original k in [0.4, 1.2] range (max observed ramping was 0.8086 at k=0.4).

k in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35], phase_1, both 48h and full year.
Same k=1.0 control gate as job6. Reuses job6's exact functions (run_rbc,
evaluate, load_baseline, interpolate_cost_at_ramping) -- not reimplemented.
"""
import json
import time

from job6_rate_sweep import (evaluate, interpolate_cost_at_ramping,
                             GATE_48H, GATE_YEAR, GATE_TOL, L3_PHASE1_48H, fmt)

K_VALUES_NEW = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
OUT = "results_rate_sweep_extension.json"


def main():
    t0 = time.time()
    prior = json.load(open("results_rate_sweep.json"))
    results = {"phase_1": {"48h": list(prior["phase_1"]["48h"]),
                           "year": list(prior["phase_1"]["year"])}}

    print("=== phase_1 sweep extension, k in", K_VALUES_NEW, "===")
    for k in K_VALUES_NEW:
        for window in ("48h", "year"):
            r = evaluate("phase_1", k, window)
            results["phase_1"][window].append(r)
            print(f"  k={k}  {window:<4}  cost={fmt(r['cost_total'])}  "
                  f"carbon={fmt(r['carbon_emissions_total'])}  "
                  f"ramping={fmt(r['ramping_average'])}  D={fmt(r['D'])}")

    # ---- re-verify the k=1.0 gate (must still hold; k=1.0 result is reused, unchanged) ----
    gate_problems = []
    for window, target in (("48h", GATE_48H), ("year", GATE_YEAR)):
        r1 = next(r for r in results["phase_1"][window] if r["k"] == 1.0)
        for metric, expect in target.items():
            got = r1[metric]
            if got is None or abs(got - expect) > GATE_TOL:
                gate_problems.append(f"{window} {metric}: got {got}, expected {expect} (tol {GATE_TOL})")
    results["gate"] = {"passed": not gate_problems, "problems": gate_problems}
    if gate_problems:
        print("\n" + "!" * 72)
        print("!! GATE FAILED on k=1.0 phase_1 control -- STOPPING")
        for pr in gate_problems:
            print("!!  ", pr)
        print("!" * 72)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        return
    print("\n[GATE PASSED] k=1.0 still reproduces the known phase_1 reference.")

    # ---- bracket L3's 48h ramping point ----
    sweep_48h = sorted(results["phase_1"]["48h"], key=lambda r: r["ramping_average"])
    l3_cost, l3_ramp = L3_PHASE1_48H["cost"], L3_PHASE1_48H["ramping"]

    below = [r for r in sweep_48h if r["ramping_average"] <= l3_ramp]
    above = [r for r in sweep_48h if r["ramping_average"] >= l3_ramp]
    bracket_below = max(below, key=lambda r: r["ramping_average"]) if below else None
    bracket_above = min(above, key=lambda r: r["ramping_average"]) if above else None
    interp_cost, bracket = interpolate_cost_at_ramping(sweep_48h, l3_ramp)

    dominated_by = [r for r in sweep_48h
                    if r["ramping_average"] <= l3_ramp and r["cost_total"] <= l3_cost
                    and (r["ramping_average"] < l3_ramp or r["cost_total"] < l3_cost)]

    if interp_cost is not None:
        classification = "L3_DOMINATES" if interp_cost > l3_cost else "L3_DOMINATED"
    else:
        classification = "STILL_OUT_OF_RANGE"

    print(f"\nFull 48h ramping range now: {sweep_48h[0]['ramping_average']:.4f} to "
          f"{sweep_48h[-1]['ramping_average']:.4f}")
    print(f"L3 point: cost={l3_cost}  ramping={l3_ramp}")
    if bracket_below:
        print(f"Bracket BELOW (ramping <= L3): k={bracket_below['k']}  "
              f"ramping={bracket_below['ramping_average']:.4f}  cost={bracket_below['cost_total']:.4f}")
    else:
        print("Bracket BELOW: none (no k has ramping <= L3's)")
    if bracket_above:
        print(f"Bracket ABOVE (ramping >= L3): k={bracket_above['k']}  "
              f"ramping={bracket_above['ramping_average']:.4f}  cost={bracket_above['cost_total']:.4f}")
    else:
        print("Bracket ABOVE: none (no k has ramping >= L3's)")
    print(f"Interpolated frontier cost at ramping={l3_ramp}: "
          f"{fmt(interp_cost) if interp_cost is not None else 'N/A'}")
    print(f"k values dominating L3 outright: {[r['k'] for r in dominated_by] or '(none)'}")
    print(f"CLASSIFICATION: {classification}")

    results["l3_bracket_analysis"] = {
        "l3_point": L3_PHASE1_48H,
        "full_ramping_range_48h": [sweep_48h[0]["ramping_average"], sweep_48h[-1]["ramping_average"]],
        "bracket_below": bracket_below,
        "bracket_above": bracket_above,
        "interpolated_cost_at_l3_ramping": interp_cost,
        "dominating_k_values": dominated_by,
        "classification": classification,
    }

    # ---- full-year table: did ANY k beat k=1.0 on cost at full year, phase_1? ----
    year_sweep = sorted(results["phase_1"]["year"], key=lambda r: r["k"])
    r1_year = next(r for r in year_sweep if r["k"] == 1.0)
    beats_on_cost_year = [r for r in year_sweep if r["k"] != 1.0 and r["cost_total"] < r1_year["cost_total"]]

    print("\n=== full-year phase_1 table, all k (original + extension) ===")
    print(f"{'k':>6}{'cost':>10}{'carbon':>10}{'ramping':>10}{'D':>10}")
    for r in year_sweep:
        print(f"{r['k']:>6}{r['cost_total']:>10.4f}{r['carbon_emissions_total']:>10.4f}"
              f"{r['ramping_average']:>10.4f}{r['D']:>10.4f}")
    print(f"\nk=1.0 full-year cost_total: {r1_year['cost_total']:.4f}")
    print(f"k values beating k=1.0 on cost at full year: "
          f"{[(r['k'], round(r['cost_total'],4)) for r in beats_on_cost_year] or '(none)'}")

    results["full_year_cost_check"] = {
        "k1_cost": r1_year["cost_total"],
        "beats_k1_on_cost": [{"k": r["k"], "cost_total": r["cost_total"]} for r in beats_on_cost_year],
        "any_k_beats_k1_on_cost": bool(beats_on_cost_year),
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ntotal seconds: {time.time() - t0:.0f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
