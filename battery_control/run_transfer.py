"""
run_transfer.py  --  RBC v5 transfer evaluation across CityLearn Challenge 2022 phases.

WHY THIS EXISTS
    Every result so far is on citylearn_challenge_2022_phase_1 -- the same five
    buildings the RBC thresholds were hand-tuned against. That is fitting to the
    test set. This script re-runs the identical controller, with identical
    parameters, on buildings it has never seen.

    Phase 1 : 5 buildings   (tuned on -- the control condition)
    Phase 2 : 5 buildings   (never seen)
    Phase 3 : 7 buildings   (never seen, different count)

WHAT IT DOES
    For each schema, runs a do-nothing baseline and RBC v5 over the full year,
    normalises RBC KPIs against the do-nothing run FROM THAT SAME SCHEMA, and
    reports the CityLearn Challenge indicators C (cost), G (carbon) and
    D (grid = mean of ramping and 1 - load factor).

    Imports RBCv5 / RBCv5Params directly from rbc_v5.py -- no embedded copy,
    so it cannot silently drift from the real controller (see git history:
    an earlier verbatim copy here did drift, and there was no signature
    mismatch to catch it since both took a positional (bid, load, solar, soc)
    call -- it just quietly ran different code).

    Controller inputs are read from the OBSERVATION VECTOR returned by
    env.reset()/env.step() -- the same source run_fullyear_rbc.py uses to
    reproduce the reference phase_1 numbers (C 0.807 / G 0.878 / D 0.874).
    An earlier version of this script read building attribute arrays instead
    (net_electricity_consumption_without_storage_and_pv, solar_generation);
    those arrays are appended to lazily as a side effect of step(), so the
    newest entry at read time is always an unfinalized placeholder -- a
    silent one-step lag with no error, which collapsed the controller to
    permanent "hold". The observation vector has no such lag: it reports
    the current step's non_shiftable_load / solar_generation directly.

USAGE
    python run_transfer.py                       # all three phases, full year
    python run_transfer.py --hours 720           # quick smoke test (~1 month)
    python run_transfer.py --schemas citylearn_challenge_2022_phase_2
    python run_transfer.py --preflight           # just report schema shapes, no run

    First run downloads Phase 2 / Phase 3 data from GitHub. Needs internet once.

OUTPUT
    results_transfer.json  +  a printed table

GATE
    phase_1 is the control condition and must reproduce run_fullyear_rbc.py's
    result (C 0.807 / G 0.878 / D 0.874, within 0.005) and show non-degenerate
    mode use (not all-hold; at least one charge and one discharge_hard). If it
    doesn't, main() reports the delta and exits before touching phase_2/phase_3
    -- a transfer number computed through a harness that can't reproduce its
    own control condition is not a transfer number, it's noise.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

try:
    from citylearn.citylearn import CityLearnEnv
except ImportError as _e:
    sys.exit(
        f"Could not import CityLearn: {_e}\n"
        "This is usually a missing dependency of CityLearn itself (torch, "
        "platformdirs), not CityLearn.\nRun: pip install -r requirements.txt"
    )

from rbc_v5 import RBCv5, RBCv5Params

START_SOC = 0.50


# ----------------------------------------------------------------------------
# Environment plumbing -- mirrors run_fullyear_rbc.py exactly (the script that
# actually reproduces the phase_1 reference numbers). Do not invent a second
# extraction path; if this drifts from run_fullyear_rbc.py, copy it again.
# ----------------------------------------------------------------------------

def resolve_schema(schema_name):
    """Prefer the already-cached local schema.json over the bare dataset name.

    CityLearnEnv(bare_name) calls DataSet.get_dataset_names(), which hits
    GitHub's API even when the schema is already cached locally -- and that
    API is rate-limited (hit during this session's debugging). Passing a
    filesystem path instead takes a different branch in CityLearnEnv's schema
    setter (os.path.isfile check, first in line) that never touches the
    network. Falls back to the bare name so first-ever download still works.
    """
    try:
        from citylearn.data import DataSet
        cache_dir = DataSet().cache_directory
        local_path = os.path.join(cache_dir, "datasets", schema_name, "schema.json")
        if os.path.isfile(local_path):
            return local_path
    except Exception:
        pass
    return schema_name


def make_env(schema):
    return CityLearnEnv(resolve_schema(schema), central_agent=True)


def flat_names(env):
    on = env.observation_names
    if on and isinstance(on[0], (list, tuple)):
        return [n for grp in on for n in grp]
    return list(on)


def reset_env(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def step_env(env, actions):
    out = env.step(actions)
    done = (out[2] or out[3]) if len(out) == 5 else out[2]
    return out[0], done


def unwrap(obs):
    if len(obs) == 1 and isinstance(obs[0], (list, tuple, np.ndarray)):
        return list(obs[0])
    return list(obs)


def build_index(names):
    """Name -> index map, built fresh from THIS schema's observation_names.

    Never hardcode positions: phase_1/phase_2 have 5 buildings, phase_3 has 7,
    so solar_generation / non_shiftable_load each occur a different number of
    times in the flattened observation list depending on which schema this is.
    occ() finds every occurrence, in building order, regardless of count.
    """
    def occ(n): return [i for i, x in enumerate(names) if x == n]
    return {"hour": names.index("hour") if "hour" in names else None,
            "price": names.index("electricity_pricing") if "electricity_pricing" in names else None,
            "solar": occ("solar_generation"),
            "load": occ("non_shiftable_load")}


def set_start_soc(env, soc):
    for bldg in env.buildings:
        bat = bldg.electrical_storage
        try:
            bat.force_set_soc(soc)
        except Exception:
            try:
                bat.initial_soc = soc * (bat.capacity or 1.0)
            except Exception:
                pass


def true_soc(env, b, t_step):
    """Read SOC off the battery object, not the observation vector.

    The electrical_storage_soc observation reads 0.000 in the 2022 phase 1
    schema even when the battery is full. bat.soc is the preallocated array for
    the whole episode, so index at t-1 (the state entering this step). Do NOT
    use soc[-1] -- that is the year-end slot, which is zero until the run ends.
    """
    bat = env.buildings[b].electrical_storage
    arr = np.asarray(bat.soc, dtype=float)
    if arr.size:
        return float(arr[0 if t_step == 0 else min(t_step - 1, arr.size - 1)])
    cap = getattr(bat, "capacity", 1.0) or 1.0
    v = float(getattr(bat, "initial_soc", 0.0))
    return v / cap if v > 1.0 else v


def building_state(env, b, t_step):
    """DIAGNOSTIC ONLY -- not used by run(). Reads the building attribute
    arrays directly, for verify_net_identity()'s cross-check below.

    These arrays are appended to lazily as a side effect of step(): the
    newest entry is an unfinalized placeholder until the NEXT step() call,
    so reading [t_step] here (or anywhere) does not, by itself, tell you
    which wall-clock hour you actually read. That's exactly why run() below
    uses the observation vector instead. This function stays only so
    verify_net_identity() can still check the arithmetic relationship
    between the two CityLearn properties.
    """
    bldg = env.buildings[b]
    load = float(bldg.net_electricity_consumption_without_storage_and_pv[t_step])
    solar = float(abs(bldg.solar_generation[t_step]))
    return load, solar, true_soc(env, b, t_step)


def verify_net_identity(env, t_step=0, tol=1e-6):
    """load - solar must reproduce CityLearn's own net-without-storage.

    NECESSARY, NOT SUFFICIENT: this validates the arithmetic relationship
    between net_electricity_consumption_without_storage_and_pv and
    net_electricity_consumption_without_storage, which holds at ANY index
    (finalized or not -- CityLearn evidently updates both properties
    together). It says nothing about whether t_step is the wall-clock hour
    you think it is. It would have passed even with the lazy-array lag bug
    that made an earlier version of this script collapse to all-hold.
    """
    bad = []
    for b in range(len(env.buildings)):
        bldg = env.buildings[b]
        load, solar, _ = building_state(env, b, t_step)
        expect = float(bldg.net_electricity_consumption_without_storage[t_step])
        if abs((load - solar) - expect) > tol:
            bad.append((b, round(load - solar, 4), round(expect, 4)))
    return bad


def kpis(env):
    """Pull district-level KPIs out of env.evaluate() as a flat dict."""
    df = env.evaluate()
    if hasattr(df, "columns") and "level" in df.columns:
        df = df[df["level"] == "district"]
    out = {}
    for _, row in df.iterrows():
        name = row.get("cost_function", row.get("name"))
        val = row.get("value")
        if name is not None and val is not None and not (
            isinstance(val, float) and math.isnan(val)
        ):
            out[str(name)] = float(val)
    return out


def run(schema, controller, hours):
    """One episode. controller=None means do-nothing.

    Mirrors run_fullyear_rbc.py's run() exactly: observation vector for
    hour/solar/load, true_soc() for SOC, controller.act(o) for the action.
    """
    env = make_env(schema)
    names = flat_names(env)
    idx = build_index(names)
    n = len(env.buildings)
    n_act = int(np.prod(env.action_space[0].shape))
    obs = reset_env(env)
    set_start_soc(env, START_SOC)
    limit = hours if hours else (env.time_steps - 1)

    t, done = 0, False
    while t < limit and not done:
        flat = unwrap(obs)
        if controller is None:
            av = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            av = []
            for b in range(n):
                o = {"hour": hour,
                     "electrical_storage_soc": true_soc(env, b, t),
                     "solar_generation": flat[idx["solar"][b]],
                     "non_shiftable_load": flat[idx["load"][b]],
                     "building_id": b, "n_buildings": n}
                av.append(float(controller.act(o)))
        obs, done = step_env(env, [av])
        t += 1

    mode_counts = dict(controller.mode_counts) if controller is not None else {}
    return kpis(env), n, t, mode_counts


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

COST = "cost_total"
CARB = "carbon_emissions_total"
RAMP = "ramping_average"
LOADF = "monthly_one_minus_load_factor_average"


def pick(d, key):
    """KPI key names have drifted across CityLearn versions; match loosely."""
    if key in d:
        return d[key]
    for k, v in d.items():
        if key.replace("_average", "") in k or k in key:
            return v
    return None


def normalise(rbc, base):
    """RBC KPIs divided by the do-nothing KPIs from the SAME schema.

    Normalising against phase 1's baseline would be meaningless -- different
    buildings, different absolute magnitudes.
    """
    out = {}
    for k, v in rbc.items():
        b = base.get(k)
        if b and abs(b) > 1e-12:
            out[k] = v / b
    return out


def challenge_kpis(norm):
    c = pick(norm, COST)
    g = pick(norm, CARB)
    r = pick(norm, RAMP)
    lf = pick(norm, LOADF)
    d = (r + lf) / 2 if (r is not None and lf is not None) else None
    return c, g, d


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -  "


# ----------------------------------------------------------------------------

PHASES = [
    "citylearn_challenge_2022_phase_1",
    "citylearn_challenge_2022_phase_2",
    "citylearn_challenge_2022_phase_3",
]

PARTICIPANT_MEDIAN = {"C": 0.792, "G": 0.940, "D": 0.996}

# phase_1 is the control condition -- run_fullyear_rbc.py's reference result.
GATE_TARGET = {"C": 0.807, "G": 0.878, "D": 0.874}
GATE_TOL = 0.005


def check_gate(c, g, d, modes):
    """Returns a list of problems; empty list means the gate passed."""
    problems = []
    total = sum(modes.values()) if modes else 0
    if total == 0:
        problems.append("no decisions recorded at all")
    elif modes.get("hold", 0) == total:
        problems.append(f"all {total} decisions were 'hold' -- controller never engaged")
    else:
        if modes.get("charge", 0) == 0:
            problems.append("zero 'charge' decisions across the run")
        if modes.get("discharge_hard", 0) == 0:
            problems.append("zero 'discharge_hard' decisions across the run")
    for label, val in (("C", c), ("G", g), ("D", d)):
        target = GATE_TARGET[label]
        if val is None:
            problems.append(f"{label}=None (target {target:.3f})")
        elif abs(val - target) > GATE_TOL:
            problems.append(f"{label}={val:.3f} vs target {target:.3f} "
                            f"(delta {val - target:+.3f}, tol {GATE_TOL})")
    return problems


def preflight(schemas):
    print("PREFLIGHT -- checking schemas load and reporting their shape\n")
    for s in schemas:
        try:
            env = make_env(s)
            reset_env(env)
            n = len(env.buildings)
            steps = env.time_steps
            soc_obs_broken = True
            try:
                bat = env.buildings[0].electrical_storage
                arr = np.asarray(bat.soc, dtype=float)
                soc_obs_broken = bool(arr.size and arr[0] == 0.0)
            except Exception:
                pass
            print(f"  OK  {s}")
            print(f"        buildings = {n}   time_steps = {steps}")
            print(f"        SOC at t=0 reads zero: {soc_obs_broken} "
                  f"({'true_soc() workaround needed' if soc_obs_broken else 'observation looks usable'})")
            bad = verify_net_identity(env)
            print(f"        net identity (load - solar == net_without_storage): "
                  f"{'CLEAN' if not bad else 'MISMATCH ' + str(bad)}  [necessary, not sufficient]")
            names = flat_names(env)
            idx = build_index(names)
            idx_ok = len(idx["solar"]) == n and len(idx["load"]) == n
            print(f"        observation index: solar x{len(idx['solar'])}  "
                  f"load x{len(idx['load'])}  (expect x{n} each)  "
                  f"{'OK' if idx_ok else 'MISMATCH'}")
        except Exception as e:
            print(f"  FAIL {s}\n        {type(e).__name__}: {e}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", nargs="*", default=PHASES)
    ap.add_argument("--hours", type=int, default=0,
                    help="0 = full year. Use 720 for a quick smoke test.")
    ap.add_argument("--preflight", action="store_true",
                    help="Only check that schemas load; run nothing.")
    ap.add_argument("--out", default="results_transfer.json")
    args = ap.parse_args()

    if args.preflight:
        preflight(args.schemas)
        return

    results = {}
    for schema in args.schemas:
        print(f"\n=== {schema} ===")
        t0 = time.time()
        try:
            base_kpi, n, steps, _ = run(schema, None, args.hours)
            print(f"  do-nothing   {n} buildings, {steps} steps")
            rbc_kpi, _, _, modes = run(schema, RBCv5(RBCv5Params()), args.hours)
            print(f"  RBC v5       modes: {modes}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results[schema] = {"error": f"{type(e).__name__}: {e}"}
            continue

        norm = normalise(rbc_kpi, base_kpi)
        c, g, d = challenge_kpis(norm)
        results[schema] = {
            "buildings": n,
            "steps": steps,
            "raw_do_nothing": base_kpi,
            "raw_rbc": rbc_kpi,
            "normalised_rbc": norm,
            "C": c, "G": g, "D": d,
            "mode_counts": modes,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  C={fmt(c)}  G={fmt(g)}  D={fmt(d)}   ({time.time()-t0:.0f}s)")

        if schema == PHASES[0]:
            problems = check_gate(c, g, d, modes)
            results[schema]["gate"] = {"passed": not problems, "problems": problems}
            with open(args.out, "w") as f:
                json.dump({"params": asdict(RBCv5Params()), "results": results}, f, indent=2)
            if problems:
                print("\n" + "!" * 72)
                print("!!  GATE FAILED on phase_1 (the control condition)")
                print("!!  target: C={:.3f} G={:.3f} D={:.3f} (tol {})".format(
                    GATE_TARGET["C"], GATE_TARGET["G"], GATE_TARGET["D"], GATE_TOL))
                print("!!  got:    C={} G={} D={}".format(fmt(c), fmt(g), fmt(d)))
                for p in problems:
                    print(f"!!    - {p}")
                print("!!")
                print("!!  Stopping. phase_2/phase_3 numbers are NOT computed or reported --")
                print("!!  a harness that can't reproduce its own control condition cannot")
                print("!!  be trusted to measure transfer on data it's never seen.")
                print("!" * 72)
                print(f"\nwrote {args.out} (phase_1 only, gate failed)")
                sys.exit(1)
            else:
                print("  [GATE PASSED] phase_1 reproduces the control condition -- "
                      "continuing to remaining schemas.")

    with open(args.out, "w") as f:
        json.dump({"params": asdict(RBCv5Params()), "results": results}, f, indent=2)

    # ---- table ----
    print("\n" + "=" * 74)
    print("RBC v5 TRANSFER  --  normalised against do-nothing on each schema")
    print("=" * 74)
    print(f"{'schema':<38}{'bldgs':>6}{'C':>9}{'G':>9}{'D':>9}")
    print("-" * 74)
    for s, r in results.items():
        if "error" in r:
            print(f"{s.replace('citylearn_challenge_2022_',''):<38}{'':>6}  {r['error'][:30]}")
            continue
        tag = s.replace("citylearn_challenge_2022_", "")
        if s == PHASES[0]:
            tag += "  (tuned on)"
        print(f"{tag:<38}{r['buildings']:>6}{fmt(r['C']):>9}{fmt(r['G']):>9}{fmt(r['D']):>9}")
    print("-" * 74)
    m = PARTICIPANT_MEDIAN
    print(f"{'2022 participant median (Phase III)':<38}{'':>6}{m['C']:>9.3f}{m['G']:>9.3f}{m['D']:>9.3f}")
    print("=" * 74)

    ok = [r for r in results.values() if "error" not in r and r.get("C")]
    if len(ok) > 1:
        base = ok[0]
        print("\nREAD THIS:")
        for s, r in list(results.items())[1:]:
            if "error" in r or not r.get("C"):
                continue
            dc = r["C"] - base["C"]
            dd = (r["D"] - base["D"]) if (r.get("D") and base.get("D")) else None
            verdict = ("holds up" if dc < 0.02 else
                       "degrades" if dc < 0.08 else "degrades badly")
            print(f"  {s.replace('citylearn_challenge_2022_','')}: "
                  f"cost {dc:+.3f}" + (f", grid {dd:+.3f}" if dd is not None else "") +
                  f"  -> {verdict}")
        print("\n  Degradation on unseen buildings is a RESULT, not a failure. It")
        print("  quantifies how much of the RBC's performance came from tuning, which")
        print("  is exactly the per-building cost an LLM controller claims to remove.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
