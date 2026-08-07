"""
SAC baseline using CityLearn's built-in agent. Writes into results_ladder.json
so it appears alongside v0 / RBC / L1-L3 in `python run_ladder.py --table`.

    python run_sac.py --episodes 5          # start here, see how long it takes
    python run_sac.py --episodes 20         # longer training
    python run_sac.py --episodes 5 --hours 24   # short episodes for a fast check

Notes
  * SAC is trained on the FULL episode by default (CityLearn's own loop), then
    evaluated. Report the training cost honestly — that is the whole point of
    the comparison: the LLM needs zero training, SAC needs episodes of it.
  * CityLearn's agent API has shifted between versions, so imports and the
    learn() signature are probed defensively. If everything fails, the error
    list printed at the end tells us exactly what to adapt.
  * A lightly-trained SAC is expected to UNDERPERFORM a good RBC — Nweye et al.
    (2022) found exactly that on flexibility KPIs. Report episodes trained so
    the comparison is honest rather than a straw man.
"""
import argparse
import json
import os
import time
import numpy as np
from config import SCHEMA

RESULTS = "results_ladder.json"
KEY_KPIS = ["ramping_average", "daily_peak_average", "all_time_peak_average",
            "cost_total", "carbon_emissions_total",
            "electricity_consumption_total",
            "daily_one_minus_load_factor_average", "zero_net_energy"]


def import_sac():
    """Try the known locations of CityLearn's SAC agent."""
    errors = []
    for path, name in [("citylearn.agents.sac", "SAC"),
                       ("citylearn.agents.sac", "SACRBC"),
                       ("citylearn.agents.rlc", "RLC")]:
        try:
            mod = __import__(path, fromlist=[name])
            cls = getattr(mod, name)
            print(f"  using {path}.{name}")
            return cls, errors
        except Exception as e:
            errors.append(f"{path}.{name}: {repr(e)[:90]}")
    return None, errors


def make_env(hours=None):
    from citylearn.citylearn import CityLearnEnv
    kw = dict(central_agent=True)
    if hours:
        # simulation_end_time_step keeps episodes short for a fast check
        try:
            return CityLearnEnv(SCHEMA, simulation_start_time_step=0,
                                simulation_end_time_step=hours - 1, **kw)
        except Exception:
            pass
    return CityLearnEnv(SCHEMA, **kw)


def kpis(env):
    df = env.evaluate()
    d = df[df["level"] == "district"] if "level" in df.columns else df
    out = {}
    for _, row in d.iterrows():
        name = row.get("cost_function", row.get("name", ""))
        val = row.get("value", None)
        if name and val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[name] = float(val)
    return out


def try_learn(model, episodes):
    """CityLearn's learn() signature has varied across versions."""
    attempts = [
        lambda: model.learn(episodes=episodes),
        lambda: model.learn(episodes=episodes, deterministic_finish=True),
        lambda: model.learn(episodes=episodes, deterministic=True),
        lambda: model.learn(),
    ]
    errs = []
    for fn in attempts:
        try:
            fn()
            return True, errs
        except TypeError as e:
            errs.append(repr(e)[:90])
        except Exception as e:
            errs.append(repr(e)[:120])
            break
    return False, errs


def save(tag, rec):
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    data[tag] = rec
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved -> {RESULTS} [{tag}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--hours", type=int, default=None,
                    help="shorten each episode (default: full year)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    print(f"SAC baseline | episodes={a.episodes} | "
          f"episode length={'full year' if not a.hours else a.hours}")

    SACcls, imp_errors = import_sac()
    if SACcls is None:
        print("\nCould not import a SAC agent from CityLearn. Tried:")
        for e in imp_errors:
            print("   ", e)
        print("\nCheck what's available:")
        print("   python -c \"import citylearn.agents as a, pkgutil; "
              "print([m.name for m in pkgutil.iter_modules(a.__path__)])\"")
        return

    env = make_env(a.hours)
    t0 = time.time()
    try:
        model = SACcls(env=env)
    except Exception as e:
        print(f"  constructor SACcls(env=env) failed: {repr(e)[:120]}")
        try:
            model = SACcls(env)
        except Exception as e2:
            print(f"  constructor SACcls(env) failed too: {repr(e2)[:120]}")
            return

    print("  training ...")
    ok, errs = try_learn(model, a.episodes)
    train_s = time.time() - t0
    if not ok:
        print("  learn() failed. Attempts:")
        for e in errs:
            print("   ", e)
        return
    print(f"  trained in {train_s:.0f}s")

    k = kpis(env)
    tag = a.tag or f"SAC_{a.episodes}ep"
    rec = {"arm": "SAC", "episodes": a.episodes,
           "hours": a.hours or "full_year",
           "train_seconds": round(train_s, 1),
           "kpi": k, "raw": {}, "note": "CityLearn built-in SAC, light training"}
    save(tag, rec)

    print(f"\n{'KPI':<38}{'SAC':>10}")
    print("-" * 48)
    for kk in KEY_KPIS:
        v = k.get(kk)
        print(f"{kk[:38]:<38}{v:>10.3f}" if v is not None else f"{kk[:38]:<38}{'-':>10}")
    print("\nRun `python run_ladder.py --table` to see SAC beside v0 / RBC / L1-L3.")


if __name__ == "__main__":
    main()
