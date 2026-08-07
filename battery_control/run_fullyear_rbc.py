"""
Full-year RBC v5 vs do-nothing. No LLM calls — pure Python, ~1-2 minutes.

Produces the headline table: does the rule-based controller beat an idle battery
over a complete year (8759 hourly steps, 5 buildings)?

    python run_fullyear_rbc.py

Notes:
  * Batteries start at 50% SOC (same convention as the 48 h comparison).
  * Uses true SOC from the battery object — the electrical_storage_soc
    observation is broken in this dataset (reads 0.000 while the battery is full).
  * warmup_steps=6 is negligible over 8759 steps (0.07%).
"""
import time
import numpy as np
from citylearn.citylearn import CityLearnEnv
from config import SCHEMA
from rbc_v5 import RBCv5, RBCv5Params

START_SOC = 0.50
PARAMS = RBCv5Params()


def make_env():
    return CityLearnEnv(SCHEMA, central_agent=True)


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
    bat = env.buildings[b].electrical_storage
    arr = np.asarray(bat.soc, dtype=float)
    if arr.size:
        i = 0 if t_step == 0 else min(t_step - 1, arr.size - 1)
        return float(arr[i])
    cap = getattr(bat, "capacity", 1.0) or 1.0
    v = float(getattr(bat, "initial_soc", 0.0))
    return v / cap if v > 1.0 else v


def kpis(env):
    df = env.evaluate()
    d = df[df["level"] == "district"] if "level" in df.columns else df
    out = {}
    for _, row in d.iterrows():
        name = row.get("cost_function", row.get("name", ""))
        val = row.get("value", None)
        if name and val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[name] = val
    return out


def run(use_rbc):
    env = make_env()
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(env.buildings)
    n_act = int(np.prod(env.action_space[0].shape))
    obs = reset_env(env)
    set_start_soc(env, START_SOC)
    ctrl = RBCv5(PARAMS) if use_rbc else None

    t, done = 0, False
    while not done:
        flat = unwrap(obs)
        if ctrl is None:
            av = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            av = []
            for b in range(n_b):
                o = {"hour": hour,
                     "electrical_storage_soc": true_soc(env, b, t),
                     "solar_generation": flat[idx["solar"][b]],
                     "non_shiftable_load": flat[idx["load"][b]],
                     "building_id": b, "n_buildings": n_b}
                av.append(float(ctrl.act(o)))
        obs, done = step_env(env, [av])
        t += 1
        if t % 2000 == 0:
            print(f"    ... {t} steps")

    return kpis(env), t, (ctrl.mode_counts if ctrl else None), \
           float(np.mean([true_soc(env, b, t) for b in range(n_b)]))


def main():
    t0 = time.time()
    print("Full-year RBC v5 vs do-nothing (no LLM calls)")
    print("\n[1/2] do-nothing ...")
    k0, steps, _, soc0_end = run(False)
    print(f"[2/2] RBC v5 ... ({steps} steps)")
    k1, _, modes, soc1_end = run(True)

    keys = sorted(set(k0) | set(k1))
    show = [k for k in keys if not k.startswith("discomfort") and "unserved" not in k]

    print("\n" + "=" * 70)
    print(f"FULL YEAR ({steps} steps, 5 buildings, start SOC {START_SOC:.0%})")
    print("=" * 70)
    print(f"{'KPI (lower = better)':<40}{'v0':>10}{'RBC v5':>11}{'verdict':>9}")
    print("-" * 70)
    wins = losses = ties = 0
    for k in show:
        a, b_ = k0.get(k), k1.get(k)
        if a is None or b_ is None:
            continue
        diff = b_ - a
        if abs(diff) < 1e-4:
            v = "tie"; ties += 1
        elif diff < 0:
            v = "BETTER"; wins += 1
        else:
            v = "worse"; losses += 1
        print(f"{k[:40]:<40}{a:>10.3f}{b_:>11.3f}{v:>9}")
    print("=" * 70)
    print(f"RBC v5 better on {wins}, worse on {losses}, tied on {ties}")
    print(f"end SOC — v0 {soc0_end:.2f}   RBC {soc1_end:.2f}   (started {START_SOC:.2f})")
    print(f"RBC decisions: {modes}")
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
