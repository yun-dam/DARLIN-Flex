"""
Compare v0 (do-nothing) vs v3 vs v4 — with the SOC-observation bug FIXED.

The dataset's electrical_storage_soc observation reads 0.000 forever even while
the battery charges (verified with probe_storage.py). So we read the TRUE soc
from each building's battery object and feed that to the controller instead.

    python run_rbc_v4.py

Needs rbc_v3.py, rbc_v4.py and this file in the same folder.
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv
from rbc_v3 import RBCv3
from rbc_v4 import RBCv4

SCHEMA = "citylearn_challenge_2022_phase_1"


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


def true_soc(env, b, t_step):
    """Real SOC from the battery object (the observation is broken)."""
    bat = env.buildings[b].electrical_storage
    if t_step == 0:
        cap = getattr(bat, "capacity", 1.0) or 1.0
        return float(getattr(bat, "initial_soc", 0.0)) / cap if cap else 0.0
    arr = np.asarray(bat.soc, dtype=float)
    i = min(t_step - 1, arr.size - 1)
    return float(arr[i])


def n_actions(env):
    return int(np.prod(env.action_space[0].shape))


def run_episode(env, controller=None):
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(env.buildings)
    n_act = n_actions(env)
    if controller is not None and n_act != n_b:
        print(f"  WARNING: {n_act} actions for {n_b} buildings — do-nothing.")
        controller = None

    obs = reset_env(env)
    done = False
    t_step = 0
    while not done:
        flat = unwrap(obs)
        if controller is None:
            av = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            price = flat[idx["price"]] if idx["price"] is not None else None
            av = []
            for b in range(n_b):
                o = {"hour": hour,
                     "electrical_storage_soc": true_soc(env, b, t_step),  # FIXED
                     "solar_generation": flat[idx["solar"][b]],
                     "non_shiftable_load": flat[idx["load"][b]],
                     "building_id": b, "n_buildings": n_b}
                if price is not None:
                    o["electricity_pricing"] = price
                av.append(float(controller.act(o)))
        obs, done = step_env(env, [av])
        t_step += 1
    return env.evaluate()


def key_kpis(df):
    d = df[df["level"] == "district"] if "level" in df.columns else df
    out = {}
    for _, row in d.iterrows():
        name = row.get("cost_function", row.get("name", ""))
        val = row.get("value", None)
        if not name or val is None:
            continue
        if isinstance(val, float) and np.isnan(val):
            continue
        out[name] = val
    return out


def main():
    print("Running v0 (do-nothing) ...")
    v0 = key_kpis(run_episode(make_env(), None))
    print("Running v3 (solar-only, coordinated) ...")
    v3 = key_kpis(run_episode(make_env(), RBCv3()))
    print("Running v4 (net-load flattening) ...")
    v4 = key_kpis(run_episode(make_env(), RBCv4()))

    keys = sorted(set(v0) | set(v3) | set(v4))
    print("\n" + "=" * 80)
    print(f"{'KPI (lower is better)':<40}{'v0':>8}{'v3':>9}{'v4':>9}{'v4 vs v0':>13}")
    print("-" * 80)
    wins = losses = 0
    for k in keys:
        a, c, d = v0.get(k), v3.get(k), v4.get(k)
        fmt = lambda x: f"{x:.3f}" if x is not None else "   -"
        verdict = ""
        if a is not None and d is not None:
            diff = d - a
            if abs(diff) < 1e-4:
                verdict = "~tie"
            elif diff < 0:
                verdict = "v4 BETTER"; wins += 1
            else:
                verdict = "v4 worse"; losses += 1
        print(f"{k:<40}{fmt(a):>8}{fmt(c):>9}{fmt(d):>9}{verdict:>13}")
    print("=" * 80)
    print(f"v4 vs do-nothing:  better on {wins} KPIs, worse on {losses}.")
    print("Targets: all_time_peak_average, daily_peak_average, ramping_average.")


if __name__ == "__main__":
    main()
