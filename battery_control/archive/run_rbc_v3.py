"""
STEP 4 — run v0 (do-nothing) vs RBC v3, PER BUILDING, print ALL KPIs.

    python run_rbc_v3.py

Passes each building its ID + the building count so the policy can stagger
charging (multi-building coordination). Default dataset already works — just run.
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv
from rbc_v3 import RBCv3

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
    obs = out[0]
    done = (out[2] or out[3]) if len(out) == 5 else out[2]
    return obs, done


def unwrap_obs(obs):
    if len(obs) == 1 and isinstance(obs[0], (list, tuple, np.ndarray)):
        return list(obs[0])
    return list(obs)


def build_index(names):
    def first(n):
        return names.index(n) if n in names else None

    def occurrences(n):
        return [i for i, x in enumerate(names) if x == n]

    return {
        "hour": first("hour"),
        "price": first("electricity_pricing"),
        "soc": occurrences("electrical_storage_soc"),
        "solar": occurrences("solar_generation"),
        "load": occurrences("non_shiftable_load"),
    }


def n_actions(env):
    return int(np.prod(env.action_space[0].shape))


def run_episode(env, controller=None):
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(idx["soc"])
    n_act = n_actions(env)

    if controller is not None and n_act != n_b:
        print(f"  WARNING: {n_act} actions for {n_b} buildings — running do-nothing.")
        controller = None

    obs = reset_env(env)
    done = False
    while not done:
        flat = unwrap_obs(obs)
        if controller is None:
            action_vals = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            price = flat[idx["price"]] if idx["price"] is not None else None
            action_vals = []
            for b in range(n_b):
                o = {
                    "hour": hour,
                    "electrical_storage_soc": flat[idx["soc"][b]],
                    "solar_generation": flat[idx["solar"][b]],
                    "non_shiftable_load": flat[idx["load"][b]],
                    "building_id": b,
                    "n_buildings": n_b,
                }
                if price is not None:
                    o["electricity_pricing"] = price
                action_vals.append(float(controller.act(o)))
        obs, done = step_env(env, [action_vals])

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
    v0 = key_kpis(run_episode(make_env(), controller=None))
    print("Running RBC v3 (staggered coordination) ...")
    v3 = key_kpis(run_episode(make_env(), controller=RBCv3()))

    print("\n" + "=" * 66)
    print(f"{'KPI':<44}{'v0':>8}{'v3':>10}")
    print("-" * 66)
    wins = losses = 0
    for k in sorted(set(v0) | set(v3)):
        a, b = v0.get(k), v3.get(k)
        a_s = f"{a:.3f}" if a is not None else "   -"
        b_s = f"{b:.3f}" if b is not None else "   -"
        flag = ""
        if a is not None and b is not None:
            diff = b - a
            if abs(diff) < 1e-4:
                flag = "  ~tie"
            elif diff < 0:
                flag = "  v3 better"; wins += 1
            else:
                flag = "  v3 worse"; losses += 1
        print(f"{k:<44}{a_s:>8}{b_s:>10}{flag}")
    print("=" * 66)
    print(f"Lower is better.  v3 better on {wins} KPIs, worse on {losses}.")
    print("Targets: all_time_peak_average, daily_peak_average, ramping_average.")


if __name__ == "__main__":
    main()
