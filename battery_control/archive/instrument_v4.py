"""
Instrument a v4 run: count charge/discharge/hold decisions, track the SOC range
the batteries actually reach, and mean action size. Reveals whether the
controller ever discharges.

    python instrument_v4.py
"""
import numpy as np
from collections import Counter
from citylearn.citylearn import CityLearnEnv
from rbc_v4 import RBCv4

SCHEMA = "citylearn_challenge_2022_phase_1"


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
            "soc": occ("electrical_storage_soc"),
            "solar": occ("solar_generation"),
            "load": occ("non_shiftable_load")}


def main():
    env = CityLearnEnv(SCHEMA, central_agent=True)
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(idx["soc"])
    ctrl = RBCv4()

    modes = Counter()
    soc_min = [9.9] * n_b
    soc_max = [-9.9] * n_b
    act_abs_sum = 0.0
    n_act_nonzero = 0

    obs = reset_env(env)
    done = False
    while not done:
        flat = unwrap(obs)
        hour = flat[idx["hour"]] if idx["hour"] is not None else 0
        av = []
        for b in range(n_b):
            soc = float(flat[idx["soc"][b]])
            soc_min[b] = min(soc_min[b], soc)
            soc_max[b] = max(soc_max[b], soc)
            o = {"hour": hour, "electrical_storage_soc": soc,
                 "solar_generation": float(flat[idx["solar"][b]]),
                 "non_shiftable_load": float(flat[idx["load"][b]]),
                 "building_id": b, "n_buildings": n_b}
            m = ctrl.decide_mode(o)
            modes[m] += 1
            a = ctrl.mode_to_action(m, soc)
            if abs(a) > 1e-9:
                act_abs_sum += abs(a); n_act_nonzero += 1
            av.append(float(a))
        obs, done = step_env(env, [av])

    total = sum(modes.values())
    print("=== decision counts over the whole year (all buildings) ===")
    for m in ("charge", "discharge_soft", "discharge_hard", "hold"):
        c = modes.get(m, 0)
        print(f"  {m:16s} {c:8d}   {100*c/total:5.1f}%")
    print(f"\n  total decisions: {total}")
    print(f"  nonzero actions: {n_act_nonzero}   mean |action|: "
          f"{act_abs_sum/max(1,n_act_nonzero):.3f}")
    print("\n=== SOC range actually reached, per building ===")
    for b in range(n_b):
        print(f"  building {b}: soc min {soc_min[b]:.3f}  max {soc_max[b]:.3f}")
    print("\nIf discharge counts are ~0 or SOC max stays low, the controller is "
          "charging but never discharging — that's the bug to fix.")


if __name__ == "__main__":
    main()
