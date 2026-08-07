"""
Find the action shape CityLearn actually applies. Charges hard for 5 steps in
several candidate formats and reports which one moves the battery SOC.

    python probe_action.py
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv

SCHEMA = "citylearn_challenge_2022_phase_1"


def flat_names(env):
    on = env.observation_names
    if on and isinstance(on[0], (list, tuple)):
        return [n for grp in on for n in grp]
    return list(on)


def reset_env(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def unwrap(obs):
    if len(obs) == 1 and isinstance(obs[0], (list, tuple, np.ndarray)):
        return list(obs[0])
    return list(obs)


def soc_from_obs(env, obs):
    names = flat_names(env)
    flat = unwrap(obs)
    i = [k for k, n in enumerate(names) if n == "electrical_storage_soc"]
    return [round(float(flat[j]), 3) for j in i]


def soc_from_building(env):
    out = []
    for b in env.buildings:
        try:
            out.append(round(float(b.electrical_storage.soc[-1]), 3))
        except Exception:
            out.append(None)
    return out


def try_format(label, make_action):
    env = CityLearnEnv(SCHEMA, central_agent=True)
    n_act = int(np.prod(env.action_space[0].shape))
    n_b = len(env.buildings)
    obs = reset_env(env)
    start = soc_from_obs(env, obs)
    err = None
    for _ in range(5):
        try:
            obs = env.step(make_action(n_act, n_b))[0]
        except Exception as e:
            err = repr(e)[:120]
            break
    end_obs = soc_from_obs(env, obs)
    end_bld = soc_from_building(env)
    moved = err is None and any(abs(a - b) > 1e-4 for a, b in zip(start, end_obs))
    print(f"\n[{label}]")
    if err:
        print(f"  ERROR: {err}")
    print(f"  soc(obs)  start={start}  ->  end={end_obs}")
    print(f"  soc(bldg) end={end_bld}")
    print(f"  >>> {'SOC MOVED — this format works' if moved else 'no change'}")


CHARGE = 0.9

def main():
    print(f"n_act / n_buildings probe (charging at {CHARGE} for 5 steps)")
    try_format("A: [[c,c,c,c,c]]  (list-in-list)",
               lambda na, nb: [[CHARGE] * na])
    try_format("B: [c,c,c,c,c]  (flat list)",
               lambda na, nb: [CHARGE] * na)
    try_format("C: np.array([[c,...]])",
               lambda na, nb: np.array([[CHARGE] * na]))
    try_format("D: [np.array([c,...])]",
               lambda na, nb: [np.array([CHARGE] * na)])
    try_format("E: [[c],[c],[c],[c],[c]]  (per-building lists)",
               lambda na, nb: [[CHARGE] for _ in range(nb)])
    print("\nUse whichever format made SOC MOVE. That's the runner fix.")


if __name__ == "__main__":
    main()
