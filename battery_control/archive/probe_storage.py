"""
Does the battery's INTERNAL soc rise when we force-charge, even though the
observation reads 0? Force +0.5 for 10 steps and inspect the battery object.

    python probe_storage.py
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv

SCHEMA = "citylearn_challenge_2022_phase_1"


def reset_env(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def flat_names(env):
    on = env.observation_names
    if on and isinstance(on[0], (list, tuple)):
        return [n for grp in on for n in grp]
    return list(on)


def unwrap(obs):
    if len(obs) == 1 and isinstance(obs[0], (list, tuple, np.ndarray)):
        return list(obs[0])
    return list(obs)


def main():
    env = CityLearnEnv(SCHEMA, central_agent=True)
    n_act = int(np.prod(env.action_space[0].shape))
    names = flat_names(env)
    soc_obs_i = [i for i, n in enumerate(names) if n == "electrical_storage_soc"][0]

    bat = env.buildings[0].electrical_storage
    print("battery: capacity", bat.capacity, "nominal_power", bat.nominal_power,
          "efficiency", bat.efficiency, "initial_soc", bat.initial_soc)

    obs = reset_env(env)
    print("\nstep | obs_soc | internal_soc(cur) | internal_soc_max | energy_bal(cur)")
    for t in range(10):
        obs = env.step([[0.5] * n_act])[0]
        flat = unwrap(obs)
        soc_arr = np.array(bat.soc, dtype=float)
        eb_arr = np.array(bat.energy_balance, dtype=float)
        # current index: number of steps taken so far
        cur = min(t, soc_arr.size - 1)
        obs_soc = flat[soc_obs_i]
        # find current as last nonzero-context: use env.time_step if available
        ts = getattr(env, "time_step", t + 1)
        cur_i = min(ts, soc_arr.size - 1)
        print(f"{t:4d} | {obs_soc:7.3f} | {soc_arr[cur_i]:16.3f} | "
              f"{soc_arr.max():16.3f} | {eb_arr[cur_i]:.3f}")

    soc_arr = np.array(bat.soc, dtype=float)
    print("\nsoc array first 12:", [round(float(x), 3) for x in soc_arr[:12]])
    print("soc array max over episode:", round(float(soc_arr.max()), 3))
    print("\nIf internal_soc_max > 0 but obs_soc stayed 0, the OBSERVATION is the "
          "bug — fix: read true soc from the battery object in the runner.")


if __name__ == "__main__":
    main()
