"""
Confirm the battery action SIGN. Uses the correct shape [[a,a,a,a,a]] and applies
positive then negative actions, printing the SOC series for building 0.

    python probe_sign.py
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv

SCHEMA = "citylearn_challenge_2022_phase_1"


def reset_env(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def soc_series(env, tail=8):
    try:
        s = np.array(env.buildings[0].electrical_storage.soc, dtype=float)
        return [round(float(x), 3) for x in s[-tail:]]
    except Exception as e:
        return f"err {e}"


def run_sign(val, steps=8):
    env = CityLearnEnv(SCHEMA, central_agent=True)
    n_act = int(np.prod(env.action_space[0].shape))
    reset_env(env)
    for _ in range(steps):
        env.step([[val] * n_act])
    print(f"\naction = {val:+.1f} for {steps} steps")
    print(f"  building0 SOC series (tail): {soc_series(env)}")
    # also report district net consumption mean over the run
    try:
        net = np.array(env.net_electricity_consumption, dtype=float)
        if net.ndim > 1:
            net = net.sum(axis=0)
        print(f"  district net consumption mean: {net.mean():.3f}")
    except Exception as e:
        print(f"  net read err: {e}")


def main():
    print("Testing battery action sign with the correct shape [[a,...]]")
    run_sign(+0.9)
    run_sign(-0.9)
    run_sign(0.0)
    print("\nWhichever sign RAISES SOC from 0 is 'charge'. If neither moves it, "
          "the battery object needs a closer look.")


if __name__ == "__main__":
    main()
