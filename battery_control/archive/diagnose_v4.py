"""
Diagnostic: is the battery actually changing the district net-load profile?
Compares RAW district net consumption (not normalized KPIs) for v0 vs v4, and
reports how much energy the batteries moved. Run:

    python diagnose_v4.py
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv
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
            "soc": occ("electrical_storage_soc"),
            "solar": occ("solar_generation"),
            "load": occ("non_shiftable_load")}


def district_net(env):
    """Raw district net electricity consumption per timestep (kWh)."""
    try:
        n = np.array(env.net_electricity_consumption, dtype=float)
        if n.ndim > 1:
            n = n.sum(axis=0)
        if n.size > 5:
            return n
    except Exception:
        pass
    return np.sum([np.array(b.net_electricity_consumption, dtype=float)
                   for b in env.buildings], axis=0)


def run(controller=None):
    env = make_env()
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(idx["soc"])
    n_act = int(np.prod(env.action_space[0].shape))
    obs = reset_env(env)
    done = False
    throughput = 0.0
    while not done:
        flat = unwrap(obs)
        if controller is None:
            av = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            price = flat[idx["price"]] if idx["price"] is not None else None
            av = []
            for b in range(n_b):
                o = {"hour": hour, "electrical_storage_soc": flat[idx["soc"][b]],
                     "solar_generation": flat[idx["solar"][b]],
                     "non_shiftable_load": flat[idx["load"][b]],
                     "building_id": b, "n_buildings": n_b}
                if price is not None:
                    o["electricity_pricing"] = price
                a = float(controller.act(o))
                av.append(a)
                throughput += abs(a)
        obs, done = step_env(env, [av])
    return district_net(env), throughput


def stats(net):
    d = np.diff(net)
    return dict(peak=float(net.max()), mean=float(net.mean()),
               std=float(net.std()), ramping=float(np.abs(d).sum()),
               steps=int(net.size))


def main():
    # obs units check
    env = make_env()
    names = flat_names(env)
    flat = unwrap(reset_env(env))
    print("=== sample observation values (check units) ===")
    for n in ["hour", "non_shiftable_load", "solar_generation",
              "electrical_storage_soc", "electricity_pricing"]:
        if n in names:
            print(f"  {n:28s} = {flat[names.index(n)]}")

    print("\n=== running v0 and v4 (raw district net consumption) ===")
    net0, _ = run(None)
    net4, thru = run(RBCv4())
    s0, s4 = stats(net0), stats(net4)

    print(f"\n{'metric':<12}{'v0':>12}{'v4':>12}{'change %':>12}")
    for k in ["peak", "mean", "std", "ramping"]:
        a, b = s0[k], s4[k]
        pct = 100 * (b - a) / a if a else 0
        print(f"{k:<12}{a:>12.3f}{b:>12.3f}{pct:>11.1f}%")
    print(f"\nbattery total throughput (sum |action|): {thru:.1f}")
    print(f"episode length: {s0['steps']} steps")
    print("\nIf peak/std/ramping barely change, the battery isn't affecting the "
          "profile — likely an action-scaling or obs-units issue, not strategy.")


if __name__ == "__main__":
    main()
