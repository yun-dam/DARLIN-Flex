"""
PER-BUILDING LLM pilot: compare v0, RBC v4, and qwen3:8b (deciding per building)
over a short horizon on raw district metrics (peak / ramping / mean).

    python run_llm.py

FIRST set MOCK = True (instant, no Ollama) to verify plumbing; then MOCK = False.

Timing: 5 buildings x HORIZON hours of LLM calls. At ~17.6 s/call, 24 h = ~35 min,
48 h = ~70 min. Start with 24. Needs rbc_v4.py, llm_controller.py, this file,
and Ollama running (qwen3:8b) for the real run.
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv
from rbc_v4 import RBCv4
from llm_controller import LLMController

SCHEMA = "citylearn_challenge_2022_phase_1"
HORIZON_HOURS = 24          # 5x calls per hour now — keep short for the first run
MOCK = False                 # True = instant plumbing test; False = real qwen3


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
    bat = env.buildings[b].electrical_storage
    if t_step == 0:
        cap = getattr(bat, "capacity", 1.0) or 1.0
        return float(getattr(bat, "initial_soc", 0.0)) / cap if cap else 0.0
    arr = np.asarray(bat.soc, dtype=float)
    return float(arr[min(t_step - 1, arr.size - 1)])


def district_net(env, horizon):
    try:
        n = np.array(env.net_electricity_consumption, dtype=float)
        if n.ndim > 1:
            n = n.sum(axis=0)
        if n.size >= horizon:
            return n[:horizon]
    except Exception:
        pass
    arr = np.sum([np.array(b.net_electricity_consumption, dtype=float)
                  for b in env.buildings], axis=0)
    return arr[:horizon]


def n_actions(env):
    return int(np.prod(env.action_space[0].shape))


def run_do_nothing(env):
    n_act = n_actions(env)
    reset_env(env)
    for _ in range(HORIZON_HOURS):
        _, done = step_env(env, [[0.0] * n_act])
        if done:
            break


def run_rbc(env):
    names = flat_names(env); idx = build_index(names)
    n_b = len(env.buildings); ctrl = RBCv4()
    obs = reset_env(env); t = 0
    for _ in range(HORIZON_HOURS):
        flat = unwrap(obs)
        hour = flat[idx["hour"]] if idx["hour"] is not None else 0
        av = []
        for b in range(n_b):
            o = {"hour": hour, "electrical_storage_soc": true_soc(env, b, t),
                 "solar_generation": flat[idx["solar"][b]],
                 "non_shiftable_load": flat[idx["load"][b]],
                 "building_id": b, "n_buildings": n_b}
            av.append(float(ctrl.act(o)))
        obs, done = step_env(env, [av]); t += 1
        if done:
            break


def run_llm(env, llm):
    names = flat_names(env); idx = build_index(names)
    n_b = len(env.buildings)
    obs = reset_env(env); t = 0
    for _ in range(HORIZON_HOURS):
        flat = unwrap(obs)
        hour = flat[idx["hour"]] if idx["hour"] is not None else 0
        price = flat[idx["price"]] if idx["price"] is not None else 0.0
        av, modes = [], []
        for b in range(n_b):
            soc = true_soc(env, b, t)
            net = float(flat[idx["load"][b]] - flat[idx["solar"][b]])
            mode = llm.decide(b, hour, net, soc, float(price))   # one call PER building
            av.append(float(llm.execute(mode, soc)))
            modes.append(mode[:4])
        if not MOCK:
            print(f"  hour {int(hour):2d}  modes {modes}")
        obs, done = step_env(env, [av]); t += 1
        if done:
            break


def stats(net):
    d = np.diff(net)
    return dict(peak=float(net.max()), mean=float(net.mean()),
                ramping=float(np.abs(d).sum()))


def main():
    print(f"Per-building LLM pilot: horizon {HORIZON_HOURS} h   "
          f"mode: {'MOCK' if MOCK else 'REAL qwen3:8b'}  "
          f"(~{'instant' if MOCK else str(round(5*HORIZON_HOURS*17.6/60)) + ' min'})")

    e0 = make_env(); run_do_nothing(e0); s0 = stats(district_net(e0, HORIZON_HOURS))
    e1 = make_env(); run_rbc(e1); s1 = stats(district_net(e1, HORIZON_HOURS))

    llm = LLMController(mock=MOCK)
    e2 = make_env()
    print("Running per-building LLM policy ..." + ("" if MOCK else " (slow part)"))
    run_llm(e2, llm); s2 = stats(district_net(e2, HORIZON_HOURS))

    print("\n" + "=" * 62)
    print(f"{'metric (raw kW, lower better)':<26}{'v0':>10}{'RBC v4':>12}{'LLM/bldg':>12}")
    print("-" * 62)
    for k in ["peak", "ramping", "mean"]:
        print(f"{k:<26}{s0[k]:>10.2f}{s1[k]:>12.2f}{s2[k]:>12.2f}")
    print("=" * 62)
    print(f"LLM decisions: {sum(llm.mode_counts.values())}   calls: {llm.calls}   "
          f"parse-failures: {llm.parse_failures}")
    print(f"LLM mode mix: {llm.mode_counts}")
    print("\nvs the district-mode pilot: per-building lets each battery react to "
          "its own load, so ramping/peak should improve toward (or past) RBC v4.")


if __name__ == "__main__":
    main()
