"""
Horizon sensitivity check — finds the shortest evaluation window at which the
KPIs are trustworthy. No LLM calls; v0 and the RBC are deterministic, so this
is pure compute (~2-4 min total).

    python check_horizon.py

WHY THIS EXISTS
The same RBC scored cost_total 2.450 at 24 h, 0.878 at 48 h and 0.807 over the
full year. A deterministic controller cannot get 3x worse by shortening the
window, so the METRIC is unstable at short horizons, not the controller.

Cause: the five batteries hold ~32 kWh while the district only consumes ~62 kWh
in 24 h. Energy pushed into storage therefore dominates the energy-total KPIs
(cost, carbon, consumption, zero_net_energy), and the benefit of discharging it
lands outside the window. Difference-based KPIs (ramping) and max-based KPIs
(peak) do not have this problem.

This prints each KPI across horizons plus the SOC drift, so you can pick the
shortest window where the numbers have converged — and cite the reason.
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv
from config import SCHEMA
from rbc_v5 import RBCv5, RBCv5Params

START_SOC = 0.50
HORIZONS = [24, 48, 72, 168, 336, 720, 2160, 8759]   # 1 day ... full year
PARAMS = RBCv5Params()

ENERGY_KPIS = ["cost_total", "carbon_emissions_total",
               "electricity_consumption_total", "zero_net_energy"]
SHAPE_KPIS = ["ramping_average", "daily_peak_average", "all_time_peak_average",
              "daily_one_minus_load_factor_average"]


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


def true_soc(env, b, t):
    bat = env.buildings[b].electrical_storage
    arr = np.asarray(bat.soc, dtype=float)
    if arr.size:
        return float(arr[0 if t == 0 else min(t - 1, arr.size - 1)])
    return 0.0


def kpis(env):
    df = env.evaluate()
    d = df[df["level"] == "district"] if "level" in df.columns else df
    out = {}
    for _, row in d.iterrows():
        n = row.get("cost_function", row.get("name", ""))
        v = row.get("value", None)
        if n and v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[n] = float(v)
    return out


def run(hours, use_rbc):
    env = make_env()
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(env.buildings)
    n_act = int(np.prod(env.action_space[0].shape))
    obs = reset_env(env)
    set_start_soc(env, START_SOC)
    ctrl = RBCv5(PARAMS) if use_rbc else None

    t = 0
    for _ in range(hours):
        flat = unwrap(obs)
        if ctrl is None:
            av = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            av = [float(ctrl.act({
                "hour": hour, "electrical_storage_soc": true_soc(env, b, t),
                "solar_generation": flat[idx["solar"][b]],
                "non_shiftable_load": flat[idx["load"][b]],
                "building_id": b, "n_buildings": n_b})) for b in range(n_b)]
        obs, done = step_env(env, [av])
        t += 1
        if done:
            break

    end_soc = float(np.mean([true_soc(env, b, t) for b in range(n_b)]))
    cap = getattr(env.buildings[0].electrical_storage, "capacity", 6.4) or 6.4
    parked_kwh = (end_soc - START_SOC) * cap * n_b
    return kpis(env), t, end_soc, parked_kwh


def main():
    print("Horizon sensitivity — RBC v5 vs do-nothing, no LLM calls\n")
    rows = {}
    for h in HORIZONS:
        k0, steps, _, _ = run(h, False)
        k1, steps, end_soc, parked = run(h, True)
        rows[steps] = dict(k0=k0, k1=k1, end_soc=end_soc, parked=parked)
        print(f"  ran {steps:>5} h   end SOC {end_soc:.2f}   "
              f"energy parked in batteries {parked:+.1f} kWh")

    hs = sorted(rows)
    W = 32 + 9 * len(hs)
    print("\n" + "=" * W)
    print("RBC v5 normalized KPI vs horizon (lower = better; watch for convergence)")
    print("-" * W)
    print(f"{'KPI':<32}" + "".join(f"{h:>9}" for h in hs))
    print("-" * W)

    def line(name):
        row = f"{name[:32]:<32}"
        for h in hs:
            v = rows[h]["k1"].get(name)
            row += f"{v:>9.3f}" if v is not None else f"{'-':>9}"
        return row

    print("  --- energy-total KPIs (suspect at short horizons) ---")
    for k in ENERGY_KPIS:
        print(line(k))
    print("  --- shape KPIs (expected horizon-robust) ---")
    for k in SHAPE_KPIS:
        print(line(k))

    print("-" * W)
    row = f"{'energy parked (kWh)':<32}"
    for h in hs:
        row += f"{rows[h]['parked']:>9.1f}"
    print(row)
    print("=" * W)

    ref = hs[-1]
    print(f"\nDeviation from the {ref} h reference value (want small):")
    print(f"{'KPI':<32}" + "".join(f"{h:>9}" for h in hs[:-1]))
    print("-" * W)
    for k in ENERGY_KPIS + SHAPE_KPIS:
        base = rows[ref]["k1"].get(k)
        if base in (None, 0):
            continue
        row = f"{k[:32]:<32}"
        for h in hs[:-1]:
            v = rows[h]["k1"].get(k)
            row += f"{'-':>9}" if v is None else f"{(v-base)/abs(base)*100:>8.0f}%"
        print(row)
    print("\nPick the shortest horizon where deviations are small (<5% say).")
    print("Below that, compare ONLY the shape KPIs (ramping / peak / load factor).")


if __name__ == "__main__":
    main()
