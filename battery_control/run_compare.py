"""
Fair 3-way comparison: do-nothing vs RBC v5 vs LLM (qwen3:8b).

METHODOLOGY (unchanged from the previous run — the comparison is identical):
  * both policies share RBCv5Params -> same warmup, thresholds, executor
  * batteries start at 50% SOC; start/end SOC reported
  * raw kW metrics AND CityLearn normalized KPIs over the same window

NEW DIAGNOSTICS in this version (measurement only — no policy changes):
  * latency distribution (median / p95 / max) — replaces an inferred ~17.6 s
  * timeouts counted SEPARATELY from parse failures (previously conflated)
  * retry-on-timeout, with retries counted
  * SHADOW-RBC AGREEMENT: what the rule would have chosen on the identical
    state, every step -> agreement % + disagreement breakdown. This diagnoses
    WHY the LLM trails the rule instead of leaving it to speculation.
  * RBC's own mode mix, for distribution comparison

    python run_compare.py
"""
import numpy as np
from citylearn.citylearn import CityLearnEnv
from config import SCHEMA
from rbc_v5 import RBCv5, RBCv5Params
from llm_controller import LLMController

HORIZON_HOURS = 48
START_SOC = 0.50
MOCK = False               # set True for an instant plumbing check
PARAMS = RBCv5Params()     # shared by BOTH policies — fairness

# ---------------------------------------------------------------- helpers --- #
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
    """Real SOC — the electrical_storage_soc OBSERVATION is broken in this
    dataset (reads 0.000 while the battery is full). Index the CURRENT step;
    never soc[-1] (that is the year-end preallocated slot)."""
    bat = env.buildings[b].electrical_storage
    arr = np.asarray(bat.soc, dtype=float)
    if arr.size:
        i = 0 if t_step == 0 else min(t_step - 1, arr.size - 1)
        return float(arr[i])
    cap = getattr(bat, "capacity", 1.0) or 1.0
    v = float(getattr(bat, "initial_soc", 0.0))
    return v / cap if v > 1.0 else v


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


def kpis(env):
    try:
        df = env.evaluate()
    except Exception:
        return {}
    d = df[df["level"] == "district"] if "level" in df.columns else df
    out = {}
    for _, row in d.iterrows():
        name = row.get("cost_function", row.get("name", ""))
        val = row.get("value", None)
        if name and val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[name] = val
    return out


# ------------------------------------------------------------------- runs --- #
def run(policy):
    env = make_env()
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(env.buildings)
    n_act = n_actions(env)
    obs = reset_env(env)
    set_start_soc(env, START_SOC)
    soc0 = [true_soc(env, b, 0) for b in range(n_b)]
    if abs(np.mean(soc0) - START_SOC) > 0.02:
        print(f"    NOTE: start SOC {np.mean(soc0):.2f} != {START_SOC:.2f}")

    t = 0
    for _ in range(HORIZON_HOURS):
        flat = unwrap(obs)
        if policy is None:
            av = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            price = flat[idx["price"]] if idx["price"] is not None else 0.0
            av = []
            for b in range(n_b):
                o = {"hour": hour,
                     "electrical_storage_soc": true_soc(env, b, t),
                     "solar_generation": flat[idx["solar"][b]],
                     "non_shiftable_load": flat[idx["load"][b]],
                     "electricity_pricing": price,
                     "building_id": b, "n_buildings": n_b}
                av.append(float(policy.act(o)))
            if isinstance(policy, LLMController) and not MOCK:
                lat = policy.latencies[-1] if policy.latencies else 0.0
                print(f"  h{t:3d} [{lat:5.1f}s] actions {[round(x, 2) for x in av]}"
                      f" | {(policy.reasons[-1] if policy.reasons else '')[:46]}")
        obs, done = step_env(env, [av])
        t += 1
        if done:
            break

    soc1 = [true_soc(env, b, t) for b in range(n_b)]
    return dict(net=district_net(env, HORIZON_HOURS), kpi=kpis(env),
                soc0=float(np.mean(soc0)), soc1=float(np.mean(soc1)))


def stats(net):
    d = np.diff(net)
    return dict(peak=float(net.max()), ramping=float(np.abs(d).sum()),
                mean=float(net.mean()))


# ------------------------------------------------------------------- main --- #
def main():
    est = "instant" if MOCK else f"~{round(5*HORIZON_HOURS*17.6/60)} min"
    print(f"Fair comparison | horizon {HORIZON_HOURS} h | start SOC {START_SOC:.0%} | "
          f"warmup {PARAMS.warmup_steps} (both) | {'MOCK' if MOCK else 'REAL qwen3:8b'} ({est})")

    print("\n[1/3] do-nothing ...")
    r0 = run(None)
    print("[2/3] RBC v5 ...")
    rbc = RBCv5(PARAMS)
    r1 = run(rbc)
    print("[3/3] LLM ...")
    llm = LLMController(params=PARAMS, mock=MOCK)
    r2 = run(llm)

    s0, s1, s2 = stats(r0["net"]), stats(r1["net"]), stats(r2["net"])

    print("\n" + "=" * 74)
    print(f"{'RAW district metrics (kW, lower better)':<34}{'v0':>12}{'RBC v5':>13}{'LLM':>13}")
    print("-" * 74)
    for k, label in [("peak", "peak"), ("ramping", "ramping"), ("mean", "mean draw")]:
        best = min(s0[k], s1[k], s2[k])
        tag = "  <- LLM" if abs(s2[k] - best) < 1e-9 else ("  <- RBC" if abs(s1[k] - best) < 1e-9 else "")
        print(f"{label:<34}{s0[k]:>12.2f}{s1[k]:>13.2f}{s2[k]:>13.2f}{tag}")

    print(f"\n{'SOC (mean over buildings)':<34}{'v0':>12}{'RBC v5':>13}{'LLM':>13}")
    print("-" * 74)
    print(f"{'start SOC':<34}{r0['soc0']:>12.2f}{r1['soc0']:>13.2f}{r2['soc0']:>13.2f}")
    print(f"{'end SOC':<34}{r0['soc1']:>12.2f}{r1['soc1']:>13.2f}{r2['soc1']:>13.2f}")

    keys = sorted(set(r0["kpi"]) | set(r1["kpi"]) | set(r2["kpi"]))
    show = [k for k in keys if not k.startswith("discomfort")
            and "unserved" not in k]
    if show:
        print(f"\n{'NORMALIZED KPI over this window':<34}{'v0':>12}{'RBC v5':>13}{'LLM':>13}")
        print("-" * 74)
        for k in show:
            a, b_, c = r0["kpi"].get(k), r1["kpi"].get(k), r2["kpi"].get(k)
            f = lambda x: f"{x:.3f}" if x is not None else "     -"
            print(f"{k[:34]:<34}{f(a):>12}{f(b_):>13}{f(c):>13}")

    # ---------------- LLM diagnostics ---------------- #
    print("\n" + "=" * 74)
    print("LLM DIAGNOSTICS")
    print("-" * 74)
    lat = llm.latency_stats()
    if lat:
        print(f"  latency (s): median {lat['median']:.1f}  mean {lat['mean']:.1f}  "
              f"p95 {lat['p95']:.1f}  max {lat['max']:.1f}   (n={lat['n']})")
    print(f"  calls {llm.calls}   timeouts {llm.timeouts}   "
          f"recovered-by-retry {llm.retry_successes}   malformed replies {llm.parse_failures}")
    print(f"  compliance (charge only on solar surplus): {llm.compliance():.0%}   "
          f"blocked grid-charges: {llm.blocked_grid_charges}")
    print(f"  LLM modes : {llm.mode_counts}")
    print(f"  RBC modes : {rbc.mode_counts}")

    print(f"\n  AGREEMENT with the rule on identical states: {llm.agreement():.1%} "
          f"({llm.agreements}/{llm.decisions_compared})")
    if llm.disagreements:
        print("  top disagreements (LLM chose -> rule would have chosen):")
        for (a, b_), n in llm.disagreements.most_common(6):
            print(f"    {a:>15} -> {b_:<15} {n:4d}")
    if llm.reasons:
        print("\n  sample reasons:")
        for r in llm.reasons[:4]:
            print(f"    - {r}")


if __name__ == "__main__":
    main()
