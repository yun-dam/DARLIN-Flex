"""
Week 6 ladder: do-nothing vs RBC vs LLM(L1/L2/L3) vs SAC.

Runs ONE arm at a time and appends to results_ladder.json, so long LLM arms can
be run across sessions without losing work. Re-run with --table any time to see
everything collected so far.

    python run_ladder.py --arm v0            # instant
    python run_ladder.py --arm rbc           # instant
    python run_ladder.py --arm L1            # ~35 min at 24 h
    python run_ladder.py --arm L2 --district # with district context
    python run_ladder.py --arm L3
    python run_ladder.py --table             # print everything collected

Options:
    --hours N        horizon (default 24; 48 doubles the LLM cost)
    --district       include MARLISA-style district context in the prompt
    --mock           instant, no network calls — verifies plumbing
    --timeout N      request timeout in seconds (default 300s, tuned on an
                     M4 — raise this on slower hardware; see check_setup.py)
    --provider NAME  ollama (default) | openai | anthropic — see providers.py.
                     openai/anthropic read their key from OPENAI_API_KEY /
                     ANTHROPIC_API_KEY; never pass a key on the command line.
    --model NAME     override the provider's default model
    --max-tokens N   generation cap (num_predict / max_tokens); default 2000
    --reason-words N cap on the model's stated reason, default 8 (tuned for
                     an 8B model; a stronger model may not need it this tight)
    --no-examples    omit the L1 few-shot examples (they anchor small models
                     but may over-constrain a stronger one)
    --tag NAME       label this run (default = arm name + flags)

Guidance levels (the "freedom" axis):
    L1 thresholds handed over   -> tests execution   (imitation)
    L2 baseline only            -> must calibrate
    L3 raw + 6 h history        -> must infer baseline AND calibrate
"""
import argparse
import json
import os
import time
import numpy as np
from citylearn.citylearn import CityLearnEnv
from config import SCHEMA
from rbc_v5 import RBCv5, RBCv5Params
from llm_controller import LLMController, DEFAULT_TIMEOUT, DEFAULT_TOKEN_LIMIT

START_SOC = 0.50
RESULTS = "results_ladder.json"
PARAMS = RBCv5Params()

KEY_KPIS = ["ramping_average", "daily_peak_average", "all_time_peak_average",
            "cost_total", "carbon_emissions_total",
            "electricity_consumption_total",
            "daily_one_minus_load_factor_average", "zero_net_energy"]

PARSE_FAILURE_WARN_THRESHOLD = 0.10  # above this, results are likely just RBC in disguise


def warn_parse_failures(tag, parse_failures, calls):
    """A high parse-failure rate means the controller silently fell back to the
    RBC action almost every hour — the run *looks* like clean LLM output but
    isn't. Loud on purpose: a 94% failure rate produced no visible signal
    otherwise (see L0_free/L0_dev1/L3_series, num_predict=400)."""
    if not calls:
        return
    rate = parse_failures / calls
    if rate > PARSE_FAILURE_WARN_THRESHOLD:
        print(f"\n{'!' * 72}\n"
              f"!!  WARNING: {tag} — {rate:.0%} PARSE FAILURE RATE "
              f"({parse_failures}/{calls} calls)\n"
              f"!!  Controller fell back to the RBC action on nearly every step.\n"
              f"!!  KPIs / net_series for this run are RBC in disguise, not real "
              f"LLM behavior.\n"
              f"{'!' * 72}\n")


# ---------------------------------------------------------------- env utils --- #
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
    def first(n): return names.index(n) if n in names else None
    return {"hour": names.index("hour") if "hour" in names else None,
            "price": names.index("electricity_pricing") if "electricity_pricing" in names else None,
            "solar": occ("solar_generation"),
            "load": occ("non_shiftable_load"),
            "price_1": first("electricity_pricing_predicted_1"),
            "price_2": first("electricity_pricing_predicted_2"),
            "price_3": first("electricity_pricing_predicted_3"),
            "sun_0":   first("direct_solar_irradiance"),
            "sun_1":   first("direct_solar_irradiance_predicted_1"),
            "sun_2":   first("direct_solar_irradiance_predicted_2"),
            "sun_3":   first("direct_solar_irradiance_predicted_3")}


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
    """The electrical_storage_soc OBSERVATION is broken in this dataset
    (reads 0.000 while the battery is full) — read the battery object."""
    bat = env.buildings[b].electrical_storage
    arr = np.asarray(bat.soc, dtype=float)
    if arr.size:
        return float(arr[0 if t == 0 else min(t - 1, arr.size - 1)])
    cap = getattr(bat, "capacity", 1.0) or 1.0
    v = float(getattr(bat, "initial_soc", 0.0))
    return v / cap if v > 1.0 else v


def district_net_series(env, horizon):
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
            out[name] = float(val)
    return out


# -------------------------------------------------------------------- run --- #
def run_arm(arm, hours, mock, district, max_dev=None, verbose=True, forecast=False,
           timeout=None, provider="ollama", model=None, token_limit=None,
           reason_words=8, few_shot=True):
    env = make_env()
    names = flat_names(env)
    idx = build_index(names)
    n_b = len(env.buildings)
    n_act = int(np.prod(env.action_space[0].shape))
    obs = reset_env(env)
    set_start_soc(env, START_SOC)

    policy = None
    llm = None
    if arm == "rbc":
        policy = RBCv5(PARAMS)
    elif arm in ("L0", "L1", "L2", "L3"):
        llm = LLMController(params=PARAMS, mock=mock,
                            guidance_level=int(arm[1]),
                            district_context=district,
                            max_deviation=max_dev,
                            forecast=forecast,
                            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
                            provider=provider, model=model,
                            token_limit=token_limit if token_limit is not None
                                       else DEFAULT_TOKEN_LIMIT,
                            reason_words=reason_words, few_shot=few_shot)
        policy = llm

    d_ema = None
    n_charging_last = 0
    n_discharging_last = 0
    t0 = time.time()
    t = 0
    for _ in range(hours):
        flat = unwrap(obs)
        if policy is None:
            av = [0.0] * n_act
        else:
            hour = flat[idx["hour"]] if idx["hour"] is not None else 0
            price = flat[idx["price"]] if idx["price"] is not None else 0.0
            nets = [flat[idx["load"][b]] - flat[idx["solar"][b]] for b in range(n_b)]
            dnet = float(sum(nets))
            d_ema = dnet if d_ema is None else 0.05 * dnet + 0.95 * d_ema
            av = []
            for b in range(n_b):
                o = {"hour": hour,
                     "electrical_storage_soc": true_soc(env, b, t),
                     "solar_generation": flat[idx["solar"][b]],
                     "non_shiftable_load": flat[idx["load"][b]],
                     "electricity_pricing": price,
                     "building_id": b, "n_buildings": n_b,
                     "district_net": dnet, "district_base": d_ema,
                     "n_charging_last_hour": n_charging_last,
                     "n_discharging_last_hour": n_discharging_last}
                for k in ("price_1", "price_2", "price_3",
                          "sun_0", "sun_1", "sun_2", "sun_3"):
                    if idx.get(k) is not None:
                        o[k] = flat[idx[k]]
                av.append(float(policy.act(o)))
            n_charging_last = sum(1 for a in av if a > 1e-9)
            n_discharging_last = sum(1 for a in av if a < -1e-9)
            if llm is not None and not mock and verbose:
                lat = llm.latencies[-1] if llm.latencies else 0.0
                print(f"  h{t:3d} [{lat:5.1f}s] {[round(x,2) for x in av]} | "
                      f"{(llm.reasons[-1] if llm.reasons else '')[:44]}")
        obs, done = step_env(env, [av])
        t += 1
        if done:
            break

    net = district_net_series(env, hours)
    rec = {
        "arm": arm, "hours": hours, "district": district, "mock": mock,
        "forecast": forecast,
        "runtime_s": round(time.time() - t0, 1),
        "raw": {"peak": float(net.max()),
                "ramping": float(np.abs(np.diff(net)).sum()),
                "mean": float(net.mean())},
        "kpi": kpis(env),
        "end_soc": float(np.mean([true_soc(env, b, t) for b in range(n_b)])),
        "net_series": [float(x) for x in net],     # hourly district net load
    }
    if llm is not None:
        rec["llm"] = {
            "level": llm.level, "provider": llm.provider_name, "model": llm.model,
            "token_limit": llm.token_limit, "reason_words": llm.reason_words,
            "few_shot": llm.few_shot,
            "calls": llm.calls, "timeouts": llm.timeouts,
            "parse_failures": llm.parse_failures,
            "agreement": round(llm.agreement(), 4),
            "mean_deviation": round(llm.mean_deviation(), 4),
            "compliance": round(llm.compliance(), 4),
            "modes": llm.mode_counts,
            "latency": llm.latency_stats(),
            "top_disagreements": [[f"{a}->{b_}", n] for (a, b_), n
                                  in llm.disagreements.most_common(5)],
            "sample_reasons": llm.reasons[:5],
            "mode_pairs": llm.mode_pairs,   # per-step [llm_mode, rbc_mode]
        }
        rec["llm"]["override_rate"] = round(llm.override_rate(), 4)
        rec["llm"]["overrides"] = llm.overrides
        rec["llm"]["clamped"] = llm.clamped
        rec["llm"]["sample_override_reasons"] = llm.override_reasons[:5]
        warn_parse_failures(arm, llm.parse_failures, llm.calls)
    if policy is not None and arm == "rbc":
        rec["rbc_modes"] = policy.mode_counts
    return rec


# ------------------------------------------------------------------ store --- #
def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save_result(tag, rec):
    data = load_results()
    data[tag] = rec
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nsaved -> {RESULTS} [{tag}]")


def print_table():
    data = load_results()
    if not data:
        print("no results yet")
        return
    tags = list(data.keys())
    print("\n" + "=" * (34 + 12 * len(tags)))
    print("NORMALIZED KPIs (lower = better)")
    print("-" * (34 + 12 * len(tags)))
    print(f"{'KPI':<34}" + "".join(f"{t[:11]:>12}" for t in tags))
    print("-" * (34 + 12 * len(tags)))
    for k in KEY_KPIS:
        row = f"{k[:34]:<34}"
        for t in tags:
            v = data[t].get("kpi", {}).get(k)
            row += f"{v:>12.3f}" if v is not None else f"{'-':>12}"
        print(row)
    print("-" * (34 + 12 * len(tags)))
    for label, path in [("raw ramping", ("raw", "ramping")),
                        ("raw peak", ("raw", "peak")),
                        ("hours", ("hours",)), ("runtime s", ("runtime_s",))]:
        row = f"{label:<34}"
        for t in tags:
            v = data[t]
            for p in path:
                v = v.get(p) if isinstance(v, dict) else None
            row += f"{v:>12.2f}" if isinstance(v, (int, float)) else f"{'-':>12}"
        print(row)
    print("=" * (34 + 12 * len(tags)))

    print("\nLLM DIAGNOSTICS")
    print(f"{'tag':<14}{'lvl':>4}{'agree':>8}{'ovr':>6}{'clmp':>6}{'dev':>7}"
          f"{'compl':>8}{'to':>4}{'med s':>7}")
    for t in tags:
        L = data[t].get("llm")
        if not L:
            continue
        lat = L.get("latency") or {}
        print(f"{t[:14]:<14}{L['level']:>4}{L['agreement']:>8.1%}"
              f"{L.get('override_rate',0):>6.0%}{L.get('clamped',0):>6}"
              f"{L['mean_deviation']:>7.2f}{L['compliance']:>8.0%}"
              f"{L['timeouts']:>4}{(L.get('latency') or {}).get('median',0):>7.1f}")
    for t in tags:
        L = data[t].get("llm")
        if L:
            warn_parse_failures(t, L.get("parse_failures", 0), L.get("calls", 0))
    for t in tags:
        L = data[t].get("llm")
        if L and L.get("top_disagreements"):
            print(f"\n{t} — modes {L['modes']}")
            print(f"  top disagreements vs rule: {L['top_disagreements'][:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["v0", "rbc", "L0", "L1", "L2", "L3"])
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--district", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-dev", type=int, default=None,
                    help="clamp the LLM to within N mode-steps of the RBC "
                         "(0 = pure RBC, 1 = one step of freedom, omit = unclamped)")
    ap.add_argument("--forecast", action="store_true",
                    help="add a price/solar forecast block to the prompt "
                         "(the RBC brain cannot see this)")
    ap.add_argument("--timeout", type=int, default=None,
                    help=f"request timeout in seconds (default {DEFAULT_TIMEOUT}s). "
                         "That default was tuned on an M4 -- on slower hardware a real "
                         "call can silently exceed it and fall back to the rule's action. "
                         "Run check_setup.py first to measure your real latency.")
    ap.add_argument("--provider", choices=["ollama", "openai", "anthropic"], default="ollama",
                    help="LLM backend (see providers.py). openai/anthropic read "
                         "OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment.")
    ap.add_argument("--model", default=None,
                    help="override the provider's default model")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help=f"generation cap: num_predict (ollama) or max_tokens "
                         f"(openai/anthropic); default {DEFAULT_TOKEN_LIMIT}")
    ap.add_argument("--reason-words", type=int, default=8,
                    help="cap on the model's stated reason length; default 8 "
                         "(tuned for an 8B model)")
    ap.add_argument("--no-examples", action="store_true",
                    help="omit the L1 few-shot examples")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    if a.table or not a.arm:
        print_table()
        return

    tag = a.tag or (a.arm + ("_dist" if a.district else "")
                    + (f"_dev{a.max_dev}" if a.max_dev is not None else "")
                    + ("_fc" if a.forecast else "")
                    + (f"_{a.provider}" if a.provider != "ollama" else "")
                    + (f"_{a.model.replace(':', '-').replace('/', '-')}" if a.model else "")
                    + ("_mock" if a.mock else ""))
    est = "instant" if (a.mock or a.arm in ("v0", "rbc")) else \
        f"~{round(5*a.hours*17.6/60)} min"
    print(f"arm={a.arm}  hours={a.hours}  district={a.district}  mock={a.mock}  "
          f"forecast={a.forecast}  provider={a.provider}  model={a.model or '(default)'}  "
          f"({est})")
    rec = run_arm(a.arm, a.hours, a.mock, a.district, a.max_dev, forecast=a.forecast,
                  timeout=a.timeout, provider=a.provider, model=a.model,
                  token_limit=a.max_tokens, reason_words=a.reason_words,
                  few_shot=not a.no_examples)
    save_result(tag, rec)
    print_table()


if __name__ == "__main__":
    main()
