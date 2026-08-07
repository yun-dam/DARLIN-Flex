"""
check_setup.py — verifies the environment before committing to a long LLM run.

    python check_setup.py                              # ollama, qwen3:8b (default)
    python check_setup.py --provider openai             # needs OPENAI_API_KEY
    python check_setup.py --provider anthropic --model claude-opus-4-5

Checks:
  1. CityLearn imports and the schema loads
  2. Provider is reachable/configured:
       ollama     - server responds at localhost:11434 and the model is present
       openai     - OPENAI_API_KEY is set
       anthropic  - ANTHROPIC_API_KEY is set
     (never pass a key on the command line — it's read from the environment)
  3. Times 3 real LLM calls using an actual L3-level prompt (district context
     + 6 h history — the longest prompt shape used in production) and reports
     median latency. A short synthetic prompt understates real latency badly
     (prompt length drives generation length): the 300 s default timeout in
     llm_controller.py (DEFAULT_TIMEOUT) was tuned on an M4 against real
     17-37 s/call production latency, not a trivial prompt. On slower
     hardware -- or a slower/larger model -- a real call can exceed it, which
     does NOT raise an error — it silently falls back to the rule's action
     (see _ask_llm). A median above ~100 s here means that headroom is thin.
     This also doubles as the model-availability check for openai/anthropic:
     a bad --model name fails here with a clear error.
  4. Confirms the electrical_storage_soc observation reads 0.000 while the
     battery object shows a real SOC — the known CityLearn bug this codebase
     works around (see true_soc() in run_ladder.py / README). If a CityLearn
     upgrade ever fixes this, true_soc() still returns the correct value
     either way, so this is informational, not blocking. Provider-independent.

Exits 0 if checks 1-3 pass, 1 otherwise. Check 4 never fails the run.
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import deque

from providers import DEFAULT_MODELS

HOST = "http://localhost:11434"
LATENCY_WARN_S = 100

try:
    from llm_controller import DEFAULT_TIMEOUT
except Exception:
    DEFAULT_TIMEOUT = 300

CHECK_CALL_TIMEOUT_S = DEFAULT_TIMEOUT + 60   # headroom above production's own timeout

_critical = []   # (name, ok) — determines the exit code
_warnings = []   # informational, never fails the run


def check(name, ok, detail=""):
    _critical.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def warn(msg):
    _warnings.append(msg)
    print(f"  [WARN] {msg}")


# --------------------------------------------------------- 1. CityLearn --- #
def check_citylearn():
    print("\n1. CityLearn / schema")
    try:
        from citylearn.citylearn import CityLearnEnv
        from config import SCHEMA
    except Exception as e:
        check("citylearn imports", False, repr(e))
        return None
    check("citylearn imports", True)
    try:
        env = CityLearnEnv(SCHEMA, central_agent=True)
    except Exception as e:
        check("schema loads", False, repr(e))
        return None
    check("schema loads", True, f"{len(env.buildings)} buildings, schema={SCHEMA}")
    return env


# ------------------------------------------------------- 2. provider config --- #
def check_provider_config(provider, model):
    print(f"\n2. Provider config: {provider}")
    if provider == "ollama":
        try:
            with urllib.request.urlopen(HOST + "/api/tags", timeout=5) as r:
                data = json.loads(r.read())
        except Exception as e:
            check("ollama responds at " + HOST, False, repr(e))
            return
        check("ollama responds at " + HOST, True)
        names = [m.get("name", "") for m in data.get("models", [])]
        present = model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)
        check(f"{model} present", present,
             "" if present else f"installed models: {names or '(none)'}")
        return
    env_key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[provider]
    present = bool(os.environ.get(env_key))
    check(f"{env_key} is set", present,
         "" if present else f"export {env_key}=... before using --provider {provider}")
    if present:
        print(f"    model availability for {model!r} is verified by the real call in "
              f"check 3 below (a bad --model name fails there with a clear error).")


# ------------------------------------------------------------ 3. timing --- #
def build_l3_prompt(provider, model):
    """The longest real prompt shape used in production: L3 (raw 6h history,
    no baseline/thresholds given) with district context on. Built via the
    actual LLMController._build_prompt() template — not a hand-rolled string —
    so it stays in sync with the real prompt, and uses the controller's real
    token_limit (2000 by default), not a trivial one that finishes early.

    mock=False on purpose: this needs the REAL backend for a real provider,
    not the mock plumbing path. For cloud providers this also means the
    LLMController constructor itself does the "is my API key valid" check
    (via providers.make_provider) before any call is made."""
    from llm_controller import LLMController
    c = LLMController(provider=provider, model=model, host=HOST,
                      timeout=CHECK_CALL_TIMEOUT_S, mock=False,
                      guidance_level=3, district_context=True)
    c.history[0] = deque([1.85, 2.10, -0.40, 0.65, 3.20, 1.10], maxlen=6)
    o = {"district_net": 9.4, "district_base": 7.1, "n_charging_last_hour": 2}
    prompt = c._build_prompt(o, hour=18, net=2.20, base=1.80, soc=0.55, price=0.22,
                             hi=2.41, mid=1.95, hours_left=1.05, bid=0,
                             rbc_mode="discharge_soft")
    return c, prompt


def check_latency(provider, model):
    print(f"\n3. Timing 3 real LLM calls ({provider}:{model}, L3 + district-context prompt)")
    try:
        c, prompt = build_l3_prompt(provider, model)
    except Exception as e:
        return check("3 LLM calls completed", False, f"could not initialize provider ({repr(e)})")

    lat = []
    for i in range(3):
        try:
            _, dt = c._one_request(prompt)
            lat.append(dt)
            print(f"    call {i + 1}/3: {dt:.1f}s")
        except Exception as e:
            print(f"    call {i + 1}/3: FAILED — {repr(e)}")

    if not lat:
        return check("3 LLM calls completed", False, "all calls failed or timed out")

    median = sorted(lat)[len(lat) // 2]
    ok = check(f"{len(lat)}/3 LLM calls completed", len(lat) == 3, f"median {median:.1f}s")
    if median > LATENCY_WARN_S:
        warn(f"median latency {median:.1f}s exceeds {LATENCY_WARN_S}s. "
             f"llm_controller.py's default timeout ({DEFAULT_TIMEOUT}s) was tuned on an "
             f"M4 running qwen3:8b — a slower/larger model or slower hardware can approach "
             f"or exceed it, which fails SILENTLY (falls back to the rule's action, no "
             f"error). Consider 'python run_ladder.py --timeout <seconds>' with real "
             f"headroom before a long run.")
    return ok


# ------------------------------------------------------ 4. SOC obs bug --- #
def check_soc_bug(env):
    print("\n4. electrical_storage_soc observation (informational, provider-independent)")
    if env is None:
        print("  [SKIP] no env from check 1")
        return
    try:
        import numpy as np
        from run_ladder import (flat_names, reset_env, step_env, unwrap,
                                set_start_soc, true_soc)
        names = flat_names(env)
        if "electrical_storage_soc" not in names:
            warn("electrical_storage_soc not found in this schema's observations "
                 "— cannot check the known-bug status")
            return
        soc_idx = names.index("electrical_storage_soc")
        reset_env(env)
        set_start_soc(env, 0.50)
        n_act = int(np.prod(env.action_space[0].shape))
        t = 0
        obs = None
        for t in range(1, 4):
            obs, done = step_env(env, [[0.15] * n_act])
            if done:
                break
        flat = unwrap(obs)
        obs_val = float(flat[soc_idx])
        true_val = true_soc(env, 0, t)
        bug_present = abs(obs_val) < 1e-9 and true_val > 0.05
        print(f"    observation electrical_storage_soc = {obs_val:.4f}")
        print(f"    true battery SOC (bat.soc array)    = {true_val:.4f}")
        if bug_present:
            print("  [PASS] bug confirmed present — true_soc() workaround is required "
                  "and correctly bypasses it")
        else:
            warn("the observation matched the true battery SOC — either this "
                 "CityLearn version fixed the bug, or the values coincided. "
                 "true_soc() still returns the correct value regardless, so this "
                 "does not block runs, but re-check KEY assumptions in the README.")
    except Exception as e:
        warn(f"could not run the SOC check ({repr(e)})")


# --------------------------------------------------------------- main --- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["ollama", "openai", "anthropic"], default="ollama",
                    help="LLM backend to verify (default: ollama, matches prior behaviour)")
    ap.add_argument("--model", default=None,
                    help="override the provider's default model")
    args = ap.parse_args()
    provider = args.provider
    model = args.model or DEFAULT_MODELS[provider]

    print("=" * 70)
    print("check_setup.py — environment verification before a long run")
    print(f"provider={provider}  model={model}")
    print("=" * 70)

    env = check_citylearn()
    check_provider_config(provider, model)
    check_latency(provider, model)
    check_soc_bug(env)

    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok in _critical if ok)
    n_total = len(_critical)
    for name, ok in _critical:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if _warnings:
        print(f"\n  {len(_warnings)} warning(s) — see [WARN] lines above")
    all_ok = n_pass == n_total
    print("=" * 70)
    print(f"{'PASS' if all_ok else 'FAIL'} — {n_pass}/{n_total} checks passed"
          + (", 0 warnings" if not _warnings else f", {len(_warnings)} warning(s)"))
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
