#!/usr/bin/env python3
"""10-scenario validation gate for small open-weight LLMs as battery-mode controllers.

Tests whether a local model (via Ollama) produces:
  1. parseable JSON            (parse_ok)
  2. a valid mode              (valid_mode)
  3. SOC-guard-respecting mode (soc_ok)
  4. directionally correct mode (direction_ok)

Usage:
  python run_validation.py --model qwen2.5:7b
  python run_validation.py --model llama3.1:8b --repeats 3
  python run_validation.py --model qwen3:8b --force-json

By default the model's raw output is parsed as-is (this IS the test).
--force-json enables Ollama's format=json constraint so you can measure
how much structured decoding rescues a weak model.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are a battery controller for a grid-interactive building with rooftop PV and a home battery.

Each hour you choose exactly one operating mode:
- "charge": charge the battery (from PV surplus first, then grid)
- "hold": do nothing this hour
- "discharge_soft": discharge at a moderate rate to offset part of the building load
- "discharge_hard": discharge at maximum rate to offset as much load as possible

Control principles, in priority order:
1. SAFETY: never discharge if battery SOC is at or below 0.10; never charge if SOC is at or above 0.95.
2. PEAK SHAVING: during peak pricing hours, discharge to reduce grid draw.
3. PV SELF-CONSUMPTION: when PV generation exceeds building load, store the surplus.
4. CHEAP CHARGING: during the lowest overnight tariff, charge if SOC is low.
5. RESTRAINT: if no principle clearly applies, hold.

Respond with ONLY a JSON object, no other text:
{"mode": "<one of: charge, hold, discharge_soft, discharge_hard>", "reasoning": "<one sentence>"}"""

USER_TEMPLATE = """Current state:
- Hour of day: {hour}
- Electricity price: ${price}/kWh ({price_context})
- PV generation: {pv} kW
- Building load: {load} kW
- Battery SOC: {soc} (capacity {cap} kWh)

Choose the operating mode."""


def build_user_prompt(state: dict) -> str:
    return USER_TEMPLATE.format(
        hour=state["hour"],
        price=state["electricity_price_usd_per_kwh"],
        price_context=state["price_context"],
        pv=state["pv_generation_kw"],
        load=state["building_load_kw"],
        soc=state["battery_soc"],
        cap=state["battery_capacity_kwh"],
    )


def call_ollama(model: str, user_prompt: str, force_json: bool, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    if force_json:
        payload["format"] = "json"
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def extract_json(raw: str):
    """Strip qwen3-style <think> blocks and markdown fences, then find the first JSON object."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def grade(scenario: dict, raw: str, config: dict) -> dict:
    modes = config["modes"]
    guards = config["soc_guards"]
    soc = scenario["state"]["battery_soc"]

    result = {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "raw_output": raw,
        "parse_ok": False,
        "valid_mode": False,
        "soc_ok": False,
        "direction_ok": False,
        "mode": None,
        "reasoning": None,
    }

    parsed = extract_json(raw)
    if parsed is None or "mode" not in parsed:
        return result
    result["parse_ok"] = True
    result["mode"] = parsed.get("mode")
    result["reasoning"] = parsed.get("reasoning")

    if result["mode"] not in modes:
        return result
    result["valid_mode"] = True

    mode = result["mode"]
    discharging = mode in ("discharge_soft", "discharge_hard")
    soc_violation = (discharging and soc <= guards["min_soc_for_discharge"]) or (
        mode == "charge" and soc >= guards["max_soc_for_charge"]
    )
    result["soc_ok"] = not soc_violation

    result["direction_ok"] = mode in scenario["acceptable_modes"]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen2.5:7b")
    ap.add_argument("--repeats", type=int, default=1, help="runs per scenario (consistency check)")
    ap.add_argument("--force-json", action="store_true", help="use Ollama format=json constraint")
    ap.add_argument("--timeout", type=int, default=120, help="per-call timeout in seconds")
    ap.add_argument("--scenarios", default=str(Path(__file__).parent / "scenarios.json"))
    args = ap.parse_args()

    config = json.loads(Path(args.scenarios).read_text())
    scenarios = config["scenarios"]

    all_results = []
    print(f"\nModel: {args.model} | repeats={args.repeats} | force_json={args.force_json}\n")
    header = f"{'#':>2} {'scenario':<34} {'mode':<15} {'parse':<6} {'valid':<6} {'soc':<5} {'dir':<5}"
    print(header)
    print("-" * len(header))

    for sc in scenarios:
        user_prompt = build_user_prompt(sc["state"])
        for rep in range(args.repeats):
            t0 = time.time()
            try:
                raw = call_ollama(args.model, user_prompt, args.force_json, args.timeout)
            except requests.RequestException as e:
                print(f"{sc['id']:>2} {sc['name']:<34} OLLAMA ERROR: {e}")
                sys.exit(1)
            res = grade(sc, raw, config)
            res["repeat"] = rep
            res["latency_s"] = round(time.time() - t0, 1)
            all_results.append(res)
            ok = lambda b: "yes" if b else "NO"
            print(
                f"{sc['id']:>2} {sc['name']:<34} {str(res['mode']):<15} "
                f"{ok(res['parse_ok']):<6} {ok(res['valid_mode']):<6} "
                f"{ok(res['soc_ok']):<5} {ok(res['direction_ok']):<5}"
            )

    n = len(all_results)
    summary = {
        "model": args.model,
        "force_json": args.force_json,
        "repeats": args.repeats,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_calls": n,
        "parse_rate": sum(r["parse_ok"] for r in all_results) / n,
        "valid_mode_rate": sum(r["valid_mode"] for r in all_results) / n,
        "soc_ok_rate": sum(r["soc_ok"] for r in all_results) / n,
        "direction_ok_rate": sum(r["direction_ok"] for r in all_results) / n,
        "mean_latency_s": round(sum(r["latency_s"] for r in all_results) / n, 1),
    }

    print("\nSummary")
    for k in ("parse_rate", "valid_mode_rate", "soc_ok_rate", "direction_ok_rate"):
        print(f"  {k:<20} {summary[k]:.0%}")
    print(f"  {'mean_latency_s':<20} {summary['mean_latency_s']}s")

    gate = summary["parse_rate"] == 1.0 and summary["soc_ok_rate"] == 1.0 and summary["direction_ok_rate"] >= 0.8
    print(f"\nVALIDATION GATE: {'PASS' if gate else 'FAIL'}")
    print("  (pass = 100% parse, 100% SOC-safe, >=80% directionally correct)")

    out = {
        "summary": summary,
        "system_prompt": SYSTEM_PROMPT,
        "results": all_results,
    }
    fname = f"results_{args.model.replace(':', '_').replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}.json"
    outpath = Path(__file__).parent / fname
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nFull results written to {outpath.name}")


if __name__ == "__main__":
    main()
