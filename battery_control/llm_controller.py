"""
LLM controller v6 — RBC-ANCHORED ARM (L0) + guidance ladder (L1-L3).

NEW IN v6 — answers the advisor's ask: "design the prompt to guarantee the RBC
baseline and maybe improve upon it."

A prompt alone cannot guarantee anything. The guarantee is built at two levels:

  1. PROMPT ANCHORING (level 0) — the LLM is shown what the rule-based
     controller would do RIGHT NOW and asked to either CONFIRM it or OVERRIDE
     it with a stated reason. RBC behaviour becomes the default; deviation
     becomes a deliberate, justified act rather than an accident.

  2. EXECUTOR CLAMPING (max_deviation) — modes are ordered
        charge(0) < hold(1) < discharge_soft(2) < discharge_hard(3)
     and the LLM's choice is clamped to within `max_deviation` steps of the
     RBC's choice. 0 = pure RBC, 1 = one step of freedom, None = unclamped.
     This is the real guarantee: worst case the controller degrades toward RBC
     behaviour, it cannot run away from it.

Combined, the arm is "RBC as floor, LLM as bounded override" — which is the
mechanism that lets it match the baseline while retaining upside.

FULL LADDER (the freedom axis = how much the rule tells the LLM):
  L0  "anchored"     the rule's actual recommendation is shown; confirm/override
  L1  "imitation"    the rule's thresholds are given
  L2  "calibration"  typical net load given, thresholds withheld
  L3  "inference"    raw state + 6 h history only

DISTRICT CONTEXT (optional, any level) — MARLISA-style limited information
sharing: district net load, district typical, batteries charging last hour.

TIMEOUT = 300 s. Evidence: three of four L3 runs were bit-identical with exactly
9 timeouts each; the outlier had 14 and differed 8.6% on ramping. qwen3 at
temperature 0 is deterministic — all run-to-run variance was timeouts becoming
fallback holds.

BACKEND: pluggable via `provider` ("ollama" default, "openai", "anthropic" —
see providers.py). Only the request/response format and the token-limit
parameter name (num_predict vs max_tokens) differ between them; the prompts,
retry/timeout/fallback logic, and clamping guarantee are identical regardless
of which model is answering.
"""
import re
from collections import Counter, deque
from providers import make_provider, DEFAULT_MODELS
from rbc_v5 import RBCv5, RBCv5Params, BaselineTracker, VALID_MODES

MODE_ORDER = {"charge": 0, "hold": 1, "discharge_soft": 2, "discharge_hard": 3}
ORDER_MODE = {v: k for k, v in MODE_ORDER.items()}
DEFAULT_TIMEOUT = 300
DEFAULT_TOKEN_LIMIT = 2000   # num_predict (ollama) / max_tokens (openai, anthropic)


def parse_mode(text: str):
    """Extract a mode. Tokenized so 'charge' inside 'discharge', <think> blocks,
    hyphens and underscores all work. Prefers text after an explicit 'Action:'."""
    t = text.lower()
    if "</think>" in t:
        t = t.split("</think>")[-1]
    if "action:" in t:
        t = t.split("action:")[-1]
    tset = set(re.findall(r"[a-z]+", t))
    if "discharge" in tset:
        if "hard" in tset:
            return "discharge_hard"
        if "soft" in tset:
            return "discharge_soft"
        return "discharge_soft"
    if "charge" in tset:
        return "charge"
    if tset & {"hold", "nothing", "idle"}:
        return "hold"
    return None


def clamp_to_rbc(llm_mode, rbc_mode, max_dev):
    """Clamp the LLM's mode to within max_dev steps of the rule's mode.
    Returns (final_mode, was_clamped)."""
    if max_dev is None:
        return llm_mode, False
    a, b = MODE_ORDER[llm_mode], MODE_ORDER[rbc_mode]
    if abs(a - b) <= max_dev:
        return llm_mode, False
    step = max_dev if a > b else -max_dev
    return ORDER_MODE[b + step], True


# --------------------------------------------------------------------------- #
_HEADER = """You control ONE building's battery in a 5-building neighborhood. Goal: flatten this building's grid load — store energy in the valleys, release it during peaks.

STATE
  hour: {hour} (0-23)
  net grid load now: {net:.2f} kW      (negative = exporting solar surplus)
  battery: {soc_pct:.0f}% full
  you can discharge_hard for about {hours_left:.1f} more hours before it is empty
  price: {price:.2f}"""

_DISTRICT = """
DISTRICT (all 5 buildings together)
  district net load now: {dnet:.2f} kW
  district typical net load: {dbase:.2f} kW
  batteries that charged last hour: {ncharge} of 5
  Charging pulls from the grid. If many batteries charge in the same hour the
  combined draw can create a NEW district peak worse than doing nothing."""

_DISCHARGE_COUNT = """
  batteries that discharged last hour: {ndischarge} of 5"""

_DESYNC = """
  The other four buildings see this same forecast. If you all discharge in
  the same hour you create a new ramp. Consider whether to act now or let
  the others go first."""

_TAIL = """
Reply with ONE short reason then the action, exactly like:
reason: <{reason_words} words max>. Action: <charge|hold|discharge_soft|discharge_hard>"""

_FORECAST = """
LOOKING AHEAD (the rule-based controller cannot see any of this — it only knows right now)
  electricity price now: {price:.3f}
  price in 1h: {p1:.3f}   in 2h: {p2:.3f}   in 3h: {p3:.3f}
  solar irradiance now: {sun0:.0f} W/m2   in 1h: {sun1:.0f}   in 2h: {sun2:.0f}   in 3h: {sun3:.0f}

The rule is purely reactive and price-blind. You are not. Override it when the
forecast gives a concrete reason — for example:
  - price rises sharply soon: keep charge in reserve now, discharge into the spike
  - price is about to fall: no need to discharge now, wait
  - solar is about to arrive: don't grid-charge, free energy is minutes away
  - solar is fading and load will climb: stop charging, prepare to discharge
State the forecast reason explicitly when you override."""

# ---- LEVEL 0: RBC-anchored. The rule's answer is shown; confirm or override. #
PROMPT_L0 = _HEADER + """
  this building's typical net load: {base:.2f} kW
{district}
A proven rule-based controller has analysed this exact situation and recommends:

    >>> {rbc_mode} <<<

That rule reliably flattens load and is the performance baseline. Your job is to
CONFIRM it, or OVERRIDE it only when you can state a concrete reason it is wrong
for this specific hour — for example it would waste free solar, or empty the
battery before a larger peak, or add to a district-wide charging spike.

If you have no specific reason to disagree, repeat the recommendation.

  charge          store energy — sensible only when net load is NEGATIVE (free solar)
  discharge_soft  release some energy during a moderate peak
  discharge_hard  release a lot during a real peak
  hold            do nothing — ordinary conditions, or battery empty (<=15%) or full""" + _TAIL

_L1_EXAMPLES = """

EXAMPLES
  net 4.10, typical 1.80, 80% full  -> reason: well above typical, plenty of charge. Action: discharge_hard
  net 2.10, typical 1.80, 60% full  -> reason: slightly above typical. Action: discharge_soft
  net -1.40, typical 1.80, 45% full -> reason: exporting solar, store it. Action: charge
  net 1.70, typical 1.80, 20% full  -> reason: near typical and low charge, save it. Action: hold"""

PROMPT_L1 = _HEADER + """
  this building's typical net load: {base:.2f} kW
{district}
DECISION RULES (use these exact numbers)
  discharge_hard  if net load > {hi:.2f} kW   (a real peak)
  discharge_soft  if net load > {mid:.2f} kW  (moderately above typical)
  charge          if net load < 0 kW          (free solar surplus ONLY)
  hold            otherwise, or if the battery is empty (<=15%) or full{examples}""" + _TAIL

PROMPT_L2 = _HEADER + """
  this building's typical net load: {base:.2f} kW
{district}
YOUR JOB
  Decide how far above typical counts as a peak worth discharging into, and how
  far below counts as a valley worth storing. You are not given thresholds.

  charge          store energy — only sensible when net load is NEGATIVE (free solar surplus)
  discharge_soft  release some energy when load is moderately above typical
  discharge_hard  release a lot when load is well above typical (a real peak)
  hold            do nothing — near typical, or battery empty (<=15%) or full

  Keep reserve: if the battery is low, prefer soft or hold so you still have
  charge for the evening peak.""" + _TAIL

PROMPT_L3 = _HEADER + """
  this building's net load over the last 6 hours: {history}
{district}
YOUR JOB
  From the recent history, judge what is normal for this building and whether
  right now is a peak, a valley, or ordinary. You are given no baseline and no
  thresholds — infer them.

  charge          store energy — only sensible when net load is NEGATIVE (free solar surplus)
  discharge_soft  release some energy during a moderate peak
  discharge_hard  release a lot during a real peak
  hold            do nothing — ordinary conditions, or battery empty (<=15%) or full

  Keep reserve: if the battery is low, prefer soft or hold so you still have
  charge for the evening peak.""" + _TAIL

PROMPTS = {0: PROMPT_L0, 1: PROMPT_L1, 2: PROMPT_L2, 3: PROMPT_L3}


class LLMController:
    """guidance_level: 0 = RBC recommendation shown (anchored), 1 = thresholds,
    2 = baseline only, 3 = raw history.
    max_deviation: clamp the LLM's mode to within N steps of the RBC's mode.
                   None = unclamped, 0 = pure RBC, 1 = one step of freedom."""

    def __init__(self, model=None, host="http://localhost:11434",
                 params: RBCv5Params = None, mock=False,
                 timeout=DEFAULT_TIMEOUT, retries=1,
                 guidance_level=1, district_context=False, max_deviation=None,
                 log_overrides=False, forecast=False,
                 provider="ollama", token_limit=DEFAULT_TOKEN_LIMIT, reason_words=8,
                 few_shot=True):
        self.p = params or RBCv5Params()
        self.provider_name = provider
        self.host = host
        self.mock = mock
        self.timeout = timeout
        self.retries = retries
        self.level = int(guidance_level)
        self.district_context = bool(district_context)
        self.max_deviation = max_deviation
        self.forecast = bool(forecast)
        self.token_limit = token_limit
        self.reason_words = reason_words
        self.few_shot = bool(few_shot)

        if self.mock:
            # no network calls in mock mode -- no provider/API key needed
            self.backend = None
            self.model = model or DEFAULT_MODELS.get(provider, provider)
        else:
            self.backend = make_provider(provider, model=model, timeout=timeout, host=host)
            self.model = self.backend.model

        self.executor = RBCv5(self.p)
        self.tracker = BaselineTracker(self.p)
        self.shadow = RBCv5(self.p)          # the rule, on identical state
        self.history = {}

        self.calls = 0
        self.timeouts = 0
        self.retry_successes = 0
        self.parse_failures = 0
        self.latencies = []
        self.mode_counts = {m: 0 for m in VALID_MODES}
        self.charge_decisions = 0
        self.charge_when_surplus = 0
        self.blocked_grid_charges = 0
        self.reasons = []
        self.decisions_compared = 0
        self.agreements = 0
        self.disagreements = Counter()
        self.deviation_sum = 0
        # L0 / clamping bookkeeping
        self.overrides = 0                   # LLM chose != rule
        self.clamped = 0                     # executor pulled it back toward rule
        self.override_reasons = []
        # diagnostic only: for each override, record whether the label swap
        # actually produced a different physical action (kW), or whether it
        # collapsed to the same value as the RBC's action would have.
        self.log_overrides = log_overrides
        self.override_log = []
        # per-step [llm_mode, rbc_mode] pairs -- lets agreement/override stats
        # be recomputed offline (e.g. against a corrected RBC brain) without
        # re-running the LLM.
        self.mode_pairs = []

    # ------------------------------------------------------------------ #
    def act(self, o) -> float:
        p = self.p
        bid = int(o.get("building_id", 0))
        load = float(o.get("non_shiftable_load", 0.0))
        solar = float(o.get("solar_generation", 0.0))
        soc = float(o.get("electrical_storage_soc", 0.5))
        price = float(o.get("electricity_pricing", 0.0))
        hour = int(o.get("hour", 0))
        net = load - solar

        self.history.setdefault(bid, deque(maxlen=6)).append(net)
        base, dev, seen = self.tracker.update(bid, net)
        rbc_mode = self.shadow.decide_mode(o)      # what the rule would do

        if seen <= p.warmup_steps:
            self.mode_counts["hold"] += 1
            self.mode_pairs.append(["hold", rbc_mode])
            return 0.0

        hi = base + p.hi_k * dev
        mid = base + p.mid_k * dev
        hours_left = max(0.0, (soc - p.soc_min) / max(1e-6, p.discharge_hard_rate))

        if self.mock:
            raw_mode = self._mock_mode(net, hi, mid, soc, rbc_mode)
        else:
            raw_mode = self._ask_llm(o, hour, net, base, soc, price, hi, mid,
                                     hours_left, bid, rbc_mode)

        # executor-level guarantee: clamp toward the rule
        mode, was_clamped = clamp_to_rbc(raw_mode, rbc_mode, self.max_deviation)
        if was_clamped:
            self.clamped += 1

        self.mode_pairs.append([mode, rbc_mode])
        self.decisions_compared += 1
        if mode == rbc_mode:
            self.agreements += 1
        else:
            self.overrides += 1
            self.disagreements[(mode, rbc_mode)] += 1
            if self.reasons:
                self.override_reasons.append(self.reasons[-1])
        self.deviation_sum += abs(MODE_ORDER[mode] - MODE_ORDER[rbc_mode])

        self.mode_counts[mode] += 1
        if mode == "charge":
            self.charge_decisions += 1
            if net < 0:
                self.charge_when_surplus += 1
        action = self.executor.mode_to_action(mode, soc, net)
        if mode == "charge" and abs(action) < 1e-9 and soc < p.soc_max:
            self.blocked_grid_charges += 1
        if self.log_overrides and mode != rbc_mode:
            rbc_action = self.executor.mode_to_action(rbc_mode, soc, net)
            entry = {"sim_step": o.get("sim_step"), "building_id": bid, "hour": hour,
                     "llm_raw_mode": raw_mode, "rbc_mode": rbc_mode,
                     "final_mode": mode, "was_clamped": was_clamped,
                     "action": action, "rbc_action": rbc_action,
                     "diff": action - rbc_action}
            self.override_log.append(entry)
            print(f"  [override] step={entry['sim_step']} b={bid} h={hour} "
                  f"llm={raw_mode}->{mode} rbc={rbc_mode} clamped={was_clamped} "
                  f"action={action:.4f} rbc_action={rbc_action:.4f} "
                  f"diff={entry['diff']:.4f}")
        return action

    def _mock_mode(self, net, hi, mid, soc, rbc_mode) -> str:
        # level 0 mock: always confirm the rule (the "no reason to deviate" case)
        if self.level == 0:
            return rbc_mode
        if net > hi and soc > self.p.soc_min:
            return "discharge_hard"
        if net > mid and soc > self.p.soc_min:
            return "discharge_soft"
        if net < 0 and soc < self.p.soc_max:
            return "charge"
        return "hold"

    def _build_prompt(self, o, hour, net, base, soc, price, hi, mid,
                      hours_left, bid, rbc_mode):
        district = ""
        if self.district_context:
            district = _DISTRICT.format(
                dnet=float(o.get("district_net", 0.0)),
                dbase=float(o.get("district_base", 0.0)),
                ncharge=int(o.get("n_charging_last_hour", 0)))
        if self.forecast:
            if self.district_context:
                district += _DISCHARGE_COUNT.format(
                    ndischarge=int(o.get("n_discharging_last_hour", 0)))
            district += _FORECAST.format(
                price=float(o.get("electricity_pricing", 0.0)),
                p1=float(o.get("price_1", 0.0)),
                p2=float(o.get("price_2", 0.0)),
                p3=float(o.get("price_3", 0.0)),
                sun0=float(o.get("sun_0", 0.0)),
                sun1=float(o.get("sun_1", 0.0)),
                sun2=float(o.get("sun_2", 0.0)),
                sun3=float(o.get("sun_3", 0.0)))
            if self.district_context:
                district += _DESYNC
        hist = ", ".join(f"{x:.2f}" for x in self.history.get(bid, []))
        examples = _L1_EXAMPLES if (self.level == 1 and self.few_shot) else ""
        return PROMPTS[self.level].format(
            hour=hour, net=net, base=base, soc_pct=100 * soc, price=price,
            hi=hi, mid=mid, hours_left=hours_left, district=district,
            history=hist or "n/a", rbc_mode=rbc_mode,
            reason_words=self.reason_words, examples=examples)

    def _one_request(self, prompt):
        return self.backend.request(prompt, self.token_limit)

    def _ask_llm(self, o, hour, net, base, soc, price, hi, mid, hours_left,
                 bid, rbc_mode) -> str:
        prompt = self._build_prompt(o, hour, net, base, soc, price, hi, mid,
                                    hours_left, bid, rbc_mode)
        text = None
        for attempt in range(self.retries + 1):
            self.calls += 1
            try:
                text, secs = self._one_request(prompt)
                self.latencies.append(secs)
                if attempt > 0:
                    self.retry_successes += 1
                break
            except Exception as e:
                is_to = "timeout" in repr(e).lower()
                if is_to:
                    self.timeouts += 1
                print(f"  [llm {'timeout' if is_to else 'error'}"
                      f"{' — retrying' if attempt < self.retries else ' — fallback to RBC'}]")
        if text is None:
            # fallback is the RULE's action, not a blind hold — keeps the floor
            return rbc_mode
        mode = parse_mode(text)
        tail = text.split("</think>")[-1].strip().replace("\n", " ")
        self.reasons.append(tail[:90])
        if mode is None:
            self.parse_failures += 1
            return rbc_mode
        return mode

    # ------------------------------------------------------------------ #
    def compliance(self):
        return 1.0 if self.charge_decisions == 0 else \
            self.charge_when_surplus / self.charge_decisions

    def agreement(self):
        return 1.0 if self.decisions_compared == 0 else \
            self.agreements / self.decisions_compared

    def override_rate(self):
        return 0.0 if self.decisions_compared == 0 else \
            self.overrides / self.decisions_compared

    def mean_deviation(self):
        return 0.0 if self.decisions_compared == 0 else \
            self.deviation_sum / self.decisions_compared

    def latency_stats(self):
        if not self.latencies:
            return None
        s = sorted(self.latencies)
        n = len(s)
        return dict(median=s[n // 2], p95=s[min(n - 1, int(0.95 * n))],
                    max=s[-1], mean=sum(s) / n, n=n)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    samples = [
        ("reason: well above typical. Action: discharge_hard", "discharge_hard"),
        ("reason: agree with the rule. Action: hold", "hold"),
        ("<think>rule says hold, but solar</think> Action: charge", "charge"),
        ("Action: Discharge-Soft", "discharge_soft"),
        ("banana", None),
    ]
    ok = sum(parse_mode(x) == w for x, w in samples)
    print(f"{ok}/{len(samples)} parse cases passed")

    print("\n--- clamping (the guarantee) ---")
    cases = [
        # LLM 3 steps below the rule, max_dev=1 -> pulled back to one step below
        ("charge", "discharge_hard", 1, "discharge_soft"),
        ("discharge_hard", "hold", 1, "discharge_soft"),
        ("discharge_hard", "hold", None, "discharge_hard"),   # unclamped
        ("hold", "hold", 0, "hold"),
        ("discharge_hard", "charge", 0, "charge"),    # 0 = pure RBC
    ]
    allok = True
    for llm, rbc, dev, want in cases:
        got, clamped = clamp_to_rbc(llm, rbc, dev)
        good = got == want
        allok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] llm={llm:<15} rule={rbc:<15} "
              f"max_dev={str(dev):<4} -> {got:<15} clamped={clamped}")
    print(f"[{'PASS' if allok else 'FAIL'}] executor clamps the LLM toward the rule")

    print("\n--- L0 prompt shows the rule's recommendation ---")
    c = LLMController(mock=True, guidance_level=0, district_context=True)
    c.history[0] = deque([1.2, 0.8, -0.4, -1.1, 0.3, 2.2], maxlen=6)
    p = c._build_prompt({"district_net": 8.4, "district_base": 6.1,
                         "n_charging_last_hour": 2},
                        18, 2.20, 1.80, 0.55, 0.22, 2.41, 1.95, 1.05, 0,
                        "discharge_soft")
    print(f"  recommendation embedded: {'>>> discharge_soft <<<' in p}")
    print(f"  asks to confirm or override: {'CONFIRM' in p and 'OVERRIDE' in p}")

    print("\n--- level information check ---")
    exp = {0: (False, True, False), 1: (True, True, False),
           2: (False, True, False), 3: (False, False, True)}
    good = True
    for lvl, want in exp.items():
        c = LLMController(mock=True, guidance_level=lvl)
        c.history[0] = deque([1.0] * 6, maxlen=6)
        p = c._build_prompt({}, 18, 2.2, 1.8, 0.55, 0.22, 2.41, 1.95, 1.05, 0, "hold")
        got = ("2.41" in p, "this building's typical net load" in p,
               "last 6 hours" in p)
        if got != want:
            good = False
            print(f"  LEVEL {lvl} MISMATCH: got {got}, expected {want}")
    print(f"[{'PASS' if good else 'FAIL'}] each level exposes the intended information")

    print("\n--- reason_words is configurable, defaults to 8 (unchanged behaviour) ---")
    c_default = LLMController(mock=True, guidance_level=2)
    c_custom = LLMController(mock=True, guidance_level=2, reason_words=20)
    c_default.history[0] = c_custom.history[0] = deque([1.0] * 6, maxlen=6)
    p_default = c_default._build_prompt({}, 18, 2.2, 1.8, 0.55, 0.22, 2.41, 1.95, 1.05, 0, "hold")
    p_custom = c_custom._build_prompt({}, 18, 2.2, 1.8, 0.55, 0.22, 2.41, 1.95, 1.05, 0, "hold")
    good = "<8 words max>" in p_default and "<20 words max>" in p_custom
    print(f"[{'PASS' if good else 'FAIL'}] default '<8 words max>', "
          f"reason_words=20 -> '<20 words max>'")

    print("\n--- L1 few-shot examples default on, can be disabled ---")
    c_on = LLMController(mock=True, guidance_level=1)
    c_off = LLMController(mock=True, guidance_level=1, few_shot=False)
    c_on.history[0] = c_off.history[0] = deque([1.0] * 6, maxlen=6)
    p_on = c_on._build_prompt({}, 18, 2.2, 1.8, 0.55, 0.22, 2.41, 1.95, 1.05, 0, "hold")
    p_off = c_off._build_prompt({}, 18, 2.2, 1.8, 0.55, 0.22, 2.41, 1.95, 1.05, 0, "hold")
    good = "EXAMPLES" in p_on and "EXAMPLES" not in p_off
    print(f"[{'PASS' if good else 'FAIL'}] few_shot=True includes EXAMPLES, "
          f"few_shot=False omits it")

    print("\n--- provider defaults unchanged; mock mode needs no API key ---")
    c = LLMController(mock=True)
    good = c.provider_name == "ollama" and c.model == "qwen3:8b" and c.backend is None
    print(f"[{'PASS' if good else 'FAIL'}] provider={c.provider_name} model={c.model} "
          f"backend={c.backend}")
    c2 = LLMController(mock=True, provider="openai")
    good2 = c2.model == "gpt-4o"   # DEFAULT_MODELS["openai"], no key needed since mock=True
    print(f"[{'PASS' if good2 else 'FAIL'}] mock + provider=openai resolves default model "
          f"({c2.model}) without requiring OPENAI_API_KEY")
