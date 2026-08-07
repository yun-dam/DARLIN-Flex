# Experiment: forecast-aware anchored control (L0F)

## The hypothesis

`RBCv5.decide_mode` uses **load, solar, SOC**. It never reads price, and it never
reads any forecast. So it is structurally incapable of two things:

1. **Price arbitrage** — discharging into expensive hours, holding through cheap ones
2. **Anticipation** — reserving charge for a peak it hasn't seen yet

CityLearn already provides `electricity_pricing_predicted_1/2/3` and solar
irradiance forecasts. Nothing in the current pipeline reads them.

**Hypothesis:** an LLM anchored to the RBC (so the baseline is preserved) but given
forecast data will override the rule *selectively and profitably*, lowering
`cost_total` below the RBC while keeping ramping near parity.

**Why this is the right experiment:** L0_free_fixed showed the model overrides only
0.48% of the time when the anchor is self-consistent — it has nothing to add,
because it sees exactly what the rule sees. Give it information the rule *cannot*
use and the override rate should rise for a principled reason.

**Falsifiable:** if the override rate stays near zero, or overrides don't improve
cost, the conclusion is that an 8B model can't exploit forecast information for
control — which is itself worth reporting.

---

## Change 1 — `llm_controller.py`: a forecast block

Add near the other prompt fragments:

```python
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
```

Add a `forecast` flag to `__init__`:

```python
                 guidance_level=1, district_context=False, max_deviation=None,
                 forecast=False):
    ...
    self.forecast = bool(forecast)
```

In `_build_prompt`, build the block and insert it. Simplest approach: append it to
the `district` string so no prompt template needs editing:

```python
        if self.forecast:
            district += _FORECAST.format(
                price=float(o.get("electricity_pricing", 0.0)),
                p1=float(o.get("price_1", 0.0)),
                p2=float(o.get("price_2", 0.0)),
                p3=float(o.get("price_3", 0.0)),
                sun0=float(o.get("sun_0", 0.0)),
                sun1=float(o.get("sun_1", 0.0)),
                sun2=float(o.get("sun_2", 0.0)),
                sun3=float(o.get("sun_3", 0.0)))
```

## Change 2 — `run_ladder.py`: pass the forecast observations through

In `build_index`, add the predicted fields (they are district-level, single
occurrence, so `.index()` is fine):

```python
    def first(n):
        return names.index(n) if n in names else None
    ...
        "price_1": first("electricity_pricing_predicted_1"),
        "price_2": first("electricity_pricing_predicted_2"),
        "price_3": first("electricity_pricing_predicted_3"),
        "sun_0":   first("direct_solar_irradiance"),
        "sun_1":   first("direct_solar_irradiance_predicted_1"),
        "sun_2":   first("direct_solar_irradiance_predicted_2"),
        "sun_3":   first("direct_solar_irradiance_predicted_3"),
```

In the per-building observation dict inside `run_arm`, add:

```python
                for k in ("price_1", "price_2", "price_3",
                          "sun_0", "sun_1", "sun_2", "sun_3"):
                    if idx.get(k) is not None:
                        o[k] = flat[idx[k]]
```

Add the CLI flag and thread it through:

```python
ap.add_argument("--forecast", action="store_true")
...
llm = LLMController(params=PARAMS, mock=mock,
                    guidance_level=int(arm[1]),
                    district_context=district,
                    max_deviation=max_dev,
                    forecast=forecast)
```

and include it in the auto-tag: `+ ("_fc" if a.forecast else "")`.

**Verify before running:** print one rendered prompt and confirm the forecast
numbers are real and differ across the three horizons. If `price_1/2/3` are all
identical to `price`, the observation names are wrong and the experiment is void.

---

## The runs

```bash
# sanity: prompt renders with real, varying forecast values
python run_ladder.py --arm L0 --hours 6 --district --forecast --mock --tag fc_smoke

# the experiment (~75 min)
caffeinate -is python run_ladder.py --arm L0 --hours 48 --district --forecast --tag L0_forecast
```

Compare against `L0_free_fixed` — same arm, same anchor, only the forecast added.
One variable changed.

---

## Reading the result

| observation | meaning |
|---|---|
| override rate rises above ~5% **and** cost drops below 0.878 | **the headline** — the LLM exploits information the rule cannot use |
| override rate rises, cost unchanged or worse | it acts on forecasts but not profitably; read the reasons |
| override rate stays ~0.5% | it ignores the forecast; the anchor dominates |
| ramping degrades badly | forecast-driven deviation costs load smoothness — a trade-off, report it |

**The reasons matter as much as the numbers.** An override that says *"price triples
in 2 hours, holding charge to discharge into the spike"* is a qualitatively different
artifact from anything a rule-based controller can produce, and it is the strongest
single slide available for a 3-minute talk — regardless of whether the KPI moves.

---

## If this works, the natural follow-up

Give the same forecast block to **L3** (no guidance). L0 is anchored, so its upside
is capped by design; L3 has room to build a genuinely different, anticipatory policy.
Run that only after L0F shows the model can use forecasts at all.
