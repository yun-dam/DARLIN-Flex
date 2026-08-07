# Project Findings Log — LLM Battery Control in CityLearn
**Lino Marrero · SUPER Fellowship · Nagy Lab (advisor: Yun Dam Ko) · through Week 6**

Running record of what was built, what was found, and what it means. Update as you go.

---

## 1. What the system is

A battery controller for a 5-building district in CityLearn, built in two layers so the decision-maker can be swapped while everything else stays identical.

```
building state (load, solar, SOC)
        │
        ▼
   [ BRAIN ]  picks one of 4 modes:  charge | hold | discharge_soft | discharge_hard
        │
        ▼
   [ EXECUTOR ]  mode → kW rate, + hard SOC guard, + solar-only charge block
        │
        ▼
   action to CityLearn
```

**The brain is either:**
- **RBC v5** — three if-statements comparing current net load to a rolling baseline
- **LLM** — qwen3:8b reads a prompt describing the same state and returns a mode word

**The executor is always RBC v5.** It converts the mode to a rate, clips so SOC stays in [0.15, 0.95], and blocks charging when net load is positive. Both brains share the same executor, thresholds, and warmup — so any performance difference is attributable to decision-making, not plumbing.

**Why 4 discrete modes and not a number:** small models choose reliably from a menu but calibrate poorly on continuous values. Evidence: 0 malformed replies across ~1,000 calls.

**Why the guard exists:** the Week-4 validation gate showed qwen2.5:7b state *"battery is already at high SOC"* and output `charge` in the same sentence. Knowing ≠ complying. The guard makes violations structurally impossible.

---

## 2. Core controller logic (be able to explain this)

```python
net  = load - solar                      # what the grid must supply
base = EMA of net      (α = 0.05)        # this building's "typical"
dev  = EMA of |net-base| (α = 0.05)      # this building's typical swing

if seen <= 6:                                    return "hold"   # warmup
if net > base + 1.20*dev and soc > 0.15:         return "discharge_hard"
if net > base + 0.35*dev and soc > 0.15:         return "discharge_soft"
if (solar > load or net < base - 0.50*dev) and soc < 0.95: return "charge"
return "hold"
```

Thresholds are in units of `dev`, so they're **scale-free** — they adapt to each building's own variability rather than being fixed kW values. Order matters: first match wins, so peaks beat valleys.

**Warmup (6 steps):** the baseline is learned from data; with one sample "typical" equals that sample and `dev` sits at its 0.05 floor, so the threshold is only 0.06 kW above base — a hair-trigger. Tested: starting a window at 16:00 with no warmup fires `discharge_hard` on 7 of the first 8 hours. It's also load-bearing for fairness, since both brains get the identical handicap.

---

## 3. Bugs found (both would have produced confidently wrong results)

### 3.1 Dataset bug — broken SOC observation
`electrical_storage_soc` reports **0.000 all year** while the battery is actually full. Verified by probing the battery object directly: internal SOC climbed 0.36 → 0.73 → 1.0 while the observation stayed 0.000.

**Consequence:** every controller believed the battery was permanently empty, so the discharge condition (`soc > 0.15`) never fired. Controllers charged ~40% of the time and discharged **0** times — adding load and never shaving.

**Fix:** read true SOC from `env.buildings[b].electrical_storage.soc`, indexed at the *current* step. Never `soc[-1]` — that's the year-end preallocated slot, always 0.

### 3.2 Methodology bug — warmup confound
RBC v4 used `warmup_steps = 24`. In a 24-hour evaluation window it was inert for 23 of 24 steps while the LLM acted from hour 1. This produced a false *"LLM beats RBC by 13%"* result, **now retracted**. Both policies now share `warmup_steps = 6`.

### 3.3 Robustness fix — ratio thresholds
v4 used `net > base * 1.15`. Since `net = load - solar`, a net-exporting building's baseline can go negative, and multiplying a negative baseline inverts the threshold. Replaced with absolute offsets (`base + k*dev`). Latent — it likely never fired on this dataset (district averages +2.6 kW) but it's wrong in principle.

---

## 4. Methodology: minimum valid evaluation horizon

Measured, not assumed. The same deterministic RBC gave cost_total **2.450 at 24 h, 0.878 at 48 h, 0.807 over the full year** — the metric was breaking, not the controller.

**Cause:** the five batteries hold ~32 kWh; the district consumes only ~62 kWh in 24 h. Energy pushed into storage dominates the energy-total KPIs, and the benefit of discharging lands outside the window.

**Deviation from the full-year value:**

| KPI | 24 h | 48 h | 168 h | verdict at 48 h |
|---|---|---|---|---|
| carbon_emissions_total | 123% | 0% | −1% | ✅ usable |
| ramping_average | 5% | −3% | 3% | ✅ usable |
| electricity_consumption_total | 180% | 4% | −1% | ✅ usable |
| cost_total | 203% | 9% | 4% | ⚠️ borderline |
| daily_one_minus_load_factor | 7% | −1% | −3% | ✅ usable |
| zero_net_energy | −12% | −12% | −6% | ❌ |
| daily_peak_average | 135% | 63% | 17% | ❌ needs >336 h |
| all_time_peak_average | 0% | 0% | 0% | useless (always 1.000) |

**Decision:** evaluate at **48 h**; report ramping, carbon, consumption, load factor. Exclude `daily_peak_average` and `zero_net_energy` with the stated reason.

Note: ramping is **horizon**-robust but **decision**-sensitive — two changed actions moved it 0.881 → 0.914. Different properties; don't conflate them.

---

## 5. Determinism: all apparent LLM randomness was infrastructure

Four L3 runs at identical settings:

| run | timeouts | ramping | cost | carbon | agreement |
|---|---|---|---|---|---|
| L3_dist | 9 | 0.881 | 0.825 | 0.839 | 39.1% |
| L3_run1 | 14 | 0.964 | 0.865 | 0.872 | 37.1% |
| L3_run2 | 9 | 0.881 | 0.825 | 0.839 | 39.1% |
| L3_run3 | 9 | 0.881 | 0.825 | 0.839 | 39.1% |
| L3_clean | 3 | 0.914 | 0.825 | 0.839 | 39.1% |

Three runs are **bit-identical** and all had exactly 9 timeouts. The outlier had 14.

**Conclusion: qwen3 at temperature 0 is deterministic.** Every observed run-to-run difference traced to timeouts converting decisions into fallback `hold`s, which shift the SOC trajectory and everything after. Raising the timeout 90 s → 300 s cut timeouts 9 → 3 but not to zero, so some generations genuinely exceed 300 s — the reasoning tail is unbounded. Next fix: cap `num_predict`.

**Why this matters:** most LLM-agent work reports run variance and assumes it's the model. Here it was the harness.

---

## 6. Week 5 result — RBC v5 vs do-nothing (full year, 8,759 steps)

| KPI | v0 | RBC v5 | |
|---|---|---|---|
| ramping_average | 1.000 | **0.780** | −22.0% |
| cost_total | 1.000 | **0.807** | −19.3% |
| daily_peak_average | 1.000 | **0.853** | −14.7% |
| carbon_emissions_total | 1.000 | **0.878** | −12.2% |
| electricity_consumption_total | 1.000 | **0.893** | −10.7% |
| daily / monthly load factor | 1.000 | 0.939 / 0.968 | better |
| all_time_peak_average | 1.000 | 1.000 | tie |
| zero_net_energy | 1.000 | 1.113 | worse |

**7 better, 1 worse, 1 tie.** Both losses are structural: `all_time_peak` ties because a reactive rule can't reserve for a peak it hasn't seen; `zero_net_energy` worsens because exported solar counts fully toward net-zero while stored solar returns only ~85% after round-trip losses.

---

## 7. Week 6 result — the guidance ladder

The "freedom" axis is **how much the rule tells the LLM**. Same executor, warmup and guard at every level; only the information changes.

- **L1 "imitation"** — the RBC's computed thresholds are handed over
- **L2 "calibration"** — typical net load given, thresholds withheld
- **L3 "inference"** — raw state + last 6 hours only; no baseline, no thresholds

All arms also received **district context** (district net load, district typical, how many batteries charged last hour) — MARLISA-style limited information sharing.

### The knob behaves as designed
| level | agreement with rule | mean deviation | compliance | median latency |
|---|---|---|---|---|
| L1 | 64.3% | 0.37 | 100% | 18.2 s |
| L2 | 44.8% | 0.63 | 98% | 27.7 s |
| L3 | 39.1% | 0.73 | 96% | 37.4 s |

Removing information monotonically reduces agreement, raises deviation, **raises latency** (more reasoning), and **degrades constraint compliance** — the executor guard catches real violations at higher freedom.

### Performance is NOT monotonic
| KPI (48 h) | v0 | RBC | L1 | L2 | L3 |
|---|---|---|---|---|---|
| ramping_average | 1.000 | **0.755** | 0.770 | 0.901 | 0.881 |
| cost_total | 1.000 | 0.878 | 0.886 | 0.901 | **0.825** |
| carbon_emissions_total | 1.000 | 0.881 | 0.884 | 0.912 | **0.839** |
| electricity_consumption_total | 1.000 | 0.924 | 0.926 | 0.957 | **0.864** |
| daily_1−load_factor | 1.000 | **0.932** | 0.944 | 0.983 | 1.008 |

**L3 beat L2** — less guidance produced *better* control, contradicting the intuition that freedom degrades performance monotonically.

### The headline finding — a strategy trade-off, replicated 4×
- **L3 beats the hand-tuned RBC on every energy metric**: cost 0.825 vs 0.878, carbon 0.839 vs 0.881, consumption 0.864 vs 0.924. These values were **identical in all four runs**.
- **The RBC beats L3 on load shape**: ramping 0.755 vs 0.881–0.964, load factor 0.932 vs 1.008. Direction held in every run.

**Mechanism:** L3 charges less (47 vs L1's 61) and holds more (155). Less cycling → less round-trip loss → better energy and carbon. But fewer actions → less active smoothing → worse ramping.

**One-line version:** the RBC is the better load-shaper; the unguided LLM is the better energy manager. Guidance level doesn't just tune quality — it selects which objective the controller pursues.

### Important caveat on "who won"
The project targets **flexibility** KPIs (that's what grid-interactive means). Ramping and load factor *are* the flexibility KPIs, and L3 lost both. So by the project's own stated success criteria, L3 did **not** beat the RBC — it won on efficiency/economics instead. **Open question for the advisor: is the target flexibility, total cost, or a weighted combination?**

---

## 8. Week 6 — SAC baseline (closes open item #1)

CityLearn's built-in SAC (`citylearn.agents.sac.SAC`), 5 episodes, default hyperparameters (`batch_size=256`, `end_exploration_time_step` = one episode's length).

### Bug: short episodes crash before SAC finishes exploring
`--hours 168` (the originally planned command) throws `AssertionError` inside `citylearn/agents/sac.py::get_normalized_observations`. Cause: `end_exploration_time_step` defaults to one episode's length (168 steps), but observation normalization only activates once the replay buffer holds `batch_size` (256) samples. A 168-step episode never fills the buffer to 256 before exploration ends, so at the start of episode 2 SAC switches to policy-based prediction with `norm_mean`/`norm_std` still `None` → crash. Confirmed with a bare-traceback repro outside `run_sac.py` — not an argument-order bug.

**Fix used:** run full-year episodes (8,759 steps), where the buffer fills well before the exploration cutoff. Separately, `run_sac.py`'s `try_learn()` fallback order was fixed — the working `learn(episodes=N)` call was being tried *after* a variant that raises a non-`TypeError` exception, and the retry loop breaks (rather than continues) on any non-`TypeError`.

### Result — full year, 5 episodes, 721s train time
| KPI | v0 | RBC v5 | SAC (5 ep) |
|---|---|---|---|
| ramping_average | 1.000 | **0.780** | 1.233 |
| cost_total | 1.000 | **0.807** | 1.039 |
| daily_peak_average | 1.000 | **0.853** | 1.048 |
| carbon_emissions_total | 1.000 | **0.878** | 1.044 |
| electricity_consumption_total | 1.000 | **0.893** | 1.045 |
| daily_one_minus_load_factor_average | 1.000 | 0.939 | 0.995 |
| all_time_peak_average | 1.000 | 1.000 | 1.000 |
| zero_net_energy | 1.000 | 1.113 | 1.061 |

**SAC underperforms v0 (do-nothing) on 5 of 8 KPIs** and loses to RBC v5 on every KPI. Consistent with `run_sac.py`'s own stated expectation (Nweye et al. 2022): a handful of episodes isn't enough for SAC to beat a hand-tuned RBC — this is the expected undertrained result, not a bug. Only `all_time_peak_average` ties and `daily_one_minus_load_factor_average` (0.995) is marginally better than v0.

**Open question:** how many episodes would it take for SAC to close the gap, and is that worth the ~144s/episode training cost at full-year length?

---

## 9. Calibration from the literature
MARLISA — a purpose-built, trained multi-agent RL system on this same 5-building CityLearn setup — beats its rule-based controller by **3.8%**. Large margins over a decent RBC are not available to anyone in this environment. MARLISA also coordinates via **shared district-level signals**, which is why district context was added to the prompts.

---

## 10. Files

| file | purpose |
|---|---|
| `rbc_v5.py` | rule brain + shared executor + SOC guard; self-tests |
| `llm_controller.py` | LLM brain, 3 guidance levels, district context, diagnostics |
| `run_ladder.py` | runs one arm at a time, appends to `results_ladder.json` |
| `run_fullyear_rbc.py` | full-year RBC vs do-nothing (no LLM calls) |
| `check_horizon.py` | horizon sensitivity sweep |
| `run_sac.py` | SAC baseline; run 2026-07-28, 5 full-year episodes, underperforms v0 (§8) |
| `results_ladder.json` | accumulated results across sessions |

---

## 11. Open items

1. ~~**SAC arm**~~ — done (§8): 5 full-year episodes, underperforms v0 and RBC v5 on most KPIs as expected for a lightly-trained agent. Open question: episode count needed to close the gap.
2. **Cap `num_predict`** to eliminate the last 3 timeouts and get fully bit-reproducible runs
3. **Confirm the objective** with the advisor (flexibility vs cost vs weighted) — determines whether L3 "won"
4. **Diagnose L2 < L3** — read the logged reasons; hypothesis is that L2's baseline gives an anchor without calibration, producing bad threshold guesses
5. **Hybrid experiment** — give L3's minimal prompt an explicit ramping objective; can it keep the energy win and recover load shape?
6. **Transfer test** — same prompt on a dataset the RBC was never tuned for. Caveat: v5's scale-free thresholds may transfer well too.
