# LLM Battery Control for Grid-Interactive Buildings

Evaluating locally-hosted LLMs as battery controllers in CityLearn, benchmarked against
a rule-based controller, a do-nothing baseline, and CityLearn's built-in SAC agent.

**Lino Marrero** · SUPER Fellowship · Jain Lab · advisor: Yun Dam Ko

---

## ⚠️ Read this before running anything

### 1. The `electrical_storage_soc` observation is broken in this dataset

In `citylearn_challenge_2022_phase_1`, the `electrical_storage_soc` observation
returns **0.000 at every timestep** even while the battery charges to full.

Verified by probing the battery object directly: internal SOC climbed
`0.36 → 0.73 → 0.98 → 1.0` while the observation stayed at 0.000.

**Consequence if you don't handle it:** any controller reading that observation believes
the battery is permanently empty, so its discharge condition never fires. Controllers
charge ~40% of the time and discharge *zero* times — adding load without ever shaving.
Every KPI looks flat and you will conclude your controller does nothing.

**The fix**, used throughout this repo:

```python
def true_soc(env, b, t_step):
    bat = env.buildings[b].electrical_storage
    arr = np.asarray(bat.soc, dtype=float)
    if arr.size:
        return float(arr[0 if t_step == 0 else min(t_step - 1, arr.size - 1)])
    cap = getattr(bat, "capacity", 1.0) or 1.0
    v = float(getattr(bat, "initial_soc", 0.0))
    return v / cap if v > 1.0 else v
```

Note `soc[-1]` is **wrong** — `bat.soc` is a preallocated full-episode array, so the last
element is the year-end slot (always 0). Index the *current* step.

### 2. `num_predict` truncation fails silently

Capping generation length too low truncates the model mid-reasoning, the reply fails to
parse, and the controller falls back to the rule's action. The run then produces
**rule-shaped output that looks like a clean result.**

Measured on the level-3 (no-guidance) prompt:

| `num_predict` | parse failure rate |
|---|---|
| 400 | **100%** |
| 2000 | 19.5% |
| uncapped | clean |

`run_ladder.py` prints a warning when parse failures exceed 10% of calls. Do not ignore it.

### 3. Evaluation windows shorter than ~48 h give invalid energy KPIs

The five batteries hold ~32 kWh; the district consumes ~62 kWh/day. Over a short window
the energy pushed into storage dominates the energy-total KPIs, and the benefit of
discharging it falls outside the window.

The same deterministic RBC scores `cost_total` **2.450 at 24 h**, **0.878 at 48 h**,
**0.807 over a full year**. Run `check_horizon.py` to reproduce the sweep.

| KPI | error at 48 h vs full year | usable at 48 h? |
|---|---|---|
| carbon_emissions_total | 0% | yes |
| ramping_average | −3% | yes |
| electricity_consumption_total | +4% | yes |
| daily_one_minus_load_factor | −1% | yes |
| cost_total | +9% | borderline |
| zero_net_energy | −12% | no |
| daily_peak_average | +63% | no — needs >336 h |

---

## Architecture

```
building state (load, solar, SOC)
        │
        ▼
   [ BRAIN ]     picks one of four modes:
                 charge | hold | discharge_soft | discharge_hard
        │
        ▼
   [ EXECUTOR ]  mode → kW rate
                 + hard SOC guard  [0.15, 0.95]
                 + solar-only charge block
        │
        ▼
   action → CityLearn
```

The brain is either the rule (`RBCv5.decide_mode`) or an LLM (`LLMController`).
**The executor is always `RBCv5`.** Both brains share the same executor, warmup,
thresholds and district context — only the decision-maker differs, which is what makes
the comparison valid.

### Why four discrete modes rather than a continuous action
Small models select reliably from a menu but calibrate poorly on continuous values.
Measured: zero malformed replies across ~1,000 calls. The cost is coarse control, which
contributes to the LLM's weaker ramping performance.

### Why the guard lives in the executor, not the prompt
A model can state a constraint and violate it in the same sentence — observed in
validation, where qwen2.5:7b wrote *"battery is already at high SOC"* and output `charge`.
Constraints enforced in the executor are structurally unviolatable.

---

## Files

| file | purpose |
|---|---|
| `config.py` | shared `SCHEMA` constant, imported by the run scripts |
| `providers.py` | pluggable LLM backends: ollama (default), openai, anthropic — see Providers below |
| `check_setup.py` | verifies CityLearn, the provider/model, latency, and the SOC bug before a long run |
| `rbc_v5.py` | rule brain + shared executor + SOC guard; self-tests in `__main__` |
| `llm_controller.py` | LLM brain: 4 guidance levels, district context, forecast block, diagnostics |
| `run_ladder.py` | runs one arm at a time, appends to `results_ladder.json` |
| `run_fullyear_rbc.py` | full-year RBC vs do-nothing (no LLM calls, ~2 min) |
| `run_transfer.py` | full-year RBC v5 across 3 CityLearn phases, gated on reproducing `run_fullyear_rbc.py`'s phase_1 result — see Transfer evaluation below |
| `job6_rate_sweep.py` | RBC v5 rate-multiplier sweep (k=0.4-1.2), gated on k=1.0 — see Rate sweep above |
| `job7_rate_sweep_extension.py` | extends the sweep down to k=0.10 to bracket L3's ramping point |
| `run_sac.py` | CityLearn's built-in SAC baseline |
| `run_compare.py` | fair 3-way do-nothing vs RBC v5 vs LLM comparison with latency/agreement diagnostics |
| `run_validation.py` | 10-scenario validation gate: parseability, valid mode, SOC guard, direction, per Ollama model |
| `check_horizon.py` | horizon sensitivity sweep (no LLM calls) |
| `plot_results.py` | district net-load figure from saved `net_series` |
| `results_ladder.json` | created by your own runs — starts empty, not shipped |
| `results_ladder_reference.json` | the author's original results, shipped for comparison; not appended to by your runs |
| `results_transfer.json` | `run_transfer.py` output: per-phase KPIs, mode counts, and the phase_1 gate result |
| `results_rate_sweep.json`, `results_rate_sweep_extension.json` | `job6`/`job7` output: per-k KPIs, gate result, L3 bracket analysis |
| `archive/` | superseded pre-v5 controllers and one-off probe scripts, kept for reference |

---

## Setup

```bash
conda create -n cl python=3.12 -y
conda activate cl
pip install "citylearn==2.5.0" --no-deps
pip install gymnasium pandas numpy scipy simplejson torch matplotlib

# local model runner
ollama pull qwen3:8b
ollama serve          # API at localhost:11434
```

Tested on Apple Silicon (M4, 36 GB). The `--no-deps` install avoids an `openstudio`
version pin that fails on ARM.

### Optional: cloud providers (OpenAI, Anthropic)

Only needed for `--provider openai` / `--provider anthropic` (see Providers below).
Keys are read from the environment — never pass one on the command line or hardcode it.

```bash
export OPENAI_API_KEY=...       # for --provider openai
export ANTHROPIC_API_KEY=...    # for --provider anthropic
```

No extra package install needed — both talk to their REST API directly over stdlib
`urllib`, same as the Ollama path.

---

## Running

```bash
# unit tests, no CityLearn needed
python rbc_v5.py
python llm_controller.py

# free baselines (instant)
python run_ladder.py --arm v0  --hours 48
python run_ladder.py --arm rbc --hours 48

# LLM arms — guidance level is the experimental variable
python run_ladder.py --arm L0 --hours 48 --district          # rule's answer shown
python run_ladder.py --arm L1 --hours 48 --district          # thresholds given
python run_ladder.py --arm L2 --hours 48 --district          # baseline only
python run_ladder.py --arm L3 --hours 48 --district          # raw history only

# options
#   --forecast          add price/solar forecast block
#   --max-dev N         clamp the LLM to within N mode-steps of the rule (0 = pure rule)
#   --mock              no network calls; instant plumbing check
#   --timeout N         request timeout in seconds (default 300, tuned on an M4)
#   --provider NAME     ollama (default) | openai | anthropic -- see Providers below
#   --model NAME        override the provider's default model
#   --max-tokens N      generation cap (num_predict / max_tokens), default 2000
#   --reason-words N    cap on the model's stated reason, default 8
#   --no-examples       omit the L1 few-shot examples
#   --tag NAME          label the run

python run_ladder.py --table    # combined comparison
```

Before a long run, verify your setup first: `python check_setup.py` (add `--provider` /
`--model` to check a non-default backend). It confirms CityLearn loads, the provider is
reachable/configured, measures 3 real call latencies against the 300 s default timeout,
and checks the known SOC-observation bug (see above) is present on your CityLearn version.

**Timing:** ~70–170 min per LLM arm (5 buildings × 48 h × 17–37 s per decision).
Use `caffeinate -is` — **macOS only.** Host sleep produces phantom timeouts that
corrupt runs; on Linux/Windows use your platform's equivalent (e.g. `systemd-inhibit`
on Linux, or just disable sleep for the duration of the run).

---

## Guidance levels

The experimental variable is **how much the rule tells the model**, not how it is instructed.

| level | the model is given | tests |
|---|---|---|
| **L0** | the rule's live recommendation; confirm or override | execution + deference |
| **L1** | the rule's numeric thresholds | execution |
| **L2** | the building's typical net load only | calibration |
| **L3** | last 6 hours of raw net load, nothing else | inference + calibration |

`--max-dev` additionally clamps the model's choice to within N steps of the rule on the
ordered scale (`charge < hold < discharge_soft < discharge_hard`). With `--max-dev 0` the
arm reproduces the RBC bit-for-bit — use this to verify the harness.

---

## Providers

The LLM backend is pluggable (`providers.py`). The prompts, retry/timeout/fallback
logic, and the executor clamping guarantee (the actual safety mechanism) are identical
no matter which model is answering — only the request format and the token-limit
parameter name differ.

| provider | default model | env var | token-limit param |
|---|---|---|---|
| `ollama` (default) | `qwen3:8b` | none — local server | `num_predict` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | `max_tokens` |
| `anthropic` | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` | `max_tokens` |

The cloud defaults are best-effort — verify against current provider docs and override
with `--model` if they've moved on. A missing API key raises immediately at construction,
not on the first call — a run that failed auth on every step would otherwise silently
fall back to the RBC action and look like a boring but "clean" result.

```bash
python run_ladder.py --arm L3 --hours 48 --provider openai --model gpt-4o
python run_ladder.py --arm L3 --hours 48 --provider anthropic --model claude-opus-4-5
python check_setup.py --provider openai        # verify before committing to a long run
```

Two prompt knobs were tuned for an 8B local model and may not suit a stronger one:

- **`--reason-words N`** (default 8) caps the model's stated reason. Tight enough to keep
  qwen3:8b terse; may hamstring a stronger model's actual reasoning.
- **`--no-examples`** drops the L1 few-shot examples. They anchor a small model toward the
  intended format but may over-constrain a larger one that would calibrate better unprompted.

All defaults reproduce the existing qwen3:8b results exactly — nothing above changes
behavior unless you pass the flag.

---

## Results summary (48 h, 5 buildings, qwen3:8b via ollama, temperature 0)

| arm | ramping | cost | carbon | agreement w/ rule |
|---|---|---|---|---|
| do nothing | 1.000 | 1.000 | 1.000 | — |
| RBC v5 | **0.755** | 0.878 | 0.881 | — |
| L0 anchored | 0.755 | 0.878 | 0.881 | 99.5% |
| L1 thresholds | 0.773 | 0.886 | 0.884 | 88.6% |
| L2 baseline only | 0.877 | 0.884 | 0.887 | 68.6% |
| L3 no guidance | 0.881 | **0.825** | **0.839** | 71.9%* |

\* inflated — parse failures fall back to the rule and count as agreement.

Results were produced by executing this code and verified against raw output.

**Full year, RBC v5 vs do-nothing:** ramping −22.0%, cost −19.3%, daily peak −14.7%,
carbon −12.2%. Ties on all-time peak; worse on zero-net-energy (round-trip losses).

**SAC (5 episodes, 721 s training):** loses to RBC v5 on all 8 KPIs and underperforms
do-nothing on 5. Expected at that training budget.

---

## Rate sweep: is L3's edge a strategy, or a hyperparameter the rule already has?

L3's phase_1 advantage over RBC v5 (cost 0.825 vs 0.878, carbon 0.839 vs 0.881, at the
cost of worse ramping: 0.881 vs 0.755) comes from acting less — fewer, smaller
charge/discharge decisions than the rule's fixed rates produce. Less cycling means less
round-trip loss. That's a rate parameter, not a strategy — so before crediting L3 with
anything, the question is whether RBC v5 already has access to the same tradeoff by
turning one knob.

**Method:** a global multiplier `k` applied jointly to `charge_rate`, `discharge_soft_rate`,
and `discharge_hard_rate` (nothing else changed — same thresholds, warmup, executor,
solar-only-charge rule). `k` swept across 14 values (0.10 through 1.2) on phase_1, at both
the 48 h ladder window and the full year, gated on `k=1.0` reproducing the known phase_1
reference at both windows before anything else was trusted.

**Result 1 — the baseline was not mis-tuned.** `k=1.0` (the shipped default) is the
full-year cost minimum across all 14 settings tested: cost_total 0.8075, versus the
nearest competitor (k=0.9) at 0.8076. No rate multiplier beats the default on cost at
full year.

**Result 2 — L3 sits outside the rule's achievable set at matched ramping.** At L3's
exact 48 h ramping value (0.8810), the rate sweep brackets it directly — k=0.25
(ramping 0.8613) below, k=0.20 (ramping 0.8860) above, a real bracket, not an
extrapolation. The interpolated rule cost at that ramping is **0.9134**. L3's actual
cost is **0.8250** — 0.088 lower than anything a rate-only retuning of the rule reaches
at the same ramping level.

**Both of the following are true, together — neither stands alone:**
- This is a **phase_1 in-sample result**. The rate sweep and the bracket above were
  computed entirely on the schema RBC v5's thresholds were tuned against.
- **L3's cost advantage over RBC v5 inverts on phase_2.** phase_1: L3 beats RBC v5 by
  0.053 on cost (0.825 vs 0.878). phase_2, same 48 h window, RBC v5 run fresh as an
  in-window comparator: L3 **trails** RBC v5 by 0.009 on cost (0.895 vs 0.886). The
  edge that survives the rate-sweep bracket on phase_1 does not survive a change of
  building.

---

## Transfer evaluation (full year, RBC v5, 3 CityLearn Challenge 2022 phases)

Every result above is on `citylearn_challenge_2022_phase_1` — the same five buildings
the RBC's thresholds were hand-tuned against. `run_transfer.py` re-runs the identical
controller, identical parameters, on buildings it has never seen.

| schema | buildings | C (cost) | G (carbon) | D (grid) |
|---|---|---|---|---|
| phase_1 (tuned on) | 5 | 0.807 | 0.878 | 0.874 |
| phase_2 | 5 | 0.801 | 0.856 | 0.887 |
| phase_3 | 7 | 0.897 | 0.921 | 0.887 |
| 2022 participant median | — | 0.792 | 0.940 | 0.996 |

`phase_1` here is the control condition, not a result: it reproduces
`run_fullyear_rbc.py`'s number **exactly** — all three KPIs (C/G/D) and all four mode
counts across the full 43,795 decisions (`charge: 11964, hold: 14310,
discharge_soft: 12147, discharge_hard: 5374`), computed through a completely
independent code path (`run_transfer.py` imports `RBCv5` from `rbc_v5.py` but reads
the environment itself differently). That exact match is the evidence this comparison
is trustworthy, not just the KPI table.

**Read it split, not as one number:** the grid indicator D — ramping and load-factor,
i.e. load shaping — is flat across all three schemas (0.874 / 0.887 / 0.887). Cost and
carbon hold on phase_2 but degrade on phase_3 (C 0.807→0.897, G 0.878→0.921).
**Load-shaping transfers to unseen buildings; energy arbitrage does not.**

### Caveats

- **Only two unseen schemas.** This is n=2, not a distribution — "holds on phase_2,
  degrades on phase_3" is two data points, not a trend.
- **phase_2 vs phase_3 confounds building count with building identity.** phase_2 has
  5 buildings like phase_1; phase_3 has 7. The phase_3 degradation could come from
  building-identity effects, building-count effects, or both — this comparison alone
  cannot separate them.
- **phase_1 is the only schema the thresholds were fitted to.** `hi_k`, `mid_k`,
  `lo_k`, and the rate parameters were hand-tuned by looking at phase_1 behavior.
  Nothing here was retuned for phase_2/phase_3 (that would defeat the point of a
  transfer test), but it also means phase_1's own number is a training-set score,
  not an unbiased estimate of anything.

---

## Reproducibility

qwen3 at temperature 0 is **deterministic**. Three of four identical L3 runs were
bit-identical, each with exactly 9 timeouts; the fourth had 14 and differed by 8.6% on
ramping. All observed run-to-run variance traced to inference timeouts converting
decisions into fallbacks — not to the model.

Runs sleep-interrupted by the host produce phantom timeouts. Use `caffeinate` (macOS only — see Running above for other platforms).

---

## Known limitations

- Single run per configuration (except L3, ×4). Deterministic, but not statistically powered.
- LLM guidance-ladder results (L0–L3) are `phase_1` only. RBC v5 has been evaluated
  across three CityLearn Challenge 2022 phases (see Transfer evaluation above); the LLM
  controller has not.
- 48 h windows; `daily_peak_average` and `zero_net_energy` are not valid at this horizon.
- Five LLM calls per hour makes full-year evaluation infeasible — call-cadence strategy is
  the open engineering problem.
- Each building's LLM call sees only its own state plus limited district aggregates.
  This is parallel single-agent control, **not** coordinated multi-agent control.
- CityLearn building attribute arrays (`net_electricity_consumption_*`,
  `solar_generation`) are finalized **lazily during `step()`** — the newest entry at
  read time is an unfinalized placeholder, valid only after the *next* `step()` call.
  Read current-step load and solar from the **observation vector**
  (`env.observation_names` → a name→index map built fresh per schema, since the layout
  changes with building count — see `build_index()` in `run_fullyear_rbc.py` /
  `run_transfer.py`) instead of these arrays. `true_soc()` is the one deliberate
  exception: `electrical_storage_soc` is broken in the observation vector itself (see
  above), so reading the battery object directly is unavoidable there — the fix is a
  *lagged* index into that array (`t_step - 1`), not a switch to the observation.

---

## References

- Song et al., *Pre-Trained Large Language Models for Industrial Control*, arXiv:2308.03028
- Vázquez-Canteli et al., *MARLISA*, BuildSys '20 — coordination via shared district signals; +3.8% over its RBC
- Nweye et al., *MERLIN*, Applied Energy 2023 — offline RL and policy transfer
- Nweye et al. 2022, *Energy and AI* — well-tuned RBCs can outperform SAC on flexibility KPIs
- Nweye et al., *CityLearn v2*, arXiv:2405.03848
