# Patch for `run_ladder.py` — add the L0 arm and clamping

Two small edits. Give this file to Claude Code and it can apply them, or do it yourself.

## Edit 1 — allow `L0` as an arm and add a `--max-dev` flag

In `main()`, change the argparse lines:

```python
ap.add_argument("--arm", choices=["v0", "rbc", "L0", "L1", "L2", "L3"])
```

and add:

```python
ap.add_argument("--max-dev", type=int, default=None,
                help="clamp the LLM to within N mode-steps of the RBC "
                     "(0 = pure RBC, 1 = one step of freedom, omit = unclamped)")
```

## Edit 2 — pass the new options through in `run_arm`

Change the signature:

```python
def run_arm(arm, hours, mock, district, max_dev=None, verbose=True):
```

and the controller construction:

```python
    elif arm in ("L0", "L1", "L2", "L3"):
        llm = LLMController(params=PARAMS, mock=mock,
                            guidance_level=int(arm[1]),
                            district_context=district,
                            max_deviation=max_dev)
        policy = llm
```

Then in `main()`, pass it:

```python
    rec = run_arm(a.arm, a.hours, a.mock, a.district, a.max_dev)
```

and include it in the tag so runs don't overwrite each other:

```python
    tag = a.tag or (a.arm + ("_dist" if a.district else "")
                    + (f"_dev{a.max_dev}" if a.max_dev is not None else "")
                    + ("_mock" if a.mock else ""))
```

## Edit 3 — record the new diagnostics

In `run_arm`, inside the `if llm is not None:` block, add:

```python
        rec["llm"]["override_rate"] = round(llm.override_rate(), 4)
        rec["llm"]["overrides"] = llm.overrides
        rec["llm"]["clamped"] = llm.clamped
        rec["llm"]["sample_override_reasons"] = llm.override_reasons[:5]
```

And in `print_table()`, add two columns to the diagnostics header/rows:

```python
    print(f"{'tag':<14}{'lvl':>4}{'agree':>8}{'ovr':>6}{'clmp':>6}{'dev':>7}"
          f"{'compl':>8}{'to':>4}{'med s':>7}")
    ...
        print(f"{t[:14]:<14}{L['level']:>4}{L['agreement']:>8.1%}"
              f"{L.get('override_rate',0):>6.0%}{L.get('clamped',0):>6}"
              f"{L['mean_deviation']:>7.2f}{L['compliance']:>8.0%}"
              f"{L['timeouts']:>4}{(L.get('latency') or {}).get('median',0):>7.1f}")
```

---

# What to run this week

Three arms, ~70 min each. Run overnight if needed.

```bash
# 1. RBC-anchored, unclamped — does showing the rule's answer keep the baseline?
caffeinate -is python run_ladder.py --arm L0 --hours 48 --district --tag L0_free

# 2. RBC-anchored, clamped to one step — the "guaranteed" version
caffeinate -is python run_ladder.py --arm L0 --hours 48 --district --max-dev 1 --tag L0_dev1

# 3. sanity check: max-dev 0 must reproduce the RBC exactly
python run_ladder.py --arm L0 --hours 48 --district --max-dev 0 --mock --tag L0_dev0_mock
```

Run #3 first — it's instant and it validates the whole mechanism. With `--max-dev 0`
the LLM's opinion is entirely overridden, so the KPIs **must** match the `rbc` arm
exactly. If they don't, the clamping is broken and nothing else is trustworthy.

# What each result means

| outcome | interpretation |
|---|---|
| L0_free ≈ RBC, low override rate | the anchor works; the model defers sensibly |
| L0_free better than RBC | **the headline** — anchored overrides improve on the rule |
| L0_free worse than RBC | overrides are harmful; the clamp is then load-bearing |
| L0_dev1 ≥ RBC on every KPI | the guarantee holds — this is what the advisor asked for |
| override rate near 0% | the model is just parroting; guidance is too strong |
| override rate very high | it's ignoring the anchor; the prompt isn't binding |

The number to watch is **override rate**. It tells you whether "confirm or override"
is producing genuine selective disagreement or just compliance. Read the
`sample_override_reasons` too — those are your interpretability evidence, and the
best material for a 3-minute talk.
