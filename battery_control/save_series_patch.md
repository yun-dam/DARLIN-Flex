# Patch: save the hourly load series so figures are possible

`results_ladder.json` currently stores only summary numbers (peak, ramping, mean).
The hourly district net-load series is computed and thrown away — so no plots can
be made without re-running an arm.

**Apply this now.** `L0_dev1` hasn't launched yet, so it will pick up the change
and capture its series for free. (Editing the file does not affect the already-
running `L0_free` process — Python loaded that module at start.)

## The edit — one line in `run_arm()`

Find where the record is built:

```python
    net = district_net_series(env, hours)
    rec = {
        "arm": arm, "hours": hours, "district": district, "mock": mock,
        "runtime_s": round(time.time() - t0, 1),
        "raw": {"peak": float(net.max()),
                "ramping": float(np.abs(np.diff(net)).sum()),
                "mean": float(net.mean())},
        "kpi": kpis(env),
        "end_soc": float(np.mean([true_soc(env, b, t) for b in range(n_b)])),
    }
```

Add one key:

```python
        "net_series": [float(x) for x in net],     # <-- hourly district net load
```

That's it. ~48 numbers per arm; the file stays small.

## Then re-capture the two free arms (instant)

```bash
python run_ladder.py --arm v0  --hours 48
python run_ladder.py --arm rbc --hours 48
```

These overwrite the existing `v0` and `rbc` entries with identical KPIs plus the
series. Deterministic, so nothing changes except the added data.

## Result

- `v0`, `rbc` — series captured immediately
- `L0_dev1` — captured automatically when it runs
- `L0_free`, `L1_dist`, `L2_dist`, `L3_dist` — no series (ran before the patch).
  Only re-run one of these if you specifically want it in the figure; `L3_dist`
  is the one worth 70 minutes since it carries the headline result.
