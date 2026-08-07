"""
Figure for the Week-10 talk: district net load over the evaluation window, one
line per controller. Reads `net_series` from results_ladder.json.

    python plot_results.py                      # auto-picks v0, rbc, and LLM arms
    python plot_results.py --arms v0 rbc L0_dev1
    python plot_results.py --out figure.png --dpi 200

Requires `net_series` in the results — see save_series_patch.md. Arms without it
are skipped with a note.

Design notes: this is for a 3-minute talk projected on a screen, so fonts are
large, lines are thick, and there is no chartjunk. One idea per figure.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results_ladder.json"

# colourblind-safe, high contrast when projected
STYLE = {
    "v0":  dict(color="#999999", ls="--", lw=2.5, label="do nothing"),
    "rbc": dict(color="#0072B2", ls="-",  lw=3.0, label="rule-based controller"),
    "L0":  dict(color="#009E73", ls="-",  lw=3.0, label="LLM (RBC-anchored)"),
    "L1":  dict(color="#56B4E9", ls="-",  lw=2.5, label="LLM (thresholds given)"),
    "L2":  dict(color="#E69F00", ls="-",  lw=2.5, label="LLM (baseline only)"),
    "L3":  dict(color="#D55E00", ls="-",  lw=3.0, label="LLM (no guidance)"),
}


def style_for(tag):
    for key in ("v0", "rbc", "L0", "L1", "L2", "L3"):
        if tag == key or tag.startswith(key + "_"):
            s = dict(STYLE[key])
            if tag != key:
                s["label"] += f"  [{tag}]" if key.startswith("L") else ""
            return s
    return dict(lw=2.0, label=tag)


def load():
    if not os.path.exists(RESULTS):
        raise SystemExit(f"{RESULTS} not found — run from the validation folder")
    with open(RESULTS) as f:
        return json.load(f)


def pick_default(data):
    """v0 and rbc, plus the LLM arms that actually have a series."""
    have = [t for t, r in data.items() if r.get("net_series")]
    out = [t for t in ("v0", "rbc") if t in have]
    out += [t for t in have if t.startswith("L")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--out", default="net_load_figure.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--hours", type=int, default=None, help="trim the x-axis")
    a = ap.parse_args()

    data = load()
    arms = a.arms or pick_default(data)

    usable, skipped = [], []
    for t in arms:
        if t not in data:
            skipped.append(f"{t} (not in results)")
        elif not data[t].get("net_series"):
            skipped.append(f"{t} (no net_series — ran before the patch)")
        else:
            usable.append(t)

    if skipped:
        print("skipped:")
        for s in skipped:
            print("   ", s)
    if not usable:
        raise SystemExit("nothing to plot — apply save_series_patch.md, then "
                         "re-run: python run_ladder.py --arm v0 --hours 48")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    peaks = {}
    for t in usable:
        y = np.asarray(data[t]["net_series"], dtype=float)
        if a.hours:
            y = y[:a.hours]
        x = np.arange(len(y))
        ax.plot(x, y, **style_for(t))
        peaks[t] = float(y.max())

    # shade evening peak hours across every day in the window
    n = max(len(data[t]["net_series"]) for t in usable)
    for day in range(int(np.ceil(n / 24))):
        lo, hi = day * 24 + 16, day * 24 + 21
        if lo < n:
            ax.axvspan(lo, min(hi, n - 1), color="#000000", alpha=0.05, zorder=0)

    ax.set_xlabel("hour of simulation", fontsize=15)
    ax.set_ylabel("district net grid load (kW)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=13, frameon=False, ncol=2, loc="upper left")

    # headline annotation: best vs do-nothing on ramping, if available
    if "v0" in usable:
        best = min((t for t in usable if t != "v0"),
                   key=lambda t: data[t]["raw"]["ramping"], default=None)
        if best:
            r0 = data["v0"]["raw"]["ramping"]
            r1 = data[best]["raw"]["ramping"]
            ax.set_title(
                f"Battery control flattens district load  —  "
                f"ramping {100*(1-r1/r0):.0f}% lower than doing nothing",
                fontsize=16, pad=14)

    fig.text(0.99, 0.01, "shaded = 16:00–21:00", ha="right", fontsize=10,
             color="#666666")
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"\nwrote {a.out}  ({', '.join(usable)})")

    print(f"\n{'arm':<14}{'peak kW':>10}{'ramping':>10}")
    for t in usable:
        print(f"{t:<14}{peaks[t]:>10.2f}{data[t]['raw']['ramping']:>10.2f}")


if __name__ == "__main__":
    main()
