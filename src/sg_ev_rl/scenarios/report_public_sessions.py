"""Out-of-sample check of the public-charging session generator, for reports/c/YYYY-MM-DD_public_sessions.md.

The profile is calibrated on 2026-10-26..11-09; generated sessions on every public port
are compared with the observed pseudo sessions of 2026-09-09..10-24 (complete days of the
real-scenario validation period).

Run:  PYTHONPATH=src python -m sg_ev_rl.scenarios.report_public_sessions --out reports/c
"""
import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from sg_ev_rl.market.report_price_diagnosis import INK, INK2, SERIES, SURFACE, _style

from . import public_charging as pc

CHECK_START, CHECK_DAYS = date(2026, 9, 9), 46


def occupancy_profile(s: pd.DataFrame, n_ports: dict[str, int], start: date, n_days: int) -> pd.DataFrame:
    """Mean occupied share by (current, day_type, half-hour) on the 5-minute grid."""
    grid = pd.date_range(pd.Timestamp(start), periods=n_days * pc.STEPS_PER_DAY, freq=f"{pc.STEP_MIN}min")
    g = grid.to_numpy()
    dtype = np.array(pc.DAY_TYPES)[pc.day_type_of(grid.normalize())]
    out = []
    for c, sc in s.groupby("current"):
        delta = np.zeros(len(g) + 1)
        np.add.at(delta, np.searchsorted(g, sc["arrival_ts"].to_numpy()), 1)
        np.add.at(delta, np.searchsorted(g, sc["departure_ts"].to_numpy()), -1)
        busy = np.cumsum(delta)[:-1] / n_ports[c]
        df = pd.DataFrame({"current": c, "day_type": dtype, "slot": grid.hour * 2 + grid.minute // 30, "share": busy})
        out.append(df.groupby(["current", "day_type", "slot"])["share"].mean())
    return pd.concat(out)


def summary(g: pd.DataFrame, r: pd.DataFrame, n_ports: dict[str, int]) -> pd.DataFrame:
    rows = []
    for c in pc.CURRENTS:
        for dt in pc.DAY_TYPES:
            row = {"current": c, "day_type": dt}
            for name, s in (("generated", g), ("observed", r)):
                s = s[(s["current"] == c)]
                s = s[np.array(pc.DAY_TYPES)[pc.day_type_of(pd.DatetimeIndex(s["arrival_ts"]).normalize())] == dt]
                n_days = sum(np.array(pc.DAY_TYPES)[pc.day_type_of(
                    pd.date_range(CHECK_START, periods=CHECK_DAYS))] == dt)
                row[f"{name}_per_port_day"] = len(s) / n_ports[c] / n_days
                for q in (25, 50, 75, 95):
                    row[f"{name}_dur_p{q}_min"] = s["duration_min"].quantile(q / 100)
                row[f"{name}_energy_mean_kwh"] = s["energy_kwh"].mean()
                row[f"_{name}_dur"] = s["duration_min"].to_numpy()
            row["ks_duration"] = ks_2samp(row.pop("_generated_dur"), row.pop("_observed_dur")).statistic
            rows.append(row)
    return pd.DataFrame(rows)


def fig_occupancy(prof: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, facecolor=SURFACE)
    for i, c in enumerate(pc.CURRENTS):
        for j, dt in enumerate(pc.DAY_TYPES):
            ax = axes[i, j]
            _style(ax)
            for k, name in enumerate(("observed", "generated")):
                y = prof.loc[(name, c, dt)] * 100
                ax.plot(y.index / 2, y.values, color=SERIES[k], lw=2, label=name)
            ax.set_title(f"{c}, {dt}", color=INK, fontsize=10, loc="left")
            ax.set_ylabel("% ports occupied", color=INK2, fontsize=9)
            ax.set_xticks(range(0, 25, 6))
    for ax in axes[1]:
        ax.set_xlabel("start of half-hour (SGT)", color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Port occupancy, observed vs generated (out of sample, 2026-09-09..10-24)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_duration(g: pd.DataFrame, r: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, c in zip(axes, pc.CURRENTS, strict=True):
        _style(ax)
        for k, (name, s) in enumerate((("observed", r), ("generated", g))):
            d = np.sort(s.loc[s["current"] == c, "duration_min"].to_numpy() / 60)
            ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=SERIES[k], lw=2, label=name)
        ax.set_xscale("log")
        ax.set_xticks([1 / 6, 0.5, 1, 2, 4, 8, 16])
        ax.set_xticklabels(["10m", "30m", "1h", "2h", "4h", "8h", "16h"])
        ax.set_title(f"{c} plug-in duration", color=INK, fontsize=10, loc="left")
        ax.set_ylabel("share of sessions ≤ x", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/c"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fig_dir, tab_dir = args.out / "figures", args.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    sess, ports, snaps = pc.load_calibration_inputs()
    prof = pc.calibrate(sess, ports, snaps)
    portfolio = pc.portfolio_from_share(ports, 1.0)  # every public port
    n_ports = {c: sum(n for (_, cc), n in portfolio.items() if cc == c) for c in pc.CURRENTS}
    g = pc.generate(prof, portfolio, CHECK_START, CHECK_DAYS, seed=args.seed)
    r = pc.real_sessions(sess, ports, portfolio, CHECK_START, CHECK_DAYS, seed=args.seed)

    tab = summary(g, r, n_ports)
    tab.round(3).to_csv(tab_dir / "public_sessions_check.csv", index=False)
    occ = pd.concat({"observed": occupancy_profile(r, n_ports, CHECK_START, CHECK_DAYS),
                     "generated": occupancy_profile(g, n_ports, CHECK_START, CHECK_DAYS)})
    occ.unstack(0).round(4).to_csv(tab_dir / "public_sessions_occupancy_profile.csv")
    gap = (occ.loc["generated"] - occ.loc["observed"]).abs()
    by_zone = pd.DataFrame({
        "generated": g.groupby(["zone", "current"]).size(),
        "observed": r.groupby(["zone", "current"]).size()})
    by_zone["ratio"] = by_zone["generated"] / by_zone["observed"]
    by_zone.round(3).to_csv(tab_dir / "public_sessions_by_zone.csv")
    meta = {
        "calibration_dates": [str(d) for d in prof.calibration_dates],
        "calibration_days": prof.meta["n_days"],
        "calibration_sessions": prof.n_sessions,
        "check_window": [str(CHECK_START), str((pd.Timestamp(CHECK_START) + pd.Timedelta(days=CHECK_DAYS - 1)).date())],
        "ports": n_ports,
        "sessions_generated": len(g),
        "sessions_observed": len(r),
        "occupancy_mae_pct_points": round(float(gap.mean() * 100), 2),
        "occupancy_max_abs_gap_pct_points": round(float(gap.max() * 100), 2),
        "vehicle_assumptions": pc.VehicleAssumptions().__dict__,
    }
    (tab_dir / "public_sessions_check.json").write_text(json.dumps(meta, indent=2, default=str))

    fig_occupancy(occ, fig_dir / "public_sessions_occupancy.png")
    fig_duration(g, r, fig_dir / "public_sessions_duration.png")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(json.dumps(meta, indent=2, default=str))
        print(tab.round(2).T)
        print(by_zone.round(3))


if __name__ == "__main__":
    main()
