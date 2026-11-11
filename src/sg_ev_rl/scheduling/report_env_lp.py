"""Benchmarks B1, B2 and the perfect-foresight LP on the charge-only environment, plus the LP replay check.

Writes tables and figures for reports/b/YYYY-MM-DD_env_lp.md. Uses the train and validation
splits only (7-day episodes, back to back). B2's percentile p is chosen on validation.

Run:  PYTHONPATH=src python -m sg_ev_rl.scheduling.report_env_lp --out reports/b
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sg_ev_rl.market.features import add_relative_price_features
from sg_ev_rl.market.loader import load_market
from sg_ev_rl.market.report_price_diagnosis import INK, INK2, SERIES, SURFACE, _style
from sg_ev_rl.scenarios import public_charging as pc
from sg_ev_rl.scenarios.episodes import build_episode, episode_starts, seed_for
from sg_ev_rl.settlement.report import settle_episode
from sg_ev_rl.sim.fleet_env import EnvConfig

from . import fleet_rules as R
from .lp import solve_charge_lp

N_DAYS = 7
P_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SPLITS = ["train", "validation"]


def run_episode(ep, b2_ps) -> list[dict]:
    rows = []

    def add(name, core, extra=None):
        rows.append({"policy": name, **settle_episode(core), **(extra or {})})

    add("B1_uncontrolled", R.run(ep, R.Uncontrolled()))
    for p in b2_ps:
        add(f"B2_p{p:g}", R.run(ep, R.PriceThreshold(p)))
    lp = solve_charge_lp(ep)
    t0 = time.perf_counter()
    per_ev = R.run_schedule(ep, lp.schedule)
    add("LP_per_ev_replay", per_ev, {"lp_objective_sgd": lp.cost_sgd, "lp_solve_s": lp.solve_s,
                                     "lp_vars": lp.n_vars, "replay_s": time.perf_counter() - t0})
    agg = lp.schedule.sum(axis=0)
    for prio in ("llf", "ratio"):
        add(f"LP_aggregate_{prio}", R.run(ep, R.ReplayAggregate(agg), EnvConfig(dispatch_priority=prio)))
    return rows


def run_split(profile, market, ports, split, n_ports, b2_ps, limit=None):
    out = []
    for k, start in enumerate(episode_starts(split, N_DAYS)[:limit]):
        ep = build_episode(profile, market, start, N_DAYS, n_ports, ports, seed=seed_for("public", split, k))
        for r in run_episode(ep, b2_ps):
            out.append({"split": split, "episode": k, "start": start, "n_ports": n_ports,
                        "mean_usep": float(ep.usep.mean()), **r})
    return pd.DataFrame(out)


def solve_stats(lp_rows: pd.DataFrame) -> dict:
    return {"median_s": float(lp_rows["lp_solve_s"].median()), "max_s": float(lp_rows["lp_solve_s"].max()),
            "vars_median": int(lp_rows["lp_vars"].median()), "sessions_median": int(lp_rows["sessions"].median())}


def vs_b1_mean(df: pd.DataFrame, policy: str) -> float:
    a = df[df["policy"] == policy]["energy_cost_sgd"].to_numpy()
    b = df[df["policy"] == "B1_uncontrolled"]["energy_cost_sgd"].to_numpy()
    return round(float((a / b - 1).mean() * 100), 3)


def fig_relative(df: pd.DataFrame, b2: str, path: Path):
    base = df[df["policy"] == "B1_uncontrolled"].set_index(["split", "episode"])["energy_cost_sgd"]
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURFACE)
    _style(ax)
    names = [(b2, "B2 price threshold"), ("LP_aggregate_llf", "LP profile via LLF"),
             ("LP_per_ev_replay", "LP per EV (bound)")]
    for k, (pol, label) in enumerate(names):
        s = df[df["policy"] == pol].set_index(["split", "episode"])
        y = (s["energy_cost_sgd"] / base - 1) * 100
        x = pd.to_datetime(s["start"])
        ax.plot(x, y.values, color=SERIES[k], lw=2, marker="o", ms=3, label=label)
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(pd.Timestamp("2026-01-01"), color=INK2, lw=1, ls=":")
    ax.text(pd.Timestamp("2026-01-04"), ax.get_ylim()[1], "validation", color=INK2, fontsize=8, va="top")
    ax.set_ylabel("energy cost vs B1 uncontrolled (%)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Weekly energy cost relative to uncontrolled charging, 100 public ports, charge only",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/b"))
    ap.add_argument("--n-ports", type=int, default=100)
    args = ap.parse_args()
    fig_dir, tab_dir = args.out / "figures", args.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    market = add_relative_price_features(load_market())
    sess, ports, snaps = pc.load_calibration_inputs()
    profile = pc.calibrate(sess, ports, snaps)

    df = pd.concat([run_split(profile, market, ports, s, args.n_ports, P_GRID) for s in SPLITS], ignore_index=True)
    # choose p on validation, report it everywhere
    val = df[(df["split"] == "validation") & df["policy"].str.startswith("B2_")]
    best_p = val.groupby("policy")["energy_cost_sgd"].sum().idxmin()
    df.to_csv(tab_dir / "env_lp_episodes.csv", index=False)

    keep = ["B1_uncontrolled", best_p, "LP_aggregate_llf", "LP_aggregate_ratio", "LP_per_ev_replay"]
    d = df[df["policy"].isin(keep)]
    base = d[d["policy"] == "B1_uncontrolled"].set_index(["split", "episode"])["energy_cost_sgd"]
    d = d.assign(vs_b1_pct=(d.set_index(["split", "episode"])["energy_cost_sgd"] / base - 1).values * 100)
    summary = d.groupby(["split", "policy"]).agg(
        episodes=("episode", "nunique"), cost_per_port_day_sgd=("cost_per_port_day_sgd", "mean"),
        avg_price_sgd_mwh=("avg_price_sgd_mwh", "mean"), vs_b1_pct_mean=("vs_b1_pct", "mean"),
        vs_b1_pct_min=("vs_b1_pct", "min"), vs_b1_pct_max=("vs_b1_pct", "max"),
        peak_kw_mean=("peak_kw", "mean"), unmet_kwh_total=("unmet_kwh", "sum")).reindex(keep, level=1)
    summary.round(3).to_csv(tab_dir / "env_lp_summary.csv")

    lp_rows = df[df["policy"] == "LP_per_ev_replay"]
    replay_err = ((lp_rows["energy_cost_sgd"] - lp_rows["lp_objective_sgd"]).abs() / lp_rows["lp_objective_sgd"])
    lp_cost = lp_rows.set_index(["split", "episode"])["energy_cost_sgd"]
    gaps = {}
    for prio in ("llf", "ratio"):
        a = df[df["policy"] == f"LP_aggregate_{prio}"].set_index(["split", "episode"])["energy_cost_sgd"]
        g = (a / lp_cost - 1) * 100
        gaps[prio] = {"mean_pct": round(float(g.mean()), 3), "median_pct": round(float(g.median()), 3),
                      "max_pct": round(float(g.max()), 3), "share_within_0p5pct": round(float((g <= 0.5).mean()), 3)}

    # scale check: 1,000 ports, first 4 training weeks
    big = run_split(profile, market, ports, "train", 1000, [], limit=4)
    big_lp = big[big["policy"] == "LP_per_ev_replay"]
    meta = {
        "n_ports": args.n_ports, "episode_days": N_DAYS,
        "episodes": int(df.groupby("split")["episode"].nunique().sum()),
        "b2_p_chosen_on_validation": best_p,
        "per_ev_replay_max_rel_error": float(replay_err.max()),
        "per_ev_replay_pass_0p5pct": bool((replay_err <= 0.005).all()),
        "aggregate_replay_gap_vs_lp": gaps,
        "unmet_kwh_all_policies": float(df["unmet_kwh"].sum()),
        "lp_solve_100_ports": solve_stats(lp_rows),
        "lp_solve_1000_ports": solve_stats(big_lp),
        "scale_1000_vs_b1_pct": {p: vs_b1_mean(big, p) for p in ("LP_aggregate_llf", "LP_per_ev_replay")},
        "sessions_per_episode_median": int(df[df["policy"] == "B1_uncontrolled"]["sessions"].median()),
        "feasible_at_arrival_note": "soc_target is capped to what the plug-in time allows; see public_charging",
        "env_config": EnvConfig().__dict__,
    }
    (tab_dir / "env_lp_check.json").write_text(json.dumps(meta, indent=2, default=str))
    fig_relative(df, best_p, fig_dir / "weekly_cost_vs_uncontrolled.png")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(json.dumps(meta, indent=2, default=str))
        print(summary.round(3))


if __name__ == "__main__":
    main()
