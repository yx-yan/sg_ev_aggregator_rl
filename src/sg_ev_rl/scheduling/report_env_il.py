"""Interruptible load in the environment: rule baselines, LP bound and LP replay, plus an activation stress test.

Same episodes as report_env_lp (100 public ports, 7-day episodes, train and validation). Rules offer
all the IL their dispatched load can back (b = 1). B2's percentile p is re-chosen on validation for
net cost (energy cost minus IL revenue).

Run:  PYTHONPATH=src python -m sg_ev_rl.scheduling.report_env_il --out reports/b
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from sg_ev_rl.market.features import add_relative_price_features
from sg_ev_rl.market.loader import load_market
from sg_ev_rl.scenarios import public_charging as pc
from sg_ev_rl.scenarios.episodes import build_episode, episode_starts, seed_for
from sg_ev_rl.settlement.report import settle_episode
from sg_ev_rl.sim.fleet_env import EnvConfig

from . import fleet_rules as R
from .service_lp import ServiceConfig, solve_service_lp

N_DAYS, N_PORTS = 7, 100
P_GRID = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]  # p = 1 is B1 with IL
SPLITS = ["train", "validation"]
STRESS_PROB = 0.05  # ASSUMPTION for the stress test only: 5 % of offered steps are activated


def run_episode(ep, k) -> list[dict]:
    rows = []

    def add(name, core, extra=None):
        rows.append({"policy": name, **settle_episode(core), **(extra or {})})

    add("B1_uncontrolled", R.run(ep, R.Uncontrolled()))
    add("B1_il", R.run(ep, R.Uncontrolled(), il_share=1.0))
    for p in P_GRID:
        add(f"B2_p{p:g}_il", R.run(ep, R.PriceThreshold(p), il_share=1.0))
    lp = solve_service_lp(ep, ServiceConfig(interruptible_load=True))
    add("LP_il_per_ev_replay", R.run_schedule(ep, lp.charge, il_kw=lp.il_kw),
        {"lp_objective_sgd": lp.total_cost_sgd, "lp_solve_s": lp.solve_s})
    agg = lp.charge.sum(axis=0)
    add("LP_il_aggregate_llf", R.run(ep, R.ReplayAggregate(agg), il_share=1.0))
    stress = EnvConfig(il_activation_prob=STRESS_PROB)
    add("B1_il_stress", R.run(ep, R.Uncontrolled(), stress, il_share=1.0, seed=k))
    add("LP_il_aggregate_llf_stress", R.run(ep, R.ReplayAggregate(agg), stress, il_share=1.0, seed=k))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/b"))
    args = ap.parse_args()
    tab_dir = args.out / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    market = add_relative_price_features(load_market())
    sess, ports, snaps = pc.load_calibration_inputs()
    profile = pc.calibrate(sess, ports, snaps)
    out = []
    for split in SPLITS:
        for k, start in enumerate(episode_starts(split, N_DAYS)):
            ep = build_episode(profile, market, start, N_DAYS, N_PORTS, ports, seed=seed_for("public", split, k))
            for r in run_episode(ep, k):
                out.append({"split": split, "episode": k, "start": start, **r})
    df = pd.DataFrame(out)
    df.to_csv(tab_dir / "env_il_episodes.csv", index=False)

    val = df[(df["split"] == "validation") & df["policy"].str.startswith("B2_")]
    best = val.groupby("policy")["net_cost_sgd"].sum().idxmin()
    keep = ["B1_uncontrolled", "B1_il", best, "LP_il_aggregate_llf", "LP_il_per_ev_replay",
            "B1_il_stress", "LP_il_aggregate_llf_stress"]
    d = df[df["policy"].isin(keep)].copy()
    base = d[d["policy"] == "B1_uncontrolled"].set_index(["split", "episode"])["net_cost_sgd"]
    d["saving_vs_b1_per_port_day"] = ((d.set_index(["split", "episode"]).index.map(base) - d["net_cost_sgd"])
                                      / N_PORTS / N_DAYS)
    summary = d.groupby(["split", "policy"]).agg(
        episodes=("episode", "nunique"),
        energy_per_port_day=("cost_per_port_day_sgd", "mean"),
        net_per_port_day=("net_cost_per_port_day_sgd", "mean"),
        saving_vs_b1_per_port_day=("saving_vs_b1_per_port_day", "mean"),
        il_mw_mean=("il_mw_mean", "mean"),
        activations=("il_activations", "sum"),
        unmet_kwh=("unmet_kwh", "sum")).reindex(keep, level=1)
    summary.round(4).to_csv(tab_dir / "env_il_summary.csv")

    lp_rows = df[df["policy"] == "LP_il_per_ev_replay"]
    err = ((lp_rows["net_cost_sgd"] - lp_rows["lp_objective_sgd"]).abs() / lp_rows["lp_objective_sgd"].abs())
    lp_net = lp_rows.set_index(["split", "episode"])["net_cost_sgd"]
    agg = df[df["policy"] == "LP_il_aggregate_llf"].set_index(["split", "episode"])["net_cost_sgd"]
    gap = (agg / lp_net - 1) * 100
    plain = df[df["policy"] == "LP_il_aggregate_llf"].set_index(["split", "episode"])
    stress = df[df["policy"] == "LP_il_aggregate_llf_stress"].set_index(["split", "episode"])
    meta = {
        "n_ports": N_PORTS, "episode_days": N_DAYS, "episodes": int(df.groupby("split")["episode"].nunique().sum()),
        "b2_p_chosen_on_validation_net_cost": best,
        "per_ev_replay_max_rel_error": float(err.max()),
        "aggregate_llf_gap_vs_lp_pct": {"mean": round(float(gap.mean()), 3), "median": round(float(gap.median()), 3),
                                        "max": round(float(gap.max()), 3)},
        "stress_activation_prob": STRESS_PROB,
        "stress_activations_total": int(df[df["policy"].str.endswith("_stress")]["il_activations"].sum()),
        "stress_unmet_kwh_total": float(df[df["policy"].str.endswith("_stress")]["unmet_kwh"].sum()),
        "stress_extra_energy_cost_pct_lp_aggregate": round(float(
            (stress["energy_cost_sgd"] / plain["energy_cost_sgd"] - 1).mean() * 100), 3),
        "unmet_kwh_all_non_stress": float(df[~df["policy"].str.endswith("_stress")]["unmet_kwh"].sum()),
        "env_config": EnvConfig().__dict__,
    }
    (tab_dir / "env_il_check.json").write_text(json.dumps(meta, indent=2, default=str))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(json.dumps(meta, indent=2, default=str))
        print(summary.round(3))


if __name__ == "__main__":
    main()
