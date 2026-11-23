"""B4 rolling MPC against the rules, PPO and the LP bounds, on the PPO report's episodes.

Same episodes as reports/d/2026-11-11_ppo_il.md (300 public ports, 7-day episodes, 12 validation
weeks and 8 train weeks, seeded as in rl.common.fixed_episodes). Baselines are recomputed and must
match the stored PPO tables, which shows the episodes are the same; PPO rows are read from those tables.

Run:  PYTHONPATH=src python -m sg_ev_rl.scheduling.report_mpc --out reports/d
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sg_ev_rl.forecast.prices import METHODS, PriceForecast
from sg_ev_rl.rl.common import baselines, fixed_episodes, load_context, rl_env_config
from sg_ev_rl.rl.report_rl import N_TRAIN_EPS
from sg_ev_rl.settlement.report import settle_episode

from .mpc import CONTROLS, RollingMPC, run_mpc

N_PORTS, N_DAYS = 300, 7
ORDER = ["B1_uncontrolled", "B1_il", "B2_p0.9_il", "PPO_mean",
         *(f"B4_mpc_{m}_{c}" for c in CONTROLS for m in METHODS if m != "perfect"),
         "LP_il_aggregate_llf", "B4_mpc_perfect_aggregate", "B4_mpc_perfect_per_ev", "LP_il_per_ev"]


def run_mpcs(ctx, eps) -> tuple[pd.DataFrame, dict]:
    rows, timing = [], {}
    env_cfg = rl_env_config()
    for method in METHODS:
        fc = PriceForecast(ctx.market, method)
        for control in CONTROLS:
            for k, ep in enumerate(eps):
                mpc = RollingMPC(fc, control)
                t0 = time.perf_counter()
                core = run_mpc(ep, mpc, env_cfg)
                wall = time.perf_counter() - t0
                rows.append({"episode": k, "start": ep.start, "policy": mpc.name, **settle_episode(core)})
                tm = timing.setdefault(mpc.name, {"wall_s": 0.0, "solves": 0, "max_evs": 0, "max_steps": 0})
                tm["wall_s"] += wall
                tm["solves"] += mpc.stats.solves
                tm["max_evs"] = max(tm["max_evs"], mpc.stats.max_evs)
                tm["max_steps"] = max(tm["max_steps"], mpc.stats.max_steps)
    return pd.DataFrame(rows), timing


def forecast_errors(ctx, eps, horizon: int = 48) -> dict:
    """Mean absolute error of each method over the next `horizon` steps, at every step of the episodes."""
    out = {}
    for col in ("usep_price", "reserve_price_CONRESA"):
        truth = PriceForecast(ctx.market, "perfect")
        for method in METHODS[1:]:
            fc, errs = PriceForecast(ctx.market, method), []
            for ep in eps:
                for t in range(0, ep.n_steps, 6):
                    steps = pd.date_range(ep.timestamps[t], periods=horizon, freq="30min")
                    errs.append(np.abs(fc.predict(col, steps[0], steps) - truth.predict(col, steps[0], steps))[1:])
            out[f"{col}_mae_{method}"] = round(float(np.mean(errs)), 3)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    ppo = df[df["policy"].str.startswith("PPO_seed")]
    d = pd.concat([df, ppo.groupby("episode", as_index=False).mean(numeric_only=True).assign(policy="PPO_mean")],
                  ignore_index=True)
    base = d[d["policy"] == "B1_uncontrolled"].set_index("episode")["net_cost_per_port_day_sgd"]
    d["saving_vs_b1"] = d["episode"].map(base) - d["net_cost_per_port_day_sgd"]
    t = d.groupby("policy").agg(net_per_port_day=("net_cost_per_port_day_sgd", "mean"),
                                energy_per_port_day=("cost_per_port_day_sgd", "mean"),
                                saving_vs_b1=("saving_vs_b1", "mean"), il_mw=("il_mw_mean", "mean"),
                                peak_kw=("peak_kw", "mean"), unmet_kwh=("unmet_kwh", "sum"))
    return t.loc[[n for n in ORDER if n in t.index]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/d"))
    args = ap.parse_args()
    tab_dir = args.out / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_context()
    val = fixed_episodes(ctx, "validation", N_PORTS, N_DAYS)
    trn = fixed_episodes(ctx, "train", N_PORTS, N_DAYS, scenario="rl_train_check", limit=N_TRAIN_EPS)
    out, meta = {}, {"n_ports": N_PORTS, "episode_days": N_DAYS, "timing": {}, "same_episodes_as_ppo": {}}
    for split, eps in (("validation", val), ("train_sample", trn)):
        base = baselines(eps, p_grid=(0.9, 1.0))
        stored = pd.read_csv(tab_dir / f"ppo_il_{split}_episodes.csv")
        check = base.merge(stored, on=["episode", "policy"], suffixes=("", "_ppo"))
        meta["same_episodes_as_ppo"][split] = bool(
            len(check) == len(base) and np.allclose(check["net_cost_sgd"], check["net_cost_sgd_ppo"], rtol=1e-9))
        mpc, meta["timing"][split] = run_mpcs(ctx, eps)
        ppo = stored[stored["policy"].str.startswith("PPO_seed")]
        df = pd.concat([base, mpc, ppo], ignore_index=True)
        df.to_csv(tab_dir / f"mpc_il_{split}_episodes.csv", index=False)
        out[split] = summarise(df)
        out[split].round(4).to_csv(tab_dir / f"mpc_il_{split}_summary.csv")
    meta["forecast_mae_48_steps_validation"] = forecast_errors(ctx, val)
    meta["forecast_mae_48_steps_train_sample"] = forecast_errors(ctx, trn)
    (tab_dir / "mpc_il_check.json").write_text(json.dumps(meta, indent=2, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(json.dumps(meta, indent=2, default=str))
        for k, v in out.items():
            print(k)
            print(v.round(4))


if __name__ == "__main__":
    main()
