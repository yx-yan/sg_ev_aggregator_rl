"""Compare trained PPO seeds with the baselines on the same validation episodes, plus a few train episodes.

Writes tables and figures for reports/d/YYYY-MM-DD_ppo_il.md. Reads models from the runs directory.

Run:  PYTHONPATH=src python -m sg_ev_rl.rl.report_rl --config configs/rl/ppo_il.yaml --out reports/d
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from sg_ev_rl.market.paths import runs_dir
from sg_ev_rl.market.report_price_diagnosis import INK, INK2, SERIES, SURFACE, _style
from sg_ev_rl.scenarios.episodes import DT_H
from sg_ev_rl.scheduling import fleet_rules as R
from sg_ev_rl.scheduling.service_lp import ServiceConfig, solve_service_lp
from sg_ev_rl.sim.fleet_env import FleetEnv

from .common import baselines, fixed_episodes, load_context, rl_env_config, run_policy

N_TRAIN_EPS = 8
ORDER = ["B1_uncontrolled", "B1_il", "B2_p0.9_il", "PPO_mean", "LP_il_aggregate_llf", "LP_il_per_ev"]


def evaluate(models: dict, eps, env_cfg) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        for k, ep in enumerate(eps):
            r = run_policy(lambda o, m=m: m.predict(o, deterministic=True)[0], ep, env_cfg)
            rows.append({"episode": k, "start": ep.start, "policy": name, **r})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    ppo = df[df["policy"].str.startswith("PPO_seed")]
    mean = ppo.groupby("episode", as_index=False).mean(numeric_only=True).assign(policy="PPO_mean")
    d = pd.concat([df, mean], ignore_index=True)
    base = d[d["policy"] == "B1_uncontrolled"].set_index("episode")["net_cost_per_port_day_sgd"]
    d["saving_vs_b1"] = d["episode"].map(base) - d["net_cost_per_port_day_sgd"]
    t = d.groupby("policy").agg(net_per_port_day=("net_cost_per_port_day_sgd", "mean"),
                                energy_per_port_day=("cost_per_port_day_sgd", "mean"),
                                saving_vs_b1=("saving_vs_b1", "mean"), il_mw=("il_mw_mean", "mean"),
                                peak_kw=("peak_kw", "mean"), unmet_kwh=("unmet_kwh", "sum"))
    names = [n for n in ORDER if n in t.index] + sorted(n for n in t.index if n.startswith("PPO_seed"))
    return t.loc[names]


def fig_learning(logs: dict, base: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    _style(ax)
    for k, lg in enumerate(logs.values()):
        ax.plot(lg["timesteps"] / 1e6, lg["val_net_per_port_day"], color=SERIES[0], lw=1.5,
                alpha=0.5 + 0.25 * (k == 0), label="PPO seeds" if k == 0 else None)
    for k, (name, label) in enumerate([("B1_il", "uncontrolled + IL (best rule)"),
                                       ("LP_il_aggregate_llf", "LP fleet profile + IL")]):
        ax.axhline(base[name], color=SERIES[k + 1], lw=1.5, ls="--")
        ax.text(ax.get_xlim()[1], base[name], f" {label}", color=INK2, fontsize=8, va="center")
    ax.set_xlabel("training steps (millions)", color=INK2)
    ax.set_ylabel("validation net cost, S$/port/day", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("PPO learning curves on the 12 validation weeks (300 public ports, smart charging + IL)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_day(model, ep, env_cfg, path: Path, day: int = 2):
    """Fleet charging power over one day: uncontrolled + IL, LP + IL fleet profile, PPO; price on top."""
    sl = slice(day * 48, (day + 1) * 48)
    hours = np.arange(48) * DT_H
    b1 = R.run(ep, R.Uncontrolled(), il_share=1.0)
    lp = solve_service_lp(ep, ServiceConfig(interruptible_load=True))
    env = FleetEnv([ep], env_cfg)
    obs, _ = env.reset(seed=0, options={"episode": ep})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(model.predict(obs, deterministic=True)[0])
    series = {"uncontrolled + IL": b1.schedule.sum(0), "LP + IL (per EV)": lp.charge.sum(0),
              "PPO + IL": env.core.schedule.sum(0)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [1, 2]})
    for ax in (a1, a2):
        _style(ax)
    a1.plot(hours, ep.usep[sl], color=INK2, lw=1.5, label="USEP")
    a1.plot(hours, ep.contingency_reserve_price[sl] * 5, color=SERIES[3], lw=1.5, label="contingency reserve x5")
    a1.set_ylabel("S$/MWh", color=INK2)
    a1.legend(frameon=False, fontsize=8, ncol=2)
    for k, (label, e) in enumerate(series.items()):
        a2.plot(hours, e[sl] / DT_H, color=SERIES[k], lw=2, label=label)
    a2.set_ylabel("fleet charging, kW", color=INK2)
    a2.set_xlabel(f"hour of {ep.timestamps[day * 48].date()} (SGT)", color=INK2)
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle("One validation day: when each controller charges", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/rl/ppo_il.yaml"))
    ap.add_argument("--out", type=Path, default=Path("reports/d"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    fig_dir, tab_dir = args.out / "figures", args.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    run_root = runs_dir().parent / "d"
    env_cfg = rl_env_config(**cfg["env"])

    ctx = load_context()
    val = fixed_episodes(ctx, "validation", cfg["n_ports"], cfg["episode_days"])
    trn = fixed_episodes(ctx, "train", cfg["n_ports"], cfg["episode_days"], scenario="rl_train_check",
                         limit=N_TRAIN_EPS)
    models, logs = {}, {}
    for s in cfg["seeds"]:
        d = run_root / f"{args.config.stem}_seed{s}"
        models[f"PPO_seed{s}"] = PPO.load(d / "best_model", device="cpu")
        logs[s] = pd.read_csv(d / "validation_log.csv")

    out = {}
    for split, eps in (("validation", val), ("train_sample", trn)):
        df = pd.concat([baselines(eps, p_grid=(0.9, 1.0)), evaluate(models, eps, env_cfg)], ignore_index=True)
        df.to_csv(tab_dir / f"ppo_il_{split}_episodes.csv", index=False)
        out[split] = summarise(df)
        out[split].round(4).to_csv(tab_dir / f"ppo_il_{split}_summary.csv")

    base = out["validation"]["net_per_port_day"]
    fig_learning(logs, base, fig_dir / "ppo_il_learning_curves.png")
    best_seed = min(cfg["seeds"], key=lambda s: logs[s]["val_net_per_port_day"].min())
    fig_day(models[f"PPO_seed{best_seed}"], val[0], env_cfg, fig_dir / "ppo_il_one_day.png")

    meta = {"config": cfg, "best_val_per_seed": {s: float(lg["val_net_per_port_day"].min()) for s, lg in logs.items()},
            "best_step_per_seed": {s: int(lg.loc[lg["val_net_per_port_day"].idxmin(), "timesteps"])
                                   for s, lg in logs.items()}}
    (tab_dir / "ppo_il_check.json").write_text(json.dumps(meta, indent=2, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(json.dumps(meta["best_val_per_seed"]), json.dumps(meta["best_step_per_seed"]))
        for k, v in out.items():
            print(k)
            print(v.round(4))


if __name__ == "__main__":
    main()
