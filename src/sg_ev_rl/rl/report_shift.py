"""Does the price-shift fix work? PPO v2 vs v3 (scale-free price features, +/- price-level randomisation).

Every controller runs on the same episodes:
  validation      the 12 fixed 2026 Q1 weeks (real reserve-price shift; also used for checkpoint choice)
  train_sample    8 fixed 2025 weeks
  train_conres_x2 the same 2025 weeks with the reserve price doubled (a synthetic shift; no test data)

Run:  PYTHONPATH=src python -m sg_ev_rl.rl.report_shift --out reports/d
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from stable_baselines3 import PPO

from sg_ev_rl.market.paths import runs_dir
from sg_ev_rl.market.report_price_diagnosis import INK, INK2, SERIES, SURFACE, _style
from sg_ev_rl.scenarios.episodes import scale_prices

from .common import baselines, fixed_episodes, load_context, rl_env_config, run_policy
from .report_rl import N_TRAIN_EPS

CONFIGS = {"v2": "ppo_il_v2", "v3_noscale": "ppo_il_v3_noscale", "v3": "ppo_il_v3"}
BASELINES = ["B1_uncontrolled", "B1_il", "B2_p0.9_il", "LP_il_aggregate_llf", "LP_il_per_ev"]


def load_runs(config_dir: Path):
    out = {}
    for label, stem in CONFIGS.items():
        cfg = yaml.safe_load((config_dir / f"{stem}.yaml").read_text())
        root = runs_dir().parent / "d"
        out[label] = {"cfg": cfg, "env": rl_env_config(**cfg["env"]),
                      "models": {s: PPO.load(root / f"{stem}_seed{s}" / "best_model", device="cpu")
                                 for s in cfg["seeds"]},
                      "logs": {s: pd.read_csv(root / f"{stem}_seed{s}" / "validation_log.csv") for s in cfg["seeds"]}}
    return out


def evaluate(runs, eps) -> pd.DataFrame:
    rows = []
    for label, r in runs.items():
        for s, m in r["models"].items():
            for k, ep in enumerate(eps):
                res = run_policy(lambda o, m=m: m.predict(o, deterministic=True)[0], ep, r["env"])
                rows.append({"episode": k, "policy": f"PPO_{label}", "seed": s, **res})
    return pd.DataFrame(rows)


def summarise(base: pd.DataFrame, ppo: pd.DataFrame) -> pd.DataFrame:
    cols = {"net_per_port_day": "net_cost_per_port_day_sgd", "il_mw": "il_mw_mean", "peak_kw": "peak_kw"}
    b = base.groupby("policy")[list(cols.values())].mean().set_axis(list(cols), axis=1).loc[BASELINES]
    per_seed = ppo.groupby(["policy", "seed"])[list(cols.values())].mean().set_axis(list(cols), axis=1)
    p = per_seed.groupby("policy").mean()
    p["net_seed_min"] = per_seed.groupby("policy")["net_per_port_day"].min()
    p["net_seed_max"] = per_seed.groupby("policy")["net_per_port_day"].max()
    t = pd.concat([b, p.loc[[f"PPO_{k}" for k in CONFIGS]]])
    t["saving_vs_b1"] = t.loc["B1_uncontrolled", "net_per_port_day"] - t["net_per_port_day"]
    t["vs_best_rule"] = t[["net_per_port_day"]].loc[["B1_il", "B2_p0.9_il"]].min().iloc[0] - t["net_per_port_day"]
    return t


def fig_learning(runs, base_val: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    _style(ax)
    for k, (label, r) in enumerate(runs.items()):
        for j, lg in enumerate(r["logs"].values()):
            ax.plot(lg["timesteps"] / 1e6, lg["val_net_per_port_day"], color=SERIES[k], lw=1.5, alpha=0.7,
                    label=f"PPO {label}" if j == 0 else None)
    for name, label in [("B1_il", "uncontrolled + IL"), ("LP_il_aggregate_llf", "LP fleet profile + IL")]:
        ax.axhline(base_val[name], color=INK2, lw=1, ls="--")
        ax.text(ax.get_xlim()[1], base_val[name], f" {label}", color=INK2, fontsize=8, va="center")
    ax.set_ylim(base_val["LP_il_aggregate_llf"] - 0.05, base_val["B1_il"] + 0.25)
    ax.set_xlabel("training steps (millions)", color=INK2)
    ax.set_ylabel("validation net cost, S$/port/day", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("PPO before (v2) and after (v3) the price-shift fix, 12 validation weeks",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, default=Path("configs/rl"))
    ap.add_argument("--out", type=Path, default=Path("reports/d"))
    args = ap.parse_args()
    fig_dir, tab_dir = args.out / "figures", args.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.configs)
    cfg = runs["v3"]["cfg"]
    ctx = load_context()
    val = fixed_episodes(ctx, "validation", cfg["n_ports"], cfg["episode_days"])
    trn = fixed_episodes(ctx, "train", cfg["n_ports"], cfg["episode_days"], scenario="rl_train_check",
                         limit=N_TRAIN_EPS)
    sets = {"validation": val, "train_sample": trn, "train_conres_x2": [scale_prices(ep, conres=2.0) for ep in trn]}

    out = {}
    for name, eps in sets.items():
        base, ppo = baselines(eps, p_grid=(0.9,)), evaluate(runs, eps)
        pd.concat([base, ppo], ignore_index=True).to_csv(tab_dir / f"price_shift_{name}_episodes.csv", index=False)
        out[name] = summarise(base, ppo)
        out[name].round(4).to_csv(tab_dir / f"price_shift_{name}_summary.csv")

    fig_learning(runs, out["validation"]["net_per_port_day"], fig_dir / "price_shift_learning_curves.png")
    meta = {label: {"best_val_per_seed": {s: float(lg["val_net_per_port_day"].min()) for s, lg in r["logs"].items()},
                    "best_step_per_seed": {s: int(lg.loc[lg["val_net_per_port_day"].idxmin(), "timesteps"])
                                           for s, lg in r["logs"].items()}}
            for label, r in runs.items()}
    (tab_dir / "price_shift_check.json").write_text(json.dumps(meta, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(json.dumps(meta))
        for k, v in out.items():
            print(k)
            print(v.round(4))


if __name__ == "__main__":
    main()
