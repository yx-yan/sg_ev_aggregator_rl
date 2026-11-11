"""Service-value table (SCENARIO STUDY) for a portfolio of public ports.

Main line = services open to EV aggregations in Singapore today (reports/b/2026-11-11_sg_service_eligibility.md):
smart charging and interruptible load (IL, paid the contingency reserve price CONRESA). Regulation and
V2G are future scenarios (VPP framework). There is no peak tariff for this setting, so peak shaving is
only studied as a sensitivity, with the assumed tariff applied to every scenario of that case.

Weeks are the Monday-Sunday weeks labelled in reports/c/tables/weeks_labelled.csv, train and
validation only.

Run:  PYTHONPATH=src python -m sg_ev_rl.scheduling.report_services --out reports/b
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sg_ev_rl.market.features import add_relative_price_features
from sg_ev_rl.market.loader import load_market
from sg_ev_rl.market.report_price_diagnosis import INK, INK2, SERIES, SURFACE, _style
from sg_ev_rl.scenarios import public_charging as pc
from sg_ev_rl.scenarios.episodes import DT_H, build_episode, seed_for

from . import fleet_rules as R
from .service_lp import ServiceConfig, peak_window, solve_service_lp

N_PORTS, N_DAYS = 100, 7
BASE_V2G_SHARE = 0.5
IL_MIN_MW = 0.1  # minimum to participate directly in NEMS
WEEKS_CSV = Path("reports/c/tables/weeks_labelled.csv")
SOC_CASES = {
    "low": pc.VehicleAssumptions(arrival_soc_ac=(0.1, 0.5), arrival_soc_dc=(0.05, 0.3)),
    "mid": pc.VehicleAssumptions(),
    "high": pc.VehicleAssumptions(arrival_soc_ac=(0.4, 0.8), arrival_soc_dc=(0.3, 0.6)),
}
MAIN = ["S1_uncontrolled", "S1b_uncontrolled_il", "S2_smart", "S6_smart_il"]
FUTURE = ["S3_smart_v2g", "S4a_smart_regulation", "S4_v2g_regulation"]
PEAK = ["S5_v2g_regulation_peak"]
SCENARIOS = MAIN + FUTURE + PEAK
COMPONENTS = ["energy", "degradation", "il", "regulation", "peak"]


def peak_slots(market: pd.DataFrame, k: int = 4) -> tuple[int, ...]:
    """Top-k half-hours of the mean daily gross-demand profile in the training split."""
    tr = market[market["split"] == "train"]
    hh = tr["interval_start"].dt.hour * 2 + tr["interval_start"].dt.minute // 30
    return tuple(sorted(int(x) for x in tr.groupby(hh)["demand_gross_mw"].mean().nlargest(k).index))


def weeks(splits=("train", "validation")) -> pd.DataFrame:
    w = pd.read_csv(WEEKS_CSV)
    w = w[w["split"].isin(splits) & (w["n"] == 336)].copy()
    w["start"] = pd.to_datetime(w["week"].str.slice(0, 10)).dt.date
    return w.reset_index(drop=True)


def evaluate(ep, cfg: ServiceConfig, slots, peak_price: float) -> list[dict]:
    """All scenarios on one episode. With peak_price > 0 every total includes the peak tariff."""
    win = peak_window(ep.timestamps, slots)
    rows = []

    def add(name, net_kwh, res=None):
        energy = float((net_kwh * ep.usep).sum() / 1000)
        peak = float(net_kwh[win].sum() * peak_price)
        row = {"scenario": name, "energy_sgd": energy, "peak_sgd": peak, "degradation_sgd": 0.0,
               "il_sgd": 0.0, "regulation_sgd": 0.0, "peak_kw": float(net_kwh.max() / DT_H)}
        if res is not None:
            il_mw = res.il_kw.sum(axis=0) / 1000
            row.update({"degradation_sgd": res.degradation_sgd, "il_sgd": res.il_revenue_sgd,
                        "regulation_sgd": res.regulation_revenue_sgd, "discharge_kwh": float(res.discharge.sum()),
                        "regulation_mw_mean": float(res.regulation_kw.sum(0).mean() / 1000),
                        "il_mw_mean": float(il_mw.mean()), "il_share_ge_min": float((il_mw >= IL_MIN_MW).mean()),
                        "simultaneous_steps": res.simultaneous_steps, "solve_s": res.solve_s})
        row["total_sgd"] = (row["energy_sgd"] + row["degradation_sgd"] + row["peak_sgd"]
                            - row["il_sgd"] - row["regulation_sgd"])
        rows.append(row)

    off = dict(v2g=False, regulation=False, interruptible_load=False, peak_price_sgd_per_kwh=0)
    core = R.run(ep, R.Uncontrolled())
    add("S1_uncontrolled", np.asarray(core.log.energy_kwh))
    r = solve_service_lp(ep, replace(cfg, **{**off, "interruptible_load": True}), fixed_charge=core.schedule)
    add("S1b_uncontrolled_il", r.charge.sum(axis=0), r)
    cases = [("S2_smart", {}), ("S6_smart_il", {"interruptible_load": True}), ("S3_smart_v2g", {"v2g": True}),
             ("S4a_smart_regulation", {"regulation": True}), ("S4_v2g_regulation", {"v2g": True, "regulation": True})]
    if peak_price > 0:
        cases.append(("S5_v2g_regulation_peak", {"v2g": True, "regulation": True,
                                                 "peak_price_sgd_per_kwh": peak_price, "peak_slots": slots}))
    for name, kw in cases:
        r = solve_service_lp(ep, replace(cfg, **{**off, **kw}))
        add(name, (r.charge - r.discharge).sum(axis=0), r)
    return rows


def run_weeks(profile, market, ports, wk, cfg, slots, peak_price, v2g_share, vehicle, tag) -> pd.DataFrame:
    out = []
    for k, w in wk.iterrows():
        ep = build_episode(profile, market, w["start"], N_DAYS, N_PORTS, ports,
                           seed=seed_for("services", w["split"], k), vehicle=vehicle, v2g_share=v2g_share)
        for r in evaluate(ep, cfg, slots, peak_price):
            out.append({"case": tag, "week": w["week"], "split": w["split"], "spike_regime": w["spike_regime"],
                        "level_regime": w["level_regime"], "sessions": len(ep.sessions),
                        "mean_usep": float(ep.usep.mean()), "mean_reg_price": float(ep.regulation_price.mean()),
                        "mean_conres_price": float(ep.contingency_reserve_price.mean()), **r})
    df = pd.DataFrame(out)
    per = N_PORTS * N_DAYS
    for c in COMPONENTS + ["total"]:
        df[f"{c}_per_port_day"] = df[f"{c}_sgd"] / per
    base = df[df["scenario"] == "S1_uncontrolled"].set_index("week")["total_sgd"]
    df["saving_vs_s1_per_port_day"] = (df["week"].map(base) - df["total_sgd"]) / per
    return df


def table(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    cols = [f"{c}_per_port_day" for c in COMPONENTS] + ["total_per_port_day", "saving_vs_s1_per_port_day"]
    keys = ([by] if by else []) + ["scenario"]
    t = df.groupby(keys)[cols].mean()
    t["weeks"] = df.groupby(keys)["week"].nunique()
    present = [s for s in SCENARIOS if s in set(df["scenario"])]
    return t.reindex(present, level=-1)


def fig_components(t: pd.DataFrame, path: Path):
    """Saving vs uncontrolled per port-day by source, for normal and spike weeks."""
    names = [s for s in SCENARIOS[1:] if s in t.index.get_level_values(-1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE, sharey=True)
    for ax, regime in zip(axes, ["normal", "spike"], strict=True):
        _style(ax)
        sub = t.loc[regime]
        s1 = sub.loc["S1_uncontrolled"]
        x = np.arange(len(names))
        parts = {"energy": s1["energy_per_port_day"] - sub.loc[names, "energy_per_port_day"],
                 "interruptible load": sub.loc[names, "il_per_port_day"],
                 "regulation": sub.loc[names, "regulation_per_port_day"],
                 "degradation": -sub.loc[names, "degradation_per_port_day"]}
        pos, neg = np.zeros(len(names)), np.zeros(len(names))
        for k, (lab, v) in enumerate(parts.items()):
            v = v.to_numpy()
            ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), color=SERIES[k], width=0.6, label=lab,
                   edgecolor=SURFACE, linewidth=1)
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
        for xi, v, top in zip(x, sub.loc[names, "saving_vs_s1_per_port_day"].to_numpy(), pos, strict=True):
            ax.text(xi, top + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(len([n for n in names if n in MAIN]) - 0.5, color=INK2, lw=1, ls=":")
        ax.text(0, 1.0, "today", transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, va="bottom")
        ax.text(len([n for n in names if n in MAIN]) - 0.4, 1.0, "future scenarios (VPP framework)",
                transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, va="bottom")
        ax.set_xticks(x, [n.split("_", 1)[1].replace("_", "\n") for n in names], fontsize=8)
        ax.set_title(f"{regime} weeks (n={int(sub['weeks'].iloc[0])})", color=INK, fontsize=10, loc="left", pad=16)
    axes[0].set_ylabel("S$ per port per day vs uncontrolled", color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.suptitle("Scenario study: saving vs uncontrolled charging by source (100 public ports)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/b"))
    ap.add_argument("--sens-every", type=int, default=4, help="sensitivity on every k-th week plus all spike weeks")
    args = ap.parse_args()
    fig_dir, tab_dir = args.out / "figures", args.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    market = add_relative_price_features(load_market())
    sess, ports, snaps = pc.load_calibration_inputs()
    profile = pc.calibrate(sess, ports, snaps)
    slots = peak_slots(market)
    wk = weeks()
    base_cfg = ServiceConfig()

    base = run_weeks(profile, market, ports, wk, base_cfg, slots, 0.0, BASE_V2G_SHARE, SOC_CASES["mid"], "base")
    base.to_csv(tab_dir / "services_base_weeks.csv", index=False)
    table(base).round(4).to_csv(tab_dir / "services_base_all.csv")
    t_spike = table(base, "spike_regime")
    t_spike.round(4).to_csv(tab_dir / "services_base_by_spike.csv")
    table(base, "level_regime").round(4).to_csv(tab_dir / "services_base_by_level.csv")
    fig_components(t_spike, fig_dir / "services_saving_sources.png")

    # one-at-a-time sensitivity on a subset of weeks
    sub = wk[(wk.index % args.sens_every == 0) | (wk["spike_regime"] == "spike")]
    cases = [("base", "base", {})]
    cases += [("il_activation_h", v, dict(cfg=replace(base_cfg, il_activation_h=v))) for v in (1.0, 2.0)]
    cases += [("arrival_soc", v, dict(vehicle=SOC_CASES[v])) for v in ("low", "high")]
    cases += [("v2g_share", v, dict(v2g_share=v)) for v in (0.25, 1.0)]
    cases += [("deg_cost", v, dict(cfg=replace(base_cfg, deg_cost_sgd_per_kwh=v))) for v in (0.0, 0.15)]
    cases += [("eta_discharge", v, dict(cfg=replace(base_cfg, eta_discharge=v))) for v in (0.88, 0.95)]
    cases += [("peak_price", v, dict(peak_price=v)) for v in (0.05, 0.10, 0.20, 0.50)]
    sens = []
    for param, val, kw in cases:
        df = run_weeks(profile, market, ports, sub, kw.get("cfg", base_cfg), slots, kw.get("peak_price", 0.0),
                       kw.get("v2g_share", BASE_V2G_SHARE), kw.get("vehicle", SOC_CASES["mid"]), f"{param}={val}")
        t = df.groupby("scenario")["saving_vs_s1_per_port_day"].mean().rename("saving").reset_index()
        t["param"], t["value"] = param, str(val)
        sens.append(t)
    sens = pd.concat(sens)
    piv = sens.pivot_table(index=["param", "value"], columns="scenario", values="saving")
    piv = piv[[s for s in SCENARIOS[1:] if s in piv.columns]]
    piv.round(4).to_csv(tab_dir / "services_sensitivity.csv")

    lp_rows = base[~base["scenario"].isin(["S1_uncontrolled"])]
    il = base[base["scenario"].isin(["S1b_uncontrolled_il", "S6_smart_il"])].groupby("scenario")
    meta = {
        "n_ports": N_PORTS, "episode_days": N_DAYS, "weeks": int(base["week"].nunique()),
        "weeks_by_spike_regime": wk["spike_regime"].value_counts().to_dict(),
        "weeks_by_level_regime": wk["level_regime"].value_counts().to_dict(),
        "sensitivity_weeks": int(len(sub)),
        "peak_window_sgt": [f"{s // 2:02d}:{30 * (s % 2):02d}" for s in slots],
        "base_v2g_share": BASE_V2G_SHARE,
        "service_config": base_cfg.__dict__,
        "il_mw_mean": il["il_mw_mean"].mean().round(4).to_dict(),
        "il_share_of_periods_ge_0p1_mw": il["il_share_ge_min"].mean().round(3).to_dict(),
        "mean_prices_sgd_mwh": {"usep": round(float(base["mean_usep"].mean()), 2),
                                "regulation": round(float(base["mean_reg_price"].mean()), 2),
                                "contingency_reserve_A": round(float(base["mean_conres_price"].mean()), 2)},
        "simultaneous_charge_discharge_steps": int(lp_rows["simultaneous_steps"].sum()),
        "lp_solve_s_max": float(lp_rows["solve_s"].max()),
        "sessions_per_week_median": int(base["sessions"].median()),
    }
    (tab_dir / "services_check.json").write_text(json.dumps(meta, indent=2, default=str))
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print(json.dumps(meta, indent=2, default=str))
        print(table(base).round(3))
        print(t_spike.round(3))
        print(piv.round(3))


if __name__ == "__main__":
    main()
