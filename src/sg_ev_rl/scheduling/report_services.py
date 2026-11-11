"""Service-value table (SCENARIO STUDY): what V2G, regulation capacity and peak shaving are worth
on top of smart charging, for a portfolio of public ports.

Weeks are the Monday-Sunday weeks labelled in reports/c/tables/weeks_labelled.csv, train and
validation only. The peak-shaving price is treated as a tariff on net import in the peak
window and applied to every scenario, so differences between scenarios are the service value.

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
BASE_PEAK_PRICE = 0.10
WEEKS_CSV = Path("reports/c/tables/weeks_labelled.csv")
SOC_CASES = {
    "low": pc.VehicleAssumptions(arrival_soc_ac=(0.1, 0.5), arrival_soc_dc=(0.05, 0.3)),
    "mid": pc.VehicleAssumptions(),
    "high": pc.VehicleAssumptions(arrival_soc_ac=(0.4, 0.8), arrival_soc_dc=(0.3, 0.6)),
}
SCENARIOS = ["S1_uncontrolled", "S2_smart", "S3_smart_v2g", "S4a_smart_regulation", "S4_v2g_regulation",
             "S5_v2g_regulation_peak"]


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


def evaluate(ep, cfg: ServiceConfig, slots, peak_price) -> list[dict]:
    """All scenarios on one episode; every total includes the peak tariff."""
    win = peak_window(ep.timestamps, slots)
    rows = []

    def add(name, net_kwh, deg=0.0, reg=0.0, extra=None):
        energy = float((net_kwh * ep.usep).sum() / 1000)
        peak = float(net_kwh[win].sum() * peak_price)
        rows.append({"scenario": name, "energy_sgd": energy, "degradation_sgd": deg, "regulation_sgd": reg,
                     "peak_sgd": peak, "total_sgd": energy + deg - reg + peak,
                     "peak_kw": float(net_kwh.max() / DT_H), "net_kw_in_window": float(net_kwh[win].mean() / DT_H),
                     **(extra or {})})

    add("S1_uncontrolled", np.asarray(R.run(ep, R.Uncontrolled()).log.energy_kwh))
    for name, c in [("S2_smart", replace(cfg, v2g=False, regulation=False, peak_price_sgd_per_kwh=0)),
                    ("S3_smart_v2g", replace(cfg, v2g=True, regulation=False, peak_price_sgd_per_kwh=0)),
                    ("S4a_smart_regulation", replace(cfg, v2g=False, regulation=True, peak_price_sgd_per_kwh=0)),
                    ("S4_v2g_regulation", replace(cfg, v2g=True, regulation=True, peak_price_sgd_per_kwh=0)),
                    ("S5_v2g_regulation_peak", replace(cfg, v2g=True, regulation=True,
                                                       peak_price_sgd_per_kwh=peak_price, peak_slots=slots))]:
        r = solve_service_lp(ep, c)
        add(name, (r.charge - r.discharge).sum(axis=0), r.degradation_sgd, r.regulation_revenue_sgd,
            {"discharge_kwh": float(r.discharge.sum()),
             "regulation_mw_mean": float(r.regulation_kw.sum(0).mean() / 1000),
             "simultaneous_steps": r.simultaneous_steps, "solve_s": r.solve_s})
    return rows


def run_weeks(profile, market, ports, wk, cfg, slots, peak_price, v2g_share, vehicle, tag) -> pd.DataFrame:
    out = []
    for k, w in wk.iterrows():
        ep = build_episode(profile, market, w["start"], N_DAYS, N_PORTS, ports,
                           seed=seed_for("services", w["split"], k), vehicle=vehicle, v2g_share=v2g_share)
        n_sess = len(ep.sessions)
        for r in evaluate(ep, cfg, slots, peak_price):
            out.append({"case": tag, "week": w["week"], "split": w["split"], "spike_regime": w["spike_regime"],
                        "level_regime": w["level_regime"], "sessions": n_sess, "mean_usep": float(ep.usep.mean()),
                        "mean_reg_price": float(ep.regulation_price.mean()), **r})
    df = pd.DataFrame(out)
    per = N_PORTS * N_DAYS
    for c in ("energy_sgd", "degradation_sgd", "regulation_sgd", "peak_sgd", "total_sgd"):
        df[c.replace("_sgd", "_per_port_day")] = df[c] / per
    base = df[df["scenario"] == "S1_uncontrolled"].set_index("week")["total_sgd"]
    df["saving_vs_s1_per_port_day"] = (df["week"].map(base) - df["total_sgd"]) / per
    return df


def table(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    cols = ["energy_per_port_day", "degradation_per_port_day", "regulation_per_port_day", "peak_per_port_day",
            "total_per_port_day", "saving_vs_s1_per_port_day"]
    keys = ([by] if by else []) + ["scenario"]
    t = df.groupby(keys)[cols].mean()
    t["weeks"] = df.groupby(keys)["week"].nunique()
    return t.reindex(SCENARIOS, level=-1)


def fig_components(t: pd.DataFrame, path: Path):
    """Saving vs uncontrolled per port-day, split into its sources, for normal and spike weeks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE, sharey=True)
    for ax, regime in zip(axes, ["normal", "spike"], strict=True):
        _style(ax)
        sub = t.loc[regime]
        s1 = sub.loc["S1_uncontrolled"]
        names = SCENARIOS[1:]
        x = np.arange(len(names))
        parts = {"energy": s1["energy_per_port_day"] - sub.loc[names, "energy_per_port_day"],
                 "regulation": sub.loc[names, "regulation_per_port_day"],
                 "peak tariff": s1["peak_per_port_day"] - sub.loc[names, "peak_per_port_day"],
                 "degradation": -sub.loc[names, "degradation_per_port_day"]}
        pos, neg = np.zeros(len(names)), np.zeros(len(names))
        for k, (lab, v) in enumerate(parts.items()):
            v = v.to_numpy()
            bottom = np.where(v >= 0, pos, neg)
            ax.bar(x, v, bottom=bottom, color=SERIES[k], width=0.6, label=lab, edgecolor=SURFACE, linewidth=1)
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
        tot = sub.loc[names, "saving_vs_s1_per_port_day"].to_numpy()
        for xi, v, top in zip(x, tot, pos, strict=True):
            ax.text(xi, top + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x, [n.split("_", 1)[1].replace("_", "\n") for n in names], fontsize=8)
        ax.set_title(f"{regime} weeks (n={int(sub['weeks'].iloc[0])})", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("S$ per port per day vs uncontrolled", color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Scenario study: where the saving comes from (100 public ports, 50% of AC sessions V2G-capable)",
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

    base = run_weeks(profile, market, ports, wk, base_cfg, slots, BASE_PEAK_PRICE, BASE_V2G_SHARE,
                     SOC_CASES["mid"], "base")
    base.to_csv(tab_dir / "services_base_weeks.csv", index=False)
    t_all, t_spike = table(base), table(base, "spike_regime")
    t_level = table(base, "level_regime")
    pd.concat({"all": t_all}, names=["group"]).round(4).to_csv(tab_dir / "services_base_all.csv")
    t_spike.round(4).to_csv(tab_dir / "services_base_by_spike.csv")
    t_level.round(4).to_csv(tab_dir / "services_base_by_level.csv")
    fig_components(t_spike, fig_dir / "services_saving_sources.png")

    # one-at-a-time sensitivity on a subset of weeks
    sub = wk[(wk.index % args.sens_every == 0) | (wk["spike_regime"] == "spike")]
    cases = [("v2g_share", v, dict(v2g_share=v)) for v in (0.0, 0.25, 1.0)]
    cases += [("eta_discharge", v, dict(cfg=replace(base_cfg, eta_discharge=v))) for v in (0.95, 0.88)]
    cases += [("deg_cost", v, dict(cfg=replace(base_cfg, deg_cost_sgd_per_kwh=v))) for v in (0.0, 0.03, 0.10, 0.15)]
    cases += [("peak_price", v, dict(peak_price=v)) for v in (0.05, 0.20, 0.50)]
    cases += [("arrival_soc", v, dict(vehicle=SOC_CASES[v])) for v in ("low", "high")]
    cases += [("activation_ratio", v, dict(cfg=replace(base_cfg, activation_ratio=v))) for v in (0.0, 0.25)]
    cases = [("base", "base", {})] + cases
    sens = []
    for param, val, kw in cases:
        df = run_weeks(profile, market, ports, sub, kw.get("cfg", base_cfg), slots,
                       kw.get("peak_price", BASE_PEAK_PRICE), kw.get("v2g_share", BASE_V2G_SHARE),
                       kw.get("vehicle", SOC_CASES["mid"]), f"{param}={val}")
        t = df.groupby("scenario")[["saving_vs_s1_per_port_day", "regulation_per_port_day",
                                    "degradation_per_port_day"]].mean()
        t["param"], t["value"] = param, str(val)
        sens.append(t.reset_index())
    sens = pd.concat(sens)
    piv = sens.pivot_table(index=["param", "value"], columns="scenario", values="saving_vs_s1_per_port_day")
    piv = piv[SCENARIOS[1:]]
    piv.round(4).to_csv(tab_dir / "services_sensitivity.csv")

    lp_rows = base[base["scenario"] != "S1_uncontrolled"]
    meta = {
        "n_ports": N_PORTS, "episode_days": N_DAYS, "weeks": int(base["week"].nunique()),
        "weeks_by_spike_regime": wk["spike_regime"].value_counts().to_dict(),
        "weeks_by_level_regime": wk["level_regime"].value_counts().to_dict(),
        "sensitivity_weeks": int(len(sub)),
        "peak_window_slots": list(slots),
        "peak_window_sgt": [f"{s // 2:02d}:{30 * (s % 2):02d}" for s in slots],
        "base_peak_price_sgd_per_kwh": BASE_PEAK_PRICE,
        "base_v2g_share": BASE_V2G_SHARE,
        "service_config": base_cfg.__dict__,
        "simultaneous_charge_discharge_steps": int(lp_rows["simultaneous_steps"].sum()),
        "lp_solve_s_max": float(lp_rows["solve_s"].max()),
        "sessions_per_week_median": int(base["sessions"].median()),
    }
    (tab_dir / "services_check.json").write_text(json.dumps(meta, indent=2, default=str))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(json.dumps(meta, indent=2, default=str))
        print(t_all.round(3))
        print(t_spike.round(3))
        print(t_level.round(3))
        print(piv.round(3))


if __name__ == "__main__":
    main()
