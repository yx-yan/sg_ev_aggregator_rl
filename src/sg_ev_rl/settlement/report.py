"""Settle a simulation run into KPIs."""
from sg_ev_rl.assets.fleet import DT_H
from sg_ev_rl.sim.simulator import SimResult


def settle(result: SimResult) -> dict[str, float]:
    energy_kwh = result.power_kw.sum() * DT_H
    cost_sgd = (result.power_kw.sum(axis=0) * DT_H * result.prices / 1000).sum()
    return {
        "energy_kwh": float(energy_kwh),
        "energy_cost_sgd": float(cost_sgd),
        "avg_price_sgd_mwh": float(cost_sgd / energy_kwh * 1000) if energy_kwh else 0.0,
        "peak_kw": float(result.power_kw.sum(axis=0).max()),
        "unmet_kwh": float(result.remaining_kwh.clip(min=0).sum()),
    }
