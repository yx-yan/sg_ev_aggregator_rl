"""Settle a simulation run into KPIs."""
import numpy as np

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


def settle_episode(core) -> dict[str, float]:
    """KPIs of a finished FleetCore run (sg_ev_rl.sim.fleet_env)."""
    log, ep = core.log, core.ep
    energy = float(np.sum(log.energy_kwh))
    cost = float(np.sum(log.cost_sgd))
    il = float(np.sum(log.il_revenue_sgd))
    days = ep.n_steps / 48
    return {
        "energy_kwh": energy,
        "energy_cost_sgd": cost,
        "avg_price_sgd_mwh": cost / energy * 1000 if energy else 0.0,
        "cost_per_port_day_sgd": cost / ep.n_ports / days,
        "cost_per_session_sgd": cost / max(len(ep.sessions), 1),
        "peak_kw": float(np.max(log.energy_kwh)) / DT_H,
        "il_revenue_sgd": il,
        "il_mw_mean": float(np.mean(log.il_kw)) / 1000,
        "il_activations": int(np.sum(log.il_activated)),
        "net_cost_sgd": cost - il,
        "net_cost_per_port_day_sgd": (cost - il) / ep.n_ports / days,
        "unmet_kwh": float(np.sum(log.unmet_kwh)),
        "unmet_share": float(np.sum(log.unmet_kwh)) / max(float(ep.need_kwh.sum()), 1e-9),
        "sessions": len(ep.sessions),
    }
