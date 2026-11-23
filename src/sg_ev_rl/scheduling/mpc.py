"""B4: rolling MPC for smart charging plus interruptible load (charge only).

At every 30-minute step the MPC re-solves the service LP (scheduling/service_lp.py, IL on) for the EVs
plugged in now, from their current state to their departures, with forecast prices
(forecast/prices.py), and applies only the first step. Information set, the same as the RL agent's:

    known      current step's USEP and contingency reserve price; each connected EV's remaining need,
               power limit and departure (the safety layer of the environment uses the same)
    forecast   all later prices
    unknown    future arrivals: the plan covers only EVs plugged in now

Two ways to apply the plan:

    aggregate  the plan's fleet energy for this step goes through FleetCore.step_energy, i.e. projection
               onto [E_min, E_max] and the least-laxity-first split, with all backable IL offered;
               the same action interface as the RL agent, so this is its direct optimisation rival
    per_ev     the plan's per-EV charge and IL offers are applied as they are (FleetCore.step_schedule)

The LP has no coupling between EVs (no fleet import cap), so with perfect prices the per-EV MPC
reproduces the perfect-foresight LP exactly (Bellman); tests/sim/test_mpc.py checks this.
With forecasts, future prices come from the market table, so episodes whose prices were rescaled
(episodes.scale_prices) are only consistent with the perfect method.
"""
import time
from dataclasses import dataclass, replace

import numpy as np

from sg_ev_rl.forecast.prices import PriceForecast
from sg_ev_rl.scenarios.episodes import DT_H, Episode
from sg_ev_rl.sim.fleet_env import TOL, EnvConfig, FleetCore

from .service_lp import ServiceConfig, solve_service_lp

CONTROLS = ("aggregate", "per_ev")


@dataclass
class MPCStats:
    solves: int = 0
    solve_s: float = 0.0
    max_evs: int = 0
    max_steps: int = 0


class RollingMPC:
    def __init__(self, forecast: PriceForecast, control: str = "aggregate"):
        if control not in CONTROLS:
            raise ValueError(f"unknown control {control!r}, expected one of {CONTROLS}")
        self.forecast, self.control = forecast, control
        self.name = f"B4_mpc_{forecast.method}_{control}"
        self.stats = MPCStats()

    def plan(self, core: FleetCore) -> tuple[np.ndarray, np.ndarray]:
        """Per-EV grid energy (kWh) and IL offer (kW) for the current step."""
        ep, t, cfg = core.ep, core.t, core.cfg
        n = len(ep.sessions)
        x, q = np.zeros(n), np.zeros(n)
        idx = np.flatnonzero((core.avail[:, t] > 0) & (core.remaining > TOL))
        if not len(idx):
            return x, q
        end = int(core.last[idx].max()) + 1
        sub = self._sub_episode(core, idx, end)
        t0 = time.perf_counter()
        res = solve_service_lp(sub, ServiceConfig(interruptible_load=True, eta_charge=cfg.eta_charge,
                                                  il_activation_h=cfg.il_activation_h))
        st = self.stats
        st.solves += 1
        st.solve_s += time.perf_counter() - t0
        st.max_evs, st.max_steps = max(st.max_evs, len(idx)), max(st.max_steps, end - t)
        x[idx], q[idx] = res.charge[:, 0], res.il_kw[:, 0]
        return x, q

    def _sub_episode(self, core: FleetCore, idx: np.ndarray, end: int) -> Episode:
        """The connected EVs from their current state to the last departure, with forecast prices."""
        ep, t, eta = core.ep, core.t, core.cfg.eta_charge
        avail = core.avail[idx, t:end]
        cap = ep.p_max[idx] * avail.sum(axis=1) * DT_H
        remaining = np.clip(core.remaining[idx], 0, None)
        need = np.minimum(remaining, eta * cap * (1 - 1e-9))  # rounding must not make the target infeasible
        s = ep.sessions.iloc[idx].copy()
        batt = s["battery_kwh"].to_numpy(float)
        level = s["soc_arrival"].to_numpy(float) * batt + s["need_kwh"].to_numpy(float) - remaining
        s["soc_arrival"] = level / batt
        s["need_kwh"] = need
        steps = ep.timestamps[t:end]
        if self.forecast.method == "perfect":
            usep, conres = ep.usep[t:end], ep.contingency_reserve_price[t:end]
        else:
            usep = self.forecast.predict("usep_price", steps[0], steps)
            conres = self.forecast.predict("reserve_price_CONRESA", steps[0], steps)
            usep[0], conres[0] = ep.usep[t], ep.contingency_reserve_price[t]
        return replace(ep, sessions=s.reset_index(drop=True), avail=avail, n_steps=end - t, usep=usep,
                       contingency_reserve_price=conres, timestamps=steps,
                       regulation_price=ep.regulation_price[t:end], demand_gross_mw=ep.demand_gross_mw[t:end],
                       usep_pct_7d=ep.usep_pct_7d[t:end])


def run_mpc(ep: Episode, mpc: RollingMPC, cfg: EnvConfig = EnvConfig(), seed: int | None = None) -> FleetCore:
    core = FleetCore(ep, cfg, seed)
    while not core.done:
        x, q = mpc.plan(core)
        if mpc.control == "per_ev":
            core.step_schedule(x, q)
        else:
            core.step_energy(float(x.sum()), il_share=1.0)
    return core
