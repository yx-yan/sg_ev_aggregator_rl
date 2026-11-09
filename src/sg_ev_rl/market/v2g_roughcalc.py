"""Rough value of V2G arbitrage per kWh of battery, from historical USEP.

One LP per plug-in window: an energy-neutral battery (end energy = start energy) buys at USEP
when charging and sells at USEP when discharging, paying a degradation cost per kWh discharged.
This isolates what V2G adds on top of the driver's own charging need, which is energy the car
needs anyway. It is an upper-bound style screen (perfect price foresight, one car, price-taker),
not a simulator result.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

DT_H = 0.5


@dataclass(frozen=True)
class Battery:
    """Normalised to 1 kWh of usable capacity. Defaults are ASSUMPTIONS (see report)."""
    soc_min: float = 0.3            # lower bound kept for the driver's mobility needs
    soc_max: float = 0.8
    c_rate_per_h: float = 7.0 / 60.0  # 7 kW AC charger on a 60 kWh battery
    round_trip_eff: float = 0.85
    deg_cost_per_kwh: float = 0.06  # S$ per kWh discharged


WINDOWS = {
    # name: (start hour, length in hours); a window starting on trading date D belongs to D
    "residential_19_07": (19.0, 12.0),
    "destination_09_18": (9.0, 9.0),
    "always_plugged_00_24": (0.0, 24.0),
}


def arbitrage_lp(price_s_per_kwh: np.ndarray, b: Battery) -> tuple[float, float]:
    """Max profit (S$ per kWh of battery) and kWh discharged for one window.

    Variables x = [c_1..c_n, d_1..d_n, e0]: c is grid energy bought and d grid energy sold per
    step (kWh); storage changes by eta*c - d/eta; e0 is the start energy.
    """
    n = len(price_s_per_kwh)
    eta = np.sqrt(b.round_trip_eff)
    pmax = b.c_rate_per_h * DT_H
    # cost = sum p*c - sum p*d + deg*d
    cost = np.concatenate([price_s_per_kwh, -price_s_per_kwh + b.deg_cost_per_kwh, [0.0]])
    lower = np.tril(np.ones((n, n)))
    # e_k = e0 + sum_{j<=k} (eta*c_j - d_j/eta)
    energy = np.hstack([eta * lower, -lower / eta, np.ones((n, 1))])
    a_ub = np.vstack([energy, -energy])
    b_ub = np.concatenate([np.full(n, b.soc_max), np.full(n, -b.soc_min)])
    a_eq = np.concatenate([np.full(n, eta), np.full(n, -1 / eta), [0.0]])[None, :]
    bounds = [(0, pmax)] * (2 * n) + [(b.soc_min, b.soc_max)]
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[0.0], bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    discharged = float(res.x[n:2 * n].sum())
    return float(-res.fun), discharged


def window_prices(market: pd.DataFrame, start_h: float, length_h: float) -> dict:
    """{trading date: USEP array in S$/kWh} for complete windows only."""
    s = market.set_index("interval_start")["usep_price"] / 1000.0
    steps = int(length_h / DT_H)
    out = {}
    for d in sorted(set(market["date"])):
        t0 = pd.Timestamp(d).tz_localize("Asia/Singapore") + pd.Timedelta(hours=start_h)
        idx = pd.date_range(t0, periods=steps, freq="30min")
        vals = s.reindex(idx).to_numpy()
        if not np.isnan(vals).any():
            out[d] = vals
    return out


def run_grid(market: pd.DataFrame, effs, degs, windows=WINDOWS, base: Battery = Battery()) -> pd.DataFrame:
    rows = []
    for wname, (start_h, length_h) in windows.items():
        prices = window_prices(market, start_h, length_h)
        for eff in effs:
            for deg in degs:
                b = Battery(base.soc_min, base.soc_max, base.c_rate_per_h, eff, deg)
                for d, p in prices.items():
                    profit, dis = arbitrage_lp(p, b)
                    rows.append({"window": wname, "round_trip_eff": eff, "deg_cost": deg,
                                 "date": d, "profit_per_kwh_batt": profit, "discharged_kwh": dis})
    return pd.DataFrame(rows)


def regulation_capacity_value(market: pd.DataFrame, start_h: float, length_h: float) -> pd.Series:
    """S$ per kW of symmetric regulation capacity held through the window, per day.
    Capacity payment only: no activation energy, no penalty (SCENARIO)."""
    prices = window_prices(market.assign(usep_price=market["regulation_price"]), start_h, length_h)
    return pd.Series({d: float(p.sum() * DT_H) for d, p in prices.items()}, name="reg_value_per_kw")
