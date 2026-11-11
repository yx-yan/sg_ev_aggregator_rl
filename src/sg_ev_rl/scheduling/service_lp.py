"""Perfect-foresight LP with V2G, regulation capacity and peak shaving (service-value study).

SCENARIO STUDY: the regulation activation profile, the peak-shaving price and whether EV
aggregations may offer regulation in Singapore are assumptions (see ServiceConfig).

Per plugged-in (EV i, step t), all energies in kWh per step and grid side unless noted:

    c_it  charge, 0..p_max a_it dt          d_it  discharge, 0..p_dis a_it dt (V2G EVs only)
    r_it  symmetric regulation capacity, kW  l_it  battery energy at the end of step t (battery side)

    l_it = l_i,t-1 + eta_c c_it - d_it / eta_d,   l before arrival = soc_arrival * B
    l at the EV's last step = soc_arrival * B + need_i (target met exactly)
    soc_min B <= l_it <= soc_max B   (widened to the arrival / target level when needed)
    r dt + c - d <= p_max a dt       room to consume more  (up-regulation of load)
    r dt - c + d <= p_dis a dt       room to consume less  (down-regulation of load)
    l + alpha eta_c r dt <= l_max, l - alpha r dt / eta_d >= l_min   energy margin for activation

    min  sum usep/1000 (c - d) + deg/eta_d * d - reg_price/1000 r dt + peak_price * (c - d) in the peak window

Activation is taken as energy-neutral over a step (no AGC signal is public); alpha only reserves
battery room for it. Charge and discharge are separate variables; simultaneous use is counted.
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from sg_ev_rl.scenarios.episodes import DT_H, Episode


@dataclass(frozen=True)
class ServiceConfig:
    v2g: bool = False
    regulation: bool = False
    peak_price_sgd_per_kwh: float = 0.0  # ASSUMPTION: value of net import in the peak window
    peak_slots: tuple[int, ...] = ()  # half-hour indices 0..47 of the peak window
    eta_charge: float = 0.92
    eta_discharge: float = 0.92  # ASSUMPTION
    deg_cost_sgd_per_kwh: float = 0.06  # ASSUMPTION: per kWh taken out of the battery
    soc_min: float = 0.2  # ASSUMPTION: floor kept for the driver
    soc_max: float = 0.95
    activation_ratio: float = 0.1  # ASSUMPTION: battery room reserved per kWh of capacity-hour


@dataclass
class ServiceResult:
    charge: np.ndarray  # (n_sessions, n_steps) kWh
    discharge: np.ndarray
    regulation_kw: np.ndarray
    energy_cost_sgd: float
    degradation_sgd: float
    regulation_revenue_sgd: float
    peak_cost_sgd: float
    solve_s: float
    n_vars: int
    simultaneous_steps: int

    @property
    def total_cost_sgd(self) -> float:
        return self.energy_cost_sgd + self.degradation_sgd - self.regulation_revenue_sgd + self.peak_cost_sgd

    @property
    def net_kw(self) -> np.ndarray:
        return (self.charge - self.discharge).sum(axis=0) / DT_H


def peak_window(timestamps, slots) -> np.ndarray:
    hh = np.asarray(timestamps.hour * 2 + timestamps.minute // 30)
    return np.isin(hh, list(slots))


def solve_service_lp(ep: Episode, cfg: ServiceConfig = ServiceConfig()) -> ServiceResult:
    s = ep.sessions
    i_idx, t_idx = np.nonzero(ep.avail > 0)  # row-major: each EV's steps are consecutive
    n = len(i_idx)
    a = ep.avail[i_idx, t_idx]
    batt = s["battery_kwh"].to_numpy(float)
    l0 = s["soc_arrival"].to_numpy(float) * batt
    target = l0 + ep.need_kwh
    lmin = np.minimum(cfg.soc_min * batt, l0)
    lmax = np.maximum(cfg.soc_max * batt, target)
    first = np.r_[True, i_idx[1:] != i_idx[:-1]]
    last = np.r_[i_idx[1:] != i_idx[:-1], True]

    p_max = ep.p_max[i_idx]
    dis_ok = s["v2g_capable"].to_numpy(bool)[i_idx] if cfg.v2g else np.zeros(n, bool)
    p_dis = np.where(dis_ok, s["p_discharge_max_kw"].to_numpy(float)[i_idx], 0.0)
    kd = np.flatnonzero(dis_ok)
    nd = len(kd)
    nr = n if cfg.regulation else 0

    # variable blocks: c [0, n), d [n, n+nd), r [n+nd, n+nd+nr), l [n+nd+nr, 2n+nd+nr)
    oc, od, orr, ol = 0, n, n + nd, n + nd + nr
    nv = ol + n
    d_of = np.full(n, -1)
    d_of[kd] = od + np.arange(nd)

    price = ep.usep[t_idx] / 1000
    win = peak_window(ep.timestamps, cfg.peak_slots)[t_idx] * cfg.peak_price_sgd_per_kwh
    cost = np.zeros(nv)
    cost[oc:oc + n] = price + win
    cost[od:od + nd] = -price[kd] - win[kd] + cfg.deg_cost_sgd_per_kwh / cfg.eta_discharge
    if nr:
        cost[orr:orr + nr] = -ep.regulation_price[t_idx] / 1000 * DT_H

    # dynamics: l_k - l_{k-1} - eta_c c_k + d_k / eta_d = l0 (first step) or 0
    rows, cols, vals = [np.arange(n), np.arange(n), np.arange(n)[~first]], \
        [ol + np.arange(n), oc + np.arange(n), ol + np.flatnonzero(~first) - 1], \
        [np.ones(n), np.full(n, -cfg.eta_charge), -np.ones((~first).sum())]
    rows.append(kd)
    cols.append(d_of[kd])
    vals.append(np.full(nd, 1 / cfg.eta_discharge))
    a_eq = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nv))
    b_eq = np.where(first, l0[i_idx], 0.0)

    a_ub, b_ub = None, None
    if nr:
        k = np.arange(n)
        r_col = orr + k
        ub_rows, ub_cols, ub_vals, rhs = [], [], [], []

        def add(row0, cols_vals, b):
            for c_, v_ in cols_vals:
                m = c_ >= 0
                ub_rows.append(row0 + k[m])
                ub_cols.append(c_[m])
                ub_vals.append(v_[m] if np.ndim(v_) else np.full(m.sum(), v_))
            rhs.append(b)

        add(0, [(r_col, DT_H), (oc + k, 1.0), (d_of, -1.0)], p_max * a * DT_H)
        add(n, [(r_col, DT_H), (oc + k, -1.0), (d_of, 1.0)], p_dis * a * DT_H)
        add(2 * n, [(ol + k, 1.0), (r_col, cfg.activation_ratio * cfg.eta_charge * DT_H)], lmax[i_idx])
        add(3 * n, [(ol + k, -1.0), (r_col, cfg.activation_ratio / cfg.eta_discharge * DT_H)], -lmin[i_idx])
        a_ub = coo_matrix((np.concatenate(ub_vals), (np.concatenate(ub_rows), np.concatenate(ub_cols))),
                          shape=(4 * n, nv)).tocsr()
        b_ub = np.concatenate(rhs)

    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    ub[oc:oc + n] = p_max * a * DT_H
    ub[od:od + nd] = p_dis[kd] * a[kd] * DT_H
    lb[ol:ol + n], ub[ol:ol + n] = lmin[i_idx], lmax[i_idx]
    lb[ol + np.flatnonzero(last)] = ub[ol + np.flatnonzero(last)] = target[i_idx[last]]

    t0 = time.perf_counter()
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq.tocsr(), b_eq=b_eq,
                  bounds=np.column_stack([lb, ub]), method="highs")
    solve_s = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(f"service LP failed: {res.message}")
    x = res.x
    shape = ep.avail.shape
    ch, dis, reg = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    ch[i_idx, t_idx] = x[oc:oc + n]
    dis[i_idx[kd], t_idx[kd]] = x[od:od + nd]
    if nr:
        reg[i_idx, t_idx] = x[orr:orr + nr]
    net = ch - dis
    t_win = peak_window(ep.timestamps, cfg.peak_slots)
    return ServiceResult(
        charge=ch, discharge=dis, regulation_kw=reg,
        energy_cost_sgd=float((net.sum(axis=0) * ep.usep).sum() / 1000),
        degradation_sgd=float(dis.sum() * cfg.deg_cost_sgd_per_kwh / cfg.eta_discharge),
        regulation_revenue_sgd=float((reg.sum(axis=0) * ep.regulation_price).sum() / 1000 * DT_H),
        peak_cost_sgd=float(net.sum(axis=0)[t_win].sum() * cfg.peak_price_sgd_per_kwh),
        solve_s=solve_s, n_vars=nv,
        simultaneous_steps=int(((ch > 1e-6) & (dis > 1e-6)).sum()))
