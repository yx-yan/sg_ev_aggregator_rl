"""Perfect-foresight LP with V2G, regulation, interruptible load and peak shaving (service-value study).

Which services are open to EV aggregations in Singapore today is set out in
reports/b/2026-11-11_sg_service_eligibility.md: interruptible load (IL, paid the contingency
reserve price) and demand response are; regulation and V2G are future scenarios.
SCENARIO STUDY: activation profiles and the peak-shaving price are assumptions (see ServiceConfig).

Per plugged-in (EV i, step t), all energies in kWh per step and grid side unless noted:

    c_it  charge, 0..p_max a_it dt          d_it  discharge, 0..p_dis a_it dt (V2G EVs only)
    r_it  symmetric regulation capacity, kW  q_it  interruptible load offered, kW (contingency reserve)
    l_it  battery energy at the end of step t (battery side)

    l_it = l_i,t-1 + eta_c c_it - d_it / eta_d,   l before arrival = soc_arrival * B
    l at the EV's last step = soc_arrival * B + need_i (target met exactly)
    soc_min B <= l_it <= soc_max B   (widened to the arrival / target level when needed)
    regulation:  r dt + c - d <= p_max a dt,  r dt - c + d <= p_dis a dt,
                 l + alpha eta_c r dt <= l_max,  l - alpha r dt / eta_d >= l_min
    IL:          q dt <= c                                  only load being drawn can be interrupted
                 q T_act + (target - l_it) / eta_c <= cap_after_it
                     an activation of T_act hours can be made up before departure; cap_after is
                     the grid energy the EV could still take after step t (charge only, so the
                     energy still to come equals (target - l) / eta_c)

    min  sum usep/1000 (c - d) + deg/eta_d d - reg_price/1000 r dt - conres_price/1000 q dt
         + peak_price (c - d) in the peak window

Regulation activation is taken as energy-neutral over a step (no AGC signal is public). IL is paid
for being scheduled only, as in the market (no payment on activation). Charge and discharge are
separate variables; simultaneous use is counted.
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
    interruptible_load: bool = False
    peak_price_sgd_per_kwh: float = 0.0  # ASSUMPTION: value of net import in the peak window
    peak_slots: tuple[int, ...] = ()  # half-hour indices 0..47 of the peak window
    eta_charge: float = 0.92
    eta_discharge: float = 0.92  # ASSUMPTION
    deg_cost_sgd_per_kwh: float = 0.06  # ASSUMPTION: per kWh taken out of the battery
    soc_min: float = 0.2  # ASSUMPTION: floor kept for the driver
    soc_max: float = 0.95
    activation_ratio: float = 0.1  # ASSUMPTION: battery room reserved per kWh of regulation capacity-hour
    il_activation_h: float = 0.5  # IL activation length the EV must be able to make up (EMA plans 30 min)


@dataclass
class ServiceResult:
    charge: np.ndarray  # (n_sessions, n_steps) kWh
    discharge: np.ndarray
    regulation_kw: np.ndarray
    il_kw: np.ndarray
    energy_cost_sgd: float
    degradation_sgd: float
    regulation_revenue_sgd: float
    il_revenue_sgd: float
    peak_cost_sgd: float
    solve_s: float
    n_vars: int
    simultaneous_steps: int

    @property
    def total_cost_sgd(self) -> float:
        return (self.energy_cost_sgd + self.degradation_sgd + self.peak_cost_sgd
                - self.regulation_revenue_sgd - self.il_revenue_sgd)

    @property
    def net_kw(self) -> np.ndarray:
        return (self.charge - self.discharge).sum(axis=0) / DT_H


def peak_window(timestamps, slots) -> np.ndarray:
    hh = np.asarray(timestamps.hour * 2 + timestamps.minute // 30)
    return np.isin(hh, list(slots))


class _Rows:
    """Collects sparse <= rows block by block."""

    def __init__(self, n_vars: int):
        self.n_vars, self.n_rows = n_vars, 0
        self.r, self.c, self.v, self.b = [], [], [], []

    def add(self, terms, rhs):
        """`terms`: list of (column index array, coefficient); a column of -1 is skipped."""
        k = np.arange(len(rhs))
        for col, coef in terms:
            m = col >= 0
            self.r.append(self.n_rows + k[m])
            self.c.append(col[m])
            self.v.append(np.broadcast_to(np.asarray(coef, float), col.shape)[m])
        self.b.append(np.asarray(rhs, float))
        self.n_rows += len(rhs)

    def matrices(self):
        if not self.n_rows:
            return None, None
        a = coo_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                       shape=(self.n_rows, self.n_vars)).tocsr()
        return a, np.concatenate(self.b)


def solve_service_lp(ep: Episode, cfg: ServiceConfig = ServiceConfig(),
                     fixed_charge: np.ndarray | None = None) -> ServiceResult:
    """`fixed_charge` (n_sessions, n_steps) pins the charging schedule, e.g. to the uncontrolled one,
    so that only service offers are optimised."""
    if cfg.interruptible_load and cfg.v2g:
        raise NotImplementedError("IL is modelled for charge-only fleets")
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
    nd, nr, nq = len(kd), n if cfg.regulation else 0, n if cfg.interruptible_load else 0

    # variable blocks: c, d, r, q, l
    oc, od = 0, n
    orr = od + nd
    oq = orr + nr
    ol = oq + nq
    nv = ol + n
    k = np.arange(n)
    c_col, l_col = oc + k, ol + k
    d_col = np.full(n, -1)
    d_col[kd] = od + np.arange(nd)

    price = ep.usep[t_idx] / 1000
    win = peak_window(ep.timestamps, cfg.peak_slots)[t_idx] * cfg.peak_price_sgd_per_kwh
    cost = np.zeros(nv)
    cost[oc:oc + n] = price + win
    cost[od:od + nd] = -price[kd] - win[kd] + cfg.deg_cost_sgd_per_kwh / cfg.eta_discharge
    if nr:
        cost[orr:orr + nr] = -ep.regulation_price[t_idx] / 1000 * DT_H
    if nq:
        cost[oq:oq + nq] = -ep.contingency_reserve_price[t_idx] / 1000 * DT_H

    # dynamics: l_k - l_{k-1} - eta_c c_k + d_k / eta_d = l0 (first step) or 0
    prev = np.flatnonzero(~first)
    a_eq = coo_matrix((np.concatenate([np.ones(n), np.full(n, -cfg.eta_charge), -np.ones(len(prev)),
                                       np.full(nd, 1 / cfg.eta_discharge)]),
                       (np.concatenate([k, k, prev, kd]),
                        np.concatenate([l_col, c_col, l_col[prev] - 1, d_col[kd]]))), shape=(n, nv)).tocsr()
    b_eq = np.where(first, l0[i_idx], 0.0)

    rows = _Rows(nv)
    if nr:
        r_col = orr + k
        rows.add([(r_col, DT_H), (c_col, 1.0), (d_col, -1.0)], p_max * a * DT_H)
        rows.add([(r_col, DT_H), (c_col, -1.0), (d_col, 1.0)], p_dis * a * DT_H)
        rows.add([(l_col, 1.0), (r_col, cfg.activation_ratio * cfg.eta_charge * DT_H)], lmax[i_idx])
        rows.add([(l_col, -1.0), (r_col, cfg.activation_ratio / cfg.eta_discharge * DT_H)], -lmin[i_idx])
    if nq:
        q_col = oq + k
        avail_after = np.cumsum(ep.avail[:, ::-1], axis=1)[:, ::-1] - ep.avail
        cap_after = p_max * avail_after[i_idx, t_idx] * DT_H
        rows.add([(q_col, DT_H), (c_col, -1.0)], np.zeros(n))
        rows.add([(q_col, cfg.il_activation_h), (l_col, -1 / cfg.eta_charge)],
                 cap_after - target[i_idx] / cfg.eta_charge)
    a_ub, b_ub = rows.matrices()

    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    ub[oc:oc + n] = p_max * a * DT_H
    ub[od:od + nd] = p_dis[kd] * a[kd] * DT_H
    lb[ol:ol + n], ub[ol:ol + n] = lmin[i_idx], lmax[i_idx]
    end = ol + np.flatnonzero(last)
    tol = 0.0
    if fixed_charge is not None:
        lb[oc:oc + n] = ub[oc:oc + n] = fixed_charge[i_idx, t_idx]
        tol = 1e-6  # the pinned schedule meets the target only to rounding
    lb[end], ub[end] = target[i_idx[last]] - tol, target[i_idx[last]] + tol

    t0 = time.perf_counter()
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                  bounds=np.column_stack([lb, ub]), method="highs")
    solve_s = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(f"service LP failed: {res.message}")
    x = res.x
    shape = ep.avail.shape
    ch, dis, reg, il = (np.zeros(shape) for _ in range(4))
    ch[i_idx, t_idx] = x[oc:oc + n]
    dis[i_idx[kd], t_idx[kd]] = x[od:od + nd]
    if nr:
        reg[i_idx, t_idx] = x[orr:orr + nr]
    if nq:
        il[i_idx, t_idx] = x[oq:oq + nq]
    net = ch - dis
    t_win = peak_window(ep.timestamps, cfg.peak_slots)
    conres = ep.contingency_reserve_price if nq else np.zeros(ep.n_steps)
    return ServiceResult(
        charge=ch, discharge=dis, regulation_kw=reg, il_kw=il,
        energy_cost_sgd=float((net.sum(axis=0) * ep.usep).sum() / 1000),
        degradation_sgd=float(dis.sum() * cfg.deg_cost_sgd_per_kwh / cfg.eta_discharge),
        regulation_revenue_sgd=float((reg.sum(axis=0) * ep.regulation_price).sum() / 1000 * DT_H),
        il_revenue_sgd=float((il.sum(axis=0) * conres).sum() / 1000 * DT_H),
        peak_cost_sgd=float(net.sum(axis=0)[t_win].sum() * cfg.peak_price_sgd_per_kwh),
        solve_s=solve_s, n_vars=nv,
        simultaneous_steps=int(((ch > 1e-6) & (dis > 1e-6)).sum()))
