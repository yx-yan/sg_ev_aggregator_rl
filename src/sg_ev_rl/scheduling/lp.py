"""Perfect-foresight LP benchmark (B3), solved with HiGHS through scipy.

Charge only, copper plate. Variables x_i,t = grid energy drawn by EV i in step t (kWh), one
per step where the EV is plugged in:

    min   sum_t usep_t / 1000 * sum_i x_i,t
    s.t.  sum_t eta * x_i,t = need_i                      every EV reaches its target
          0 <= x_i,t <= p_max_i * avail_i,t * dt

Targets are met exactly, as in the environment (EVs never charge beyond their need). With no
fleet-level coupling this decomposes per EV; the coupling terms (V2G, regulation headroom,
peak window) are the next step and enter as extra rows.
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from sg_ev_rl.scenarios.episodes import DT_H, Episode


@dataclass
class LPResult:
    schedule: np.ndarray  # (n_sessions, n_steps) grid energy, kWh
    cost_sgd: float
    solve_s: float
    n_vars: int
    status: str


def solve_charge_lp(ep: Episode, eta: float = 0.92) -> LPResult:
    i_idx, t_idx = np.nonzero(ep.avail > 0)
    n = len(i_idx)
    cost = ep.usep[t_idx] / 1000
    ub = ep.p_max[i_idx] * ep.avail[i_idx, t_idx] * DT_H
    a_eq = coo_matrix((np.full(n, eta), (i_idx, np.arange(n))), shape=(len(ep.sessions), n)).tocsr()
    t0 = time.perf_counter()
    res = linprog(cost, A_eq=a_eq, b_eq=ep.need_kwh, bounds=np.column_stack([np.zeros(n), ub]), method="highs")
    solve_s = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    sched = np.zeros_like(ep.avail)
    sched[i_idx, t_idx] = res.x
    return LPResult(sched, float(res.fun), solve_s, n, res.message)
