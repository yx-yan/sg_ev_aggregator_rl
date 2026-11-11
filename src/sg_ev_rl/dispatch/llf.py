"""Feasible fleet envelope and least-laxity-first split of a fleet setpoint (charge only).

All quantities are grid-side energy in one step (kWh); battery energy = eta * grid energy.
For each connected EV with remaining battery need e_i:

    cap_now_i    = p_max_i * avail_i,t * dt                 what it can take this step
    cap_future_i = p_max_i * sum_{s>t} avail_i,s * dt       what it can take later
    must_i       = clip(e_i / eta - cap_future_i, 0, cap_now_i)
    upper_i      = min(cap_now_i, e_i / eta)

E_min = sum must_i, E_max = sum upper_i. A setpoint inside [E_min, E_max] is given to
the must-charge part first and the rest in priority order:

    "llf"    least laxity first: spare hours (cap_now + cap_future - need) / p_max
    "ratio"  largest share of remaining capacity still needed: need / (cap_now + cap_future)

Each EV's must-charge depends only on its own future capacity, so every target that was
feasible at arrival is met whatever setpoint is chosen (charge only, no fleet import cap).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Envelope:
    must: np.ndarray
    upper: np.ndarray
    laxity_h: np.ndarray  # spare hours before departure; inf for EVs with nothing left to do
    priority: np.ndarray  # dispatch order, lowest first; inf for EVs with nothing left to do

    @property
    def e_min(self) -> float:
        return float(self.must.sum())

    @property
    def e_max(self) -> float:
        return float(self.upper.sum())


def envelope(remaining: np.ndarray, p_max: np.ndarray, avail_now: np.ndarray, avail_future: np.ndarray,
             eta: float, dt_h: float, priority: str = "llf") -> Envelope:
    """`avail_future` is the sum of availability over steps after this one, per EV."""
    grid_need = np.maximum(remaining, 0) / eta
    cap_now = p_max * avail_now * dt_h
    cap_future = p_max * avail_future * dt_h
    must = np.clip(grid_need - cap_future, 0, cap_now)
    upper = np.minimum(cap_now, grid_need)
    with np.errstate(divide="ignore", invalid="ignore"):
        laxity = np.where(upper > 0, (cap_now + cap_future - grid_need) / p_max, np.inf)
        if priority == "llf":
            key = laxity
        elif priority == "ratio":
            key = np.where(upper > 0, -grid_need / (cap_now + cap_future), np.inf)
        else:
            raise ValueError(f"unknown dispatch priority {priority!r}")
    return Envelope(must, upper, laxity, key)


def dispatch(env: Envelope, energy: float) -> np.ndarray:
    """Split a fleet energy setpoint (clipped to the envelope) across EVs in priority order."""
    e = min(max(energy, env.e_min), env.e_max)
    out = env.must.copy()
    rest = e - env.e_min
    if rest <= 0:
        return out
    head = env.upper - env.must
    order = np.argsort(env.priority, kind="stable")
    filled = np.cumsum(head[order])
    give = np.clip(rest - (filled - head[order]), 0, head[order])
    out[order] += give
    return out
