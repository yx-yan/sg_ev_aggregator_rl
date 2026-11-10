"""Fleet simulator: policies propose power, the simulator enforces physical limits."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sg_ev_rl.assets.fleet import DT_H, Session


@dataclass
class FleetState:
    """What a policy sees at period t (arrays are per vehicle)."""

    t: int
    present: np.ndarray  # bool, plugged in this period
    remaining_kwh: np.ndarray
    periods_left: np.ndarray  # periods until departure, 0 if not present
    max_kw: np.ndarray


class Policy(Protocol):
    name: str

    def act(self, state: FleetState, prices: np.ndarray) -> np.ndarray:
        """Return requested charging power in kW per vehicle."""


@dataclass
class SimResult:
    sessions: list[Session]
    prices: np.ndarray  # SGD/MWh per period
    power_kw: np.ndarray  # shape (n_vehicles, n_periods)
    remaining_kwh: np.ndarray  # unmet energy at the end, per vehicle


class FleetSimulator:
    def __init__(self, sessions: list[Session], n_periods: int):
        if any(s.departure > n_periods for s in sessions):
            raise ValueError("a session departs after the simulation horizon")
        self.sessions = sessions
        self.n_periods = n_periods
        self.arrival = np.array([s.arrival for s in sessions])
        self.departure = np.array([s.departure for s in sessions])
        self.max_kw = np.array([s.max_kw for s in sessions])
        self.energy = np.array([s.energy_kwh for s in sessions])

    def run(self, policy: Policy, prices: np.ndarray) -> SimResult:
        if len(prices) != self.n_periods:
            raise ValueError("prices must cover every period")
        remaining = self.energy.copy()
        power = np.zeros((len(self.sessions), self.n_periods))
        for t in range(self.n_periods):
            present = (self.arrival <= t) & (t < self.departure)
            state = FleetState(t, present, remaining.copy(),
                               np.where(present, self.departure - t, 0), self.max_kw)
            p = np.clip(np.asarray(policy.act(state, prices), dtype=float), 0, self.max_kw)
            p = np.minimum(np.where(present, p, 0), remaining / DT_H)
            power[:, t] = p
            remaining = remaining - p * DT_H
        return SimResult(self.sessions, prices, power, remaining)
