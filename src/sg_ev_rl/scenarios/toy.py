"""Toy scenario for smoke tests and demos.

Placeholder until the session generator is calibrated to Singapore statistics;
numbers here are illustrative only.
"""
import numpy as np

from sg_ev_rl.assets.fleet import DT_H, Session


def toy_sessions(n_vehicles: int, n_periods: int = 48, seed: int = 0) -> list[Session]:
    """Workplace-style sessions: arrive 07:00-11:00, stay 6-10 h, always feasible."""
    rng = np.random.default_rng(seed)
    sessions = []
    for i in range(n_vehicles):
        arrival = int(rng.integers(14, 22))
        departure = min(n_periods, arrival + int(rng.integers(12, 21)))
        max_kw = float(rng.choice([7.4, 11.0]))
        cap = max_kw * (departure - arrival) * DT_H
        energy = float(min(rng.uniform(5, 30), cap))
        sessions.append(Session(f"ev{i:03d}", arrival, departure, energy, max_kw))
    return sessions


def toy_prices(n_periods: int = 48, seed: int = 0) -> np.ndarray:
    """USEP-like price path in SGD/MWh: night trough, evening peak, noise."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n_periods) * DT_H
    shape = 180 + 40 * np.sin((hours - 9) / 24 * 2 * np.pi) + 30 * np.exp(-((hours - 19) ** 2) / 4)
    return shape + rng.normal(0, 10, n_periods)
