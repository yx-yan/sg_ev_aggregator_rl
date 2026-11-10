"""Rule-based charging policies used as baselines."""
import numpy as np

from sg_ev_rl.assets.fleet import DT_H
from sg_ev_rl.sim.simulator import FleetState


class Uncontrolled:
    """Charge at full power from plug-in until the energy need is met."""

    name = "uncontrolled"

    def act(self, state: FleetState, prices: np.ndarray) -> np.ndarray:
        return np.where(state.present, state.max_kw, 0.0)


class CheapestPeriods:
    """Charge in the cheapest periods before departure (assumes known prices).

    At each period, a vehicle charges if the current period is among the k cheapest
    remaining ones, where k is the number of full-power periods it still needs.
    """

    name = "cheapest_periods"

    def act(self, state: FleetState, prices: np.ndarray) -> np.ndarray:
        power = np.zeros_like(state.max_kw)
        for i in np.flatnonzero(state.present & (state.remaining_kwh > 0)):
            window = prices[state.t:state.t + state.periods_left[i]]
            k = int(np.ceil(state.remaining_kwh[i] / (state.max_kw[i] * DT_H) - 1e-9))
            if 0 in np.argsort(window, kind="stable")[:k]:
                power[i] = state.max_kw[i]
        return power
