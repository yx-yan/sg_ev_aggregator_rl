"""Fleet-level rule baselines and the runner shared by rules and LP replay.

Each rule maps the core's state to a fleet energy setpoint for the step; FleetCore projects it
onto [E_min, E_max] and splits it in dispatch-priority order, exactly as for the RL agent.
"""
import numpy as np

from sg_ev_rl.scenarios.episodes import Episode
from sg_ev_rl.sim.fleet_env import EnvConfig, FleetCore


class Uncontrolled:
    """B1: every connected EV charges at full power until its need is met."""

    name = "B1_uncontrolled"

    def setpoint(self, core: FleetCore) -> float:
        return np.inf


class PriceThreshold:
    """B2: charge fully when USEP is at or below its trailing 7-day percentile p, else only must-charge.

    p is tuned on the validation split. The percentile uses past prices only.
    """

    def __init__(self, p: float = 0.3):
        self.p = p
        self.name = f"B2_price_threshold_p{p:g}"

    def setpoint(self, core: FleetCore) -> float:
        pct = core.ep.usep_pct_7d[core.t]
        return np.inf if (np.isnan(pct) or pct <= self.p) else 0.0


class ReplayAggregate:
    """Replays a fleet energy profile (e.g. the LP's) through the projection and LLF split."""

    def __init__(self, energy_kwh: np.ndarray, name: str = "LP_replay_aggregate"):
        self.energy, self.name = energy_kwh, name

    def setpoint(self, core: FleetCore) -> float:
        return float(self.energy[core.t])


def run(ep: Episode, policy, cfg: EnvConfig = EnvConfig()) -> FleetCore:
    core = FleetCore(ep, cfg)
    while not core.done:
        core.step_energy(policy.setpoint(core))
    return core


def run_schedule(ep: Episode, schedule: np.ndarray, cfg: EnvConfig = EnvConfig()) -> FleetCore:
    """Replay a per-EV schedule step by step (limits checked by the core)."""
    core = FleetCore(ep, cfg)
    while not core.done:
        core.step_schedule(schedule[:, core.t])
    return core
