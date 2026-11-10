import numpy as np
import pytest

from sg_ev_rl.assets.fleet import DT_H, Session
from sg_ev_rl.demo import run
from sg_ev_rl.scenarios.toy import toy_prices, toy_sessions
from sg_ev_rl.scheduling.rules import CheapestPeriods, Uncontrolled
from sg_ev_rl.settlement.report import settle
from sg_ev_rl.sim.simulator import FleetSimulator

N = 48


@pytest.fixture
def setup():
    sessions = toy_sessions(30, N, seed=1)
    return sessions, FleetSimulator(sessions, N), toy_prices(N, seed=1)


class Greedy:
    """Asks for far more than any limit allows."""

    name = "greedy"

    def act(self, state, prices):
        return np.full(len(state.max_kw), 1e6)


@pytest.mark.parametrize("policy", [Uncontrolled(), CheapestPeriods(), Greedy()])
def test_limits_and_energy_balance(setup, policy):
    sessions, sim, prices = setup
    res = sim.run(policy, prices)
    max_kw = np.array([s.max_kw for s in sessions])[:, None]
    assert (res.power_kw >= 0).all() and (res.power_kw <= max_kw + 1e-9).all()
    for i, s in enumerate(sessions):
        outside = np.r_[res.power_kw[i, :s.arrival], res.power_kw[i, s.departure:]]
        assert (outside == 0).all()
    delivered = res.power_kw.sum(axis=1) * DT_H
    energy = np.array([s.energy_kwh for s in sessions])
    np.testing.assert_allclose(delivered + res.remaining_kwh, energy)
    assert (res.remaining_kwh >= -1e-9).all()


@pytest.mark.parametrize("policy", [Uncontrolled(), CheapestPeriods()])
def test_feasible_sessions_fully_served(setup, policy):
    _, sim, prices = setup
    assert settle(sim.run(policy, prices))["unmet_kwh"] < 1e-6


def test_cheapest_not_more_expensive(setup):
    _, sim, prices = setup
    smart = settle(sim.run(CheapestPeriods(), prices))
    dumb = settle(sim.run(Uncontrolled(), prices))
    assert smart["energy_cost_sgd"] <= dumb["energy_cost_sgd"] + 1e-9


def test_session_validation():
    with pytest.raises(ValueError):
        Session("bad", 5, 5, 10, 7.4)


def test_demo_runs():
    df = run({"seed": 0, "n_vehicles": 5, "n_periods": N,
              "policies": ["uncontrolled", "cheapest_periods"]})
    assert list(df.index) == ["uncontrolled", "cheapest_periods"]
