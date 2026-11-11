import numpy as np
import pytest
from test_fleet_env import random_episode

from sg_ev_rl.scenarios.episodes import DT_H
from sg_ev_rl.scheduling import fleet_rules as R
from sg_ev_rl.scheduling.service_lp import ServiceConfig, solve_service_lp
from sg_ev_rl.settlement.report import settle_episode
from sg_ev_rl.sim.fleet_env import EnvConfig, FleetEnv


@pytest.fixture
def ep():
    return random_episode(n=50, seed=21)


def test_lp_with_il_replays_exactly(ep):
    lp = solve_service_lp(ep, ServiceConfig(interruptible_load=True))
    k = settle_episode(R.run_schedule(ep, lp.charge, il_kw=lp.il_kw))
    assert k["il_revenue_sgd"] == pytest.approx(lp.il_revenue_sgd, rel=1e-7)
    assert k["energy_cost_sgd"] == pytest.approx(lp.energy_cost_sgd, rel=1e-7)
    assert k["net_cost_sgd"] == pytest.approx(lp.total_cost_sgd, rel=1e-7)
    assert k["unmet_kwh"] < 1e-6


@pytest.mark.parametrize("policy", [R.Uncontrolled(), R.PriceThreshold(0.3), R.ReplayAggregate(np.zeros(96))])
def test_targets_hold_even_if_every_offer_is_activated(ep, policy):
    cfg = EnvConfig(il_activation_prob=1.0)
    core = R.run(ep, policy, cfg, il_share=1.0, seed=0)
    k = settle_episode(core)
    assert k["il_revenue_sgd"] > 0
    assert k["il_activations"] > 0
    assert k["unmet_kwh"] < 1e-6


def test_il_offer_never_exceeds_load(ep):
    core = R.run(ep, R.Uncontrolled(), il_share=1.0)
    load_kw = core.schedule.sum(axis=0) / DT_H
    assert np.all(np.asarray(core.log.il_kw) <= load_kw + 1e-6)


def test_il_share_scales_offer(ep):
    full = settle_episode(R.run(ep, R.Uncontrolled(), il_share=1.0))["il_revenue_sgd"]
    half = settle_episode(R.run(ep, R.Uncontrolled(), il_share=0.5))["il_revenue_sgd"]
    assert half == pytest.approx(full / 2)


def test_lp_il_bound_dominates_rules(ep):
    lp = solve_service_lp(ep, ServiceConfig(interruptible_load=True))
    for pol in (R.Uncontrolled(), R.PriceThreshold(0.3)):
        assert settle_episode(R.run(ep, pol, il_share=1.0))["net_cost_sgd"] >= lp.total_cost_sgd - 1e-6


def test_gym_reward_includes_il(ep):
    env = FleetEnv([ep], EnvConfig(reward_scale_sgd=1.0, unmet_penalty_sgd_per_kwh=0.0))
    env.reset(seed=0)
    total, rev = 0.0, 0.0
    done = False
    while not done:
        _, r, done, _, info = env.step(np.array([1.0, 1.0], np.float32))
        total += r
        rev += info["il_revenue_sgd"]
    k = settle_episode(env.core)
    assert rev > 0
    assert total == pytest.approx(-(k["energy_cost_sgd"] - k["il_revenue_sgd"]))


def test_envelope_action_mode_spans_feasible_band(ep):
    for u, pick in ((-1.0, "e_min"), (1.0, "e_max")):
        env = FleetEnv([ep], EnvConfig(action_mode="envelope"))
        env.reset(seed=0)
        done = False
        while not done:
            _, _, done, _, info = env.step(np.array([u, 0.0], np.float32))
            assert info["energy_kwh"] == pytest.approx(info[pick], abs=1e-6)
        assert settle_episode(env.core)["unmet_kwh"] < 1e-6
