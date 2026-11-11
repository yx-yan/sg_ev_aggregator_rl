from dataclasses import replace

import numpy as np
import pytest
from test_fleet_env import random_episode

from sg_ev_rl.rl.common import baselines, rl_env_config, run_policy
from sg_ev_rl.scenarios.episodes import scale_prices
from sg_ev_rl.sim.fleet_env import N_RELATIVE, OBS_DIM, FleetEnv, obs_dim


def test_run_policy_and_baselines_on_synthetic_episode():
    ep = random_episode(n=30, seed=4)
    cfg = rl_env_config()
    lo = run_policy(lambda o: np.array([-1.0, -1.0], np.float32), ep, cfg)
    hi = run_policy(lambda o: np.array([1.0, 1.0], np.float32), ep, cfg)
    assert lo["unmet_kwh"] < 1e-6 and hi["unmet_kwh"] < 1e-6
    assert lo["il_revenue_sgd"] == 0 and hi["il_revenue_sgd"] > 0
    b = baselines([ep], p_grid=(0.9,)).set_index("policy")["net_cost_sgd"]
    # u = 1 everywhere charges as early as possible like B1 and offers all backable IL
    assert hi["net_cost_sgd"] == pytest.approx(b["B1_il"])
    assert b["LP_il_per_ev"] <= b.drop("LP_il_per_ev").min() + 1e-6


def _with_regime_features(ep):
    n = ep.n_steps
    return replace(ep, conres_pct_7d=np.full(n, 0.9), conres_ratio_7d=np.full(n, 2.0),
                   conres_fc_ratio=np.full(n, np.nan), usep_ratio_7d=np.ones(n), usep_fc_ratio=np.ones(n))


def test_relative_price_observation():
    ep = _with_regime_features(random_episode(n=10, seed=1))
    base, rel = FleetEnv([ep], rl_env_config()), FleetEnv([ep], rl_env_config(relative_prices=True))
    o0, o1 = base.reset(seed=0)[0], rel.reset(seed=0)[0]
    assert rel.observation_space.shape[0] == o1.size == obs_dim(rel.cfg) == OBS_DIM + N_RELATIVE
    np.testing.assert_array_equal(o1[:OBS_DIM], o0)
    assert o1[OBS_DIM:].tolist() == pytest.approx([0.9, 2.0, 1.0, 1.0, 1.0])  # NaN forecast -> neutral 1


def test_scale_prices_changes_level_not_sessions():
    ep = random_episode(n=10, seed=2)
    s = scale_prices(ep, usep=2.0, conres=0.5)
    np.testing.assert_allclose(s.usep, 2 * ep.usep)
    np.testing.assert_allclose(s.contingency_reserve_price, 0.5 * ep.contingency_reserve_price)
    assert s.sessions is ep.sessions and s.usep_pct_7d is ep.usep_pct_7d
