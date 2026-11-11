import numpy as np
import pytest
from test_fleet_env import random_episode

from sg_ev_rl.rl.common import baselines, rl_env_config, run_policy


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
