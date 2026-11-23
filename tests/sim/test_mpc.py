from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from test_fleet_env import START, fake_market, random_episode

from sg_ev_rl.forecast.prices import STEP, PriceForecast
from sg_ev_rl.scenarios import episodes as E
from sg_ev_rl.scheduling.mpc import RollingMPC, run_mpc
from sg_ev_rl.scheduling.service_lp import ServiceConfig, solve_service_lp
from sg_ev_rl.settlement.report import settle_episode
from sg_ev_rl.sim.fleet_env import EnvConfig

HISTORY_DAYS = 8


def market_with_history(days=3, seed=0):
    """fake_market shifted back so that HISTORY_DAYS of prices precede START, with a varying reserve price."""
    m = fake_market(days + HISTORY_DAYS, seed)
    m["interval_start"] = m["interval_start"] - timedelta(days=HISTORY_DAYS)
    rng = np.random.default_rng(seed + 1)
    m["reserve_price_CONRESA"] = 12 + 8 * rng.random(len(m))
    return m


@pytest.fixture
def market():
    return market_with_history()


@pytest.fixture
def ep(market):
    base = random_episode(n=40, seed=5)
    return E.episode_from_sessions(base.sessions.drop(columns=["need_kwh", "truncated"]), market, START, 2,
                                   n_ports=40, fleet_kw=400.0, scenario_id="mpc")


def test_per_ev_mpc_with_perfect_prices_matches_lp(ep, market):
    lp = solve_service_lp(ep, ServiceConfig(interruptible_load=True))
    k = settle_episode(run_mpc(ep, RollingMPC(PriceForecast(market, "perfect"), "per_ev")))
    assert k["net_cost_sgd"] == pytest.approx(lp.total_cost_sgd, rel=1e-6)
    assert k["unmet_kwh"] < 1e-6


@pytest.mark.parametrize("method", ["last_day", "last_week", "profile7"])
@pytest.mark.parametrize("control", ["aggregate", "per_ev"])
def test_forecast_mpc_meets_targets_and_respects_bound(ep, market, method, control):
    lp = solve_service_lp(ep, ServiceConfig(interruptible_load=True))
    mpc = RollingMPC(PriceForecast(market, method), control)
    k = settle_episode(run_mpc(ep, mpc))
    assert k["unmet_kwh"] < 1e-6
    assert k["net_cost_sgd"] >= lp.total_cost_sgd - 1e-6
    assert mpc.stats.solves > 0


def test_targets_hold_when_every_offer_is_activated(ep, market):
    core = run_mpc(ep, RollingMPC(PriceForecast(market, "profile7")), EnvConfig(il_activation_prob=1.0), seed=0)
    k = settle_episode(core)
    assert k["il_activations"] > 0
    assert k["unmet_kwh"] < 1e-6


@pytest.mark.parametrize("method", ["last_day", "last_week", "profile7"])
def test_forecast_reads_no_future_prices(market, method):
    now = pd.Timestamp(market["interval_start"].iloc[HISTORY_DAYS * 48 + 17])
    steps = pd.date_range(now, periods=60, freq="30min")
    before = PriceForecast(market, method).predict("usep_price", now, steps)
    later = market.copy()
    later.loc[later["interval_start"] > now, "usep_price"] = 1e6
    after = PriceForecast(later, method).predict("usep_price", now, steps)
    np.testing.assert_array_equal(before, after)
    assert before[0] == market.loc[market["interval_start"] == now, "usep_price"].item()


def test_forecast_shifts(market):
    now = pd.Timestamp(market["interval_start"].iloc[HISTORY_DAYS * 48])
    s = market.set_index("interval_start")["usep_price"]
    steps = pd.DatetimeIndex([now + 3 * STEP, now + 50 * STEP])
    day = PriceForecast(market, "last_day").predict("usep_price", now, steps)
    assert day[0] == s[now + 3 * STEP - 48 * STEP]
    assert day[1] == s[now + 50 * STEP - 96 * STEP]
    week = PriceForecast(market, "last_week").predict("usep_price", now, steps)
    assert week[1] == s[now + 50 * STEP - 336 * STEP]
    prof = PriceForecast(market, "profile7").predict("usep_price", now, steps[:1])
    assert prof[0] == pytest.approx(np.mean([s[now + 3 * STEP - j * 48 * STEP] for j in range(1, 8)]))


def test_short_history_falls_back(market):
    first = pd.Timestamp(market["interval_start"].iloc[0])
    s = market.set_index("interval_start")["usep_price"]
    steps = pd.DatetimeIndex([first + STEP, first + 50 * STEP])
    assert PriceForecast(market, "last_day").predict("usep_price", first, steps)[0] == s[first]
    week = PriceForecast(market, "last_week").predict("usep_price", first + 49 * STEP, steps[1:])
    assert week[0] == s[first + 2 * STEP]  # one day back
    prof = PriceForecast(market, "profile7").predict("usep_price", first + 97 * STEP,
                                                     pd.DatetimeIndex([first + 98 * STEP]))
    assert prof[0] == pytest.approx(np.mean([s[first + 2 * STEP], s[first + 50 * STEP]]))  # the two days it has
    with pytest.raises(ValueError, match="history"):
        PriceForecast(market, "last_day").predict("usep_price", first - STEP, pd.DatetimeIndex([first]))
