from datetime import date

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from sg_ev_rl.dispatch.llf import dispatch, envelope
from sg_ev_rl.market.loader import TZ
from sg_ev_rl.scenarios import episodes as E
from sg_ev_rl.scheduling import fleet_rules as R
from sg_ev_rl.scheduling.lp import solve_charge_lp
from sg_ev_rl.settlement.report import settle_episode
from sg_ev_rl.sim.fleet_env import EnvConfig, FleetEnv

START = date(2025, 3, 3)
ETA = 0.92


def fake_market(days=3, seed=0):
    rng = np.random.default_rng(seed)
    steps = pd.date_range(pd.Timestamp(START).tz_localize(TZ), periods=48 * days, freq="30min")
    hours = np.asarray(steps.hour + steps.minute / 60)
    usep = 120 + 40 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 15, len(steps))
    return pd.DataFrame({"interval_start": steps, "usep_price": usep, "regulation_price": 10.0,
                         "reserve_price_CONRESA": 12.0,
                         "demand_gross_mw": 6000.0, "usep_price_pct_336": rng.uniform(0, 1, len(steps))})


def make_sessions(rows):
    t0 = pd.Timestamp(START).tz_localize(TZ)
    out = []
    for k, (arr_h, dep_h, kw, need) in enumerate(rows):
        out.append({"session_id": f"s{k}", "port_id": f"p{k}", "arrival_ts": t0 + pd.Timedelta(hours=arr_h),
                    "departure_ts": t0 + pd.Timedelta(hours=dep_h), "segment": "destination", "zone": "central",
                    "battery_kwh": 60.0, "soc_arrival": 0.3, "soc_target": 0.3 + need / 60,
                    "p_charge_max_kw": kw, "charger_type": "AC", "energy_kwh": need,
                    "feasible_at_arrival": True, "v2g_capable": False, "p_discharge_max_kw": 0.0})
    return pd.DataFrame(out)


def random_episode(n=60, days=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        arr = rng.uniform(0, 24 * days - 1)
        dwell = rng.uniform(0.3, 14)
        kw = float(rng.choice([7.4, 11.0, 50.0]))
        need = ETA * kw * dwell * rng.uniform(0.1, 1.0)  # always feasible
        rows.append((arr, arr + dwell, kw, need))
    return E.episode_from_sessions(make_sessions(rows), fake_market(days + 1, seed), START, days,
                                   n_ports=n, fleet_kw=n * 10.0, scenario_id="test")


def test_availability_fractions_and_truncation():
    # 10:10 -> 11:00, and 23:00 -> 02:00 next day with a 1-day episode
    ep = E.episode_from_sessions(make_sessions([(10 + 1 / 6, 11, 7.4, 3.0), (23, 26, 7.4, 9.0)]),
                                 fake_market(), START, 1, n_ports=2, fleet_kw=14.8, scenario_id="t")
    assert ep.avail[0, 20] == pytest.approx(2 / 3)
    assert ep.avail[0, 21] == pytest.approx(1.0)
    assert ep.avail[0].sum() == pytest.approx(5 / 3)
    # second session: 1 of 3 hours inside the episode -> a third of its need
    assert ep.need_kwh[1] == pytest.approx(3.0)
    assert bool(ep.sessions["truncated"][1])
    assert ep.last_step.tolist() == [21, 47]


def test_envelope_must_charge_and_projection():
    # EV0 needs exactly its full capacity -> all must; EV1 has slack
    remaining = np.array([2 * 7.4 * 0.5 * ETA, 1.0])
    env = envelope(remaining, np.array([7.4, 7.4]), np.array([1.0, 1.0]), np.array([1.0, 5.0]), ETA, 0.5)
    assert env.must[0] == pytest.approx(3.7)
    assert env.must[1] == 0
    assert dispatch(env, 0.0).sum() == pytest.approx(env.e_min)
    assert dispatch(env, 1e9).sum() == pytest.approx(env.e_max)
    x = dispatch(env, env.e_min + 0.5)
    assert x.sum() == pytest.approx(env.e_min + 0.5)
    assert np.all(x <= env.upper + 1e-12)


def test_dispatch_order_is_least_laxity_first():
    remaining = np.array([3.0, 3.0])
    env = envelope(remaining, np.array([7.4, 7.4]), np.array([1.0, 1.0]), np.array([10.0, 2.0]), ETA, 0.5)
    x = dispatch(env, 1.0)  # nothing is must-charge; EV1 has less slack
    assert x[1] == pytest.approx(1.0) and x[0] == 0


@pytest.mark.parametrize("policy", [R.Uncontrolled(), R.PriceThreshold(0.3), R.ReplayAggregate(np.zeros(96))])
@pytest.mark.parametrize("priority", ["llf", "ratio"])
def test_safety_layer_meets_every_target(policy, priority):
    ep = random_episode()
    core = R.run(ep, policy, EnvConfig(dispatch_priority=priority))
    k = settle_episode(core)
    assert k["unmet_kwh"] < 1e-6
    assert k["energy_kwh"] * ETA == pytest.approx(ep.need_kwh.sum())
    assert np.all(core.schedule <= ep.p_max[:, None] * ep.avail * 0.5 + 1e-9)


def test_penalty_mode_counts_unmet_energy():
    ep = random_episode()
    core = R.run(ep, R.ReplayAggregate(np.zeros(ep.n_steps)), EnvConfig(safety_layer=False))
    assert settle_episode(core)["unmet_kwh"] == pytest.approx(ep.need_kwh.sum())


def test_lp_is_a_lower_bound_and_replays_exactly():
    ep = random_episode(seed=3)
    lp = solve_charge_lp(ep, ETA)
    replay = settle_episode(R.run_schedule(ep, lp.schedule))
    assert replay["energy_cost_sgd"] == pytest.approx(lp.cost_sgd, rel=1e-9)
    assert replay["unmet_kwh"] < 1e-6
    for pol in (R.Uncontrolled(), R.PriceThreshold(0.3), R.ReplayAggregate(lp.schedule.sum(0))):
        assert settle_episode(R.run(ep, pol))["energy_cost_sgd"] >= lp.cost_sgd - 1e-6


def test_lp_matches_per_ev_cheapest_fill():
    """Without coupling, the LP optimum is each EV filling its cheapest steps first."""
    ep = random_episode(seed=5)
    lp = solve_charge_lp(ep, ETA)
    total = 0.0
    for i in range(len(ep.sessions)):
        cap = ep.p_max[i] * ep.avail[i] * 0.5
        need = ep.need_kwh[i] / ETA
        for t in np.argsort(ep.usep, kind="stable"):
            take = min(cap[t], need)
            total += take * ep.usep[t] / 1000
            need -= take
    assert lp.cost_sgd == pytest.approx(total, rel=1e-7)


def test_gym_env_api():
    ep = random_episode()
    env = FleetEnv([ep])
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1)
    assert np.all(np.isfinite(obs))
    steps, done = 0, False
    while not done:
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r) and not trunc
        steps += 1
    assert steps == ep.n_steps
    assert settle_episode(env.core)["unmet_kwh"] < 1e-6


def test_seeds_and_portfolio():
    assert E.seed_for("public", "train", 0) == E.seed_for("public", "train", 0)
    assert len({E.seed_for("public", s, i) for s in ("train", "validation") for i in range(50)}) == 100
    ports = pd.DataFrame({"zone": ["CENTRAL"] * 7 + ["WEST"] * 3, "current": ["AC"] * 5 + ["DC"] * 2 + ["AC"] * 3})
    pf = E.portfolio_of_size(ports, 20)
    assert sum(pf.values()) == 20
    assert pf[("CENTRAL", "AC")] == 10
