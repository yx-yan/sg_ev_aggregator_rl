import numpy as np
import pytest
from test_fleet_env import ETA, random_episode

from sg_ev_rl.scenarios.episodes import DT_H
from sg_ev_rl.scheduling.lp import solve_charge_lp
from sg_ev_rl.scheduling.service_lp import ServiceConfig, peak_window, solve_service_lp

PEAK = (31, 32, 33, 34)


@pytest.fixture
def ep():
    e = random_episode(n=40, seed=11)
    s = e.sessions
    rng = np.random.default_rng(0)
    s["v2g_capable"] = rng.random(len(s)) < 0.6
    s["p_discharge_max_kw"] = np.where(s["v2g_capable"], s["p_charge_max_kw"], 0.0)
    return e


def levels(ep, res, cfg):
    s = ep.sessions
    l0 = (s["soc_arrival"] * s["battery_kwh"]).to_numpy()
    flow = cfg.eta_charge * res.charge - res.discharge / cfg.eta_discharge
    return l0[:, None] + np.cumsum(flow, axis=1)


def test_no_services_equals_charge_lp(ep):
    res = solve_service_lp(ep, ServiceConfig(eta_charge=ETA))
    assert res.total_cost_sgd == pytest.approx(solve_charge_lp(ep, ETA).cost_sgd, rel=1e-7)
    assert res.discharge.sum() == 0 and res.regulation_kw.sum() == 0


def test_options_never_raise_cost(ep):
    base = solve_service_lp(ep).total_cost_sgd
    v2g = solve_service_lp(ep, ServiceConfig(v2g=True)).total_cost_sgd
    both = solve_service_lp(ep, ServiceConfig(v2g=True, regulation=True)).total_cost_sgd
    assert v2g <= base + 1e-6
    assert both <= v2g + 1e-6


def test_prohibitive_degradation_stops_discharge(ep):
    res = solve_service_lp(ep, ServiceConfig(v2g=True, deg_cost_sgd_per_kwh=1e3))
    assert res.discharge.sum() < 1e-6


def test_physics_targets_and_headroom(ep):
    cfg = ServiceConfig(v2g=True, regulation=True, peak_price_sgd_per_kwh=0.2, peak_slots=PEAK)
    res = solve_service_lp(ep, cfg)
    s = ep.sessions
    batt = s["battery_kwh"].to_numpy()
    lev = levels(ep, res, cfg)
    target = (s["soc_arrival"] * batt).to_numpy() + ep.need_kwh
    assert np.allclose(lev[np.arange(len(s)), ep.last_step], target, atol=1e-5)
    present = ep.avail > 0
    lmin = np.minimum(cfg.soc_min * batt, (s["soc_arrival"] * batt).to_numpy())[:, None]
    assert np.all(lev[present] >= np.broadcast_to(lmin, lev.shape)[present] - 1e-6)
    cap = ep.p_max[:, None] * ep.avail * DT_H
    dcap = s["p_discharge_max_kw"].to_numpy()[:, None] * ep.avail * DT_H
    assert np.all(res.charge <= cap + 1e-6)
    assert np.all(res.discharge[~s["v2g_capable"].to_numpy()] == 0)
    up = res.regulation_kw * DT_H + res.charge - res.discharge
    down = res.regulation_kw * DT_H - res.charge + res.discharge
    assert np.all(up <= cap + 1e-6) and np.all(down <= dcap + 1e-6)
    assert np.all(res.regulation_kw[~present] == 0)


def test_peak_price_shifts_load_out_of_window(ep):
    win = peak_window(ep.timestamps, PEAK)
    base = solve_service_lp(ep, ServiceConfig(v2g=True))
    peak = solve_service_lp(ep, ServiceConfig(v2g=True, peak_price_sgd_per_kwh=1.0, peak_slots=PEAK))
    assert peak.net_kw[win].sum() <= base.net_kw[win].sum() + 1e-6
