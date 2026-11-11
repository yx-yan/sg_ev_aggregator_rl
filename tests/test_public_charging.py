from datetime import date

import numpy as np
import pandas as pd
import pytest

from sg_ev_rl.assets.fleet import DT_H
from sg_ev_rl.market.paths import ev_charging_dir
from sg_ev_rl.scenarios import public_charging as pc
from sg_ev_rl.sim.simulator import FleetSimulator

CAL = (date(2026, 10, 26), date(2026, 11, 8))  # Mon..Sun, two weeks


def fake_inputs(seed=0):
    """Two ports per group; AC stays 4 h from 18:00, DC 40 min from 12:00, every day."""
    rng = np.random.default_rng(seed)
    ports, sessions = [], []
    days = pd.date_range(CAL[0], CAL[1], freq="D")
    for z in pc.ZONES:
        for c, (hour, dur) in {"AC": (18, 240), "DC": (12, 40)}.items():
            for k in range(2):
                pid = f"{z}-{c}-{k}"
                ports.append({"ev_cp_id": pid, "zone": z, "current": c, "power_kw": 7.4 if c == "AC" else 60.0,
                              "first_seen": days[0], "last_seen": days[-1] + pd.Timedelta(hours=23)})
                for d in days:
                    start = d + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 6)) * 5)
                    sessions.append({"ev_cp_id": pid, "zone": z, "current": c, "start": start,
                                     "end": start + pd.Timedelta(minutes=dur), "duration_min": float(dur),
                                     "is_clean": True})
    ts = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23, minutes=55), freq="5min")
    snaps = pd.DataFrame({"ts": ts})
    return pd.DataFrame(sessions), pd.DataFrame(ports), snaps


@pytest.fixture
def profile():
    return pc.calibrate(*fake_inputs(), dates=CAL)


def test_calibration_shapes(profile):
    assert profile.meta["n_days"] == [10, 4]
    ac, dc = pc.CURRENTS.index("AC"), pc.CURRENTS.index("DC")
    # one arrival per port per day, all in the 18:00/18:30 slots for AC
    assert profile.hazard[0, ac, 0, :36].sum() == 0
    assert profile.hazard[0, ac, 0, 36:38].sum() > 0
    assert profile.hazard[0, dc, 0, 24:26].sum() > 0
    # AC ports are occupied 18:00-22:00, i.e. 8 of 48 half-hours
    assert profile.occupancy[0, ac, 0, 40] == pytest.approx(1.0)
    assert profile.occupancy[0, ac, 0, 10] == 0
    assert np.all((profile.hazard >= 0) & (profile.hazard <= 1))


def test_generate_respects_one_vehicle_per_port(profile):
    pf = {("CENTRAL", "AC"): 20, ("WEST", "DC"): 10}
    g = pc.generate(profile, pf, date(2026, 9, 14), 7, seed=3)
    assert len(g) > 0
    assert (g["arrival_ts"] >= pd.Timestamp("2026-09-14")).all()
    for _, p in g.groupby("port_id"):
        p = p.sort_values("arrival_ts")
        assert (p["arrival_ts"].iloc[1:].to_numpy() >= p["departure_ts"].iloc[:-1].to_numpy()).all()
    assert set(g.loc[g["current"] == "AC", "arrival_ts"].dt.hour) <= {18}


def test_generate_is_reproducible(profile):
    pf = {("EAST", "AC"): 5}
    a = pc.generate(profile, pf, date(2026, 9, 14), 3, seed=7)
    b = pc.generate(profile, pf, date(2026, 9, 14), 3, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_energy_is_feasible_and_bounded(profile):
    g = pc.generate(profile, {("NORTH", "AC"): 10, ("NORTH", "DC"): 10}, date(2026, 9, 14), 5, seed=1)
    v = pc.VehicleAssumptions()
    assert (g["energy_kwh"] >= 0).all()
    assert (g["energy_kwh"] <= g["max_kw"] * g["duration_min"] / 60 + 1e-9).all()
    assert (g["energy_kwh"] <= (v.target_soc_ac - v.arrival_soc_ac[0]) * max(v.battery_kwh) + 1e-9).all()
    assert (g["max_kw"] <= g["port_kw"]).all()


def test_to_sessions_runs_in_simulator(profile):
    g = pc.generate(profile, {("CENTRAL", "AC"): 10, ("CENTRAL", "DC"): 5}, date(2026, 9, 14), 2, seed=2)
    n = 2 * 48
    sessions = pc.to_sessions(g, date(2026, 9, 14), n)
    assert len(sessions) == len(g)
    assert all(s.departure <= n for s in sessions)
    assert all(s.energy_kwh <= s.max_deliverable_kwh + 1e-9 for s in sessions)
    sim = FleetSimulator(sessions, n)

    class Full:
        name = "full"

        def act(self, state, prices):
            return state.max_kw

    res = sim.run(Full(), np.full(n, 100.0))
    assert res.remaining_kwh.max() < 1e-6
    assert res.power_kw.sum() * DT_H == pytest.approx(sum(s.energy_kwh for s in sessions))


def test_calibration_needs_both_day_types():
    s, p, snaps = fake_inputs()
    with pytest.raises(ValueError):
        pc.calibrate(s, p, snaps, dates=(date(2026, 10, 26), date(2026, 10, 30)))


def test_real_data_matches_out_of_sample():
    if not (ev_charging_dir() / "pseudo_sessions.parquet").exists():
        pytest.skip("research data not available")
    sess, ports, snaps = pc.load_calibration_inputs()
    prof = pc.calibrate(sess, ports, snaps)
    pf = pc.portfolio_from_share(ports, 0.05)
    start, n_days = date(2026, 9, 14), 14  # inside real_validation, outside calibration
    g = pc.generate(prof, pf, start, n_days, seed=0)
    r = pc.real_sessions(sess, ports, pf, start, n_days, seed=0)
    assert len(g) / len(r) == pytest.approx(1, abs=0.1)
    for c in pc.CURRENTS:
        mg = g.loc[g["current"] == c, "duration_min"].median()
        mr = r.loc[r["current"] == c, "duration_min"].median()
        assert mg / mr == pytest.approx(1, abs=0.2)
