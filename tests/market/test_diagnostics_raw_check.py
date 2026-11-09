import numpy as np
import pandas as pd
import pytest

from sg_ev_rl.market import diagnostics as d
from sg_ev_rl.market.raw_check import compare_months, raw_month_file


def test_daily_spread_known_values(grid):
    g = grid.copy()
    g.loc[g["period"] <= 8, "usep_price"] = 50.0     # bottom 4 h
    g.loc[g["period"] > 40, "usep_price"] = 250.0    # top 4 h
    spread = d.daily_spread(g)
    assert np.allclose(spread.values, 200.0)


def test_price_summary_spike_contribution(grid):
    g = grid.copy()
    g.loc[0, "usep_price"] = 4500.0
    g["day"] = 1
    s = d.price_summary(g, "day").iloc[0]
    assert s["share_gt_500"] == pytest.approx(1 / len(g))
    assert s["spike_contrib_gt_500"] == pytest.approx((4500.0 - 500.0) / len(g))


def test_lag_correlation_finds_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=500))
    df = pd.DataFrame({"a": a, "b": a.shift(1)})   # b lags a by one step
    corr = d.lag_correlation(df, "a", "b")
    assert corr.idxmax() == 1


def test_solar_peak_centre():
    periods = np.arange(1, 49)
    solar = np.where(periods == 27, 1.0, 0.0)      # only 13:00-13:30
    df = pd.DataFrame({"period": periods, "solar_mw": solar})
    s = d.solar_peak(df)
    assert s["peak_period"] == 27
    assert s["centre_hours_if_period_start_based"] == pytest.approx(13.25)


def _write_usep(root, month, rows):
    path = raw_month_file("emc_usep_demand_forecast", "USEP", month, root)
    path.parent.mkdir(parents=True)
    lines = ['"INFORMATION TYPE","DATE","PERIOD","USEP ($/MWh)","DEMAND (MW)","SOLAR(MW)"']
    lines += [f'"USEP","{dt:%d-%b-%Y}","{p}","{v}","6000","0"' for dt, p, v in rows]
    path.write_text("\n".join(lines) + "\n")


def test_compare_months_detects_mismatch(tmp_path, grid):
    month = pd.Period("2025-01", "M")
    days = pd.to_datetime(["2025-01-01", "2025-01-02"])
    rows = [(dt, p, 100.0) for dt in days for p in range(1, 49)]
    rows[3] = (days[0], 4, 101.0)
    _write_usep(tmp_path, month, rows)
    table = grid.assign(demand_mw=6000.0, solar_mw=0.0)
    import sg_ev_rl.market.raw_check as rc
    saved = rc.CHECKS
    rc.CHECKS = {k: v for k, v in saved.items() if k in ("usep_price", "demand_mw")}
    try:
        out = compare_months(table, [month], root=tmp_path).set_index("column")
    finally:
        rc.CHECKS = saved
    assert out.loc["usep_price", "mismatches"] == 1
    assert out.loc["usep_price", "max_abs_diff"] == pytest.approx(1.0)
    assert out.loc["demand_mw", "mismatches"] == 0
