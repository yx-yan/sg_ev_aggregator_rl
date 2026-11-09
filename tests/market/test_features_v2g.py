import numpy as np
import pandas as pd
import pytest

from sg_ev_rl.market.features import add_relative_price_features
from sg_ev_rl.market.v2g_roughcalc import Battery, arbitrage_lp


def _prices(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"usep_price": 100 + 30 * rng.standard_normal(n)})


@pytest.mark.parametrize("t", [100, 500, 998])
def test_features_do_not_use_future_values(t):
    base = _prices()
    perturbed = base.copy()
    perturbed.loc[t + 1:, "usep_price"] *= 10.0       # change only the future
    a = add_relative_price_features(base)
    b = add_relative_price_features(perturbed)
    cols = [c for c in a.columns if c != "usep_price"]
    pd.testing.assert_frame_equal(a.loc[:t, cols], b.loc[:t, cols])


def test_features_ignore_values_older_than_window():
    base = _prices()
    perturbed = base.copy()
    perturbed.loc[:100, "usep_price"] = 0.0           # outside a 336-step window seen from t=500
    a = add_relative_price_features(base)
    b = add_relative_price_features(perturbed)
    cols = [c for c in a.columns if c != "usep_price"]
    pd.testing.assert_frame_equal(a.loc[500:, cols], b.loc[500:, cols])


def test_percentile_range_and_warmup():
    f = add_relative_price_features(_prices())
    pct = f["usep_price_pct_336"]
    assert pct.iloc[:47].isna().all() and pct.iloc[47:].notna().all()
    assert pct.dropna().between(0, 1).all()


def test_arbitrage_zero_for_flat_price():
    profit, _ = arbitrage_lp(np.full(24, 0.2), Battery(round_trip_eff=0.85, deg_cost_per_kwh=0.06))
    assert profit == pytest.approx(0.0, abs=1e-9)


def test_arbitrage_two_level_price_matches_hand_calculation():
    # store 0.5 kWh (bounds 0.3..0.8) at 0.1, release it at 0.3; eta = 0.9 each way, deg 0.05
    p = np.r_[np.full(12, 0.1), np.full(12, 0.3)]
    profit, sold = arbitrage_lp(p, Battery(round_trip_eff=0.81, deg_cost_per_kwh=0.05))
    expected = 0.5 * 0.9 * (0.3 - 0.05) - 0.5 / 0.9 * 0.1
    assert profit == pytest.approx(expected, rel=1e-6)
    assert sold == pytest.approx(0.45, rel=1e-6)


def test_arbitrage_respects_power_limit():
    # only one step of price difference: at most c_rate*0.5h can be moved
    p = np.r_[0.1, np.full(10, 0.2), 0.5]
    b = Battery(round_trip_eff=1.0, deg_cost_per_kwh=0.0)
    profit, _ = arbitrage_lp(p, b)
    step = b.c_rate_per_h * 0.5
    assert profit == pytest.approx(step * (0.5 - 0.1), rel=1e-6)
