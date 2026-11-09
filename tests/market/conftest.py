import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sg_ev_rl.market.paths import market_table_path  # noqa: E402

TZ = "Asia/Singapore"


def make_grid(dates, value=100.0):
    """Minimal market table on a correct period-ending grid."""
    rows = []
    for d in pd.to_datetime(dates):
        for p in range(1, 49):
            rows.append({"date": d.date(), "period": p,
                         "timestamp": (d + pd.Timedelta(minutes=30 * p)).tz_localize(TZ),
                         "usep_price": value})
    return pd.DataFrame(rows)


@pytest.fixture
def grid():
    return make_grid(["2025-01-01", "2025-01-02"])


@pytest.fixture(scope="session")
def real_market():
    if not market_table_path().exists():
        pytest.skip("research data not available")
    from sg_ev_rl.market.loader import load_market
    return load_market()
