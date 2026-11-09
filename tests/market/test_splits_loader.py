from datetime import date, timedelta

import pandas as pd
import pytest

from sg_ev_rl.market.loader import check_grid
from sg_ev_rl.market.splits import SPLITS, TUNING_CUTOFF, split_of


def test_splits_are_contiguous_and_ordered():
    spans = list(SPLITS.values())
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start == end + timedelta(days=1)


def test_split_boundaries():
    assert split_of(date(2025, 12, 31)) == "train"
    assert split_of(date(2026, 1, 1)) == "validation"
    assert split_of(date(2026, 4, 1)) == "test"
    assert split_of(date(2026, 9, 3)) == "test"
    assert split_of(date(2026, 9, 4)) == "real_validation"
    assert split_of(date(2026, 10, 26)) is None
    assert TUNING_CUTOFF == SPLITS["test"][0]


def test_check_grid_accepts_period_ending_grid(grid):
    check_grid(grid)


def test_check_grid_rejects_missing_period(grid):
    with pytest.raises(ValueError):
        check_grid(grid.drop(index=5).reset_index(drop=True))


def test_check_grid_rejects_period_start_timestamps(grid):
    shifted = grid.assign(timestamp=grid["timestamp"] - pd.Timedelta(minutes=30))
    with pytest.raises(ValueError, match="end of period"):
        check_grid(shifted)


def test_check_grid_rejects_naive_timestamps(grid):
    with pytest.raises(ValueError, match="tz-aware"):
        check_grid(grid.assign(timestamp=grid["timestamp"].dt.tz_localize(None)))


def test_real_table(real_market):
    m = real_market
    assert len(m) == 31824
    assert m["date"].min() == date(2025, 1, 1) and m["date"].max() == date(2026, 10, 25)
    assert (m["interval_end"] - m["interval_start"] == pd.Timedelta(minutes=30)).all()
    first = m.iloc[0]
    assert first["period"] == 1 and first["interval_start"].strftime("%H:%M") == "00:00"
    assert m["split"].notna().all()
