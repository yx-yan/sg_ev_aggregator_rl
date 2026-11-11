"""Episodes: generated sessions on the half-hourly market grid, shared by every controller.

An episode covers `n_days` from 00:00 SGT of its start date (T = 48 * n_days steps). Each
session gets an availability fraction per step (the share of the half-hour it is plugged
in), so a 40-minute DC session is not rounded to whole periods.

Terminal handling (the same for every controller): a session still plugged in at the end
of the episode is cut there, and its energy need is scaled by the share of its plug-in
time that falls inside the episode. Sessions that arrive before the start are not in the
episode (the generator drops them).

Session fields follow the common sessions interface (reports/b): segment, battery_kwh,
soc_arrival, soc_target, p_charge_max_kw, p_discharge_max_kw, charger_type, v2g_capable,
feasible_at_arrival. Energy is battery side; `need_kwh` is what the episode must deliver.
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from sg_ev_rl.market.loader import TZ
from sg_ev_rl.market.splits import SPLITS

from . import public_charging as pc

DT_H = 0.5
STEPS_PER_DAY = 48
SEGMENT = {"AC": "destination", "DC": "public_fast"}


@dataclass
class Episode:
    scenario_id: str
    start: date
    n_steps: int
    sessions: pd.DataFrame  # one row per session, interface fields + need_kwh
    avail: np.ndarray  # (n_sessions, n_steps) plugged-in share of each half-hour, 0..1
    usep: np.ndarray  # S$/MWh per step
    regulation_price: np.ndarray
    demand_gross_mw: np.ndarray
    usep_pct_7d: np.ndarray  # trailing 7-day percentile of USEP (no look-ahead)
    timestamps: pd.DatetimeIndex  # interval start of each step, SGT
    n_ports: int
    fleet_kw: float  # sum of port ratings: scale for actions and observations
    contingency_reserve_price: np.ndarray | None = None  # CONRESA, S$/MWh (load facilities are group A)

    @property
    def p_max(self) -> np.ndarray:
        return self.sessions["p_charge_max_kw"].to_numpy(float)

    @property
    def need_kwh(self) -> np.ndarray:
        return self.sessions["need_kwh"].to_numpy(float)

    @property
    def last_step(self) -> np.ndarray:
        """Index of the last step with any availability, per session."""
        return self.n_steps - 1 - np.argmax(self.avail[:, ::-1] > 0, axis=1)


def seed_for(scenario: str, split: str, index: int) -> int:
    """Seed derived from (scenario, split, episode index), stable across runs."""
    words = [int.from_bytes(s.encode()[:8].ljust(8, b"\0"), "little") for s in (scenario, split)]
    return int(np.random.SeedSequence([*words, index]).generate_state(1)[0])


def portfolio_of_size(ports: pd.DataFrame, n_ports: int) -> dict[tuple[str, str], int]:
    """Split `n_ports` over (zone, AC/DC) in proportion to the real public ports (largest remainder)."""
    n = ports[ports["zone"].isin(pc.ZONES)].groupby(["zone", "current"]).size()
    raw = n / n.sum() * n_ports
    out = np.floor(raw).astype(int)
    out[(raw - out).sort_values(ascending=False).index[: n_ports - out.sum()]] += 1
    return {k: int(v) for k, v in out.items() if v > 0}


def to_interface(df: pd.DataFrame, rng: np.random.Generator, v2g_share: float = 0.0) -> pd.DataFrame:
    """Generator output -> common sessions interface. V2G only for AC sessions, Bernoulli(v2g_share)."""
    out = pd.DataFrame({
        "session_id": df["session_id"],
        "port_id": df["port_id"],
        "arrival_ts": df["arrival_ts"].dt.tz_localize(TZ),
        "departure_ts": df["departure_ts"].dt.tz_localize(TZ),
        "segment": df["current"].map(SEGMENT),
        "zone": df["zone"].str.lower(),
        "battery_kwh": df["battery_kwh"],
        "soc_arrival": df["arrival_soc"],
        "soc_target": df["soc_target"],
        "p_charge_max_kw": df["max_kw"],
        "charger_type": df["current"],
        "energy_kwh": df["energy_kwh"],
        "feasible_at_arrival": df["target_feasible"],
    })
    out["v2g_capable"] = (out["charger_type"] == "AC") & (rng.random(len(out)) < v2g_share)
    out["p_discharge_max_kw"] = np.where(out["v2g_capable"], out["p_charge_max_kw"], 0.0)
    return out


def availability(sessions: pd.DataFrame, t0: pd.Timestamp, n_steps: int) -> np.ndarray:
    """Plugged-in share of each half-hour step, per session."""
    step = pd.Timedelta(hours=DT_H)
    a = ((sessions["arrival_ts"] - t0) / step).to_numpy(float)
    d = np.minimum(((sessions["departure_ts"] - t0) / step).to_numpy(float), n_steps)
    edges = np.arange(n_steps)
    return np.clip(np.minimum(d[:, None], edges + 1) - np.maximum(a[:, None], edges), 0, 1)


def build_episode(profile: pc.PublicChargingProfile, market: pd.DataFrame, start: date, n_days: int,
                  n_ports: int, ports: pd.DataFrame, seed: int, scenario: str = "public",
                  vehicle: pc.VehicleAssumptions = pc.VehicleAssumptions(), v2g_share: float = 0.0,
                  holidays=frozenset()) -> Episode:
    rng = np.random.default_rng(seed)
    portfolio = portfolio_of_size(ports, n_ports)
    raw = pc.generate(profile, portfolio, start, n_days, seed=int(rng.integers(2**63)), vehicle=vehicle,
                      holidays=holidays)
    s = to_interface(raw, rng, v2g_share)
    return episode_from_sessions(s, market, start, n_days, n_ports=sum(portfolio.values()),
                                 fleet_kw=_fleet_kw(profile, portfolio), scenario_id=f"{scenario}-{seed}")


def episode_from_sessions(s: pd.DataFrame, market: pd.DataFrame, start: date, n_days: int, n_ports: int,
                          fleet_kw: float, scenario_id: str) -> Episode:
    t0 = pd.Timestamp(start).tz_localize(TZ)
    n_steps = n_days * STEPS_PER_DAY
    s = s[(s["arrival_ts"] >= t0) & (s["arrival_ts"] < t0 + pd.Timedelta(hours=DT_H * n_steps))]
    s = s.reset_index(drop=True)
    avail = availability(s, t0, n_steps)
    plugged = (s["departure_ts"] - s["arrival_ts"]) / pd.Timedelta(hours=DT_H)
    share = np.clip(avail.sum(axis=1) / plugged.to_numpy(float), 0, 1)
    s["need_kwh"] = s["energy_kwh"] * share
    s["truncated"] = share < 1 - 1e-9

    m = market.set_index("interval_start")
    steps = pd.date_range(t0, periods=n_steps, freq="30min")
    if not steps.isin(m.index).all():
        raise ValueError(f"market data does not cover {start} + {n_days} days")
    if "usep_price_pct_336" not in m.columns:
        raise ValueError("market needs add_relative_price_features() first")
    m = m.loc[steps]
    return Episode(scenario_id, start, n_steps, s, avail, m["usep_price"].to_numpy(float),
                   m["regulation_price"].to_numpy(float), m["demand_gross_mw"].to_numpy(float),
                   m["usep_price_pct_336"].to_numpy(float), steps, n_ports, fleet_kw,
                   m["reserve_price_CONRESA"].to_numpy(float))


def _fleet_kw(profile: pc.PublicChargingProfile, portfolio) -> float:
    """Expected sum of port ratings for the portfolio (actions are scaled by it)."""
    return float(sum(n * float(np.dot(*profile.port_kw[k])) for k, n in portfolio.items()))


def episode_starts(split: str, n_days: int, every_days: int | None = None) -> list[date]:
    """Episode start dates inside a split, back to back unless `every_days` is given."""
    lo, hi = SPLITS[split]
    last = pd.Timestamp(hi) - pd.Timedelta(days=n_days - 1)
    return [d.date() for d in pd.date_range(lo, last, freq=f"{every_days or n_days}D")]
