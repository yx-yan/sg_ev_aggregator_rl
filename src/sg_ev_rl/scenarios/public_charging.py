"""EV sessions at public chargers, calibrated to Singapore charger occupancy (SGT).

Scope (decision 2026-11-11): the aggregator controls a portfolio of public charging
ports; home and workplace charging are not modelled.

Calibration data are the pseudo sessions built in the research workspace
(`datasets/interim/ev_charging/`, from 5-minute port status snapshots). A pseudo
session is one run of "occupied" on one port, so it measures plug-in time, not
energy. What comes from the data and what is assumed:

    from data   arrival hazard by zone x AC/DC x weekday/weekend x half-hour,
                plug-in duration by AC/DC x day type x arrival hour,
                port nameplate power by zone x AC/DC
    assumed     battery size, arrival SoC, target SoC, vehicle AC/DC acceptance
                and DC taper (VehicleAssumptions)

Arrival model: each port is a single server. While it is free, a vehicle arrives in
a 5-minute step with probability h = lambda / (1 - occupancy), where lambda is the
observed arrival rate per port and occupancy the observed occupied share for that
group and half-hour. Dividing by the free share makes the simulated arrival rate and
occupancy both match the data, instead of losing the arrivals that would land on a
busy port.

By default the profile is calibrated on 2026-10-26..11-09 only, so that the
real-scenario validation period (2026-09-04..10-25, see market/splits.py) stays
out of sample.
"""
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from sg_ev_rl.assets.fleet import DT_H, Session
from sg_ev_rl.market.paths import ev_charging_dir

STEP_MIN = 5
STEPS_PER_DAY = 24 * 60 // STEP_MIN
SLOTS = 48  # half-hours, aligned with market periods
CURRENTS = ("AC", "DC")
DAY_TYPES = ("weekday", "weekend")
ZONES = ("CENTRAL", "EAST", "NORTH", "NORTH_EAST", "WEST")  # Jurong Island has 2 public ports

CALIBRATION_DATES = (date(2026, 10, 26), date(2026, 11, 9))
MIN_DURATION_MIN = 10  # 5-minute runs are mostly single-poll flicker
MIN_FREE_SHARE = 0.05
MIN_DURATION_SAMPLES = 50


@dataclass(frozen=True)
class VehicleAssumptions:
    """Vehicle-side assumptions; none of these are in the occupancy data."""

    battery_kwh: tuple[float, ...] = (50.0, 60.0, 75.0)
    battery_p: tuple[float, ...] = (0.3, 0.4, 0.3)
    ac_accept_kw: tuple[float, ...] = (7.0, 11.0)  # on-board charger
    ac_accept_p: tuple[float, ...] = (0.4, 0.6)
    dc_accept_kw: float = 100.0
    dc_taper: float = 0.75  # mean DC power / limit over a 20-80 % charge
    arrival_soc_ac: tuple[float, float] = (0.2, 0.7)  # uniform range
    arrival_soc_dc: tuple[float, float] = (0.1, 0.5)
    target_soc_ac: float = 0.9
    target_soc_dc: float = 0.8
    eta_charge: float = 0.92  # grid -> battery


@dataclass
class PublicChargingProfile:
    hazard: np.ndarray  # (zone, current, day_type, slot) arrival prob per 5-min step while free
    durations: dict[tuple, np.ndarray]  # (current, day_type, hour) or (current, day_type) -> minutes
    port_kw: dict[tuple, tuple[np.ndarray, np.ndarray]]  # (zone, current) -> (kW values, probabilities)
    occupancy: np.ndarray  # (zone, current, day_type, slot) observed occupied share, for checks
    calibration_dates: tuple[date, date] = CALIBRATION_DATES
    n_sessions: int = 0
    meta: dict = field(default_factory=dict)

    def sample_duration(self, rng, current: str, day_type: str, hour: int, n: int) -> np.ndarray:
        d = self.durations.get((current, day_type, hour))
        if d is None:
            d = self.durations[(current, day_type)]
        # pseudo-session ends are known to +-5 min; spread the 5-minute lattice
        return np.maximum(rng.choice(d, n) + rng.uniform(-STEP_MIN / 2, STEP_MIN / 2, n), 1.0)


def day_type_of(days: pd.DatetimeIndex, holidays=frozenset()) -> np.ndarray:
    weekend = (days.day_of_week >= 5) | days.normalize().isin(pd.to_datetime(list(holidays)))
    return np.where(weekend, 1, 0)


def _full_days(snaps: pd.DataFrame, start: date, end: date) -> pd.DatetimeIndex:
    per_day = snaps.groupby(snaps["ts"].dt.normalize()).size()
    per_day = per_day[(per_day.index >= pd.Timestamp(start)) & (per_day.index <= pd.Timestamp(end))]
    return pd.DatetimeIndex(per_day[per_day >= 0.95 * STEPS_PER_DAY].index)


def calibrate(sessions: pd.DataFrame, ports: pd.DataFrame, snaps: pd.DataFrame,
              dates: tuple[date, date] = CALIBRATION_DATES,
              min_duration_min: float = MIN_DURATION_MIN) -> PublicChargingProfile:
    """Build the profile from pseudo sessions, the port register and the snapshot index."""
    days = _full_days(snaps, *dates)
    if len(days) == 0:
        raise ValueError(f"no complete snapshot days in {dates}")
    dtype_of_day = dict(zip(days, day_type_of(days), strict=True))
    n_days = np.bincount(list(dtype_of_day.values()), minlength=2)
    if (n_days == 0).any():
        raise ValueError("calibration window needs both weekdays and weekend days")

    lo, hi = days.min(), days.max() + pd.Timedelta(days=1)
    alive = ports[(ports["first_seen"] < hi) & (ports["last_seen"] >= lo) & ports["zone"].isin(ZONES)]
    n_ports = alive.groupby(["zone", "current"]).size()

    s = sessions[sessions["zone"].isin(ZONES) & (sessions["duration_min"] >= min_duration_min)]
    s = s[(s["end"] > lo) & (s["start"] < hi)]
    zi = {z: i for i, z in enumerate(ZONES)}
    ci = {c: i for i, c in enumerate(CURRENTS)}
    shape = (len(ZONES), len(CURRENTS), 2, SLOTS)

    # observed arrivals: clean sessions starting on a complete day
    a = s[s["is_clean"]]
    a = a[a["start"].dt.normalize().isin(days)]
    arrivals = np.zeros(shape)
    slot = a["start"].dt.hour * 2 + a["start"].dt.minute // 30
    np.add.at(arrivals, (a["zone"].map(zi).to_numpy(), a["current"].map(ci).to_numpy(),
                         a["start"].dt.normalize().map(dtype_of_day).to_numpy(int), slot.to_numpy()), 1)

    # observed occupancy on the 5-minute grid of the complete days (all sessions, clean or not)
    grid = (days.values[:, None] + np.arange(STEPS_PER_DAY) * np.timedelta64(STEP_MIN, "m")).ravel()
    g_dtype = np.repeat(day_type_of(days), STEPS_PER_DAY)
    g_slot = np.tile(np.arange(STEPS_PER_DAY) // (30 // STEP_MIN), len(days))
    occupancy = np.zeros(shape)
    for (z, c), g in s.groupby(["zone", "current"]):
        delta = np.zeros(len(grid) + 1)
        np.add.at(delta, np.searchsorted(grid, g["start"].to_numpy()), 1)
        np.add.at(delta, np.searchsorted(grid, g["end"].to_numpy()), -1)
        busy = np.cumsum(delta)[:-1]
        for d in (0, 1):
            m = g_dtype == d
            occupancy[zi[z], ci[c], d] = np.bincount(g_slot[m], busy[m], SLOTS) / np.bincount(g_slot[m], None, SLOTS)
    port_counts = np.zeros((len(ZONES), len(CURRENTS)))
    for (z, c), n in n_ports.items():
        port_counts[zi[z], ci[c]] = n
    with np.errstate(divide="ignore", invalid="ignore"):
        occupancy = np.nan_to_num(occupancy / port_counts[:, :, None, None])
        rate = arrivals / (port_counts[:, :, None, None] * n_days[None, None, :, None] * (30 // STEP_MIN))
        hazard = np.nan_to_num(rate / np.maximum(1 - occupancy, MIN_FREE_SHARE))
    hazard = np.clip(hazard, 0, 1)

    durations = {}
    a_dt = a["start"].dt.normalize().map(dtype_of_day)
    for c in CURRENTS:
        for d, dname in enumerate(DAY_TYPES):
            sel = a[(a["current"] == c) & (a_dt == d)]
            if len(sel) == 0:
                raise ValueError(f"no calibration sessions for {c} {dname}")
            durations[(c, dname)] = sel["duration_min"].to_numpy()
            for h, gh in sel.groupby(sel["start"].dt.hour):
                if len(gh) >= MIN_DURATION_SAMPLES:
                    durations[(c, dname, int(h))] = gh["duration_min"].to_numpy()

    port_kw = {}
    for (z, c), g in alive.groupby(["zone", "current"]):
        vc = g["power_kw"].dropna().value_counts(normalize=True)
        port_kw[(z, c)] = (vc.index.to_numpy(float), vc.to_numpy(float))

    return PublicChargingProfile(hazard, durations, port_kw, occupancy, dates, len(a),
                                 {"days": len(days), "n_days": n_days.tolist(),
                                  "n_ports": {f"{z}/{c}": int(n) for (z, c), n in n_ports.items()}})


def load_calibration_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = ev_charging_dir()
    sessions = pd.read_parquet(d / "pseudo_sessions.parquet")
    ports = pd.read_parquet(d / "charger_ports.parquet")
    snaps = pd.read_parquet(d / "snapshot_index.parquet")
    return sessions, ports, snaps


def load_profile(dates: tuple[date, date] = CALIBRATION_DATES) -> PublicChargingProfile:
    return calibrate(*load_calibration_inputs(), dates=dates)


def portfolio_from_share(ports: pd.DataFrame, share: float) -> dict[tuple[str, str], int]:
    """Ports per (zone, current) for an aggregator holding `share` of every group."""
    n = ports[ports["zone"].isin(ZONES)].groupby(["zone", "current"]).size()
    return {k: int(round(v * share)) for k, v in n.items() if round(v * share) > 0}


def assign_energy(df: pd.DataFrame, rng: np.random.Generator,
                  vehicle: VehicleAssumptions = VehicleAssumptions()) -> pd.DataFrame:
    """Add battery, vehicle power limit, arrival SoC and energy need to plug-in sessions.

    Needs columns current, port_kw, duration_min. Energy need (battery side) is what the
    driver would take in the observed plug-in time at full power, capped by the SoC
    headroom, so every session is feasible on its own. `target_feasible` records whether
    the desired target SoC was reachable; `soc_target` is the target actually requested.
    """
    df = df.copy()
    n = len(df)
    dc = (df["current"] == "DC").to_numpy()
    df["battery_kwh"] = rng.choice(vehicle.battery_kwh, n, p=vehicle.battery_p)
    accept = np.where(dc, vehicle.dc_accept_kw, rng.choice(vehicle.ac_accept_kw, n, p=vehicle.ac_accept_p))
    df["max_kw"] = np.minimum(df["port_kw"].to_numpy(), accept)
    lo = np.where(dc, vehicle.arrival_soc_dc[0], vehicle.arrival_soc_ac[0])
    hi = np.where(dc, vehicle.arrival_soc_dc[1], vehicle.arrival_soc_ac[1])
    df["arrival_soc"] = rng.uniform(lo, hi)
    target = np.where(dc, vehicle.target_soc_dc, vehicle.target_soc_ac)
    headroom = np.clip(target - df["arrival_soc"], 0, None) * df["battery_kwh"]
    deliverable = (vehicle.eta_charge * df["max_kw"] * np.where(dc, vehicle.dc_taper, 1.0)
                   * df["duration_min"] / 60)
    df["energy_kwh"] = np.minimum(headroom, deliverable)
    df["target_feasible"] = headroom <= deliverable
    df["soc_target"] = df["arrival_soc"] + df["energy_kwh"] / df["battery_kwh"]
    return df


def generate(profile: PublicChargingProfile, portfolio: dict[tuple[str, str], int], start: date,
             n_days: int, seed: int = 0, vehicle: VehicleAssumptions = VehicleAssumptions(),
             holidays=frozenset()) -> pd.DataFrame:
    """Simulate plug-in sessions on a portfolio of public ports for `n_days` from `start` (00:00 SGT)."""
    rng = np.random.default_rng(seed)
    zi = {z: i for i, z in enumerate(ZONES)}
    ci = {c: i for i, c in enumerate(CURRENTS)}
    rows = []
    for (z, c), n in sorted(portfolio.items()):
        if z not in zi or c not in ci:
            raise ValueError(f"unknown portfolio group {(z, c)}")
        kw_vals, kw_p = profile.port_kw[(z, c)]
        for kw in rng.choice(kw_vals, n, p=kw_p):
            rows.append((f"{z[:2]}{c}{len(rows):05d}", z, c, kw))
    ports = pd.DataFrame(rows, columns=["port_id", "zone", "current", "port_kw"])
    if ports.empty:
        raise ValueError("empty portfolio")
    gz, gc = ports["zone"].map(zi).to_numpy(), ports["current"].map(ci).to_numpy()

    # one warm-up day so that ports are not all free at the start; its arrivals are dropped
    t0 = pd.Timestamp(start)
    days = pd.date_range(t0 - pd.Timedelta(days=1), periods=n_days + 1, freq="D")
    dtypes = day_type_of(days, holidays)
    busy_until = np.full(len(ports), -np.inf)  # minutes from t0
    out = []
    for k in range((n_days + 1) * STEPS_PER_DAY):
        now = k * STEP_MIN - 24 * 60
        d, slot = dtypes[k // STEPS_PER_DAY], (k % STEPS_PER_DAY) // (30 // STEP_MIN)
        free = busy_until <= now
        arrive = free & (rng.random(len(ports)) < profile.hazard[gz, gc, d, slot])
        if not arrive.any():
            continue
        idx = np.flatnonzero(arrive)
        t_arr = now + rng.uniform(0, STEP_MIN, len(idx))
        dur = np.empty(len(idx))
        hour = (k % STEPS_PER_DAY) * STEP_MIN // 60
        for c in CURRENTS:
            m = gc[idx] == ci[c]
            if m.any():
                dur[m] = profile.sample_duration(rng, c, DAY_TYPES[d], hour, int(m.sum()))
        busy_until[idx] = t_arr + dur
        out.append(pd.DataFrame({"port_idx": idx, "arrival_min": t_arr, "duration_min": dur}))

    cols = ["session_id", "port_id", "zone", "current", "port_kw", "arrival_ts", "departure_ts", "duration_min"]
    if not out:
        return assign_energy(pd.DataFrame(columns=cols), rng, vehicle)
    s = pd.concat(out, ignore_index=True)
    s = s[s["arrival_min"] >= 0].join(ports, on="port_idx").drop(columns="port_idx")
    s["arrival_ts"] = t0 + pd.to_timedelta(s.pop("arrival_min"), unit="min")
    s["departure_ts"] = s["arrival_ts"] + pd.to_timedelta(s["duration_min"], unit="min")
    s = s.sort_values(["arrival_ts", "port_id"]).reset_index(drop=True)
    s.insert(0, "session_id", [f"s{i:06d}" for i in range(len(s))])
    return assign_energy(s[cols], rng, vehicle)


def real_sessions(sessions: pd.DataFrame, ports: pd.DataFrame, portfolio: dict[tuple[str, str], int],
                  start: date, n_days: int, seed: int = 0,
                  vehicle: VehicleAssumptions = VehicleAssumptions()) -> pd.DataFrame:
    """Observed pseudo sessions on a random port sample of the same portfolio size (real-scenario check)."""
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=n_days)
    chosen = []
    for (z, c), n in sorted(portfolio.items()):
        pool = ports.loc[(ports["zone"] == z) & (ports["current"] == c), "ev_cp_id"].to_numpy()
        chosen.extend(rng.choice(pool, min(n, len(pool)), replace=False))
    s = sessions[sessions["ev_cp_id"].isin(chosen) & (sessions["start"] >= t0) & (sessions["start"] < t1)
                 & sessions["is_clean"] & (sessions["duration_min"] >= MIN_DURATION_MIN)]
    s = s.rename(columns={"ev_cp_id": "port_id", "power_kw": "port_kw",
                          "start": "arrival_ts", "end": "departure_ts"})
    s = s.sort_values(["arrival_ts", "port_id"]).reset_index(drop=True)
    s.insert(0, "session_id", [f"r{i:06d}" for i in range(len(s))])
    cols = ["session_id", "port_id", "zone", "current", "port_kw", "arrival_ts", "departure_ts", "duration_min"]
    return assign_energy(s[cols], rng, vehicle)


def to_sessions(df: pd.DataFrame, start: date, n_periods: int) -> list[Session]:
    """Map to simulator sessions on the half-hour grid starting at `start` 00:00.

    A vehicle counts as present for every period it overlaps. Sessions that depart
    after the horizon are cut at the horizon with energy capped to what still fits.
    """
    t0 = pd.Timestamp(start)
    period = pd.Timedelta(hours=DT_H)
    out = []
    for r in df.itertuples(index=False):
        a = int((r.arrival_ts - t0) // period)
        dep = int(-((t0 - r.departure_ts) // period))  # ceil
        if a < 0 or a >= n_periods:
            continue
        dep = min(max(dep, a + 1), n_periods)
        plugged_h = (min(r.departure_ts, t0 + dep * period) - r.arrival_ts) / pd.Timedelta(hours=1)
        energy = min(r.energy_kwh, r.max_kw * plugged_h)
        out.append(Session(r.session_id, a, dep, max(float(energy), 0.0), float(r.max_kw)))
    return out
