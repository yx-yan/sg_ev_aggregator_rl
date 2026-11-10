"""End-to-end smoke run: toy scenario -> policies -> settlement.

    python -m sg_ev_rl.demo --config configs/demo.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from sg_ev_rl.scenarios.toy import toy_prices, toy_sessions
from sg_ev_rl.scheduling.rules import CheapestPeriods, Uncontrolled
from sg_ev_rl.settlement.report import settle
from sg_ev_rl.sim.simulator import FleetSimulator

POLICIES = {p.name: p for p in (Uncontrolled, CheapestPeriods)}


def run(cfg: dict) -> pd.DataFrame:
    n_periods = cfg["n_periods"]
    sessions = toy_sessions(cfg["n_vehicles"], n_periods, cfg["seed"])
    prices = toy_prices(n_periods, cfg["seed"])
    sim = FleetSimulator(sessions, n_periods)
    rows = {name: settle(sim.run(POLICIES[name](), prices)) for name in cfg["policies"]}
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    cfg = yaml.safe_load(parser.parse_args().config.read_text())
    print(run(cfg).round(2).to_string())


if __name__ == "__main__":
    main()
