"""Shared setup for RL training and evaluation: data context, episode samplers, baselines.

Training episodes are drawn from the train split only, with a fresh session draw per episode;
validation episodes are a fixed list (seeded by split and index) used for model selection and
for the baselines, so every controller is scored on the same episodes.
"""
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from sg_ev_rl.market.features import add_regime_features, add_relative_price_features
from sg_ev_rl.market.loader import load_market
from sg_ev_rl.market.splits import SPLITS
from sg_ev_rl.scenarios import public_charging as pc
from sg_ev_rl.scenarios.episodes import Episode, build_episode, episode_starts, scale_prices, seed_for
from sg_ev_rl.scheduling import fleet_rules as R
from sg_ev_rl.scheduling.service_lp import ServiceConfig, solve_service_lp
from sg_ev_rl.settlement.report import settle_episode
from sg_ev_rl.sim.fleet_env import EnvConfig, FleetEnv


@dataclass
class Context:
    market: pd.DataFrame
    profile: pc.PublicChargingProfile
    ports: pd.DataFrame


def load_context() -> Context:
    market = add_relative_price_features(load_market())
    for col in ("usep_price", "reserve_price_CONRESA"):
        market = add_regime_features(market, col)
    sess, ports, snaps = pc.load_calibration_inputs()
    return Context(market, pc.calibrate(sess, ports, snaps), ports)


def train_sampler(ctx: Context, n_ports: int, n_days: int, split: str = "train",
                  price_scale: dict | None = None):
    """Callable rng -> Episode with a random start day inside `split` and fresh sessions.

    `price_scale` = {"usep": [lo, hi], "conres": [lo, hi]}: each episode's price level is multiplied by a
    log-uniform draw from that range, so the policy sees price regimes other than the training year's.
    """
    lo, hi = SPLITS[split]
    n_starts = (hi - lo).days - n_days + 2
    ranges = {k: np.log(v) for k, v in (price_scale or {}).items()}

    def sample(rng: np.random.Generator) -> Episode:
        start = lo + timedelta(days=int(rng.integers(n_starts)))
        ep = build_episode(ctx.profile, ctx.market, start, n_days, n_ports, ctx.ports,
                           seed=int(rng.integers(2**63)), scenario=f"rl-{split}")
        if not ranges:
            return ep
        return scale_prices(ep, **{k: float(np.exp(rng.uniform(*r))) for k, r in ranges.items()})

    return sample


def fixed_episodes(ctx: Context, split: str, n_ports: int, n_days: int, scenario: str = "rl_eval",
                   limit: int | None = None) -> list[Episode]:
    return [build_episode(ctx.profile, ctx.market, start, n_days, n_ports, ctx.ports,
                          seed=seed_for(scenario, split, k), scenario=scenario)
            for k, start in enumerate(episode_starts(split, n_days)[:limit])]


def run_policy(predict, ep: Episode, cfg: EnvConfig) -> dict:
    """Roll out `predict(obs) -> action` on one episode and settle it."""
    env = FleetEnv([ep], cfg)
    obs, _ = env.reset(seed=0, options={"episode": ep})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(predict(obs))
    return settle_episode(env.core)


def baselines(eps: list[Episode], p_grid=(0.7, 0.9, 1.0)) -> pd.DataFrame:
    """B1 + IL, B2 + IL over a p grid, LP fleet profile via LLF + IL, LP + IL per EV (bound)."""
    rows = []
    for k, ep in enumerate(eps):
        lp = solve_service_lp(ep, ServiceConfig(interruptible_load=True))
        cores = {"B1_uncontrolled": R.run(ep, R.Uncontrolled()),
                 "B1_il": R.run(ep, R.Uncontrolled(), il_share=1.0),
                 **{f"B2_p{p:g}_il": R.run(ep, R.PriceThreshold(p), il_share=1.0) for p in p_grid},
                 "LP_il_aggregate_llf": R.run(ep, R.ReplayAggregate(lp.charge.sum(axis=0)), il_share=1.0),
                 "LP_il_per_ev": R.run_schedule(ep, lp.charge, il_kw=lp.il_kw)}
        rows += [{"episode": k, "start": ep.start, "policy": name, **settle_episode(core)}
                 for name, core in cores.items()]
    return pd.DataFrame(rows)


def rl_env_config(**kw) -> EnvConfig:
    return replace(EnvConfig(action_mode="envelope"), **kw)


def start_dates(eps: list[Episode]) -> list[date]:
    return [ep.start for ep in eps]
