"""Train PPO (Stable-Baselines3) on the fleet environment and keep the best model on validation.

Model selection uses the fixed validation episodes only (net cost = energy cost - IL revenue);
no test-split data is touched. Models and logs go to the runs directory, not into Git.

Run:  PYTHONPATH=src python -m sg_ev_rl.rl.train_ppo --config configs/rl/ppo_il.yaml --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sg_ev_rl.market.paths import runs_dir
from sg_ev_rl.sim.fleet_env import FleetEnv

from .common import fixed_episodes, load_context, rl_env_config, run_policy, train_sampler


class ValidationCallback(BaseCallback):
    """Every `every` steps: deterministic rollout on the validation episodes; keep the best model."""

    def __init__(self, eps, cfg, every: int, out: Path):
        super().__init__()
        self.eps, self.cfg, self.every, self.out = eps, cfg, every, out
        self.next_eval, self.best, self.log = 0, np.inf, []

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        self.evaluate()

    def evaluate(self):
        def predict(obs):
            return self.model.predict(obs, deterministic=True)[0]

        rows = [run_policy(predict, ep, self.cfg) for ep in self.eps]
        net = float(np.mean([r["net_cost_per_port_day_sgd"] for r in rows]))
        self.log.append({"timesteps": self.num_timesteps, "val_net_per_port_day": net,
                         "val_il_mw": float(np.mean([r["il_mw_mean"] for r in rows])),
                         "val_unmet_kwh": float(np.sum([r["unmet_kwh"] for r in rows])),
                         "wall_s": time.perf_counter()})
        if net < self.best:
            self.best = net
            self.model.save(self.out / "best_model")
        pd.DataFrame(self.log).to_csv(self.out / "validation_log.csv", index=False)
        if self.verbose:
            print(f"{self.num_timesteps:>9d}  val net {net:.4f} S$/port/day  best {self.best:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/rl/ppo_il.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    steps = args.timesteps or cfg["total_timesteps"]
    out = runs_dir().parent / "d" / f"{args.config.stem}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "seed": args.seed, "total_timesteps": steps}, indent=2))

    ctx = load_context()
    env_cfg = rl_env_config(**cfg["env"])
    sampler = train_sampler(ctx, cfg["n_ports"], cfg["episode_days"], price_scale=cfg.get("price_scale"))
    venv = make_vec_env(lambda: FleetEnv(sampler, env_cfg), n_envs=cfg["n_envs"], seed=args.seed)
    if cfg.get("norm_reward"):
        # rewards only: observations stay as built, so evaluation needs no saved statistics
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg["ppo"]["gamma"])
    val_eps = fixed_episodes(ctx, "validation", cfg["n_ports"], cfg["episode_days"])

    p = dict(cfg["ppo"])
    arch = p.pop("net_arch")
    model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0,
                policy_kwargs={"net_arch": arch}, **p)
    cb = ValidationCallback(val_eps, env_cfg, cfg["eval_every_steps"], out)
    cb.verbose = 1
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=cb)
    model.save(out / "final_model")
    print(json.dumps({"out": str(out), "best_val_net_per_port_day": cb.best,
                      "train_minutes": round((time.perf_counter() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
