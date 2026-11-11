"""Copper-plate, charge-only fleet environment on the half-hourly market grid.

FleetCore holds the physics and settlement; every controller goes through it:
rule baselines and LP replay call `step_energy` / `step_schedule` directly, the RL agent
uses the Gymnasium wrapper FleetEnv. One step = one 30-minute settlement period.

Physics per EV: battery need e_i decreases by eta * x_i, where x_i is grid energy drawn in
the step, 0 <= x_i <= p_max_i * avail_i,t * dt. Unmet energy is counted when an EV leaves
(its last step with availability). Energy cost = USEP * sum_i x_i.

Interruptible load (IL, contingency reserve; see reports/b/2026-11-11_sg_service_eligibility.md):
after dispatch, each EV can offer up to

    q_max_i = min(x_i / dt, (cap_after_i - e_after_i / eta) / T_act)

kW, i.e. no more than it is drawing, and only what it could make up before departure after an
activation of T_act hours (the same bound as in scheduling/service_lp.py). The fleet offers a share
of sum q_max_i and is paid CONRESA * offer * dt. Activations are rare and not in the data; they are
drawn with an assumed probability per step (0 by default) and cut each EV's charging by its offer.
"""
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from sg_ev_rl.dispatch.llf import Envelope, dispatch, envelope
from sg_ev_rl.scenarios.episodes import DT_H, Episode

LAXITY_BINS_H = np.array([0, 1, 2, 4, 8, np.inf])
TOL = 1e-6
OBS_CLIP = 10.0  # observations are scaled to O(1); USEP spikes are clipped here


@dataclass(frozen=True)
class EnvConfig:
    eta_charge: float = 0.92  # must not exceed the generator's eta_charge, or targets may be infeasible
    unmet_penalty_sgd_per_kwh: float = 1.0  # ASSUMPTION; reward only, not part of energy cost
    reward_scale_sgd: float = 10.0  # rewards are divided by this to stay near unit range
    safety_layer: bool = True  # project setpoints onto [E_min, E_max] (method M2); False = penalty only (M1)
    dispatch_priority: str = "llf"  # or "ratio", see dispatch.llf
    il_activation_h: float = 0.5  # IL activation an EV must be able to make up (EMA plans 30 min)
    il_activation_prob: float = 0.0  # ASSUMPTION: chance that an IL offer is activated in a step
    action_mode: str = "absolute"  # "absolute": P = u * fleet_kw; "envelope": u spans [E_min, E_max] of the step


@dataclass
class StepLog:
    energy_kwh: list = field(default_factory=list)  # grid energy per step
    cost_sgd: list = field(default_factory=list)
    unmet_kwh: list = field(default_factory=list)  # counted at departure
    e_min: list = field(default_factory=list)
    e_max: list = field(default_factory=list)
    setpoint_kwh: list = field(default_factory=list)  # requested before projection
    il_kw: list = field(default_factory=list)  # IL offered (scheduled contingency reserve)
    il_revenue_sgd: list = field(default_factory=list)
    il_activated: list = field(default_factory=list)


class FleetCore:
    def __init__(self, episode: Episode, cfg: EnvConfig = EnvConfig(), seed: int | None = None):
        self.ep, self.cfg = episode, cfg
        self.rng = np.random.default_rng(seed)
        self.p_max = episode.p_max
        self.avail = episode.avail
        # availability summed over the steps after t, per EV
        self.avail_after = np.cumsum(self.avail[:, ::-1], axis=1)[:, ::-1] - self.avail
        self.last = episode.last_step
        self.remaining = episode.need_kwh.copy()
        self.t = 0
        self.log = StepLog()
        self.schedule = np.zeros_like(self.avail)  # grid energy per EV and step

    @property
    def done(self) -> bool:
        return self.t >= self.ep.n_steps

    def connected(self) -> np.ndarray:
        return self.avail[:, self.t] > 0

    def envelope(self) -> Envelope:
        return envelope(self.remaining, self.p_max, self.avail[:, self.t], self.avail_after[:, self.t],
                        self.cfg.eta_charge, DT_H, self.cfg.dispatch_priority)

    def step_energy(self, energy_kwh: float, il_share: float = 0.0) -> dict:
        """Apply a fleet energy setpoint for this step, split in dispatch-priority order, and offer
        `il_share` of the IL the dispatched load can back."""
        env = self.envelope()
        if not self.cfg.safety_layer:
            env = Envelope(np.zeros_like(env.must), env.upper, env.laxity_h, env.priority)
        x = dispatch(env, energy_kwh)
        q = float(np.clip(il_share, 0, 1)) * self.il_max(x)
        return self._apply(x, env, energy_kwh, q)

    def step_schedule(self, x: np.ndarray, il_kw: np.ndarray | None = None) -> dict:
        """Apply a per-EV grid-energy schedule (and IL offer) for this step (LP replay); limits are checked."""
        env = self.envelope()
        cap_now = self.p_max * self.avail[:, self.t] * DT_H
        if (x < -TOL).any() or (x > cap_now + TOL).any() or (x > env.upper + TOL).any():
            raise ValueError(f"schedule violates EV limits at step {self.t}")
        x = np.clip(x, 0, env.upper)
        q = np.zeros_like(x) if il_kw is None else np.asarray(il_kw, float)
        if (q < -TOL).any() or (q > self.il_max(x) + 1e-5).any():
            raise ValueError(f"IL offer exceeds what the load can back at step {self.t}")
        return self._apply(x, env, float(x.sum()), np.clip(q, 0, None))

    def il_max(self, x: np.ndarray) -> np.ndarray:
        """Largest IL offer per EV (kW) given this step's dispatch x."""
        cap_after = self.p_max * self.avail_after[:, self.t] * DT_H
        after = np.clip(self.remaining - self.cfg.eta_charge * x, 0, None)
        makeup = (cap_after - after / self.cfg.eta_charge) / self.cfg.il_activation_h
        return np.clip(np.minimum(x / DT_H, makeup), 0, None)

    def _apply(self, x: np.ndarray, env: Envelope, setpoint: float, q: np.ndarray) -> dict:
        t = self.t
        activated = bool(q.sum() > 0 and self.rng.random() < self.cfg.il_activation_prob)
        if activated:
            x = x - q * min(self.cfg.il_activation_h, DT_H)
        il_rev = float(q.sum() * self.ep.contingency_reserve_price[t] / 1000 * DT_H)
        self.log.il_kw.append(float(q.sum()))
        self.log.il_revenue_sgd.append(il_rev)
        self.log.il_activated.append(activated)
        self.remaining = self.remaining - self.cfg.eta_charge * x
        self.schedule[:, t] = x
        leaving = self.last == t
        unmet = float(np.clip(self.remaining[leaving], 0, None).sum())
        e = float(x.sum())
        cost = self.ep.usep[t] * e / 1000
        for k, v in (("energy_kwh", e), ("cost_sgd", cost), ("unmet_kwh", unmet), ("e_min", env.e_min),
                     ("e_max", env.e_max), ("setpoint_kwh", setpoint)):
            getattr(self.log, k).append(v)
        self.t += 1
        return {"energy_kwh": e, "cost_sgd": cost, "unmet_kwh": unmet, "e_min": env.e_min, "e_max": env.e_max,
                "il_kw": float(q.sum()), "il_revenue_sgd": il_rev, "il_activated": activated}


class FleetEnv(gym.Env):
    """Aggregate action a = (u, b) in [-1, 1] x [0, 1]; b = share of the backable interruptible load offered.

    action_mode "absolute": fleet setpoint P = u * fleet_kw, projected onto the envelope (charge only, so
    u < 0 means "as little as allowed"). "envelope": E = E_min + (u + 1) / 2 * (E_max - E_min), so the whole
    action range is feasible; used for RL because the feasible band is a few percent of fleet_kw.
    """

    metadata = {"render_modes": []}

    def __init__(self, episodes, cfg: EnvConfig = EnvConfig()):
        """`episodes`: a list of Episode, or a callable rng -> Episode."""
        self.episodes, self.cfg = episodes, cfg
        self.action_space = gym.spaces.Box(np.array([-1.0, 0.0], np.float32), np.array([1.0, 1.0], np.float32))
        self.observation_space = gym.spaces.Box(-OBS_CLIP, OBS_CLIP, (OBS_DIM,), np.float32)
        self.core: FleetCore | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "episode" in options:
            ep = options["episode"]
        elif callable(self.episodes):
            ep = self.episodes(self.np_random)
        else:
            ep = self.episodes[int(self.np_random.integers(len(self.episodes)))]
        self.core = FleetCore(ep, self.cfg, seed=int(self.np_random.integers(2**63)))
        return observe(self.core), {}

    def step(self, action):
        core = self.core
        u = float(np.clip(action[0], -1, 1))
        if self.cfg.action_mode == "envelope":
            env = core.envelope()
            energy = env.e_min + (u + 1) / 2 * (env.e_max - env.e_min)
        elif self.cfg.action_mode == "absolute":
            energy = max(u, 0.0) * core.ep.fleet_kw * DT_H
        else:
            raise ValueError(f"unknown action_mode {self.cfg.action_mode!r}")
        info = core.step_energy(energy, float(action[1]))
        penalty = self.cfg.unmet_penalty_sgd_per_kwh * info["unmet_kwh"]
        reward = -(info["cost_sgd"] - info["il_revenue_sgd"] + penalty) / self.cfg.reward_scale_sgd
        terminated = core.done
        obs = np.zeros(OBS_DIM, np.float32) if terminated else observe(core)
        return obs, float(reward), terminated, False, info


OBS_DIM = 6 + 8 + 4 + (len(LAXITY_BINS_H) - 1)


def observe(core: FleetCore) -> np.ndarray:
    ep, t = core.ep, core.t
    ts = ep.timestamps[t]
    hh = (ts.hour * 2 + ts.minute // 30) / 48 * 2 * np.pi
    dow = ts.day_of_week / 7 * 2 * np.pi
    time = [np.sin(hh), np.cos(hh), np.sin(dow), np.cos(dow), float(ts.day_of_week >= 5), 1 - t / ep.n_steps]
    lags = [ep.usep[max(t - k, 0)] / 1000 for k in range(5)]
    pct = ep.usep_pct_7d[t]
    market = [*lags, 0.5 if np.isnan(pct) else pct, ep.regulation_price[t] / 100,
              ep.contingency_reserve_price[t] / 100]
    con = core.connected()
    env = core.envelope()
    scale = ep.fleet_kw * DT_H
    fleet = [con.sum() / ep.n_ports, np.clip(core.remaining[con], 0, None).sum() / scale,
             env.e_min / scale, env.e_max / scale]
    lax = env.laxity_h[con & np.isfinite(env.laxity_h)]
    hist = np.histogram(lax, LAXITY_BINS_H)[0] / max(con.sum(), 1)
    return np.clip(np.asarray([*time, *market, *fleet, *hist], np.float32), -OBS_CLIP, OBS_CLIP)
