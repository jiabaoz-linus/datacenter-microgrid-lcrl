"""Method configurations and batched training loops.

Queue-enabled variants use virtual-queue observations and queue-augmented
rewards. PCM correction is enabled separately for each method.
"""
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch

from .config import SystemParams, TrainParams
from .env import DCMGEnv, OBS_DIM, ACT_DIM
from .disturbance import make_pool, sample_equipment
from .agents.onpolicy import OnPolicyAgent
from .agents.offpolicy import OffPolicyAgent

METHODS: Dict[str, Dict] = {
    # LCRL configuration
    "LCRL":       dict(kind="on", algo="ppo", use_pcm=True, use_queue=True, rerun=True),
    # Reinforcement-learning baselines
    "PPO-Lag":    dict(kind="on", algo="ppo_lag", use_pcm=True, use_queue=False),
    "CPO":        dict(kind="on", algo="cpo", use_pcm=True, use_queue=False),
    "SAC":        dict(kind="off", algo="sac", use_pcm=True, use_queue=False),
    "TD3":        dict(kind="off", algo="td3", use_pcm=True, use_queue=False),
    "DDPG":       dict(kind="off", algo="ddpg", use_pcm=True, use_queue=False),
    "PPO":        dict(kind="on", algo="ppo", use_pcm=True, use_queue=False),
    "TRPO":       dict(kind="on", algo="trpo", use_pcm=True, use_queue=False),
    # Ablation variants; the +PCM alias resolves to PPO below.
    "Base":       dict(kind="on", algo="ppo", use_pcm=False, use_queue=False),
    "+Lyapunov":  dict(kind="on", algo="ppo", use_pcm=False, use_queue=True, rerun=True),
    # Model predictive control benchmark
    "MPC":        dict(kind="mpc", use_pcm=True, use_queue=False),
}
ALIASES = {"+PCM": "PPO"}
RL_BASELINES = ["PPO-Lag", "CPO", "SAC", "TD3", "DDPG", "PPO", "TRPO"]


def resolve(name: str) -> str:
    return ALIASES.get(name, name)


def make_env(sp: SystemParams, method: str, pcc_mode="adaptive", eta_Z=None, g_bar=None, solver="fast"):
    m = METHODS[resolve(method)]
    return DCMGEnv(sp, use_pcm=m["use_pcm"], use_queue=m["use_queue"], pcc_mode=pcc_mode,
                   solver=solver, eta_Z=eta_Z, g_bar=g_bar)


def make_agent(sp: SystemParams, tp: TrainParams, method: str, g_bar=None):
    m = METHODS[resolve(method)]
    if m["kind"] == "on":
        return OnPolicyAgent(OBS_DIM, ACT_DIM, tp, sp, algo=m["algo"], rerun_rule=m.get("rerun", False),
                             g_bar=g_bar)
    if m["kind"] == "off":
        return OffPolicyAgent(OBS_DIM, ACT_DIM, tp, sp, algo=m["algo"])
    from .mpc import MPCController
    return MPCController(sp)


def run_dir(root, method, seed, tag=""):
    return os.path.join(root, f"{resolve(method)}{tag}", f"seed{seed}")


def _log_row(ep, env, rew_aug, r0, g, cost, corr, dt, extra, mod=None):
    row = dict(episode=ep, cum_reward=float(r0.sum(0).mean()), cum_reward_aug=float(rew_aug.sum(0).mean()),
               mean_g=float(g.mean()), Z_T=float(env.Z.mean()), cost_day=float(cost.sum(0).mean()),
               R_train=float(env.alive.mean()), f_corr=float(corr.mean()),
               f_mod=float(mod.mean()) if mod is not None else float("nan"), time_s=dt)
    row.update({k: float(v) for k, v in extra.items() if np.isscalar(v)})
    return row


def train(method: str, seed: int, sp: SystemParams, tp: TrainParams, out_root: str,
          pcc_mode: str = "adaptive", eta_Z: Optional[float] = None, g_bar: Optional[float] = None,
          tag: str = "", verbose: bool = True):
    """Train one method with one seed. Writes model.pt, log.csv and config.json."""
    method = resolve(method)
    m = METHODS[method]
    assert m["kind"] in ("on", "off"), "MPC needs no training"
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = run_dir(out_root, method, seed, tag)
    os.makedirs(out, exist_ok=True)
    days = make_pool(sp, "train", tp.n_train_days)
    env = make_env(sp, method, pcc_mode, eta_Z, g_bar)
    agent = make_agent(sp, tp, method, g_bar=env.g_bar)
    T, B = sp.T, tp.n_envs
    rows = []
    total_steps = 0
    for ep in range(1, tp.n_episodes + 1):
        t0 = time.time()
        sel = [days[i] for i in rng.integers(0, len(days), B)]
        obs = env.reset(sel, sample_equipment(sp, rng, B))
        O = np.zeros((T, B, OBS_DIM), np.float32); A = np.zeros((T, B, ACT_DIM), np.float32)
        LP = np.zeros((T, B), np.float32); RW = np.zeros((T, B)); R0 = np.zeros((T, B))
        G = np.zeros((T, B)); CST = np.zeros((T, B)); COR = np.zeros((T, B)); MOD = np.zeros((T, B))
        extra = {}
        for t in range(T):
            if m["kind"] == "on":
                a, lp = agent.act(obs)
            else:
                a, lp = agent.act(obs, random=total_steps < tp.warmup_steps)
            obs2, r, done, info = env.step(a)
            O[t], A[t], RW[t], R0[t], G[t] = obs, a, r, info["r0"], info["g"]
            CST[t], COR[t], MOD[t] = info["cost"], info["corrected"], info["modified"]
            if lp is not None:
                LP[t] = lp
            if m["kind"] == "off":
                # Store candidate actions, base rewards and the resulting next states.
                agent.buf.add(obs, np.clip(a, -1, 1), info["r0"].astype(np.float32), obs2,
                              np.full(B, float(done), np.float32))
                total_steps += B
                for _ in range(tp.updates_per_step):
                    extra = agent.update() or extra
            obs = obs2
        if m["kind"] == "on":
            rew = RW if m["use_queue"] else R0
            extra = agent.update(dict(obs=O, act=A, logp=LP, rew=rew, cost=G))
            if tp.adapt_eta and m["use_queue"] and G.mean() > env.g_bar:
                env.eta_Z *= 1.05
        row = _log_row(ep, env, RW, R0, G, CST, COR, time.time() - t0, extra, MOD)
        row["eta_Z"] = env.eta_Z
        rows.append(row)
        if verbose and (ep % max(1, tp.n_episodes // 20) == 0 or ep == 1):
            print(f"[{method}{tag} s{seed}] ep {ep:4d} R0sum {row['cum_reward']:7.2f} g {row['mean_g']:.3f} "
                  f"Z_T {row['Z_T']:.3f} cost {row['cost_day']:8.0f} Rtr {row['R_train']:.2f} "
                  f"fcorr {row['f_corr']:.2f} {row['time_s']:.2f}s", flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out, "log.csv"), index=False)
    torch.save(agent.state_dict(), os.path.join(out, "model.pt"))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(method=method, seed=seed, pcc_mode=pcc_mode, eta_Z=env.eta_Z, g_bar=env.g_bar,
                       sp=sp.to_dict(), tp=asdict(tp)), f, indent=1, default=str)
    return agent, out


def load_agent(sp: SystemParams, tp: TrainParams, method: str, seed: int, root: str, tag: str = ""):
    method = resolve(method)
    if METHODS[method]["kind"] == "mpc":
        return make_agent(sp, tp, method)
    path = os.path.join(run_dir(root, method, seed, tag), "model.pt")
    agent = make_agent(sp, tp, method)
    agent.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return agent


def policy_fn(agent, method):
    """Deterministic action function used in evaluation."""
    kind = METHODS[resolve(method)]["kind"]
    if kind == "mpc":
        return lambda obs, env: ("phys", agent.act(env))
    return lambda obs, env: ("norm", agent.act(obs, deterministic=True)[0])
