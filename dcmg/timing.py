"""Action-correction solver timing and objective-value comparison."""
import os
import time
import numpy as np

from .config import SystemParams, TrainParams, SCENARIO_NAMES
from .disturbance import make_pool, sample_equipment
from .pcm import action_bounds, h_vector, action_ranges, correct_fast
from .pcm_exact import correct_exact_batch, HAS_SCIP
from .train import make_env, load_agent, run_dir


def collect_instances(sp, tp, root, seed, n_per_scenario, rng):
    """Collect infeasible candidate actions from stochastic LCRL rollouts.
    Use uniformly sampled actions when no trained model is available."""
    have_model = os.path.exists(os.path.join(run_dir(root, "LCRL", seed), "model.pt"))
    agent = load_agent(sp, tp, "LCRL", seed, root) if have_model else None
    inst = []
    for scen in SCENARIO_NAMES:
        got = []
        days = make_pool(sp, "test", 8, [scen])
        env = make_env(sp, "LCRL")
        obs = env.reset(days, sample_equipment(sp, rng, len(days)))
        R = action_ranges(sp)
        for t in range(sp.T):
            a = agent.act(obs)[0] if agent else rng.uniform(-1, 1, (len(days), 7))
            a0 = env.decode(a, t)
            ctx = env.checking_state(t)
            lo, hi = action_bounds(sp, ctx)
            h, _ = h_vector(sp, a0, ctx, lo, hi, R)
            for b in np.where((h > 1e-9).any(1))[0]:
                got.append((a0[b], {k: (v[b] if np.ndim(v) else v) for k, v in ctx.items()}, lo[b], hi[b]))
            obs, _, _, _ = env.step(a)
        idx = rng.choice(len(got), min(n_per_scenario, len(got)), replace=False) if got else []
        inst += [got[i] for i in idx]
    return inst


def correction_timing(sp: SystemParams, tp: TrainParams, root: str, seed: int, n_per_scenario=200, quick=False):
    rng = np.random.default_rng(seed)
    inst = collect_instances(sp, tp, root, seed, n_per_scenario, rng)
    R = action_ranges(sp)
    backend = "scip" if HAS_SCIP else "slsqp"
    t_exact, t_fast, gap = [], [], []
    for a0, c, lo, hi in inst:
        cc = {k: np.array([v]) for k, v in c.items()}
        t0 = time.perf_counter()
        ae, se = correct_exact_batch(sp, a0[None], cc, lo[None], hi[None], R, backend=backend)
        t_exact.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        af, sf = correct_fast(sp, a0[None], cc, lo[None], hi[None], R)
        t_fast.append((time.perf_counter() - t0) * 1e3)
        w = sp.slack_w_factor / np.array([sp.m_ref_IT, sp.m_ref_C, sp.m_ref_IPCS])
        fe = (((ae[0] - a0) / R) ** 2).sum() + (se[0] * w).sum()
        ff = (((af[0] - a0) / R) ** 2).sum() + (sf[0] * w).sum()
        gap.append((ff - fe) / max(abs(fe), 1e-9))
    q = lambda x: dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))
    return dict(n_instances=len(inst), backend=backend, exact_ms=q(t_exact), fast_ms=q(t_fast),
                fast_vs_exact_rel_obj_gap=q(gap))
