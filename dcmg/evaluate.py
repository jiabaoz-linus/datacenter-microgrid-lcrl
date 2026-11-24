"""
Monte Carlo evaluation of deterministic policies and failure components.

For each test day, equipment states remain fixed throughout the day. Samples
with identical server, cooling-unit and IPCS-unit failure counts share one
trajectory, weighted by the number of samples with that signature.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .config import SystemParams
from .disturbance import Day, Equip, sample_equipment, equipment_signatures, day_seed
from .train import make_env, policy_fn


def _jobs(sp, days: List[Day], n_mc: int, equip_base: int):
    jobs = []
    for di, d in enumerate(days):
        rng = np.random.default_rng(day_seed("equip", d.scenario, di, equip_base))
        uniq, cnt = equipment_signatures(sample_equipment(sp, rng, n_mc))
        for s, c in zip(uniq, cnt):
            jobs.append((di, s, int(c)))
    return jobs


def evaluate(sp: SystemParams, method: str, agent, days: List[Day], n_mc: Optional[int] = None,
             equip_base: int = 0, pcc_mode: str = "adaptive", eta_Z=None, g_bar=None,
             solver: str = "fast", chunk: int = 512, traj_day: Optional[int] = 0):
    """Return per-day metrics and an available failure-free trajectory for traj_day."""
    n_mc = n_mc or sp.N_MC
    jobs = _jobs(sp, days, n_mc, equip_base)
    act = policy_fn(agent, method)
    recs = []
    traj = None
    for start in range(0, len(jobs), chunk):
        J = jobs[start:start + chunk]
        env = make_env(sp, method, pcc_mode, eta_Z, g_bar, solver)
        eq = Equip(np.array([j[1][0] for j in J]), np.array([j[1][1] for j in J]),
                   np.array([j[1][2] for j in J]))
        obs = env.reset([days[j[0]] for j in J], eq)
        B, T = len(J), sp.T
        acc = {k: np.ones(B, bool) for k in ("F_IT", "F_C", "F_P", "dev", "op")}
        cost = np.zeros(B); gsum = np.zeros(B); corr = np.zeros(B); mod = np.zeros(B)
        rt = np.zeros(B); dfc = np.zeros(B); dump = np.zeros(B)
        mIT = np.zeros(B); mC = np.zeros(B); mP = np.zeros(B); peak = np.zeros(B)
        tr_idx = None
        if traj_day is not None:
            cand = [i for i, j in enumerate(J) if j[0] == traj_day and not any(j[1])]
            tr_idx = cand[0] if cand else None
        tr = {k: [] for k in ("P_C2D", "T", "P_out", "P_IT", "P_cool", "grid_cap", "P_G", "P_DG", "P_B",
                              "SOC_D", "q", "d", "lam", "g", "C2D_max")} if tr_idx is not None else None
        for t in range(T):
            kind, a = act(obs, env)
            if kind == "norm":
                obs, r, done, info = env.step(a)
            else:
                obs, r, done, info = env.step(a_phys=a)
            for k in ("F_IT", "F_C", "F_P"):
                acc[k] &= info[k]
            acc["dev"] &= info["E_dev_IT"] & info["E_dev_C"] & info["E_dev_P"]
            acc["op"] &= info["E_op_IT"] & info["E_op_C"] & info["E_op_P"]
            cost += info["cost"]; gsum += info["g"]; corr += info["corrected"]; mod += info["modified"]
            rt += info["rt_adj"] * sp.dt_h * 1e3; dfc += info["deficit"] * sp.dt_h * 1e3
            dump += info["dumped"] * sp.dt_h * 1e3
            mIT += np.clip(info["m_IT"], -sp.d_max, sp.d_max); mC += info["m_C"]; mP += info["m_P"] * 1e3
            peak = np.maximum(peak, info["P_out"] / sp.P_IPCS_max)
            if tr is not None:
                i = tr_idx
                ex = info["exec"]
                tr["P_C2D"].append(ex["P_C2D"][i]); tr["T"].append(info["T_next"][i])
                tr["P_out"].append(info["P_out"][i]); tr["P_IT"].append(info["P_IT"][i])
                tr["P_cool"].append(ex["P_cool"][i]); tr["grid_cap"].append(info["grid_cap"][i])
                tr["P_G"].append(ex["P_G"][i]); tr["P_DG"].append(ex["P_DG"][i]); tr["P_B"].append(ex["P_B"][i])
                tr["SOC_D"].append(info["SOC_D"][i]); tr["q"].append(info["q_next"][i]); tr["d"].append(info["d"][i])
                tr["lam"].append(env.D["lam"][i, t]); tr["g"].append(info["g"][i])
                tr["C2D_max"].append(info["P_C2D_max"][i])
        if tr is not None and traj is None:
            traj = {k: np.array(v) for k, v in tr.items()}
        for i, (di, sig, c) in enumerate(J):
            recs.append(dict(day=di, w=c, sys=acc["F_IT"][i] & acc["F_C"][i] & acc["F_P"][i],
                             IT=acc["F_IT"][i], C=acc["F_C"][i], P=acc["F_P"][i], dev=acc["dev"][i],
                             op=acc["op"][i], cost=cost[i], g=gsum[i] / T, fcorr=corr[i] / T, fmod=mod[i] / T,
                             rt_kWh=rt[i], deficit_kWh=dfc[i], dumped_kWh=dump[i],
                             m_IT=mIT[i] / T, m_C=mC[i] / T, m_P=mP[i] / T, peak_ipcs=peak[i]))
    df = pd.DataFrame(recs)
    rows = []
    for di, grp in df.groupby("day"):
        w = grp["w"].values / grp["w"].sum()
        f = lambda col: float((grp[col].values.astype(float) * w).sum())
        rows.append(dict(day=di, scenario=days[di].scenario, R_sys=f("sys"), R_IT=f("IT"), R_C=f("C"),
                         R_P=f("P"), P_devc=1 - f("dev"),                       # P[(E_dev)^c]
                         P_dev_opc=float(((grp["dev"] & ~grp["op"]).astype(float) * w).sum()),
                         cost=f("cost"), g=f("g"), f_corr=f("fcorr"), f_mod=f("fmod"),
                         rt_kWh=f("rt_kWh"), deficit_kWh=f("deficit_kWh"), dumped_kWh=f("dumped_kWh"), m_IT=f("m_IT"), m_C=f("m_C"),
                         m_P=f("m_P"), peak_ipcs=f("peak_ipcs"), n_sig=len(grp)))
    return pd.DataFrame(rows), traj


def summarize(day_df: pd.DataFrame) -> Dict[str, float]:
    """Average per-day reliability, cost, action and operating metrics."""
    cols = ["R_sys", "R_IT", "R_C", "R_P", "P_devc", "P_dev_opc", "cost", "g", "f_corr", "f_mod",
            "rt_kWh", "deficit_kWh", "dumped_kWh",
            "m_IT", "m_C", "m_P", "peak_ipcs"]
    return {c: float(day_df[c].mean()) for c in cols}
