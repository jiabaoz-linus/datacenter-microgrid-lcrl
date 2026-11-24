"""Solver-based action correction with continuous actions and integer server count.

The objective combines squared normalised action displacement with weighted
functional slacks. Bounds, storage limits, power balance and functional
constraints are imposed using the conservative checking state.

The SCIP backend uses PySCIPOpt. The SLSQP backend solves a continuous
relaxation and re-solves at the floor and ceiling of its server count; this
is a local heuristic. If a backend cannot produce a solution, correction
falls back to SLSQP or the structured projection, respectively.

When the delay target is attainable, an auxiliary nonnegative queue variable
represents the end-of-slot backlog in the delay constraints. Otherwise the
server count and utilisation are fixed at their available maxima."""
import numpy as np
from scipy.optimize import minimize

from .config import SystemParams
from . import physics as ph
from .pcm import U, N, PC, DG, PB, PG, C2D, evaluate

try:
    import pyscipopt  # noqa: F401
    HAS_SCIP = True
except Exception:  # pragma: no cover
    HAS_SCIP = False


def _weights(sp):
    return sp.slack_w_factor / np.array([sp.m_ref_IT, sp.m_ref_C, sp.m_ref_IPCS])


def _it_feasible(sp, c):
    mu_hi = sp.mu0 * sp.u_max * c["N_pool"]
    _, d, _ = ph.queue_and_delay(sp, np.array([c["q"]]), np.array([c["lam"]]), np.array([mu_hi]))
    return d[0] <= sp.d_max


def _solve_slsqp(sp, a0, c, lo, hi, R, n_fixed=None, it_on=True):
    w = _weights(sp)
    nvar = 11  # 7 actions, 3 slacks, q~
    x0 = np.concatenate([np.clip(a0, lo, hi), [0, 0, 0], [0]])
    blo = np.concatenate([lo, [0, 0, 0], [0]])
    bhi = np.concatenate([hi, [ph.D_CLIP, 50, 5], [1e6]])
    bhi[N] = min(hi[N], c["N_pool"])
    bhi[PC] = min(hi[PC], c["Q_ava"] / sp.COP)
    if n_fixed is not None:
        blo[N] = bhi[N] = n_fixed
        x0[N] = n_fixed
    if not it_on:  # delay target unreachable: fix the service controls at their available maxima
        blo[U] = bhi[U] = sp.u_max
        blo[N] = bhi[N] = min(hi[N], c["N_pool"])
    x0 = np.clip(x0, blo, bhi)
    cap = min(c["P_ava"], sp.P_IPCS_max)

    def parts(x):
        mu = sp.mu0 * x[U] * x[N]
        P_IT = x[N] * (sp.P_idle + (sp.P_peak - sp.P_idle) * x[U])
        P_out = P_IT + x[PC]
        Q_load = sp.alpha_h * P_IT + sp.k_loss * P_out + sp.Q_env
        T1 = c["T"] + sp.dt_h / sp.C_th * (Q_load - sp.COP * x[PC] + (c["Tamb"] - c["T"]) / sp.R_th)
        demand = P_out / sp.eta_IPCS + sp.P_base
        supply = c["res"] + x[C2D] + x[PG] + x[DG] + x[PB]
        return mu, P_out, T1, demand, supply

    def obj(x):
        return float((((x[:7] - a0) / R) ** 2).sum() + (w * x[7:10]).sum())

    def ineq(x):
        mu, P_out, T1, demand, supply = parts(x)
        cons = [sp.T_max + x[8] - T1,
                cap + x[9] - P_out,
                sp.bal_tol - (demand - supply - x[9] / sp.eta_IPCS),
                sp.bal_tol - (supply - demand)]
        if it_on:
            cons += [x[10] - (c["q"] + (c["lam"] - mu) * sp.dt_min),
                     (sp.d_max + x[7]) * (mu - c["lam"]) - (x[10] + sp.q_res),
                     mu - c["lam"] - sp.eps_d]
        return np.array(cons)

    res = minimize(obj, x0, method="SLSQP", bounds=list(zip(blo, bhi)),
                   constraints=[dict(type="ineq", fun=ineq)], options=dict(maxiter=200, ftol=1e-9))
    ok = res.success and (ineq(res.x) >= -1e-6).all()
    return res.x, obj(res.x), ok


def correct_one_slsqp(sp: SystemParams, a0, c, lo, hi, R):
    it_on = _it_feasible(sp, c)
    x, f, ok = _solve_slsqp(sp, a0, c, lo, hi, R, it_on=it_on)
    best = None
    for nf in sorted({np.floor(x[N]), np.ceil(x[N])}):
        xi, fi, oki = _solve_slsqp(sp, a0, c, lo, hi, R, n_fixed=float(nf), it_on=it_on)
        if oki and (best is None or fi < best[1]):
            best = (xi, fi)
    if best is None:
        from .pcm import correct_fast
        cc = {k: np.array([v]) for k, v in c.items()}
        ab, sg = correct_fast(sp, a0[None], cc, lo[None], hi[None], R)
        return ab[0], sg[0]
    xb = best[0]
    a = xb[:7].copy()
    cc = {k: np.array([v]) for k, v in c.items()}
    ev = evaluate(sp, a[None], cc)
    sigma = np.array([max(-ev["m_IT"][0], 0), max(-ev["m_C"][0], 0), max(-ev["m_IPCS"][0], 0)])
    return a, sigma


def correct_one_scip(sp: SystemParams, a0, c, lo, hi, R, feastol=1e-7):
    from pyscipopt import Model, quicksum
    m = Model()
    m.hideOutput()
    m.setParam("numerics/feastol", feastol)
    names = ["u", "n", "pc", "dg", "pb", "pg", "c2d"]
    it_ok = _it_feasible(sp, c)
    x = []
    for i, nm in enumerate(names):
        ub = hi[i]
        if i == N:
            ub = min(hi[N], c["N_pool"])
        if i == PC:
            ub = min(hi[PC], c["Q_ava"] / sp.COP)
        lb = lo[i]
        if not it_ok and i == U:
            lb = ub = sp.u_max
        if not it_ok and i == N:
            lb = ub
        x.append(m.addVar(nm, vtype="I" if i == N else "C", lb=lb, ub=ub))
    sIT = m.addVar("sIT", lb=0, ub=ph.D_CLIP)
    sC = m.addVar("sC", lb=0)
    sP = m.addVar("sP", lb=0)
    qt = m.addVar("qt", lb=0)
    mu = sp.mu0 * x[U] * x[N]
    P_IT = x[N] * (sp.P_idle + (sp.P_peak - sp.P_idle) * x[U])
    P_out = P_IT + x[PC]
    Q_load = sp.alpha_h * P_IT + sp.k_loss * P_out + sp.Q_env
    T1 = c["T"] + sp.dt_h / sp.C_th * (Q_load - sp.COP * x[PC] + (c["Tamb"] - c["T"]) / sp.R_th)
    demand = P_out / sp.eta_IPCS + sp.P_base
    supply = c["res"] + x[C2D] + x[PG] + x[DG] + x[PB]
    cap = min(c["P_ava"], sp.P_IPCS_max)
    m.addCons(T1 <= sp.T_max + sC)
    m.addCons(P_out <= cap + sP)
    m.addCons(demand - supply - sP / sp.eta_IPCS <= sp.bal_tol)
    m.addCons(supply - demand <= sp.bal_tol)
    if it_ok:
        m.addCons(qt >= c["q"] + (c["lam"] - mu) * sp.dt_min)
        m.addCons(qt + sp.q_res <= (sp.d_max + sIT) * (mu - c["lam"]))
        m.addCons(mu >= c["lam"] + sp.eps_d)
    w = _weights(sp)
    # quadratic objective via epigraph variable
    tobj = m.addVar("tobj", lb=0)
    m.addCons(quicksum(((x[i] - a0[i]) / R[i]) * ((x[i] - a0[i]) / R[i]) for i in range(7))
              + w[0] * sIT + w[1] * sC + w[2] * sP <= tobj)
    m.setObjective(tobj, "minimize")
    m.optimize()
    if m.getNSols() == 0:
        return correct_one_slsqp(sp, a0, c, lo, hi, R)
    a = np.array([m.getVal(v) for v in x])
    cc = {k: np.array([v]) for k, v in c.items()}
    ev = evaluate(sp, a[None], cc)
    sigma = np.array([max(-ev["m_IT"][0], 0), max(-ev["m_C"][0], 0), max(-ev["m_IPCS"][0], 0)])
    return a, sigma


def correct_exact_batch(sp: SystemParams, a0, ctx, lo, hi, R, backend=None):
    backend = backend or ("scip" if HAS_SCIP else "slsqp")
    fn = correct_one_scip if backend == "scip" else correct_one_slsqp
    out_a = np.zeros_like(a0)
    out_s = np.zeros((a0.shape[0], 3))
    for i in range(a0.shape[0]):
        c = {k: (float(v[i]) if np.ndim(v) else float(v)) for k, v in ctx.items()}
        out_a[i], out_s[i] = fn(sp, a0[i], c, lo[i], hi[i], R)
    return out_a, out_s
