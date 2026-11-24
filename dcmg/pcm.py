"""Pre-Constraint Mechanism (PCM) for checking and correcting candidate actions.

The environment supplies the conservative state as ctx, with one array entry
per batch member. Candidate checks cover action bounds, storage SOC, power
balance and functional margins. The fast correction uses a structured
approximate projection; solver-based correction is provided in pcm_exact.
Residual violations are recorded as slack variables.

Action order: [u, n, P_cool, P_DG, P_B, P_G, P_C2D], in physical units."""
import numpy as np
from .config import SystemParams
from . import physics as ph

U, N, PC, DG, PB, PG, C2D = range(7)
ACT_NAMES = ["u", "n", "P_cool", "P_DG", "P_B", "P_G", "P_C2D"]


def action_ranges(sp: SystemParams) -> np.ndarray:
    """Return action normalisation ranges for W = diag(1 / range**2)."""
    return np.array([sp.u_max - sp.u_min, sp.N_S, sp.P_cool_max, sp.P_DG_max,
                     2 * sp.P_B_D_max, sp.P_sell_max + sp.P_grid_max, sp.P_PCC_max])


def action_bounds(sp: SystemParams, ctx):
    """Build per-sample action bounds, including SOC-dependent battery limits."""
    B = ctx["q"].shape[0]
    dis, ch = ph.battery_limits(ctx["SOC_D"], sp.P_B_D_max, sp.E_B_D, sp.eta_ch_D, sp.eta_dis_D,
                                sp.SOC_D_min, sp.SOC_D_max, sp.dt_h)
    lo = np.zeros((B, 7))
    hi = np.zeros((B, 7))
    lo[:, U], hi[:, U] = sp.u_min, sp.u_max
    lo[:, N], hi[:, N] = 0, sp.N_S
    lo[:, PC], hi[:, PC] = 0, sp.P_cool_max
    lo[:, DG], hi[:, DG] = 0, sp.P_DG_max
    lo[:, PB], hi[:, PB] = -ch, dis
    lo[:, PG], hi[:, PG] = -sp.P_sell_max, ctx["grid"]
    # rule-based PCC mode fixes the support power (lo = hi = rule value)
    lo[:, C2D], hi[:, C2D] = ctx.get("c2d_lo", 0.0), ctx["c2d"]
    return lo, hi


def evaluate(sp: SystemParams, a, ctx):
    """Evaluate functional margins and power mismatch for action a under ctx."""
    N_act = ph.active_servers(a[:, N], ctx["N_pool"])
    P_IT = ph.it_power(sp, N_act, a[:, U])
    mu = ph.service_rate(sp, N_act, a[:, U])
    _, d, _ = ph.queue_and_delay(sp, ctx["q"], ctx["lam"], mu)
    P_out = P_IT + a[:, PC]
    Q_load = ph.heat_load(sp, P_IT, P_out)
    Q_act = np.minimum(sp.COP * a[:, PC], ctx["Q_ava"])
    T1 = ph.temp_next(sp, ctx["T"], Q_load, Q_act, ctx["Tamb"])
    cap = np.minimum(ctx["P_ava"], sp.P_IPCS_max)
    demand = P_out / sp.eta_IPCS + sp.P_base
    supply = ctx["res"] + a[:, C2D] + a[:, PG] + a[:, DG] + a[:, PB]
    return dict(m_IT=sp.d_max - d, m_C=sp.T_max - T1, m_IPCS=cap - P_out,
                mismatch=demand - supply, P_IT=P_IT, P_out=P_out, d=d, T1=T1, mu=mu)


def h_vector(sp: SystemParams, a, ctx, lo, hi, R):
    """Return clipped, normalised positive constraint excesses and action metrics."""
    box = (np.maximum(lo - a - 1e-9, 0) + np.maximum(a - hi - 1e-9, 0)) / R          # (B,7)
    ev = evaluate(sp, a, ctx)
    bal = np.maximum(np.abs(ev["mismatch"]) - sp.bal_tol, 0) / 1.0
    it = np.maximum(-ev["m_IT"], 0) / sp.m_ref_IT
    co = np.maximum(-ev["m_C"], 0) / sp.m_ref_C
    ip = np.maximum(-ev["m_IPCS"], 0) / sp.m_ref_IPCS
    h = np.concatenate([box, np.stack([bal, it, co, ip], 1)], 1)
    return np.minimum(h, 3.0), ev


def _balance(sp, a, ctx, lo, hi, R, passes=5):
    """Redistribute power mismatch over grid, PCC, diesel and battery power.
    
    Each pass allocates changes in proportion to squared normalisation ranges
    among variables with available headroom, then clips to their bounds."""
    idx = [PG, C2D, DG, PB]
    w = R[idx] ** 2
    for _ in range(passes):
        mis = evaluate(sp, a, ctx)["mismatch"]               # >0 -> need more supply
        up = mis > 0
        free = np.where(up[:, None], a[:, idx] < hi[:, idx] - 1e-12, a[:, idx] > lo[:, idx] + 1e-12)
        ww = free * w[None, :]
        tot = ww.sum(1, keepdims=True)
        share = np.where(tot > 0, ww / np.maximum(tot, 1e-12), 0.0) * mis[:, None]
        a[:, idx] = np.clip(a[:, idx] + share, lo[:, idx], hi[:, idx])
    return a


def correct_fast(sp: SystemParams, a0, ctx, lo, hi, R):
    """Apply a structured approximate action projection.
    
    Process box bounds, IT service, cooling, IPCS capacity and supply balance
    in sequence. Returns corrected actions and residual slacks of shape (B, 3)."""
    a = np.clip(a0.copy(), lo, hi)
    a[:, N] = np.round(a[:, N])
    Bn = a.shape[0]
    sigma = np.zeros((Bn, 3))

    # ---- IT delay margin: find minimum service rate, then closest (u, n)
    mu_hi = sp.mu0 * sp.u_max * ctx["N_pool"]
    mu_req, feas = ph.required_service_rate(sp, ctx["q"], ctx["lam"], mu_hi)
    mu_req = mu_req * (1 + 1e-6)
    mu_cur = ph.service_rate(sp, ph.active_servers(a[:, N], ctx["N_pool"]), a[:, U])
    need = mu_cur < mu_req
    if need.any():
        ug = np.linspace(sp.u_min, sp.u_max, 25)[None, :]
        ug = np.concatenate([ug.repeat(Bn, 0), a[:, U:U + 1]], 1)            # (B,26)
        n_need = np.ceil(mu_req[:, None] / (sp.mu0 * ug))
        n_new = np.maximum(a[:, N:N + 1], n_need)
        ok = n_new <= ctx["N_pool"][:, None]
        dist = ((ug - a0[:, U:U + 1]) / R[U]) ** 2 + ((n_new - a0[:, N:N + 1]) / R[N]) ** 2
        dist = np.where(ok, dist, np.inf)
        j = np.argmin(dist, 1)
        any_ok = np.isfinite(dist[np.arange(Bn), j])
        u_sel = np.where(any_ok, ug[np.arange(Bn), j], sp.u_max)
        n_sel = np.where(any_ok, n_new[np.arange(Bn), j], ctx["N_pool"])
        a[:, U] = np.where(need, u_sel, a[:, U])
        a[:, N] = np.where(need, np.minimum(n_sel, sp.N_S), a[:, N])
    ev = evaluate(sp, a, ctx)
    sigma[:, 0] = np.maximum(-ev["m_IT"], 0)

    # ---- cooling margin
    P_IT = ev["P_IT"]
    pc_req = ph.cooling_power_required(sp, P_IT, ctx["T"], ctx["Tamb"])
    pc_cap = np.minimum(sp.P_cool_max, ctx["Q_ava"] / sp.COP)
    a[:, PC] = np.where(a[:, PC] < pc_req, np.minimum(pc_req, pc_cap), a[:, PC])

    # ---- IPCS margin: trim cooling above its requirement first
    cap = np.minimum(ctx["P_ava"], sp.P_IPCS_max)
    over = P_IT + a[:, PC] - cap
    trim = np.clip(over, 0, np.maximum(a[:, PC] - pc_req, 0))
    a[:, PC] -= trim

    # ---- supply balance
    a = _balance(sp, a, ctx, lo, hi, R)
    ev = evaluate(sp, a, ctx)
    sigma[:, 0] = np.maximum(-ev["m_IT"], 0)
    sigma[:, 1] = np.maximum(-ev["m_C"], 0)
    deficit = np.maximum(ev["mismatch"] - sp.bal_tol, 0) * sp.eta_IPCS
    sigma[:, 2] = np.maximum(-ev["m_IPCS"], 0) + deficit
    sigma[:, 0] = np.minimum(sigma[:, 0], ph.D_CLIP)
    return a, sigma


def violation_degree(sp: SystemParams, h, a0, a_bar, sigma, R, infeasible):
    """Combine constraint excess, correction distance and residual slacks."""
    dist = np.sqrt((((a_bar - a0) / R) ** 2).sum(1))
    sbar = np.array([sp.m_ref_IT, sp.m_ref_C, sp.m_ref_IPCS])
    s_term = np.minimum(sigma / sbar, 3.0).sum(1)
    G = sp.alpha1 * np.linalg.norm(h, axis=1) + sp.alpha2 * dist + sp.alpha3 * s_term
    return np.where(infeasible, np.minimum(sp.g_max, G), 0.0)


def slack_objective(sp: SystemParams, a0, a_bar, sigma, R):
    """Return squared normalised correction distance plus weighted residual slacks."""
    w = sp.slack_w_factor / np.array([sp.m_ref_IT, sp.m_ref_C, sp.m_ref_IPCS])
    return (((a_bar - a0) / R) ** 2).sum(1) + (sigma * w).sum(1)


class PCM:
    def __init__(self, sp: SystemParams, enabled: bool = True, solver: str = "fast"):
        self.sp = sp
        self.enabled = enabled
        self.solver = solver
        self.R = action_ranges(sp)

    def __call__(self, a0, ctx):
        """Return executed actions, violation degree, correction flags, slacks and excesses.
        
        When PCM is disabled, candidates pass through without correction. Their
        violation degree is still computed for the virtual queue; the environment
        applies physical actuator limits during execution."""
        sp = self.sp
        lo, hi = action_bounds(sp, ctx)
        h, _ = h_vector(sp, a0, ctx, lo, hi, self.R)
        infeasible = (h > 1e-9).any(1)
        a_bar = a0.copy()
        sigma = np.zeros((a0.shape[0], 3))
        if infeasible.any():
            idx = np.where(infeasible)[0]
            sub = {k: v[idx] for k, v in ctx.items()}
            if self.solver == "exact":
                from .pcm_exact import correct_exact_batch
                ab, sg = correct_exact_batch(sp, a0[idx], sub, lo[idx], hi[idx], self.R)
            else:
                ab, sg = correct_fast(sp, a0[idx], sub, lo[idx], hi[idx], self.R)
            a_bar[idx] = ab
            sigma[idx] = sg
        g = violation_degree(sp, h, a0, a_bar, sigma, self.R, infeasible)
        if not self.enabled:
            a_exec = a0.copy()
            corrected = np.zeros(a0.shape[0], bool)
        else:
            a_exec = a_bar
            corrected = infeasible
        return a_exec, g, corrected, sigma, h
