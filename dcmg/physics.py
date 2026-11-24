"""Vectorised physical relations for networked data center microgrids.

Functions operate on NumPy arrays and support broadcasting."""
import numpy as np
from .config import SystemParams

D_CLIP = 100.0  # numerical cap for the waiting delay (min)


# ---------------- IT layer ----------------
def active_servers(n, N_pool):
    """Limit the active server count to the available healthy servers."""
    return np.minimum(n, N_pool)


def it_power(sp: SystemParams, N_act, u):
    """Return IT power demand in MW."""
    return N_act * (sp.P_idle + (sp.P_peak - sp.P_idle) * u)


def service_rate(sp: SystemParams, N_act, u):
    """Return the task service rate in tasks per minute."""
    return sp.mu0 * u * N_act


def queue_and_delay(sp: SystemParams, q, lam, mu):
    """Update the queue and compute the capped waiting delay.
    
    The delay numerator is the end-of-slot backlog plus q_res. The denominator
    is the service-rate surplus, bounded below by eps_d. Returns the next queue,
    delay in minutes and number of tasks served during the slot."""
    A = lam * sp.dt_min                                    
    D = np.minimum(q + A, mu * sp.dt_min)                  
    q_next = np.maximum(q + A - D, 0.0)                    
    d = (q_next + sp.q_res) / np.maximum(mu - lam, sp.eps_d)  
    return q_next, np.minimum(d, D_CLIP), D


def required_service_rate(sp: SystemParams, q, lam, mu_hi, d_target=None, iters=40):
    """Find the required service rate by vectorised bisection.
    
    Returns (mu_req, feasible). If the delay target cannot be met at mu_hi,
    mu_req is set to mu_hi and feasible is False."""
    d_target = sp.d_max if d_target is None else d_target
    lo = np.zeros_like(q, dtype=float)
    hi = np.asarray(mu_hi, dtype=float).copy()
    _, d_hi, _ = queue_and_delay(sp, q, lam, hi)
    feas = d_hi <= d_target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, d_mid, _ = queue_and_delay(sp, q, lam, mid)
        ok = d_mid <= d_target
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return np.where(feas, hi, mu_hi), feas


# ---------------- thermal layer ----------------
def heat_load(sp: SystemParams, P_IT_sup, P_out_sup):
    """Return heat from supplied IT power, IPCS losses and external sources."""
    return sp.alpha_h * P_IT_sup + sp.k_loss * P_out_sup + sp.Q_env


def heat_required(sp: SystemParams, Q_load, Tamb, T):
    """Return the nonnegative cooling load including ambient heat exchange."""
    return np.maximum(0.0, Q_load + (Tamb - T) / sp.R_th)


def temp_next(sp: SystemParams, T, Q_load, Q_act, Tamb):
    """Advance the hotspot temperature by one scheduling interval."""
    return T + sp.dt_h / sp.C_th * (Q_load - Q_act + (Tamb - T) / sp.R_th)


def cooling_power_required(sp: SystemParams, P_IT, T, Tamb, T_target=None):
    """Compute cooling power required to meet the next-slot temperature target.
    
    The heat balance includes IPCS losses associated with cooling power."""
    T_target = sp.T_max if T_target is None else T_target
    num = (sp.alpha_h + sp.k_loss) * P_IT + sp.Q_env + (Tamb - T) / sp.R_th \
        - sp.C_th * (T_target - T) / sp.dt_h
    return np.maximum(0.0, num / (sp.COP - sp.k_loss))


# ---------------- storage ----------------
def soc_next(SOC, P_net, E, eta_ch, eta_dis, dt_h):
    """Update storage SOC; positive P_net discharges and negative P_net charges."""
    dis = np.maximum(P_net, 0.0)
    ch = np.maximum(-P_net, 0.0)
    return SOC - dt_h / E * (dis / eta_dis - eta_ch * ch)


def battery_limits(SOC, P_max, E, eta_ch, eta_dis, soc_min, soc_max, dt_h):
    """Return discharge and charge limits that respect power and SOC bounds."""
    dis = np.clip((SOC - soc_min) * E * eta_dis / dt_h, 0.0, P_max)
    ch = np.clip((soc_max - SOC) * E / (eta_ch * dt_h), 0.0, P_max)
    return dis, ch


# ---------------- CMG ----------------
def cmg_support_max(sp: SystemParams, SOC_C, res_C, L_cr, L_ba, P_g_C=None):
    """Return PCC support capacity after reserving supply for local CMG loads."""
    P_g_C = sp.P_grid_C_max if P_g_C is None else P_g_C
    dis_ava, _ = battery_limits(SOC_C, sp.P_B_C_max, sp.E_B_C, sp.eta_ch_C, sp.eta_dis_C,
                                sp.SOC_C_min, sp.SOC_C_max, sp.dt_h)
    P_sur = np.maximum(P_g_C + res_C + dis_ava - L_cr - L_ba, 0.0)
    return np.minimum(sp.P_PCC_max, P_sur)


def cmg_dispatch(sp: SystemParams, SOC_C, res_C, L_cr, L_ba, C2D, c_buy):
    """Dispatch the CMG battery and external-grid import.
    
    Charge from renewable surplus or available valley-period grid capacity;
    discharge during peak periods or to reduce grid-interface shortfalls.
    Returns net battery power, grid import and updated SOC."""
    dis_max, ch_max = battery_limits(SOC_C, sp.P_B_C_max, sp.E_B_C, sp.eta_ch_C, sp.eta_dis_C,
                                     sp.SOC_C_min, sp.SOC_C_max, sp.dt_h)
    net = L_cr + L_ba + C2D - res_C               # power still needed after renewables
    peak = c_buy >= sp.c_peak - 1e-9
    valley = c_buy <= sp.c_valley + 1e-9
    PB = np.zeros_like(net)
    # surplus renewable -> charge
    PB = np.where(net < 0, -np.minimum(-net, ch_max), PB)
    # peak hours -> discharge to cover net load
    PB = np.where((net > 0) & peak, np.minimum(net, dis_max), PB)
    # valley hours -> charge from grid with headroom
    head = np.maximum(sp.P_grid_C_max - np.maximum(net, 0), 0)
    PB = np.where((net > 0) & valley, -np.minimum(ch_max, head) * 0.5, PB)
    grid = np.maximum(net - PB, 0.0)
    # interface insufficient -> extra discharge
    short = np.maximum(grid - sp.P_grid_C_max, 0.0)
    extra = np.minimum(short, np.maximum(dis_max - np.maximum(PB, 0), 0))
    PB = PB + extra
    grid = np.minimum(np.maximum(net - PB, 0.0), sp.P_grid_C_max)
    SOC_new = soc_next(SOC_C, PB, sp.E_B_C, sp.eta_ch_C, sp.eta_dis_C, sp.dt_h)
    return PB, grid, np.clip(SOC_new, 0.0, 1.0)


# ---------------- reward helpers ----------------
def saturate(m, m_ref):
    """Normalise a margin and clip it to the interval [0, 1]."""
    return np.clip(m / m_ref, 0.0, 1.0)
