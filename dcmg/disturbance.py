"""Generate stochastic task days, forecasts and equipment availability.

Workload and temperature use autoregressive disturbances with periodic means
and a delayed cross term. Scenario windows modify workload, temperature and
grid capacity. Renewable output uses clipped daily curves with random noise.
Equipment failures are sampled once per task day, without repair."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import zlib
import numpy as np

from .config import SystemParams, SCENARIOS


def _h(sp: SystemParams) -> np.ndarray:
    """Return slot midpoints in hours."""
    return (np.arange(sp.T) + 0.5) * sp.dt_h


def window_weight(sp: SystemParams, win: Optional[Tuple[float, float]], ramp_h: float = 0.5) -> np.ndarray:
    """Return disturbance-window weights with linear transitions at the edges."""
    h = _h(sp)
    if win is None:
        return np.zeros(sp.T)
    a, b = win
    w = np.clip((h - (a - ramp_h)) / ramp_h, 0, 1) * np.clip(((b + ramp_h) - h) / ramp_h, 0, 1)
    return w


def baseline_arrival(sp: SystemParams) -> np.ndarray:
    """Return the daily workload template with mean lam_mean and peak lam_peak."""
    h = _h(sp)
    s = 0.5 * (1 - np.cos(2 * np.pi * (h - 4.0) / 24.0))
    bump = np.exp(-((h - 13.0) / 2.2) ** 2)
    s2 = 0.55 * s + 0.45 * bump
    s2 = (s2 - s2.min()) / (s2.max() - s2.min())
    m = s2.mean()
    low = (sp.lam_mean - sp.lam_peak * m) / (1 - m)
    return low + (sp.lam_peak - low) * s2


def baseline_ambient(sp: SystemParams) -> np.ndarray:
    h = _h(sp)
    mid = 0.5 * (sp.T_amb_min + sp.T_amb_max)
    amp = 0.5 * (sp.T_amb_max - sp.T_amb_min)
    return mid - amp * np.cos(2 * np.pi * (h - 15.0) / 24.0)


def solar_shape(sp: SystemParams) -> np.ndarray:
    h = _h(sp)
    return np.clip(np.sin(np.pi * (h - 6.0) / 12.0), 0, None) ** 1.2


def cmg_load_shape(sp: SystemParams) -> np.ndarray:
    h = _h(sp)
    e = np.exp(-((h - 19.5) / 3.0) ** 2) + 0.45 * np.exp(-((h - 12.0) / 3.0) ** 2)
    return (e - e.min()) / (e.max() - e.min())


def tou_prices(sp: SystemParams) -> Tuple[np.ndarray, np.ndarray]:
    h = _h(sp)
    buy = np.full(sp.T, sp.c_flat)
    buy[h < 8.0] = sp.c_valley
    buy[((h >= 11.0) & (h < 15.0)) | ((h >= 18.0) & (h < 21.0))] = sp.c_peak
    sell = np.full(sp.T, sp.c_sell)
    return buy, sell


def mu_lam(h):  # daily periodic mean of zeta_lambda
    return 0.01 * np.sin(2 * np.pi * (h - 10.0) / 24.0)


def mu_th(h):   # daily periodic mean of zeta_theta
    return 0.30 * np.sin(2 * np.pi * (h - 9.0) / 24.0)


@dataclass
class Day:
    """A task day containing time series of length T and scenario metadata."""
    lam: np.ndarray        # actual arrival rate lambda_t (task/min) incl. pulses
    lam_hat: np.ndarray    # forecast hat-lambda_t
    lam_tmpl: np.ndarray   # periodic template incl. scenario multiplier
    Tamb: np.ndarray       # actual ambient temperature
    Tamb_hat: np.ndarray
    Tamb_tmpl: np.ndarray
    zeta_lam: np.ndarray
    zeta_th: np.ndarray
    xi_g: np.ndarray       # actual DCMG utility capacity coefficient
    xi_g_hat: np.ndarray
    res_D: np.ndarray      # actual DCMG renewable, MW
    res_D_hat: np.ndarray
    res_C: np.ndarray
    res_C_hat: np.ndarray
    L_cr: np.ndarray       # CMG critical load, MW
    L_ba: np.ndarray       # CMG general base load, MW
    c_buy: np.ndarray
    c_sell: np.ndarray
    scenario: str
    seed: int


def _trunc_normal(rng, size, sigma, clip):
    return np.clip(rng.standard_normal(size), -clip, clip) * sigma


def make_day(sp: SystemParams, scen: Dict, seed: int, name: str = "") -> Day:
    rng = np.random.default_rng(seed)
    T = sp.T
    h = _h(sp)
    w = window_weight(sp, scen.get("win"))
    task_mult = 1.0 + (scen["task"] - 1.0) * w
    dT = scen["dT"] * w
    grid_coef = 1.0 - (1.0 - scen["grid"]) * w

    lam_tmpl = baseline_arrival(sp) * task_mult
    Tamb_tmpl = baseline_ambient(sp) + dT

    # ---- AR processes with daily periodic means and a delayed cross term
    e_l = _trunc_normal(rng, T, sp.sig_lam, sp.eps_clip)
    e_t = _trunc_normal(rng, T, sp.sig_th, sp.eps_clip)
    z_l = np.zeros(T)
    z_t = np.zeros(T)
    z_l_hat = np.zeros(T)
    z_t_hat = np.zeros(T)
    prev_l, prev_t, prev_el = mu_lam(h[0] - sp.dt_h), mu_th(h[0] - sp.dt_h), 0.0
    prev_h = h[0] - sp.dt_h
    for t in range(T):
        # one-step forecast (conditional mean given information up to t-1)
        z_l_hat[t] = mu_lam(h[t]) + sp.rho_lam * (prev_l - mu_lam(prev_h))
        z_t_hat[t] = mu_th(h[t]) + sp.rho_th * (prev_t - mu_th(prev_h)) + sp.rho_lamth * prev_el
        z_l[t] = z_l_hat[t] + e_l[t]
        z_t[t] = z_t_hat[t] + e_t[t]
        prev_l, prev_t, prev_el, prev_h = z_l[t], z_t[t], e_l[t], h[t]

    # Random task pulses are excluded from the workload forecast.
    pulse = np.zeros(T)
    t = 0
    while t < T:
        if rng.random() < sp.pulse_rate:
            L = rng.integers(sp.pulse_len[0], sp.pulse_len[1] + 1)
            pulse[t:t + L] += rng.uniform(*sp.pulse_amp)
            t += L
        else:
            t += 1

    lam = lam_tmpl * (1.0 + z_l) * (1.0 + pulse)                 
    lam_hat = lam_tmpl * (1.0 + z_l_hat)
    Tamb = Tamb_tmpl + z_t                                         
    Tamb_hat = Tamb_tmpl + z_t_hat

    # ---- DCMG utility-grid capacity coefficient
    xi_g = np.clip(grid_coef + _trunc_normal(rng, T, sp.sig_grid, sp.eps_clip) * w
                   - np.abs(_trunc_normal(rng, T, sp.sig_grid / 2, sp.eps_clip)), 0.0, 1.0)
    xi_g_hat = grid_coef.copy()

    # ---- renewables: truncated daily curves with AR(1) relative noise
    sol = solar_shape(sp)

    def ar_noise(rho=0.7):
        e = _trunc_normal(rng, T, sp.sig_res, sp.eps_clip)
        x = np.zeros(T)
        for k in range(T):
            x[k] = (rho * x[k - 1] if k else 0.0) + e[k]
        return x

    res_D_hat = sp.P_RES_D_max * sol
    res_C_hat = sp.P_RES_C_max * sol
    res_D = np.clip(res_D_hat * (1 + ar_noise()), 0, sp.P_RES_D_max)
    res_C = np.clip(res_C_hat * (1 + ar_noise()), 0, sp.P_RES_C_max)

    ls = cmg_load_shape(sp)
    L_cr = sp.L_cr_range[0] + (sp.L_cr_range[1] - sp.L_cr_range[0]) * ls
    L_ba = sp.L_ba_range[0] + (sp.L_ba_range[1] - sp.L_ba_range[0]) * ls
    L_cr = L_cr * (1 + _trunc_normal(rng, T, 0.02, sp.eps_clip))
    L_ba = L_ba * (1 + _trunc_normal(rng, T, 0.03, sp.eps_clip))

    buy, sell = tou_prices(sp)
    return Day(lam=lam, lam_hat=lam_hat, lam_tmpl=lam_tmpl, Tamb=Tamb, Tamb_hat=Tamb_hat,
               Tamb_tmpl=Tamb_tmpl, zeta_lam=z_l, zeta_th=z_t, xi_g=xi_g, xi_g_hat=xi_g_hat,
               res_D=res_D, res_D_hat=res_D_hat, res_C=res_C, res_C_hat=res_C_hat,
               L_cr=L_cr, L_ba=L_ba, c_buy=buy, c_sell=sell, scenario=name, seed=seed)


def day_seed(pool: str, scenario: str, idx: int, base: int = 0) -> int:
    return (zlib.crc32(f"{pool}|{scenario}|{idx}|{base}".encode()) & 0x7FFFFFFF)


def make_pool(sp: SystemParams, pool: str, n_days: int, scenarios=None, base: int = 0,
              scen_dict: Optional[Dict] = None, name: str = "custom") -> List[Day]:
    """Create reproducible task-day pools.
    
    Choose uniformly from the supplied scenario names, or use scen_dict for
    every day. Custom scenarios share seeds across severity settings."""
    days = []
    names = scenarios or list(SCENARIOS.keys())
    for i in range(n_days):
        if scen_dict is not None:
            days.append(make_day(sp, scen_dict, day_seed(pool, "custom", i, base), name))
            continue
        else:
            rng = np.random.default_rng(day_seed(pool, "pick", i, base))
            name = names[rng.integers(len(names))] if len(names) > 1 else names[0]
            sc = SCENARIOS[name]
        days.append(make_day(sp, sc, day_seed(pool, name, i, base), name))
    return days


# ---------------- equipment states ----------------
@dataclass
class Equip:
    """Failure counts for servers, cooling units and IPCS units during each day."""
    nS_fail: np.ndarray    # failed servers
    nC_fail: np.ndarray    # failed cooling units
    nP_fail: np.ndarray    # failed IPCS units


def sample_equipment(sp: SystemParams, rng: np.random.Generator, n: int) -> Equip:
    return Equip(nS_fail=rng.binomial(sp.N_S, sp.p_S, n),
                 nC_fail=rng.binomial(sp.n_cool, sp.p_C, n),
                 nP_fail=rng.binomial(sp.n_ipcs, sp.p_IPCS, n))


def no_failure(n: int) -> Equip:
    z = np.zeros(n, dtype=int)
    return Equip(z.copy(), z.copy(), z.copy())


def equipment_signatures(eq: Equip):
    """Return unique equipment-failure counts and their multiplicities.
    
    For a deterministic policy and fixed task day, identical failure counts
    produce identical trajectories and can be simulated together."""
    keys = np.stack([eq.nS_fail, eq.nC_fail, eq.nP_fail], 1)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return uniq, counts


def stack_days(days: List[Day]) -> Dict[str, np.ndarray]:
    fields = ["lam", "lam_hat", "Tamb", "Tamb_hat", "xi_g", "xi_g_hat", "res_D", "res_D_hat",
              "res_C", "res_C_hat", "L_cr", "L_ba", "c_buy", "c_sell", "lam_tmpl", "Tamb_tmpl"]
    return {f: np.stack([getattr(d, f) for d in days], 0) for f in fields}


def forecast_horizon(sp: SystemParams, D: Dict[str, np.ndarray], b: int, t: int, H: int,
                     last_zl: float, last_zt: float, last_el: float):
    """Forecast workload and temperature using templates and decaying AR residuals."""
    h = _h(sp)
    idx = np.minimum(np.arange(t, t + H), sp.T - 1)
    k = np.arange(1, H + 1)
    hp = h[t] - sp.dt_h
    zl = mu_lam(h[idx]) + sp.rho_lam ** k * (last_zl - mu_lam(hp))
    zt = mu_th(h[idx]) + sp.rho_th ** k * (last_zt - mu_th(hp))
    zt[0] += sp.rho_lamth * last_el
    lam = D["lam_tmpl"][b, idx] * (1 + zl)
    Tamb = D["Tamb_tmpl"][b, idx] + zt
    return idx, lam, Tamb
