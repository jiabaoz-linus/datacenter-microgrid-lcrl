"""Forecast-based rolling-horizon MPC benchmark.

The controller solves a linear program with HiGHS and applies the first
decision. CPU utilisation is fixed at u_max, and server count is continuous
in the optimisation. Service-rate and temperature constraints include soft
slacks. The first server count is rounded up, with the additional demand
assigned to the grid command. The environment then processes the action.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .config import SystemParams
from . import physics as ph
from .pcm import U, N, PC, DG, PB, PG, C2D

# variable layout per slot
VN, VPC, VDG, VBD, VBC, VGP, VGM, VC2D, VSOC, VT, VST, VSI = range(12)
NV = 12


class MPCController:
    def __init__(self, sp: SystemParams, H: int = 24, soc_value: float = 0.60,
                 M_T: float = 5e4, M_IT: float = 50.0, T_backoff: float = 1.5, mu_backoff: float = 0.03):
        """Tighten temperature by T_backoff (degC) and the forecast service
        requirement by the relative margin mu_backoff."""
        self.sp, self.H = sp, H
        self.T_backoff, self.mu_backoff = T_backoff, mu_backoff
        self.soc_value = soc_value   # RMB/kWh value of terminal stored energy
        self.M_T, self.M_IT = M_T, M_IT

    def _forecasts(self, env, b, t):
        sp, D, H = self.sp, env.D, self.H
        idx = np.minimum(np.arange(t, t + H), sp.T - 1)
        k = np.arange(H)
        z_l = D["lam_hat"][b, t] / D["lam_tmpl"][b, t] - 1.0
        z_t = D["Tamb_hat"][b, t] - D["Tamb_tmpl"][b, t]
        lam = D["lam_tmpl"][b, idx] * (1 + z_l * sp.rho_lam ** k) * (1 + sp.buf_lam)
        Tamb = D["Tamb_tmpl"][b, idx] + z_t * sp.rho_th ** k + sp.buf_Tamb
        res = D["res_D_hat"][b, idx] * (1 - sp.buf_res)
        grid = D["xi_g_hat"][b, idx] * sp.P_grid_max * (1 - sp.buf_grid)
        c2d = ph.cmg_support_max(sp, np.full(H, env.SOC_C[b]), D["res_C_hat"][b, idx] * (1 - sp.buf_res),
                                 D["L_cr"][b, idx], D["L_ba"][b, idx])
        if env.pcc_mode == "none":
            c2d = np.zeros(H)
        return idx, lam, Tamb, res, grid, c2d, D["c_buy"][b, idx], D["c_sell"][b, idx]

    def solve_one(self, env, b, t):
        sp, H = self.sp, self.H
        idx, lam, Tamb, res, grid, c2d, cb, cs = self._forecasts(env, b, t)
        u = sp.u_max
        pit = sp.P_idle + (sp.P_peak - sp.P_idle) * u        # MW per active server
        nv = NV * H
        v = lambda k, j: k * NV + j
        c = np.zeros(nv)
        lb = np.zeros(nv)
        ub = np.full(nv, np.inf)
        kwh = sp.dt_h * 1e3
        cap_ipcs = min(env.P_ava[b], sp.P_IPCS_max)
        for k in range(H):
            c[v(k, VGP)] = cb[k] * kwh
            c[v(k, VGM)] = -cs[k] * kwh
            c[v(k, VDG)] = sp.c_DG * kwh
            c[v(k, VC2D)] = sp.c_PCC * kwh
            c[v(k, VBD)] = sp.c_B * kwh
            c[v(k, VBC)] = sp.c_B * kwh
            c[v(k, VST)] = self.M_T
            c[v(k, VSI)] = self.M_IT
            ub[v(k, VN)] = env.N_pool[b]
            ub[v(k, VPC)] = min(sp.P_cool_max, env.Q_ava[b] / sp.COP)
            ub[v(k, VDG)] = sp.P_DG_max
            ub[v(k, VBD)] = sp.P_B_D_max
            ub[v(k, VBC)] = sp.P_B_D_max
            ub[v(k, VGP)] = grid[k]
            ub[v(k, VGM)] = sp.P_sell_max
            ub[v(k, VC2D)] = c2d[k]
            lb[v(k, VSOC)], ub[v(k, VSOC)] = sp.SOC_D_min, sp.SOC_D_max
            lb[v(k, VT)], ub[v(k, VT)] = -np.inf, np.inf
        c[v(H - 1, VSOC)] -= self.soc_value * sp.E_B_D * 1e3

        n_eq = 3 * H
        Aeq = lil_matrix((n_eq, nv)); beq = np.zeros(n_eq)
        n_ub = 3 * H
        Aub = lil_matrix((n_ub, nv)); bub = np.zeros(n_ub)
        a_th = sp.dt_h / sp.C_th
        q0 = env.q[b]
        for k in range(H):
            # power balance: res + C2D + Gp - Gm + DG + Bd - Bc - (P_IT + Pc)/eta = P_base
            r = 3 * k
            for j, s in ((VC2D, 1), (VGP, 1), (VGM, -1), (VDG, 1), (VBD, 1), (VBC, -1)):
                Aeq[r, v(k, j)] = s
            Aeq[r, v(k, VN)] = -pit / sp.eta_IPCS
            Aeq[r, v(k, VPC)] = -1 / sp.eta_IPCS
            beq[r] = sp.P_base - res[k]
            # SOC dynamics
            Aeq[r + 1, v(k, VSOC)] = 1
            Aeq[r + 1, v(k, VBD)] = sp.dt_h / (sp.E_B_D * sp.eta_dis_D)
            Aeq[r + 1, v(k, VBC)] = -sp.dt_h * sp.eta_ch_D / sp.E_B_D
            if k == 0:
                beq[r + 1] = env.SOC_D[b]
            else:
                Aeq[r + 1, v(k - 1, VSOC)] = -1
            # Temperature dynamics with Q_act = COP * P_cool.
            # T_{k+1} = (1 - a/R) T_k + a[(alpha+kl) pit N + kl Pc + Qenv - COP Pc + Tamb/R]
            Aeq[r + 2, v(k, VT)] = 1
            Aeq[r + 2, v(k, VN)] = -a_th * (sp.alpha_h + sp.k_loss) * pit
            Aeq[r + 2, v(k, VPC)] = -a_th * (sp.k_loss - sp.COP)
            const = a_th * (sp.Q_env + Tamb[k] / sp.R_th)
            if k == 0:
                beq[r + 2] = (1 - a_th / sp.R_th) * env.T[b] + const
            else:
                Aeq[r + 2, v(k - 1, VT)] = -(1 - a_th / sp.R_th)
                beq[r + 2] = const
            # inequalities
            # Service requirement includes forecast backoff and the initial queue.
            Aub[r, v(k, VN)] = -sp.mu0 * u
            Aub[r, v(k, VSI)] = -1
            bub[r] = -(lam[k] * (1 + self.mu_backoff) + sp.q_res / sp.d_max + (q0 / sp.dt_min if k == 0 else 0.0))
            # temperature limit
            Aub[r + 1, v(k, VT)] = 1
            Aub[r + 1, v(k, VST)] = -1
            bub[r + 1] = sp.T_max - self.T_backoff
            # IPCS capacity
            Aub[r + 2, v(k, VN)] = pit
            Aub[r + 2, v(k, VPC)] = 1
            bub[r + 2] = cap_ipcs
        res_lp = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                         bounds=list(zip(lb, ub)), method="highs")
        a = np.zeros(7)
        if res_lp.status != 0:
            return None
        x = res_lp.x
        a[U] = u
        a[N] = min(np.ceil(x[v(0, VN)] - 1e-6), sp.N_S)
        a[PC] = x[v(0, VPC)]
        a[DG] = x[v(0, VDG)]
        a[PB] = x[v(0, VBD)] - x[v(0, VBC)]
        a[PG] = x[v(0, VGP)] - x[v(0, VGM)]
        a[C2D] = x[v(0, VC2D)]
        # rounding N up adds a little demand: absorb with the grid
        a[PG] += (a[N] - x[v(0, VN)]) * pit / sp.eta_IPCS
        return a

    def act(self, env):
        t = env.t
        out = np.zeros((env.B, 7))
        for b in range(env.B):
            a = self.solve_one(env, b, t)
            if a is None:   # Use the decoded zero policy output if the LP fails.
                a = env.decode(np.zeros((env.B, 7)), t)[b]
            out[b] = a
        return out
