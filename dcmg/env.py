"""Vectorised networked DCMG environment.

Each instance simulates a batch of task days and equipment states. A step
decodes the candidate action, applies PCM checking and correction, executes
the action under realised disturbances, evaluates functional success and
rewards, and updates the Lyapunov virtual queue."""
from typing import Dict, List, Optional
import numpy as np

from .config import SystemParams
from . import physics as ph
from .pcm import PCM, U, N, PC, DG, PB, PG, C2D
from .disturbance import Day, Equip, stack_days, no_failure

OBS_DIM = 15
ACT_DIM = 7
OBS_NAMES = ["q", "T", "SOC_D", "SOC_C", "P_DG_prev", "lam_hat", "Tamb_hat", "Pgrid_hat",
             "PC2D_hat", "c_buy", "c_sell", "rho_IT", "rho_C", "rho_IPCS", "Z"]


class DCMGEnv:
    def __init__(self, sp: SystemParams, use_pcm: bool = True, use_queue: bool = False,
                 pcc_mode: str = "adaptive", solver: str = "fast",
                 eta_Z: Optional[float] = None, g_bar: Optional[float] = None):
        assert pcc_mode in ("adaptive", "none", "rule")
        self.sp = sp
        self.use_queue = use_queue
        self.pcc_mode = pcc_mode
        self.pcm = PCM(sp, enabled=use_pcm, solver=solver)
        self.eta_Z = sp.eta_Z if eta_Z is None else eta_Z
        self.g_bar = sp.g_bar if g_bar is None else g_bar

    # ------------------------------------------------------------------
    def reset(self, days: List[Day], equip: Optional[Equip] = None):
        sp = self.sp
        self.D = stack_days(days)
        self.B = len(days)
        eq = equip if equip is not None else no_failure(self.B)
        self.N_pool = (sp.N_S - eq.nS_fail).astype(float)
        self.Q_ava = (sp.n_cool - eq.nC_fail) * sp.Q_cool_unit           
        self.P_ava = (sp.n_ipcs - eq.nP_fail) * sp.P_ipcs_unit           
        self.t = 0
        B = self.B
        self.q = np.zeros(B)
        self.T = np.full(B, sp.T0)
        self.SOC_D = np.full(B, sp.SOC_D0)
        self.SOC_C = np.full(B, sp.SOC_C0)
        self.DG_prev = np.zeros(B)
        self.Z = np.zeros(B)
        self.alive = np.ones(B, bool)       # task-period success so far
        return self._obs()

    # ------------------------------------------------------------------
    def _c2d_forecast(self, t, conservative=False):
        sp, D = self.sp, self.D
        res_C = D["res_C_hat"][:, t] * ((1 - sp.buf_res) if conservative else 1.0)
        c2d = ph.cmg_support_max(sp, self.SOC_C, res_C, D["L_cr"][:, t], D["L_ba"][:, t])
        if self.pcc_mode == "none":
            c2d = np.zeros_like(c2d)
        return c2d

    def _rule_c2d(self, t, c2d_max):
        """Request up to 0.30 MW of PCC support when forecast grid capacity is below 95%."""
        limited = self.D["xi_g_hat"][:, t] < 0.95
        return np.where(limited, np.minimum(0.30, c2d_max), 0.0)

    def _obs(self):
        sp, D = self.sp, self.D
        t = min(self.t, sp.T - 1)
        o = np.zeros((self.B, OBS_DIM), dtype=np.float32)
        o[:, 0] = np.clip(self.q / 2000.0, 0, 5)
        o[:, 1] = self.T / sp.T_max
        o[:, 2] = self.SOC_D
        o[:, 3] = self.SOC_C
        o[:, 4] = self.DG_prev / sp.P_DG_max
        o[:, 5] = D["lam_hat"][:, t] / sp.lam_peak
        o[:, 6] = D["Tamb_hat"][:, t] / 40.0
        o[:, 7] = D["xi_g_hat"][:, t]
        o[:, 8] = self._c2d_forecast(t) / sp.P_PCC_max
        o[:, 9] = D["c_buy"][:, t] / sp.c_peak
        o[:, 10] = D["c_sell"][:, t] / sp.c_peak
        o[:, 11] = self.N_pool / sp.N_S
        o[:, 12] = self.Q_ava / (sp.n_cool * sp.Q_cool_unit)
        o[:, 13] = self.P_ava / (sp.n_ipcs * sp.P_ipcs_unit)
        o[:, 14] = np.clip(self.Z / sp.Z_norm, 0, 10) if self.use_queue else 0.0
        return o

    # ------------------------------------------------------------------
    def checking_state(self, t):
        """Build the forecast state with conservative disturbance buffers."""
        sp, D = self.sp, self.D
        ctx = dict(q=self.q.copy(), T=self.T.copy(), SOC_D=self.SOC_D.copy(),
                   N_pool=self.N_pool, Q_ava=self.Q_ava, P_ava=self.P_ava,
                   lam=D["lam_hat"][:, t] * (1 + sp.buf_lam),
                   Tamb=D["Tamb_hat"][:, t] + sp.buf_Tamb,
                   res=D["res_D_hat"][:, t] * (1 - sp.buf_res),
                   grid=D["xi_g_hat"][:, t] * sp.P_grid_max * (1 - sp.buf_grid),
                   c2d=self._c2d_forecast(t, conservative=True))
        if self.pcc_mode == "rule":
            r = self._rule_c2d(t, ctx["c2d"])
            ctx["c2d"], ctx["c2d_lo"] = r, r
        return ctx

    def decode(self, a_norm, t):
        """Map a clipped policy output in [-1, 1]^7 to physical candidate actions.
        
        Grid exchange uses the forecast balancing residual plus a learned deviation
        within +/- G_offset_scale MW."""
        sp, D = self.sp, self.D
        a = np.clip(a_norm, -1.0, 1.0)
        x = (a + 1.0) / 2.0
        c2d_hat = self._c2d_forecast(t)
        out = np.zeros((self.B, 7))
        out[:, U] = np.clip(sp.u_min + x[:, 0] * (sp.u_max - sp.u_min), sp.u_min, sp.u_max)
        out[:, N] = np.round(x[:, 1] * sp.N_S)
        out[:, PC] = x[:, 2] * sp.P_cool_max
        out[:, DG] = x[:, 3] * sp.P_DG_max
        out[:, PB] = a[:, 4] * sp.P_B_D_max
        if self.pcc_mode == "adaptive":
            out[:, C2D] = x[:, 6] * c2d_hat
        elif self.pcc_mode == "rule":
            out[:, C2D] = self._rule_c2d(t, c2d_hat)
        N_act = ph.active_servers(out[:, N], self.N_pool)
        P_out = ph.it_power(sp, N_act, out[:, U]) + out[:, PC]
        resid = P_out / sp.eta_IPCS + sp.P_base - D["res_D_hat"][:, t] * (1 - sp.buf_res) \
            - out[:, DG] - out[:, PB] - out[:, C2D]
        out[:, PG] = resid + a[:, 5] * sp.G_offset_scale
        return out

    # ------------------------------------------------------------------
    def execute(self, a, t):
        """Execute the action under the realised disturbances of slot t."""
        sp, D = self.sp, self.D
        lam = D["lam"][:, t]
        Tamb = D["Tamb"][:, t]
        res = D["res_D"][:, t]
        cap_g = D["xi_g"][:, t] * sp.P_grid_max                                
        c2d_max = ph.cmg_support_max(sp, self.SOC_C, D["res_C"][:, t], D["L_cr"][:, t], D["L_ba"][:, t])
        if self.pcc_mode == "none":
            c2d_max = np.zeros_like(c2d_max)
        dis_lim, ch_lim = ph.battery_limits(self.SOC_D, sp.P_B_D_max, sp.E_B_D, sp.eta_ch_D,
                                            sp.eta_dis_D, sp.SOC_D_min, sp.SOC_D_max, sp.dt_h)
        u = np.clip(a[:, U], sp.u_min, sp.u_max)
        n = np.clip(np.round(a[:, N]), 0, sp.N_S)
        Pc = np.clip(a[:, PC], 0, sp.P_cool_max)
        Pdg = np.clip(a[:, DG], 0, sp.P_DG_max)
        Pb = np.clip(a[:, PB], -ch_lim, dis_lim)
        Pg = np.clip(a[:, PG], -sp.P_sell_max, cap_g)
        Pc2d = np.clip(a[:, C2D], 0, c2d_max)

        N_act = ph.active_servers(n, self.N_pool)
        P_IT = ph.it_power(sp, N_act, u)                                        
        mu = ph.service_rate(sp, N_act, u)                                      
        P_out_req = P_IT + Pc                                                   
        P_out_cap = np.minimum(P_out_req, self.P_ava)                           

        # actuator saturation to the physical limits of the realised slot
        sat = ((np.abs(u - a[:, U]) > 1e-9) | (np.abs(n - a[:, N]) > 1e-9) | (np.abs(Pc - a[:, PC]) > 1e-9)
               | (np.abs(Pdg - a[:, DG]) > 1e-9) | (np.abs(Pb - a[:, PB]) > 1e-9)
               | (np.abs(Pg - a[:, PG]) > 1e-9) | (np.abs(Pc2d - a[:, C2D]) > 1e-9))

        # ---- realised-slot power balance
        # Apply the same grid-balancing rule for every method and PCM setting.
        # Grid exchange can vary within its physical interface capacity and the
        # configured deviation band augmented by renewable forecast error.
        
        # The clipped diesel, battery and PCC powers remain fixed during balancing.
        # Remaining deficits reduce supplied IT and cooling power.
        # Remaining surplus is curtailed from renewables, then dissipated.
        demand = P_out_cap / sp.eta_IPCS + sp.P_base                            
        delta = demand - (res + Pc2d + Pg + Pdg + Pb)
        # Band = configured slack plus absolute renewable error relative to
        # the conservative checking forecast.
        res_err = np.abs(res - D["res_D_hat"][:, t] * (1 - sp.buf_res))
        band = np.inf if sp.rt_slack_max is None else sp.rt_slack_max + res_err
        Pg_cmd = Pg.copy()
        up = np.clip(delta, 0, np.minimum(band, np.maximum(cap_g - Pg, 0)))
        down = np.clip(-delta, 0, np.minimum(band, np.maximum(Pg + sp.P_sell_max, 0)))
        Pg = Pg + up - down
        deficit = np.maximum(delta - up, 0)
        surplus = np.maximum(-delta - down, 0)
        spill = np.minimum(surplus, res)                                        # renewable curtailment
        dumped = surplus - spill                                                # excess dispatch, wasted
        rt_adj = np.abs(Pg - Pg_cmd)

        # ---- propagate supply shortage to IT and cooling service
        P_out_sup = np.maximum(P_out_cap - deficit * sp.eta_IPCS, 0)
        kappa = np.where(P_out_req > 1e-9, P_out_sup / np.maximum(P_out_req, 1e-9), 1.0)
        mu_sup = kappa * mu                                                     
        Q_sup = sp.COP * kappa * Pc                                             
        Q_act = np.minimum(Q_sup, self.Q_ava)                                   

        q_next, d, Dt = ph.queue_and_delay(sp, self.q, lam, mu_sup)            
        Q_load = ph.heat_load(sp, kappa * P_IT, P_out_sup)                      
        Q_req = ph.heat_required(sp, Q_load, Tamb, self.T)                      
        T_next = ph.temp_next(sp, self.T, Q_load, Q_act, Tamb)                  
        SOC_D_next = ph.soc_next(self.SOC_D, Pb, sp.E_B_D, sp.eta_ch_D, sp.eta_dis_D, sp.dt_h)
        PB_C, Pg_C, SOC_C_next = ph.cmg_dispatch(sp, self.SOC_C, D["res_C"][:, t], D["L_cr"][:, t],
                                                 D["L_ba"][:, t], Pc2d, D["c_buy"][:, t])

        # ---- slot cost: convert MW and hours to kWh
        k = sp.dt_h * 1e3
        Gp, Gm = np.maximum(Pg, 0), np.maximum(-Pg, 0)                          
        cost = (D["c_buy"][:, t] * Gp - D["c_sell"][:, t] * Gm + sp.c_DG * Pdg
                + sp.c_PCC * Pc2d + sp.c_B * np.abs(Pb)) * k

        # ---- functional margins
        cap_ipcs = np.minimum(self.P_ava, sp.P_IPCS_max)
        m_IT = sp.d_max - d
        m_C = sp.T_max - T_next
        m_P = cap_ipcs - P_out_req

        # ---- success events
        k_IT = np.ceil(lam / (sp.mu0 * sp.u_max))
        E_dev_IT = self.N_pool >= k_IT                                          
        E_op_IT = d <= sp.d_max                                                 
        E_dev_C = self.Q_ava >= Q_req - 1e-9                                    
        E_op_C = T_next <= sp.T_max + 1e-9                                      
        E_dev_P = self.P_ava >= P_out_req - 1e-9                                
        E_op_P = P_out_req <= sp.P_IPCS_max + 1e-9                              
        # Supply adequacy: the IPCS power-supply function also fails when the
        # upstream supply cannot deliver the requested input (shortfall above the
        # balance tolerance), even if IPCS capacity itself is sufficient.
        E_sup_P = deficit <= sp.bal_tol + 1e-9
        m_sup = sp.bal_tol - deficit                                            # supply margin, MW

        ex = dict(u=u, n=n, P_cool=Pc, P_DG=Pdg, P_B=Pb, P_G=Pg, P_C2D=Pc2d)
        self.q, self.T, self.SOC_D, self.SOC_C, self.DG_prev = q_next, T_next, SOC_D_next, SOC_C_next, Pdg
        return dict(cost=cost, m_IT=m_IT, m_C=m_C, m_P=m_P, d=d, T_next=T_next, q_next=q_next,
                    P_IT=P_IT, P_out=P_out_req, P_out_sup=P_out_sup, kappa=kappa, deficit=deficit,
                    spill=spill, dumped=dumped, rt_adj=rt_adj, saturated=sat, P_C2D_max=c2d_max, grid_cap=cap_g, Q_req=Q_req, Q_act=Q_act,
                    E_dev_IT=E_dev_IT, E_op_IT=E_op_IT, E_dev_C=E_dev_C, E_op_C=E_op_C,
                    E_dev_P=E_dev_P, E_op_P=E_op_P & E_sup_P, E_sup_P=E_sup_P, m_sup=m_sup,
                    F_IT=E_dev_IT & E_op_IT, F_C=E_dev_C & E_op_C, F_P=E_dev_P & E_op_P & E_sup_P,
                    SOC_D=SOC_D_next, SOC_C=SOC_C_next, P_B_C=PB_C, P_G_C=Pg_C, exec=ex)

    # ------------------------------------------------------------------
    def step(self, a_norm=None, a_phys=None):
        """Advance one slot using normalised or physical candidate actions.
        
        If a_phys is supplied it takes precedence over a_norm."""
        sp, t = self.sp, self.t
        a0 = self.decode(a_norm, t) if a_phys is None else a_phys
        ctx = self.checking_state(t)
        a_exec, g, corrected, sigma, h = self.pcm(a0, ctx)
        info = self.execute(a_exec, t)

        r_rel = (sp.beta_IT * ph.saturate(info["m_IT"], sp.m_ref_IT)
                 + sp.beta_C * ph.saturate(info["m_C"], sp.m_ref_C)
                 + sp.beta_IPCS * ph.saturate(np.minimum(info["m_P"], info["m_sup"]), sp.m_ref_IPCS))  
        r_viol = np.zeros_like(r_rel)
        for m, ref in ((info["m_IT"], sp.m_ref_IT), (info["m_C"], sp.m_ref_C), (np.minimum(info["m_P"], info["m_sup"]), sp.m_ref_IPCS)):
            r_viol += sp.beta_fail * (m < 0) + sp.beta_viol * np.clip(-m / ref, 0.0, sp.viol_cap)
        r0 = r_rel - sp.beta_cost * info["cost"] / sp.C0 - r_viol               # reliability reward minus cost and negative-margin penalties
        Zt = self.Z.copy()
        r_tilde = r0 - self.eta_Z * Zt * g if self.use_queue else r0            
        self.Z = np.maximum(self.Z + g - self.g_bar, 0.0)                       

        slot_ok = info["F_IT"] & info["F_C"] & info["F_P"]
        self.alive &= slot_ok
        # corrected records the PCM infeasibility flag when correction is enabled.
        # modified also includes actuator saturation or grid balancing above
        # the power-balance tolerance. The grid-adjustment magnitude is
        # recorded separately as rt_adj.
        modified = corrected | info["saturated"] | (info["rt_adj"] > sp.bal_tol + 1e-9)
        info.update(r0=r0, r_rel=r_rel, r_viol=r_viol, reward=r_tilde, g=g, corrected=corrected, modified=modified, sigma=sigma,
                    Z=Zt, Z_next=self.Z.copy(), a0=a0, a_exec=a_exec, slot_ok=slot_ok)
        self.t += 1
        done = self.t >= sp.T
        return self._obs(), r_tilde.astype(np.float32), done, info
