"""Model, scenario and training parameters.

Power is expressed in MW, energy in MWh, temperature in degrees Celsius,
queueing time in minutes, energy-accounting time in hours and cost in RMB."""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple


@dataclass
class SystemParams:
    # ---------------- time ----------------
    dt_min: float = 15.0          # scheduling interval, min
    T: int = 96                   # slots per task day

    # ---------------- IT layer ----------------
    N_S: int = 420                # server fleet size
    P_idle: float = 0.80e-3       # idle power per server, MW
    P_peak: float = 2.20e-3       # peak power per server, MW
    u_min: float = 0.30           # minimum CPU utilisation
    u_max: float = 0.90
    mu0: float = 28.0             # tasks/min per server at full utilisation
    lam_mean: float = 6200.0      # baseline mean arrival rate, tasks/min
    lam_peak: float = 8200.0      # baseline peak arrival rate, tasks/min
    d_max: float = 1.8            # waiting-delay limit, min
    eps_d: float = 1e-3           # delay denominator lower bound
    q_res: float = 100.0          # constant backlog added to the delay numerator, tasks
    P_base: float = 0.05          # base load supplied outside the IPCS, MW

    # ---------------- cooling layer ----------------
    P_cool_max: float = 0.35      # maximum cooling electrical power, MW
    COP: float = 3.0              # effective cooling coefficient of performance
    n_cool: int = 4               # number of cooling units
    Q_cool_unit: float = 0.30
    alpha_h: float = 0.85         # share of IT power entering the hotspot thermal node
    Q_env: float = 0.02           # external heat input, MW
    R_th: float = 50.0            # hotspot thermal resistance, degC/MW
    C_th: float = 0.015           # hotspot thermal capacitance, MWh/degC
    T0: float = 31.0              # initial hotspot temperature, degC
    T_max: float = 38.0           # hotspot temperature limit, degC
    T_amb_min: float = 24.0       # minimum baseline ambient temperature, degC
    T_amb_max: float = 32.0

    # ---------------- IPCS ----------------
    P_IPCS_max: float = 1.40      # IPCS operating output limit, MW
    n_ipcs: int = 4               # number of IPCS units
    P_ipcs_unit: float = 0.40
    eta_IPCS: float = 0.965       # IPCS conversion efficiency

    # ---------------- DCMG supply ----------------
    E_B_D: float = 1.20           # DCMG battery capacity, MWh
    P_B_D_max: float = 1.00       # DCMG battery charge/discharge power limit, MW
    SOC_D0: float = 0.60          # initial DCMG storage SOC
    SOC_D_min: float = 0.15
    SOC_D_max: float = 0.90
    eta_ch_D: float = 0.95        # DCMG charging efficiency
    eta_dis_D: float = 0.95       # DCMG discharging efficiency
    P_DG_max: float = 0.90        # diesel-generator power limit, MW
    P_RES_D_max: float = 0.30     # DCMG renewable-power limit, MW
    P_grid_max: float = 1.00      # rated DCMG grid-import capacity, MW
    P_sell_max: float = 0.50      # grid-export power limit, MW
    P_PCC_max: float = 0.65       # PCC support power limit, MW

    # ---------------- CMG ----------------
    E_B_C: float = 1.00           # CMG battery capacity, MWh
    P_B_C_max: float = 0.60       # CMG battery power limit, MW
    SOC_C0: float = 0.60
    SOC_C_min: float = 0.20
    SOC_C_max: float = 0.90
    eta_ch_C: float = 0.95        # CMG charging efficiency
    eta_dis_C: float = 0.95       # CMG discharging efficiency
    P_RES_C_max: float = 0.55     # CMG renewable-power limit, MW
    L_cr_range: Tuple[float, float] = (0.35, 0.55)   # CMG critical-load range, MW
    L_ba_range: Tuple[float, float] = (0.25, 0.45)   # CMG general-load range, MW
    P_grid_C_max: float = 1.20    # CMG grid-import capacity, MW

    # ---------------- equipment failures ----------------
    p_S: float = 0.002            # server failure probability per task day
    p_C: float = 0.006
    p_IPCS: float = 0.004
    N_MC: int = 200               # equipment-state draws per test day

    # ---------------- prices (RMB/kWh) ----------------
    c_valley: float = 0.40        # 00:00-08:00
    c_flat: float = 0.80          # flat-period purchase price
    c_peak: float = 1.25          # 11:00-15:00 and 18:00-21:00
    c_sell: float = 0.35          # grid-export price
    c_DG: float = 1.60            # diesel-generation price
    c_PCC: float = 0.80           # PCC support price
    c_B: float = 0.05             # storage degradation per kWh of throughput

    # ---------------- disturbance process ----------------
    rho_lam: float = 0.65         # workload autoregression coefficient
    rho_th: float = 0.75          # temperature autoregression coefficient
    rho_lamth: float = 0.10       # delayed workload-to-temperature cross coefficient
    sig_lam: float = 0.015        # workload innovation scale, relative
    sig_th: float = 0.45          # temperature innovation scale, degC
    eps_clip: float = 2.5         # normal-draw clipping threshold before scaling
    pulse_rate: float = 0.01      # pulse-start probability at each eligible slot
    pulse_amp: Tuple[float, float] = (0.02, 0.04)   # relative pulse-amplitude range
    pulse_len: Tuple[int, int] = (1, 3)             # pulse-duration range, slots
    sig_res: float = 0.10         # relative renewable innovation scale
    sig_grid: float = 0.02        # grid-capacity innovation scale

    # ---------------- PCM conservative buffer ----------------
    buf_lam: float = 0.05         # relative upward workload buffer
    buf_Tamb: float = 0.6         # upward ambient-temperature buffer, degC
    buf_res: float = 0.10         # relative downward renewable-output buffer
    buf_grid: float = 0.05        # relative downward grid-capacity buffer
    bal_tol: float = 0.02         # power-balance tolerance, MW
    # Grid-balancing band, MW, augmented by the realised renewable forecast error.
    # Grid import and export remain limited by the available interface capacity.
    # A remaining deficit reduces supply; surplus is curtailed or dissipated.
    # None removes the deviation-band limit while retaining interface limits.
    # Actuator commands are clipped before the grid-balancing stage.
    rt_slack_max: Optional[float] = 0.05
    G_offset_scale: float = 0.05  # learned grid-power deviation range, MW

    # ---------------- margins / reward ----------------
    m_ref_IT: float = 0.40        # IT margin normalisation scale, min
    m_ref_C: float = 3.0
    m_ref_IPCS: float = 0.150     # IPCS margin normalisation scale, MW
    beta_IT: float = 1.0          # IT reward weight
    beta_C: float = 1.0
    beta_IPCS: float = 1.0
    beta_cost: float = 1.25       # normalised cost weight
    # Negative-margin penalty applied to each functional margin:
    #   r_viol = sum_l [ beta_fail * 1{m_l < 0} + beta_viol * min(-m_l / m_ref_l, viol_cap)_+ ]
    # beta_fail weights the negative-margin indicator.
    # beta_viol weights the clipped normalised shortfall.
    # viol_cap limits each normalised shortfall contribution.
    beta_fail: float = 1.0
    beta_viol: float = 1.0
    viol_cap: float = 3.0
    C0: float = 160.0             # cost normalisation scale, RMB per slot

    # ---------------- violation degree ----------------
    alpha1: float = 0.45          # constraint-excess weight
    alpha2: float = 0.35
    alpha3: float = 0.20
    g_max: float = 1.0            # violation-degree cap
    slack_w_factor: float = 4.0   # slack weight factor divided by each margin scale

    # ---------------- Lyapunov queue ----------------
    eta_Z: float = 1.60           # queue-penalty weight
    g_bar: float = 0.055          # virtual-queue violation target
    Z_norm: float = 1.0           # virtual-queue observation scale

    @property
    def dt_h(self) -> float:
        return self.dt_min / 60.0

    @property
    def k_loss(self) -> float:            # (1/eta - 1)
        return 1.0 / self.eta_IPCS - 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------- disturbance scenarios ----------------
# Each scenario specifies a task multiplier, temperature rise, grid coefficient, window and test-day count.
SCENARIOS: Dict[str, Dict] = {
    "Normal":       dict(task=1.00, dT=0.0, grid=1.00, win=None,          n_test=56),
    "Task burst":   dict(task=1.16, dT=0.5, grid=0.97, win=(11.0, 15.0),  n_test=60),
    "High ambient": dict(task=1.06, dT=4.3, grid=0.97, win=(11.0, 17.0),  n_test=60),
    "Grid limit":   dict(task=1.08, dT=0.8, grid=0.72, win=(11.5, 15.75), n_test=64),
    "Combined":     dict(task=1.21, dT=4.5, grid=0.64, win=(11.0, 16.0),  n_test=68),
}
SCENARIO_NAMES = list(SCENARIOS.keys())


def severity_scenario(xi: float) -> Dict:
    """Construct a compound-disturbance scenario from severity index xi."""
    return dict(task=1.0 + 0.021 * xi, dT=0.45 * xi, grid=1.0 - 0.036 * xi,
                win=(11.0, 16.0) if xi > 0 else None, n_test=68)


@dataclass
class TrainParams:
    # Network and policy-update parameters
    hidden: int = 128
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.20
    epochs: int = 10
    batch_size: int = 1024
    # Training budget and optimisation settings
    n_envs: int = 16              # parallel task days per training iteration
    n_episodes: int = 1200        # training iterations
    minibatch: int = 256
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03       # KL threshold for repeated policy updates
    max_reruns: int = 2
    adapt_eta: bool = False       # increase eta_Z when mean violation exceeds g_bar
    init_log_std: float = -1.0
    # off-policy
    buffer_size: int = 200_000
    off_batch: int = 256
    tau: float = 0.005
    updates_per_step: int = 1
    warmup_steps: int = 2_000
    # Training and validation day pools
    n_train_days: int = 800
    n_val_days: int = 160
    seed: int = 1101


def quick_overrides(tp: TrainParams, sp: SystemParams):
    """Reduce the training and sampling budgets for a workflow check."""
    tp.n_episodes = 20
    tp.n_envs = 4
    tp.epochs = 3
    tp.warmup_steps = 200
    tp.n_train_days = 40
    tp.n_val_days = 8
    sp.N_MC = 20
    return tp, sp
