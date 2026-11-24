"""Plots for training, reliability, PCC support and parameter studies."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"LCRL": "#d62728", "PPO-Lag": "#1f77b4", "CPO": "#2ca02c", "SAC": "#9467bd", "TD3": "#8c564b",
          "DDPG": "#e377c2", "PPO": "#7f7f7f", "TRPO": "#bcbd22", "MPC": "#17becf"}


def _smooth(x, k=15):
    if len(x) < k:
        return np.asarray(x)
    return pd.Series(x).rolling(k, min_periods=1).mean().values


def fig3_training(logs, path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    keys = [("cum_reward", "(a) Cumulative reward per task day (sum of $r_t^0$)"),
            ("mean_g", "(b) Mean candidate-action violation degree $g_t$"),
            ("Z_T", "(c) End-of-episode virtual queue $Z_T$"),
            ("time_s", "(d) Computation time per training episode (s)")]
    for (k, title), a in zip(keys, ax.ravel()):
        if k == "time_s":
            names = list(logs)
            vals = [np.mean([d[k].mean() for d in logs[m]]) for m in names]
            a.bar(names, vals, color=[COLORS.get(m, "k") for m in names])
            a.tick_params(axis="x", rotation=45)
        else:
            for m, L in logs.items():
                n = min(len(d) for d in L)
                y = np.mean([d[k].values[:n] for d in L], 0)
                a.plot(np.arange(1, n + 1), _smooth(y), label=m, color=COLORS.get(m), lw=1.6 if m == "LCRL" else 1)
            a.set_xlabel("Episode")
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig4_normal(per, path):
    names = list(per)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, col, title in zip(ax, ["cost", "R_sys", "g"],
                             ["(a) Daily operating cost (RMB)", "(b) Task-period reliability $R_{sys}$ per day",
                              "(c) Mean $g_t$ per day"]):
        bp = a.boxplot([per[m][col].values for m in names], labels=names, patch_artist=True, showfliers=False)
        for p, m in zip(bp["boxes"], names):
            p.set_facecolor(COLORS.get(m, "w")); p.set_alpha(0.6)
        a.set_title(title, fontsize=10); a.tick_params(axis="x", rotation=45); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig5_pcc(res, path):
    names = list(res)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, col, title in zip(ax, ["cost", "R_sys", "peak_ipcs"],
                             ["(a) Daily operating cost (RMB)", "(b) $R_{sys}$ per test day",
                              "(c) Peak IPCS loading ratio"]):
        a.boxplot([res[m][col].values for m in names], labels=names, showfliers=False)
        a.set_title(title, fontsize=10); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig6_traj(trajs, sp, path):
    h = (np.arange(sp.T) + 0.5) * sp.dt_h
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for m, tr in trajs.items():
        ax[0].plot(h, tr["P_C2D"] * 1e3, label=m)
        ax[1].plot(h, tr["T"], label=m)
        ax[2].plot(h, tr["P_out"] * 1e3, label=m)
    ax[1].axhline(sp.T_max, ls="--", c="k", lw=0.8)
    ax[2].axhline(sp.P_IPCS_max * 1e3, ls="--", c="k", lw=0.8)
    for a, yl in zip(ax, ["$P^{C2D}_t$ (kW)", "Hotspot $T_t$ (°C)", "$P^{out}_{IPCS,t}$ (kW)"]):
        a.axvspan(11, 16, color="gold", alpha=0.2)
        a.set_ylabel(yl); a.grid(alpha=0.3)
    ax[0].legend(); ax[2].set_xlabel("Hour of day")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig7_radar(t7, path):
    cols = ["R_sys", "Cost (RMB/day)", "Mean g_t", "m_IT (min)", "m_C (degC)", "m_IPCS (kW)"]
    df = t7.copy()
    df["R_sys"] = df["R_sys [95% CI]"].str.split(" ").str[0].astype(float)
    vals = df[cols].values.astype(float)
    # Scale metrics to [0.2, 1], reversing cost and violation degree.
    lo, hi = vals.min(0), vals.max(0)
    nrm = (vals - lo) / np.where(hi > lo, hi - lo, 1)
    nrm[:, 1] = 1 - nrm[:, 1]; nrm[:, 2] = 1 - nrm[:, 2]
    nrm = 0.2 + 0.8 * nrm
    ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    fig = plt.figure(figsize=(6, 6)); a = plt.subplot(111, polar=True)
    for i, v in enumerate(df["Variant"]):
        r = nrm[i].tolist()
        a.plot(ang + ang[:1], r + r[:1], label=v); a.fill(ang + ang[:1], r + r[:1], alpha=0.1)
    labels = ["$R_{sys}$", "Cost (inv.)", "$g_t$ (inv.)", "$m_{IT}$", "$m_C$", "$m_{IPCS}$"]
    a.set_xticks(ang); a.set_xticklabels(labels); a.set_yticklabels([])
    a.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig8_sens(t, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col, title in zip(ax, ["cost", "R_sys", "g"], ["Daily cost (RMB)", "$R_{sys}$", "Mean $g_t$"]):
        p = t.pivot(index="eta_Z", columns="g_bar", values=col)
        im = a.imshow(p.values, aspect="auto", origin="lower", cmap="viridis")
        a.set_xticks(range(len(p.columns))); a.set_xticklabels([f"{c:.3f}" for c in p.columns])
        a.set_yticks(range(len(p.index))); a.set_yticklabels([f"{e:.2f}" for e in p.index])
        a.set_xlabel(r"$\bar g$"); a.set_ylabel(r"$\eta_Z$"); a.set_title(title)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                a.text(j, i, f"{p.values[i, j]:.3f}" if col != "cost" else f"{p.values[i, j]:.0f}",
                       ha="center", va="center", fontsize=7, color="w")
        fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig9_severity(t, path):
    g = t.groupby(["method", "xi"])[["R_sys", "g"]].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, d in g.groupby("method"):
        ax[0].plot(d.xi, d.R_sys, "o-", label=m, color=COLORS.get(m))
        ax[1].plot(d.xi, d.g, "o-", label=m, color=COLORS.get(m))
    ax[0].set_title("(a) Task-period reliability"); ax[1].set_title("(b) Mean candidate-action violation degree")
    for a in ax:
        a.set_xlabel(r"Disturbance severity index $\xi$"); a.grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
