"""
Command-line interface for training, evaluation and experiment summaries.

  python run.py train     [--methods ...] [--seeds ...]      train RL policies
  python run.py eval      [--methods ...] [--seeds ...]      evaluate policies by Monte Carlo sampling
  python run.py tables                                     summarize evaluation results
  python run.py fig3                                       plot training curves and time action correction
  python run.py pcc                                        compare PCC support modes
  python run.py sens                                       evaluate the eta_Z and g_bar parameter grid
  python run.py severity                                   evaluate disturbance severity xi = 0..10
  python run.py tune      --methods X                      search hyperparameters on validation days
  python run.py all                                        run train through severity in the order above

Use --quick for reduced experiment budgets and --jobs N for parallel training.
"""
import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from dcmg.config import SystemParams, TrainParams, SCENARIOS, SCENARIO_NAMES, severity_scenario, quick_overrides
from dcmg.train import METHODS, RL_BASELINES, train, load_agent, resolve, run_dir
from dcmg.disturbance import make_pool
from dcmg.evaluate import evaluate, summarize
from dcmg.bootstrap import hierarchical_bootstrap
from dcmg import plots

MAIN_METHODS = ["LCRL"] + RL_BASELINES
ABLATION = ["Base", "+PCM", "+Lyapunov", "LCRL"]
DEFAULT_SEEDS = list(range(1101, 1111))


# ---------------------------------------------------------------------------
def setup(args):
    sp, tp = SystemParams(), TrainParams()
    if args.quick:
        tp, sp = quick_overrides(tp, sp)
    if args.episodes:
        tp.n_episodes = args.episodes
    os.makedirs(args.out, exist_ok=True)
    for sub in ("runs", "eval", "tables", "figures"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    return sp, tp


def n_test(args, scen):
    return 4 if args.quick else SCENARIOS[scen]["n_test"]


def seeds_of(args):
    return args.seeds or ([1101] if args.quick else DEFAULT_SEEDS)


def _train_job(job):
    import torch
    torch.set_num_threads(1)
    method, seed, sp, tp, root, kw = job
    path = os.path.join(run_dir(root, method, seed, kw.get("tag", "")), "model.pt")
    if os.path.exists(path) and not kw.pop("force", False):
        print(f"skip {method}{kw.get('tag', '')} seed {seed} (exists)")
        return
    kw.pop("force", None)
    train(method, seed, sp, tp, root, **kw)


def run_jobs(jobs, n_jobs):
    if n_jobs <= 1:
        for j in jobs:
            _train_job(j)
    else:
        with Pool(n_jobs) as p:
            p.map(_train_job, jobs)


def eval_path(args, method, seed, scen, tag=""):
    s = scen.replace(" ", "_")
    return os.path.join(args.out, "eval", f"{resolve(method)}{tag}_s{seed}_{s}.csv")


def run_eval(args, sp, tp, method, seed, scen, days, tag="", **kw):
    path = eval_path(args, method, seed, scen, tag)
    if os.path.exists(path) and not args.force:
        return pd.read_csv(path)
    agent = load_agent(sp, tp, method, seed, os.path.join(args.out, "runs"), tag)
    kw.setdefault("solver", args.solver)
    df, traj = evaluate(sp, method, agent, days, equip_base=0, **kw)
    df["method"], df["seed"] = resolve(method) + tag, seed
    df.to_csv(path, index=False)
    if traj is not None:
        np.savez(path.replace(".csv", "_traj.npz"), **traj)
    return df


# ---------------------------------------------------------------------------
def cmd_train(args):
    sp, tp = setup(args)
    methods = [resolve(m) for m in (args.methods or MAIN_METHODS + ["Base", "+Lyapunov"])]
    jobs = [(m, s, sp, tp, os.path.join(args.out, "runs"), dict(force=args.force))
            for m in methods if METHODS[m]["kind"] != "mpc" for s in seeds_of(args)]
    run_jobs(jobs, args.jobs)


def cmd_eval(args):
    sp, tp = setup(args)
    methods = args.methods or MAIN_METHODS + ["Base", "+Lyapunov", "MPC"]
    for scen in args.scenarios or SCENARIO_NAMES:
        days = make_pool(sp, "test", n_test(args, scen), [scen])
        for m in methods:
            seeds = [seeds_of(args)[0]] if METHODS[resolve(m)]["kind"] == "mpc" else seeds_of(args)
            for s in seeds:
                t0 = time.time()
                df = run_eval(args, sp, tp, m, s, scen, days)
                print(f"eval {m:10s} seed {s} {scen:12s} R_sys {df.R_sys.mean():.4f} "
                      f"cost {df.cost.mean():8.1f} g {df.g.mean():.4f} ({time.time() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
def load_eval(args, method, scen, tag=""):
    """Load available per-seed evaluation tables; load one table for MPC."""
    seeds = seeds_of(args)
    if METHODS.get(resolve(method), {}).get("kind") == "mpc":
        seeds = seeds[:1]
    dfs = []
    for s in seeds:
        p = eval_path(args, method, s, scen, tag)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    return dfs


def mat(dfs, col):
    return np.stack([d.sort_values("day")[col].values for d in dfs], 0)


def fmt_ci(v):
    return f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"


def cmd_tables(args):
    sp, tp = setup(args)
    tdir = os.path.join(args.out, "tables")
    # Cost and reliability across scenarios, averaged over seeds and days.
    rows = []
    for m in MAIN_METHODS:
        for scen in SCENARIO_NAMES:
            dfs = load_eval(args, m, scen)
            if not dfs:
                continue
            d = pd.concat(dfs)
            s = summarize(d)
            rows.append(dict(Algorithm=m, Scenario=scen, Cost=s["cost"], R_IT=s["R_IT"], R_C=s["R_C"],
                             R_P=s["R_P"], R_sys=s["R_sys"], mean_g=s["g"], f_corr=s["f_corr"],
                             f_mod=s["f_mod"], rt_kWh=s["rt_kWh"], deficit_kWh=s["deficit_kWh"]))
    t4 = pd.DataFrame(rows)
    if len(t4):
        t4.to_csv(os.path.join(tdir, "table4_long.csv"), index=False)
        wide = t4.melt(id_vars=["Algorithm", "Scenario"], value_vars=["Cost", "R_IT", "R_C", "R_P", "R_sys", "mean_g"],
                       var_name="Metric").pivot_table(index=["Algorithm", "Metric"], columns="Scenario", values="value")
        wide = wide.reindex(columns=[s for s in SCENARIO_NAMES if s in wide.columns])
        order = [m for m in MAIN_METHODS if m in wide.index.get_level_values(0)]
        wide = wide.reindex(order, level=0)
        wide.to_csv(os.path.join(tdir, "table4.csv"))
        with open(os.path.join(tdir, "table4.md"), "w") as f:
            f.write(wide.round(4).to_markdown())
        print("Table 4\n", wide.round(4))

    # Matched method comparison under Combined with bootstrap intervals.
    names = [m for m in ["LCRL", "PPO-Lag", "CPO", "MPC", "PPO", "TRPO", "SAC", "TD3", "DDPG"]
             if load_eval(args, m, "Combined")]
    if names:
        S = len(load_eval(args, "LCRL", "Combined"))
        R = {m: mat(load_eval(args, m, "Combined")[:S], "R_sys") for m in names}
        ci, diff = hierarchical_bootstrap(R, n_boot=args.n_boot)
        rows = []
        for m in names:
            dfs = load_eval(args, m, "Combined")[:S]
            rows.append(dict(Method=m, n_runs=len(dfs), R_sys_CI=fmt_ci(ci[m]), Cost=mat(dfs, "cost").mean(),
                             mean_g=mat(dfs, "g").mean(),
                             diff_vs_LCRL=("" if m == "LCRL" else fmt_ci(diff[m]))))
        t5 = pd.DataFrame(rows)
        t5.to_csv(os.path.join(tdir, "table5.csv"), index=False)
        with open(os.path.join(tdir, "table5.md"), "w") as f:
            f.write(t5.to_markdown(index=False, floatfmt=".4f"))
        print("Table 5\n", t5)

    # LCRL reliability and disjoint equipment and operational failure components.
    rows = []
    for scen in SCENARIO_NAMES:
        dfs = load_eval(args, "LCRL", scen)
        if not dfs:
            continue
        ci, _ = hierarchical_bootstrap({"LCRL": mat(dfs, "R_sys")}, n_boot=args.n_boot)
        a, b = mat(dfs, "P_devc").mean() * 100, mat(dfs, "P_dev_opc").mean() * 100
        rows.append({"Scenario": scen, "R_sys [95% CI]": fmt_ci(ci["LCRL"]), "P(E_dev^c) %": a,
                     "P(E_dev & E_op^c) %": b, "1-R_sys %": (1 - mat(dfs, "R_sys").mean()) * 100,
                     "check: sum %": a + b})
    if rows:
        t6 = pd.DataFrame(rows)
        t6.to_csv(os.path.join(tdir, "table6.csv"), index=False)
        with open(os.path.join(tdir, "table6.md"), "w") as f:
            f.write(t6.to_markdown(index=False, floatfmt=".4f"))
        print("Table 6\n", t6)

    # Ablation comparison under Combined.
    names = [m for m in ABLATION if load_eval(args, m, "Combined")]
    if names:
        S = min(len(load_eval(args, m, "Combined")) for m in names)
        R = {m: mat(load_eval(args, m, "Combined")[:S], "R_sys") for m in names}
        ci, _ = hierarchical_bootstrap(R, n_boot=args.n_boot)
        rows = []
        for m in names:
            dfs = load_eval(args, m, "Combined")[:S]
            rows.append({"Variant": m, "Cost (RMB/day)": mat(dfs, "cost").mean(), "R_sys [95% CI]": fmt_ci(ci[m]),
                         "Mean g_t": mat(dfs, "g").mean(), "m_IT (min)": mat(dfs, "m_IT").mean(),
                         "m_C (degC)": mat(dfs, "m_C").mean(), "m_IPCS (kW)": mat(dfs, "m_P").mean()})
        t7 = pd.DataFrame(rows)
        t7.to_csv(os.path.join(tdir, "table7.csv"), index=False)
        with open(os.path.join(tdir, "table7.md"), "w") as f:
            f.write(t7.to_markdown(index=False, floatfmt=".4f"))
        print("Table 7\n", t7)
        plots.fig7_radar(t7, os.path.join(args.out, "figures", "fig7_ablation_radar.png"))

    # Per-day distributions under Normal.
    per = {m: pd.concat(load_eval(args, m, "Normal")) for m in MAIN_METHODS if load_eval(args, m, "Normal")}
    if per:
        plots.fig4_normal(per, os.path.join(args.out, "figures", "fig4_normal.png"))


# ---------------------------------------------------------------------------
def cmd_fig3(args):
    sp, tp = setup(args)
    logs = {}
    for m in MAIN_METHODS:
        L = []
        for s in seeds_of(args):
            p = os.path.join(run_dir(os.path.join(args.out, "runs"), m, s), "log.csv")
            if os.path.exists(p):
                L.append(pd.read_csv(p))
        if L:
            logs[m] = L
    if logs:
        plots.fig3_training(logs, os.path.join(args.out, "figures", "fig3_training.png"))
    # Compare per-slot action-correction solve times and objective values.
    from dcmg.timing import correction_timing
    n_per = 20 if args.quick else 200
    res = correction_timing(sp, tp, os.path.join(args.out, "runs"), seeds_of(args)[0], n_per_scenario=n_per,
                            quick=args.quick)
    with open(os.path.join(args.out, "tables", "correction_timing.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("Correction solve time (ms):", res)


# ---------------------------------------------------------------------------
PCC_MODES = [("No PCC", "none", "_nopcc"), ("Rule-based PCC", "rule", "_rulepcc"), ("Adaptive PCC", "adaptive", "")]


def cmd_pcc(args):
    sp, tp = setup(args)
    scen = args.pcc_scenario
    seeds = seeds_of(args)[: (1 if args.quick else args.pcc_seeds)]
    jobs = [("LCRL", s, sp, tp, os.path.join(args.out, "runs"), dict(pcc_mode=mode, tag=tag, force=args.force))
            for _, mode, tag in PCC_MODES if tag for s in seeds]
    run_jobs(jobs, args.jobs)
    days = make_pool(sp, "test", n_test(args, scen), [scen])
    res, trajs = {}, {}
    for label, mode, tag in PCC_MODES:
        dfs = [run_eval(args, sp, tp, "LCRL", s, scen, days, tag=tag, pcc_mode=mode) for s in seeds]
        res[label] = pd.concat(dfs)
        tp_path = eval_path(args, "LCRL", seeds[0], scen, tag).replace(".csv", "_traj.npz")
        if os.path.exists(tp_path):
            trajs[label] = dict(np.load(tp_path))
        print(label, {k: round(v, 4) for k, v in summarize(res[label]).items() if k in ("R_sys", "cost", "peak_ipcs")})
    pd.concat([d.assign(mode=k) for k, d in res.items()]).to_csv(
        os.path.join(args.out, "tables", "exp3_pcc.csv"), index=False)
    plots.fig5_pcc(res, os.path.join(args.out, "figures", "fig5_pcc.png"))
    if trajs:
        plots.fig6_traj(trajs, sp, os.path.join(args.out, "figures", "fig6_pcc_trajectories.png"))


# ---------------------------------------------------------------------------
def cmd_sens(args):
    sp, tp = setup(args)
    etas = [0.4, 1.6, 2.8] if args.quick else [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    gbars = [0.03, 0.055, 0.08] if args.quick else [0.025, 0.04, 0.055, 0.07, 0.085]
    seed = seeds_of(args)[0]
    jobs = []
    for e in etas:
        for gb in gbars:
            tag = f"_eta{e:.2f}_g{gb:.3f}"
            jobs.append(("LCRL", seed, sp, tp, os.path.join(args.out, "runs"),
                         dict(eta_Z=e, g_bar=gb, tag=tag, force=args.force)))
    run_jobs(jobs, args.jobs)
    days = make_pool(sp, "test", n_test(args, "Combined"), ["Combined"])
    rows = []
    for e in etas:
        for gb in gbars:
            tag = f"_eta{e:.2f}_g{gb:.3f}"
            df = run_eval(args, sp, tp, "LCRL", seed, "Combined", days, tag=tag, eta_Z=e, g_bar=gb)
            s = summarize(df)
            rows.append(dict(eta_Z=e, g_bar=gb, cost=s["cost"], R_sys=s["R_sys"], g=s["g"]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(args.out, "tables", "fig8_sensitivity.csv"), index=False)
    plots.fig8_sens(t, os.path.join(args.out, "figures", "fig8_sensitivity.png"))


def cmd_severity(args):
    sp, tp = setup(args)
    methods = ["LCRL", "PPO-Lag", "CPO", "SAC"]
    xis = list(range(0, 11, 5 if args.quick else 1))
    rows = []
    for xi in xis:
        sc = severity_scenario(xi)
        days = make_pool(sp, "severity", 4 if args.quick else sc["n_test"], scen_dict=sc, name=f"xi{xi}")
        for m in methods:
            for s in seeds_of(args):
                p = os.path.join(args.out, "eval", f"{m}_s{s}_xi{xi}.csv")
                if os.path.exists(p) and not args.force:
                    df = pd.read_csv(p)
                else:
                    if not os.path.exists(os.path.join(run_dir(os.path.join(args.out, "runs"), m, s), "model.pt")):
                        continue
                    agent = load_agent(sp, tp, m, s, os.path.join(args.out, "runs"))
                    df, _ = evaluate(sp, m, agent, days, traj_day=None)
                    df.to_csv(p, index=False)
                rows.append(dict(xi=xi, method=m, seed=s, R_sys=df.R_sys.mean(), g=df.g.mean(), cost=df.cost.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(args.out, "tables", "fig9_severity.csv"), index=False)
    if len(t):
        plots.fig9_severity(t, os.path.join(args.out, "figures", "fig9_severity.png"))
        print(t.groupby(["method", "xi"])[["R_sys", "g"]].mean().unstack(0).round(4))


# ---------------------------------------------------------------------------
def cmd_tune(args):
    """Evaluate 12 hyperparameter configurations per method, or four in quick mode.
    Rank validation results by reliability, breaking ties by lower cost."""
    sp, tp = setup(args)
    rng = np.random.default_rng(0)
    val_days = make_pool(sp, "val", tp.n_val_days)
    for m in [resolve(x) for x in (args.methods or MAIN_METHODS)]:
        rows = []
        for k in range(4 if args.quick else 12):
            tpk = copy.deepcopy(tp)
            tpk.lr_actor = float(10 ** rng.uniform(-4.0, -3.0))
            tpk.lr_critic = float(10 ** rng.uniform(-3.5, -2.5))
            tpk.init_log_std = float(rng.choice([-1.5, -1.0, -0.5]))
            if METHODS[m]["kind"] == "on":
                tpk.clip = float(rng.choice([0.1, 0.2, 0.3]))
                tpk.gae_lambda = float(rng.choice([0.9, 0.95, 0.98]))
            else:
                tpk.tau = float(rng.choice([0.002, 0.005, 0.01]))
            tag = f"_tune{k}"
            train(m, seeds_of(args)[0], sp, tpk, os.path.join(args.out, "runs"), tag=tag, verbose=False)
            agent = load_agent(sp, tpk, m, seeds_of(args)[0], os.path.join(args.out, "runs"), tag)
            df, _ = evaluate(sp, m, agent, val_days, n_mc=20, traj_day=None)
            s = summarize(df)
            rows.append(dict(config=k, **{f: getattr(tpk, f) for f in ("lr_actor", "lr_critic", "init_log_std",
                                                                         "clip", "gae_lambda", "tau")},
                             R_sys=s["R_sys"], cost=s["cost"], g=s["g"]))
            print(m, rows[-1], flush=True)
        t = pd.DataFrame(rows).sort_values(["R_sys", "cost"], ascending=[False, True])
        t.to_csv(os.path.join(args.out, "tables", f"tune_{m}.csv"), index=False)
        print("best", m, t.iloc[0].to_dict())


def cmd_all(args):
    for f in (cmd_train, cmd_eval, cmd_tables, cmd_fig3, cmd_pcc, cmd_sens, cmd_severity):
        print(f"\n===== {f.__name__} =====", flush=True)
        f(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "eval", "tables", "fig3", "pcc", "sens", "severity", "tune", "all"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pcc-scenario", default="Combined")
    ap.add_argument("--pcc-seeds", type=int, default=3)
    ap.add_argument("--solver", choices=["fast", "exact"], default="fast",
                    help="action-correction solver used in evaluation (exact = SCIP/SLSQP, Eq. 69)")
    args = ap.parse_args()
    if args.scenarios:
        args.scenarios = [s.replace("_", " ") for s in args.scenarios]
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
