# DCMG-LCRL

Research code for **Reliability Modeling and Optimization of Networked Data Center Microgrids Based on Lyapunov Constrained Reinforcement Learning**.

Hongyan Dui, Sihan Li, Jiabao Zhai, and Liudong Xing

This repository implements a simulator for a data center microgrid connected to a community microgrid and a Lyapunov-constrained reinforcement learning (LCRL) controller. The experiment suite compares LCRL with seven RL baselines and a model predictive control (MPC) benchmark, evaluates task-period reliability by Monte Carlo simulation, and includes ablations and sensitivity studies.

## Installation

Use Python 3.10 in a fresh environment. From the repository root, run:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Train one LCRL policy and evaluate it on the combined disturbance scenario:

```bash
python run.py train --methods LCRL --quick --out outputs_quick
python run.py eval --methods LCRL --scenarios Combined --quick --out outputs_quick
```

The quick configuration uses one seed, 20 training iterations, four test days per scenario, and 20 equipment-state draws per test day. Check `outputs_quick/runs/LCRL/seed1101/` for the model, training log, and configuration, and `outputs_quick/eval/LCRL_s1101_Combined.csv` for day-level reliability and cost metrics. Use the standard configuration for full-budget experiments.

## Reproduce the experiments

```bash
python run.py all --jobs 4 --out outputs_full
```

The full workflow runs the stages below in order. To run a single stage, replace `all` with the stage name and keep the same `--out` directory and seed selection for dependent stages.

| Stage | Experiment or output |
| --- | --- |
| `train` | Train LCRL, the seven RL baselines, and the ablation variants. |
| `eval` | Compare the trained policies and MPC across normal, task burst, high ambient, grid limit, and combined scenarios. |
| `tables` | Summarize operating metrics, reliability with bootstrap intervals, reliability decomposition, and component ablations. |
| `fig3` | Plot training curves and benchmark per-slot action correction. |
| `pcc` | Compare disabled, rule-based, and adaptive power exchange at the point of common coupling. |
| `sens` | Sweep the Lyapunov penalty weight and violation threshold. |
| `severity` | Evaluate policies over the compound disturbance severity range. |

The standard configuration uses seeds 1101–1110, 96 scheduling slots per day, and 200 equipment-state draws per test day. The optional `tune` command searches hyperparameters on the validation pool. Run `python run.py --help` for CLI options. Use a fresh output directory when changing the experiment budget, model settings, or solver backend. Existing checkpoints and evaluation CSVs can be reused by subsequent runs.

Evaluation uses vectorized action correction by default. For `eval`, `pcc`, and `sens`, `--solver exact` selects solver-based correction. The implementation uses SCIP through PySCIPOpt when available and falls back to a local SciPy SLSQP heuristic over integer server-count candidates.

## Data and outputs

The simulator generates synthetic workloads, ambient temperatures, renewable generation, grid supply conditions, and equipment states from the parameters and seeds in the code. The workflow requires no external dataset. All run artifacts are generated locally under the directory passed to `--out`.

| Directory | Contents |
| --- | --- |
| `runs/` | Trained models, logs, and the configuration for each run. |
| `eval/` | Day-level evaluation CSVs and representative trajectories when available. |
| `tables/` | Summary CSVs, bootstrap intervals, ablation tables, and timing results. |
| `figures/` | Training and experiment plots. |

## Code map

| Path | Role |
| --- | --- |
| `run.py` | Training, evaluation, and experiment commands. |
| `dcmg/config.py`, `dcmg/disturbance.py` | Simulator settings, scenario definitions, and synthetic sample generation. |
| `dcmg/env.py`, `dcmg/physics.py` | Environment dynamics, operating costs, and task-period success conditions. |
| `dcmg/agents/`, `dcmg/train.py` | LCRL and RL baseline implementations and training. |
| `dcmg/pcm.py`, `dcmg/pcm_exact.py`, `dcmg/mpc.py` | Action correction and the MPC benchmark. |
| `dcmg/evaluate.py`, `dcmg/bootstrap.py` | Monte Carlo reliability evaluation and statistical comparisons. |

## Citation

If you use this code in your research, please cite the accompanying manuscript:

> Hongyan Dui, Sihan Li, Jiabao Zhai, and Liudong Xing. *Reliability Modeling and Optimization of Networked Data Center Microgrids Based on Lyapunov Constrained Reinforcement Learning*.
