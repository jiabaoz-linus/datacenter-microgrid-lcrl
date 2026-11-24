# DCMG-LCRL

Research code for **Reliability Modeling and Optimization of Networked Data Center Microgrids Based on Lyapunov Constrained Reinforcement Learning**.

Hongyan Dui, Sihan Li, Jiabao Zhai, and Liudong Xing

This repository implements a simulator for a data center microgrid connected to a community microgrid and a Lyapunov-constrained reinforcement learning (LCRL) controller. The experiment suite compares LCRL with seven RL baselines and a model predictive control (MPC) benchmark, evaluates task-period reliability by Monte Carlo simulation, and includes ablations and sensitivity studies.

## Data

The simulator generates synthetic workloads, ambient temperatures, renewable generation, grid supply conditions, and equipment states from the parameters and seeds in the code. No external dataset is required.

## Code map

| Path | Role |
| --- | --- |
| `run.py` | Experiment orchestration. |
| `dcmg/config.py`, `dcmg/disturbance.py` | Simulator settings, scenario definitions, and synthetic sample generation. |
| `dcmg/env.py`, `dcmg/physics.py` | Environment dynamics, operating costs, and task-period success conditions. |
| `dcmg/agents/`, `dcmg/train.py` | LCRL and RL baseline implementations and training. |
| `dcmg/pcm.py`, `dcmg/pcm_exact.py`, `dcmg/mpc.py` | Action correction and the MPC benchmark. |
| `dcmg/evaluate.py`, `dcmg/bootstrap.py` | Monte Carlo reliability evaluation and statistical comparisons. |

## Citation

If you use this code in your research, please cite the accompanying manuscript.

> Hongyan Dui, Sihan Li, Jiabao Zhai, and Liudong Xing. *Reliability Modeling and Optimization of Networked Data Center Microgrids Based on Lyapunov Constrained Reinforcement Learning*.
