"""On-policy actor-critic updates for PPO, PPO-Lagrangian, TRPO and CPO.

LCRL uses PPO with queue-augmented observations and rewards supplied by the
environment, PCM correction, and a KL-based retry rule for policy updates.
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from .common import (GaussianActor, ValueNet, gae, flat_grad, flat_params, set_flat_params,
                     conjugate_gradient)


class OnPolicyAgent:
    def __init__(self, obs_dim, act_dim, tp, sp, algo="ppo", rerun_rule=False,
                 g_bar=None, device="cpu"):
        assert algo in ("ppo", "ppo_lag", "trpo", "cpo")
        self.tp, self.sp, self.algo = tp, sp, algo
        self.device = device
        self.actor = GaussianActor(obs_dim, act_dim, tp.hidden, tp.init_log_std).to(device)
        self.critic = ValueNet(obs_dim, tp.hidden).to(device)
        self.cost_critic = ValueNet(obs_dim, tp.hidden).to(device) if algo in ("ppo_lag", "cpo") else None
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=tp.lr_actor)
        params_c = list(self.critic.parameters()) + (list(self.cost_critic.parameters()) if self.cost_critic else [])
        self.opt_c = torch.optim.Adam(params_c, lr=tp.lr_critic)
        self.rerun_rule = rerun_rule
        self.g_bar = sp.g_bar if g_bar is None else g_bar
        self.lag = 0.0            # PPO-Lagrangian multiplier
        self.lr_lag = 2.0
        self.max_kl = 0.01        # TRPO / CPO trust region
        self.damping = 0.1

    # ------------------------------------------------------------------
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a, logp = self.actor.act(o, deterministic)
        return a.cpu().numpy(), logp.cpu().numpy()

    def values(self, obs):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            v = self.critic(o).cpu().numpy()
            vc = self.cost_critic(o).cpu().numpy() if self.cost_critic else np.zeros_like(v)
        return v, vc

    # ------------------------------------------------------------------
    def _prepare(self, buf):
        tp = self.tp
        T, B = buf["rew"].shape
        v, vc = self.values(buf["obs"].reshape(T * B, -1))
        v, vc = v.reshape(T, B), vc.reshape(T, B)
        zero = np.zeros(B)
        adv, ret = gae(buf["rew"], v, zero, tp.gamma, tp.gae_lambda)
        cadv, cret = gae(buf["cost"], vc, zero, tp.gamma, tp.gae_lambda)
        f = lambda x: torch.as_tensor(x.reshape(T * B, *x.shape[2:]), dtype=torch.float32, device=self.device)
        return dict(obs=f(buf["obs"]), act=f(buf["act"]), logp=f(buf["logp"]), adv=f(adv), ret=f(ret),
                    cadv=f(cadv), cret=f(cret), mean_cost=float(buf["cost"].mean()), n=T * B)

    def _value_update(self, d, epochs=None):
        tp = self.tp
        n = d["n"]
        n_mb = max(1, int(round(n / tp.batch_size)))
        for _ in range(epochs or tp.epochs):
            perm = torch.randperm(n)
            for idx in perm.chunk(n_mb):
                loss = ((self.critic(d["obs"][idx]) - d["ret"][idx]) ** 2).mean()          # Value-regression loss.
                if self.cost_critic is not None:
                    loss = loss + ((self.cost_critic(d["obs"][idx]) - d["cret"][idx]) ** 2).mean()
                self.opt_c.zero_grad()
                (tp.vf_coef * loss).backward()
                nn.utils.clip_grad_norm_(self.opt_c.param_groups[0]["params"], tp.max_grad_norm)
                self.opt_c.step()

    # ------------------------------------------------------------------
    def update(self, buf):
        d = self._prepare(buf)
        if self.algo in ("ppo", "ppo_lag"):
            stats = self._ppo_update(d)
        elif self.algo == "trpo":
            stats = self._trpo_update(d, constrained=False)
        else:
            stats = self._trpo_update(d, constrained=True)
        self._value_update(d)
        return stats

    def _ppo_update(self, d):
        tp = self.tp
        adv = d["adv"]
        if self.algo == "ppo_lag":
            # multiplier update on the per-slot violation degree
            self.lag = max(0.0, self.lag + self.lr_lag * (d["mean_cost"] - self.g_bar))
            cadv = (d["cadv"] - d["cadv"].mean()) / (d["cadv"].std() + 1e-8)
            radv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = (radv - self.lag * cadv) / (1.0 + self.lag)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        snapshot = (copy.deepcopy(self.actor.state_dict()), copy.deepcopy(self.opt_a.state_dict()))
        lr0 = self.opt_a.param_groups[0]["lr"]
        reruns = 0
        while True:
            kl = self._ppo_epochs(d, adv)
            # Retry from the saved state at a lower learning rate if KL is too large.
            if self.rerun_rule and kl > tp.target_kl and reruns < tp.max_reruns:
                self.actor.load_state_dict(snapshot[0]); self.opt_a.load_state_dict(snapshot[1])
                reruns += 1
                for g in self.opt_a.param_groups:
                    g["lr"] = lr0 * 0.5 ** reruns
                continue
            break
        for g in self.opt_a.param_groups:
            g["lr"] = lr0
        return dict(kl=kl, reruns=reruns, lag=self.lag)

    def _ppo_epochs(self, d, adv):
        tp = self.tp
        n = d["n"]
        n_mb = max(1, int(round(n / tp.batch_size)))
        kl = 0.0
        for _ in range(tp.epochs):
            perm = torch.randperm(n)
            for idx in perm.chunk(n_mb):
                dist = self.actor.dist(d["obs"][idx])
                logp = dist.log_prob(d["act"][idx]).sum(-1)
                ratio = torch.exp(logp - d["logp"][idx])                               # Candidate-action likelihood ratio.
                a = adv[idx]
                l_clip = torch.min(ratio * a, ratio.clamp(1 - tp.clip, 1 + tp.clip) * a).mean()  # Clipped PPO surrogate.
                loss = -l_clip - tp.ent_coef * dist.entropy().sum(-1).mean()
                self.opt_a.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), tp.max_grad_norm)
                self.opt_a.step()
            with torch.no_grad():
                logp_all = self.actor.dist(d["obs"]).log_prob(d["act"]).sum(-1)
                kl = float((d["logp"] - logp_all).mean())
            if not self.rerun_rule and kl > 1.5 * tp.target_kl:
                break   # standard PPO early stopping
        return kl

    # ------------------------------------------------------------------
    def _trpo_update(self, d, constrained):
        obs, act, logp_old = d["obs"], d["act"], d["logp"]
        adv = (d["adv"] - d["adv"].mean()) / (d["adv"].std() + 1e-8)
        cadv = d["cadv"] - d["cadv"].mean()
        with torch.no_grad():
            old = self.actor.dist(obs)
            old_mu, old_std = old.mean.clone(), old.stddev.clone()

        def surr(cost=False):
            lp = self.actor.dist(obs).log_prob(act).sum(-1)
            ratio = torch.exp(lp - logp_old)
            return (ratio * (cadv if cost else adv)).mean()

        def kl_fn():
            new = self.actor.dist(obs)
            oldd = torch.distributions.Normal(old_mu, old_std)
            return torch.distributions.kl_divergence(oldd, new).sum(-1).mean()

        def Hvp(v):
            k = kl_fn()
            g = flat_grad(k, self.actor, create_graph=True)
            return flat_grad(g @ v.detach(), self.actor, retain_graph=True).detach() + self.damping * v

        L = surr()
        g = flat_grad(L, self.actor, retain_graph=True)
        x = conjugate_gradient(Hvp, g)                         # H^{-1} g
        q = float(g @ x)
        delta = self.max_kl
        params0 = flat_params(self.actor)
        case = 4
        if constrained:
            Lc = surr(cost=True)
            b = flat_grad(Lc, self.actor, retain_graph=True)
            # constraint value in discounted-sum units, J_c - d
            c = (d["mean_cost"] - self.g_bar) / (1 - self.tp.gamma)
            if float(b @ b) < 1e-8 and c < 0:
                case = 4
            else:
                w = conjugate_gradient(Hvp, b)
                r, s = float(g @ w), float(b @ w)
                A = q - r ** 2 / (s + 1e-8)
                Bq = 2 * delta - c ** 2 / (s + 1e-8)
                if c < 0 and Bq < 0:
                    case = 3
                elif c < 0 <= Bq:
                    case = 2
                elif c >= 0 and Bq >= 0:
                    case = 1
                else:
                    case = 0
            if case in (3, 4):
                lam, nu = np.sqrt(q / (2 * delta)), 0.0
            elif case in (1, 2):
                LA, LB = [0, r / c], [r / c, np.inf]
                if c < 0:
                    LA, LB = LB, LA
                proj = lambda v, Lr: max(Lr[0], min(Lr[1], v))
                lam_a = proj(np.sqrt(max(A, 1e-12) / max(Bq, 1e-12)), LA)
                lam_b = proj(np.sqrt(q / (2 * delta)), LB)
                f_a = -0.5 * (A / (lam_a + 1e-8) + Bq * lam_a) - r * c / (s + 1e-8)
                f_b = -0.5 * (q / (lam_b + 1e-8) + 2 * delta * lam_b)
                lam = lam_a if f_a >= f_b else lam_b
                nu = max(0.0, lam * c - r) / (s + 1e-8)
            else:
                lam, nu = 0.0, np.sqrt(2 * delta / (s + 1e-8))
            if case > 0:
                step = (x - nu * w) / (lam + 1e-8) if case in (1, 2) else x / (lam + 1e-8)
            else:
                step = -nu * w        # recovery step: purely decrease cost
            # The cost surrogate is minimised and the reward surrogate is maximised.
        else:
            step = float(np.sqrt(2 * delta / (q + 1e-8))) * x
        with torch.no_grad():
            L0 = float(L)
            Lc0 = float(surr(cost=True)) if constrained else 0.0
        accepted = False
        for frac in 0.8 ** np.arange(10):
            set_flat_params(self.actor, params0 + frac * step)
            with torch.no_grad():
                kl = float(kl_fn())
                Lnew = float(surr())
                Lcnew = float(surr(cost=True)) if constrained else 0.0
            ok_kl = kl <= 1.5 * delta
            ok_r = (Lnew > L0) or (constrained and case == 0)
            ok_c = (not constrained) or (Lcnew - Lc0 <= max(-c, 0.0) + 1e-6)
            if ok_kl and ok_r and ok_c:
                accepted = True
                break
        if not accepted:
            set_flat_params(self.actor, params0)
            kl = 0.0
        return dict(kl=kl, case=case, accepted=accepted)

    # ------------------------------------------------------------------
    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), lag=self.lag,
                    cost_critic=self.cost_critic.state_dict() if self.cost_critic else None)

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        if self.cost_critic is not None and sd.get("cost_critic") is not None:
            self.cost_critic.load_state_dict(sd["cost_critic"])
        self.lag = sd.get("lag", 0.0)
