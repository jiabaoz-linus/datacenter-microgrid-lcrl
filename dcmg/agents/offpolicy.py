"""SAC, TD3 and DDPG agents with replay-buffer updates.

Transitions contain candidate actions and base rewards. The environment
applies the configured action correction when producing the next state.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .common import QNet, TanhGaussianActor, DetActor


class Replay:
    def __init__(self, obs_dim, act_dim, size):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, o, a, r, o2, d):
        n = o.shape[0]
        idx = (self.ptr + np.arange(n)) % self.size
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx] = o, a, r, o2, d
        self.ptr = (self.ptr + n) % self.size
        self.full = self.full or self.ptr < n

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n):
        idx = np.random.randint(0, len(self), n)
        t = lambda x: torch.as_tensor(x[idx])
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class OffPolicyAgent:
    def __init__(self, obs_dim, act_dim, tp, sp, algo="sac", device="cpu"):
        assert algo in ("sac", "td3", "ddpg")
        self.tp, self.algo, self.act_dim = tp, algo, act_dim
        self.q1 = QNet(obs_dim, act_dim)
        self.q2 = QNet(obs_dim, act_dim) if algo in ("sac", "td3") else None
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2) if self.q2 is not None else None
        qparams = list(self.q1.parameters()) + (list(self.q2.parameters()) if self.q2 else [])
        self.opt_q = torch.optim.Adam(qparams, lr=tp.lr_critic)
        if algo == "sac":
            self.actor = TanhGaussianActor(obs_dim, act_dim)
            self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=tp.lr_actor)
            self.target_ent = -float(act_dim)
        else:
            self.actor = DetActor(obs_dim, act_dim)
            self.actor_t = copy.deepcopy(self.actor)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=tp.lr_actor)
        self.buf = Replay(obs_dim, act_dim, tp.buffer_size)
        self.n_updates = 0
        self.expl_noise = 0.1
        self.policy_noise, self.noise_clip, self.policy_delay = 0.2, 0.5, 2

    def act(self, obs, deterministic=False, random=False):
        n = obs.shape[0]
        if random:
            return np.random.uniform(-1, 1, (n, self.act_dim)).astype(np.float32), None
        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            if self.algo == "sac":
                a, _ = self.actor(o, deterministic=deterministic, with_logp=False)
                a = a.numpy()
            else:
                a = self.actor(o).numpy()
                if not deterministic:
                    a = np.clip(a + self.expl_noise * np.random.randn(*a.shape), -1, 1)
        return a.astype(np.float32), None

    def update(self):
        tp = self.tp
        if len(self.buf) < max(tp.off_batch, tp.warmup_steps):
            return {}
        o, a, r, o2, d = self.buf.sample(tp.off_batch)
        g = tp.gamma
        with torch.no_grad():
            if self.algo == "sac":
                a2, lp2 = self.actor(o2)
                alpha = self.log_alpha.exp()
                qt = torch.min(self.q1_t(o2, a2), self.q2_t(o2, a2)) - alpha * lp2
            elif self.algo == "td3":
                noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a2 = (self.actor_t(o2) + noise).clamp(-1, 1)
                qt = torch.min(self.q1_t(o2, a2), self.q2_t(o2, a2))
            else:
                qt = self.q1_t(o2, self.actor_t(o2))
            y = r + g * (1 - d) * qt
        lq = F.mse_loss(self.q1(o, a), y)
        if self.q2 is not None:
            lq = lq + F.mse_loss(self.q2(o, a), y)
        self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()
        self.n_updates += 1
        stats = dict(q_loss=float(lq))

        do_actor = self.algo != "td3" or self.n_updates % self.policy_delay == 0
        if do_actor:
            if self.algo == "sac":
                an, lp = self.actor(o)
                qn = torch.min(self.q1(o, an), self.q2(o, an))
                la = (self.log_alpha.exp().detach() * lp - qn).mean()
                self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
                lal = -(self.log_alpha * (lp.detach() + self.target_ent)).mean()
                self.opt_alpha.zero_grad(); lal.backward(); self.opt_alpha.step()
            else:
                la = -self.q1(o, self.actor(o)).mean()
                self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
            self._soft_update()
        return stats

    def _soft_update(self):
        tau = self.tp.tau
        pairs = [(self.q1, self.q1_t)]
        if self.q2 is not None:
            pairs.append((self.q2, self.q2_t))
        if self.algo != "sac":
            pairs.append((self.actor, self.actor_t))
        with torch.no_grad():
            for net, tgt in pairs:
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(),
                    q2=self.q2.state_dict() if self.q2 else None)

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        if self.q2 is not None and sd.get("q2") is not None:
            self.q2.load_state_dict(sd["q2"])
