"""Shared neural-network building blocks."""
import numpy as np
import torch
import torch.nn as nn


def mlp(inp, out, hidden=128, n_layers=2, act=nn.Tanh, out_act=None):
    layers, d = [], inp
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


def init_orthogonal(module, gain=np.sqrt(2)):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)


class GaussianActor(nn.Module):
    """Diagonal Gaussian over the unsquashed action; the environment clips to
    [-1, 1] when decoding. The policy likelihood is that of the sampled
    candidate action before clipping."""

    def __init__(self, obs_dim, act_dim, hidden=128, init_log_std=-0.5):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, hidden)
        init_orthogonal(self.mu)
        nn.init.orthogonal_(self.mu[-1].weight, 0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.log_std.clamp(-5, 1)).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            d = self.dist(obs)
            a = d.mean if deterministic else d.sample()
            return a, d.log_prob(a).sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.v = mlp(obs_dim, 1, hidden)
        init_orthogonal(self.v)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, 1, hidden, act=nn.ReLU)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], -1)).squeeze(-1)


class TanhGaussianActor(nn.Module):
    """SAC actor."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.body = mlp(obs_dim, 2 * act_dim, hidden, act=nn.ReLU)
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False, with_logp=True):
        out = self.body(obs)
        mu, log_std = out[..., :self.act_dim], out[..., self.act_dim:].clamp(-5, 2)
        std = log_std.exp()
        d = torch.distributions.Normal(mu, std)
        u = mu if deterministic else d.rsample()
        a = torch.tanh(u)
        logp = None
        if with_logp:
            logp = d.log_prob(u).sum(-1) - (2 * (np.log(2) - u - nn.functional.softplus(-2 * u))).sum(-1)
        return a, logp


class DetActor(nn.Module):
    """TD3 / DDPG actor."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim, hidden, act=nn.ReLU, out_act=nn.Tanh)

    def forward(self, obs):
        return self.pi(obs)


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def flat_grad(y, model, retain_graph=None, create_graph=False):
    g = torch.autograd.grad(y, list(model.parameters()), retain_graph=retain_graph,
                            create_graph=create_graph, allow_unused=True)
    return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                      for gi, p in zip(g, model.parameters())])


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / (p @ Ap + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        if rr_new < tol:
            break
        p = r + rr_new / rr * p
        rr = rr_new
    return x


def gae(rew, val, last_val, gamma, lam):
    """Generalised advantage estimation on (T, B) arrays with last_val
    supplying the bootstrap value after the final time step."""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1])
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val
