from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from simulator import Ti as TI
from simulator import alpha as ALPHA
from simulator import k as K_COND
from simulator import source_radius as SOURCE_RADIUS
import config as C
SOURCE_PARAM_NAMES = ("xs", "ys", "log_q")


# ----------------------------------------------------------------------------
# Config (stored inside the checkpoint so modelviz.py rebuilds the same model)
# ----------------------------------------------------------------------------
@dataclass
class Config:
    latent_dim: int = C.LATENT_DIM
    dyn_hidden: int = C.DYN_HIDDEN
    hidden: int = C.HIDDEN
    depth: int = C.DEPTH
    ode_steps: int = C.ODE_STEPS
    t_max: float = C.TIME_MAX
    temp_scale: float = C.TEMP_SCALE
    x_min: float = C.X_MIN
    x_max: float = C.X_MAX
    y_min: float = C.Y_MIN
    y_max: float = C.Y_MAX
    q_ref: float = C.Q_REF
    source_features: bool = C.SOURCE_FEATURES
    n_sources: int = C.N_SOURCES

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _as_vec(v, k: int, name: str) -> np.ndarray:
    """Scalar -> broadcast to k values; sequence -> must have exactly k values."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size == 1:
        a = np.repeat(a, k)
    if a.size != k:
        raise ValueError(f"{name}: expected 1 or {k} values, got {a.size}")
    return a


def default_layout(n_sources: int, radius: float = 0.5):
    """Distinct fallback positions: origin for K=1, otherwise a ring."""
    if n_sources == 1:
        return np.zeros(1), np.zeros(1)
    ang = 2.0 * np.pi * np.arange(n_sources) / n_sources
    return radius * np.cos(ang), radius * np.sin(ang)


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
class LatentDynamics(nn.Module):
    """dz/dtau = f(z), integrated from 0 to tau with fixed-step RK4.

    tau may differ per sample (shape (B, 1)), so each row gets its own step.
    """

    def __init__(self, latent_dim: int, hidden: int, steps: int):
        super().__init__()
        self.steps = steps
        self.f = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        dt = tau / self.steps
        z = z0
        for _ in range(self.steps):
            k1 = self.f(z)
            k2 = self.f(z + 0.5 * dt * k1)
            k3 = self.f(z + 0.5 * dt * k2)
            k4 = self.f(z + dt * k3)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z


class Decoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class PILNODE(nn.Module):
    def __init__(self, cfg: Config, xs0=None, ys0=None, q0=2.0e4):
        """xs0 / ys0 / q0: scalar (broadcast) or length-K sequences."""
        super().__init__()
        self.cfg = cfg
        n_src = cfg.n_sources
        if n_src < 1:
            raise ValueError("n_sources must be >= 1.")

        # The initial field is the known constant Ti, so an encoder of it would
        # only ever output one constant vector: learn that vector directly.
        self.z0 = nn.Parameter(0.1 * torch.randn(cfg.latent_dim))
        self.dynamics = LatentDynamics(cfg.latent_dim, cfg.dyn_hidden, cfg.ode_steps)

        in_dim = cfg.latent_dim + 2 + (3 * n_src if cfg.source_features else 0)
        self.decoder = Decoder(in_dim, cfg.hidden, cfg.depth)

        # Unknown sources: positions and (log) strengths, each of shape (K,).
        dx0, dy0 = default_layout(n_src)
        xs0 = dx0 if xs0 is None else _as_vec(xs0, n_src, "xs0")
        ys0 = dy0 if ys0 is None else _as_vec(ys0, n_src, "ys0")
        q0 = _as_vec(q0, n_src, "q0")
        self.xs = nn.Parameter(torch.tensor(xs0, dtype=torch.float32))
        self.ys = nn.Parameter(torch.tensor(ys0, dtype=torch.float32))
        self.log_q = nn.Parameter(torch.tensor(np.log(q0 / cfg.q_ref), dtype=torch.float32))

    # -- sources ---------------------------------------------------------
    @property
    def q(self) -> torch.Tensor:

        return self.cfg.q_ref * torch.exp(self.log_q)

    def source_estimate(self) -> dict:

        return {
            "x": self.xs.detach().cpu().tolist(),
            "y": self.ys.detach().cpu().tolist(),
            "q": self.q.detach().cpu().tolist(),
        }

    def source_term(self, x, y, t):

        safe_t = t.clamp_min(1e-8)                                   # (N, 1)
        R2 = (x - self.xs).square() + (y - self.ys).square() + SOURCE_RADIUS ** 2   # (N, K)
        s = (R2 / (4.0 * ALPHA * safe_t)).clamp(max=80.0)            # (N, K)
        amp = self.q / (4.0 * math.pi * K_COND)                      # (K,)
        q_j = amp * torch.exp(-s) * SOURCE_RADIUS ** 2 * (
            1.0 / (safe_t * R2) + 4.0 * ALPHA / R2.square()
        )
        return q_j.sum(dim=-1, keepdim=True)

    # -- field -----------------------------------------------------------
    def theta(self, x, y, t):

        tau = t / self.cfg.t_max
        z0 = self.z0.unsqueeze(0).expand(x.shape[0], -1)
        z = self.dynamics(z0, tau)

        feats = [z, x, y]
        if self.cfg.source_features:
            dx = x - self.xs                                         # (N, K)
            dy = y - self.ys                                         # (N, K)
            ln_r2 = torch.log(dx.square() + dy.square() + SOURCE_RADIUS ** 2)
            feats += [dx, dy, (ln_r2 + 4.0) / 3.0]

        return tau * self.decoder(torch.cat(feats, dim=-1))

    def forward(self, x, y, t):
        """Temperature in degC."""
        return TI + self.cfg.temp_scale * self.theta(x, y, t)


# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------
def pde_residual(model: PILNODE, x, y, t):
    """Nondimensional residual of u_t = alpha*lap(u) + sum_j Q_j, in units of theta per unit tau."""
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    th = model.theta(x, y, t)
    ones = torch.ones_like(th)

    th_x, th_y, th_t = torch.autograd.grad(th, (x, y, t), grad_outputs=ones, create_graph=True)
    th_xx = torch.autograd.grad(th_x, x, grad_outputs=torch.ones_like(th_x), create_graph=True)[0]
    th_yy = torch.autograd.grad(th_y, y, grad_outputs=torch.ones_like(th_y), create_graph=True)[0]

    q_term = model.source_term(x, y, t) / model.cfg.temp_scale
    return model.cfg.t_max * (th_t - ALPHA * (th_xx + th_yy) - q_term)


def sample_collocation(n, model: PILNODE, args, device):
    """Uniform points + points log-spaced in radius around the CURRENT source estimates.

    The source cores are ~1 cm wide inside a 2 m domain, so uniform sampling would
    practically never see Q and could not constrain (xs_j, ys_j, q_j). The near-source
    points are split evenly over the K sources, so raise --n-colloc roughly in
    proportion to K to keep the same number of points per source.
    """
    cfg = model.cfg
    n_near = int(round(n * args.near_frac))
    n_uni = n - n_near

    xu = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(n_uni, 1, device=device)
    yu = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(n_uni, 1, device=device)

    xs, ys = model.xs.detach(), model.ys.detach()                    # (K,)
    owner = torch.arange(n_near, device=device) % cfg.n_sources      # source assigned to each point
    r = args.r_min * (args.r_max / args.r_min) ** torch.rand(n_near, 1, device=device)
    ang = 2.0 * math.pi * torch.rand(n_near, 1, device=device)
    xn = (xs[owner].unsqueeze(1) + r * torch.cos(ang)).clamp(cfg.x_min, cfg.x_max)
    yn = (ys[owner].unsqueeze(1) + r * torch.sin(ang)).clamp(cfg.y_min, cfg.y_max)

    x = torch.cat([xu, xn], dim=0)
    y = torch.cat([yu, yn], dim=0)

    # Half uniform in time, half log-uniform over [1e-3, 1] * t_max (early times are sharp).
    t_lin = cfg.t_max * torch.rand(n, 1, device=device)
    t_log = cfg.t_max * 10.0 ** (-3.0 * torch.rand(n, 1, device=device))
    t = torch.where(torch.rand(n, 1, device=device) < 0.5, t_lin, t_log)
    return x, y, t


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_observations(path: str):
    df = pd.read_csv(path)
    required = ["x", "y", "t", "temperature", "sigma"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("CSV contains non-finite values.")
    return df


def _weighted_kmeans(pts: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator,
                     n_iter: int = 100) -> np.ndarray:
    """Weighted k-means with weighted k-means++ seeding. pts: (N, 2), w: (N,) >= 0."""
    centers = [pts[rng.choice(len(pts), p=w / w.sum())]]
    for _ in range(1, k):
        d2 = np.min([((pts - c) ** 2).sum(axis=1) for c in centers], axis=0)
        prob = w * d2
        if prob.sum() <= 0:                      # fewer hot points than clusters
            prob = d2 + 1e-12
        centers.append(pts[rng.choice(len(pts), p=prob / prob.sum())])
    centers = np.array(centers)

    for _ in range(n_iter):
        lab = ((pts[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            m = lab == j
            if w[m].sum() > 0:
                new[j] = (w[m, None] * pts[m]).sum(axis=0) / w[m].sum()
        if np.allclose(new, centers):
            break
        centers = new
    return centers[np.argsort(centers[:, 0])]    # deterministic source ordering


def initial_source_guess(df: pd.DataFrame, n_sources: int = 1, power: float = 1.0, seed: int = 0):
    """Initial (xs0, ys0), each of length K (uses data only).

    K=1: rise-weighted centroid of the sensors at the latest time.
    K>1: rise-weighted k-means with K clusters over the same sensors, so the sources
         start apart (identical starts would receive identical gradients and never
         separate). `power` > 1 sharpens the weights towards the hottest sensors.
    """
    last = df[df["t"] == df["t"].max()]
    pts = last[["x", "y"]].to_numpy(dtype=float)
    rise = np.clip(last["temperature"].to_numpy(dtype=float) - TI, 0.0, None)
    w = rise ** power

    if w.sum() <= 0 or len(pts) < n_sources:
        return default_layout(n_sources)
    if n_sources == 1:
        c = (w[:, None] * pts).sum(axis=0) / w.sum()
        return c[:1], c[1:]

    centers = _weighted_kmeans(pts, w, n_sources, np.random.default_rng(seed))
    return centers[:, 0], centers[:, 1]


def format_sources(est: dict) -> str:
    return "  ".join(
        f"[{j}] x={x:+.4f} y={y:+.4f} q={q:.5g}"
        for j, (x, y, q) in enumerate(zip(est["x"], est["y"], est["q"]))
    )


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=C.OBS_CSV)
    p.add_argument("--out", default=C.CHECKPOINT)
    p.add_argument("--history", default=C.HISTORY_CSV)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--n-sources", type=int, default=C.N_SOURCES)
    p.add_argument("--latent-dim", type=int, default=C.LATENT_DIM)
    p.add_argument("--dyn-hidden", type=int, default=C.DYN_HIDDEN)
    p.add_argument("--hidden", type=int, default=C.HIDDEN)
    p.add_argument("--depth", type=int, default=C.DEPTH)
    p.add_argument("--ode-steps", type=int, default=C.ODE_STEPS)
    p.add_argument("--no-source-features", action="store_true", default=not C.SOURCE_FEATURES)
    p.add_argument("--temp-scale", type=float, default=C.TEMP_SCALE)
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LR_NET)
    p.add_argument("--lr-source", type=float, default=C.LR_SOURCE)
    p.add_argument("--lambda-pde", type=float, default=C.LAMBDA_PDE)
    p.add_argument("--sigma-floor", type=float, default=C.SIGMA_FLOOR)
    p.add_argument("--n-colloc", type=int, default=C.N_COLLOC)
    p.add_argument("--near-frac", type=float, default=C.NEAR_FRAC)
    p.add_argument("--r-min", type=float, default=C.R_MIN)
    p.add_argument("--r-max", type=float, default=C.R_MAX)
    p.add_argument("--grad-clip", type=float, default=C.GRAD_CLIP)
    p.add_argument("--log-every", type=int, default=C.LOG_EVERY)
    p.add_argument("--xs0", type=float, nargs="+", default=C.XS0)
    p.add_argument("--ys0", type=float, nargs="+", default=C.YS0)
    p.add_argument("--q0", type=float, nargs="+", default=C.Q0)
    p.add_argument("--init-power", type=float, default=C.INIT_POWER)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--t-max", type=float, default=None,
               help="defaults to the last observation time; larger values extrapolate in time")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    n_src = args.n_sources
    if n_src < 1:
        raise ValueError("--n-sources must be >= 1.")

    df = load_observations(args.data)
    t_max = float(args.t_max) if args.t_max is not None else float(df["t"].max())
    if t_max < float(df["t"].max()):
        raise ValueError("--t-max must be >= the last observation time.")

    cfg = Config(
        latent_dim=args.latent_dim,
        dyn_hidden=args.dyn_hidden,
        hidden=args.hidden,
        depth=args.depth,
        ode_steps=args.ode_steps,
        t_max=t_max,
        temp_scale=args.temp_scale,
        source_features=not args.no_source_features,
        n_sources=n_src,
    )

    xs0, ys0 = initial_source_guess(df, n_src, args.init_power, args.seed)
    if args.xs0 is not None:
        xs0 = _as_vec(args.xs0, n_src, "--xs0")
    if args.ys0 is not None:
        ys0 = _as_vec(args.ys0, n_src, "--ys0")
    q0 = _as_vec(args.q0, n_src, "--q0")

    model = PILNODE(cfg, xs0, ys0, q0).to(device)

    x_d = torch.tensor(df[["x"]].to_numpy(), dtype=torch.float32, device=device)
    y_d = torch.tensor(df[["y"]].to_numpy(), dtype=torch.float32, device=device)
    t_d = torch.tensor(df[["t"]].to_numpy(), dtype=torch.float32, device=device)
    th_obs = (
        torch.tensor(df[["temperature"]].to_numpy(), dtype=torch.float32, device=device) - TI
    ) / cfg.temp_scale
    sig = torch.tensor(df[["sigma"]].to_numpy(), dtype=torch.float32, device=device)
    sig = sig.clamp_min(args.sigma_floor) / cfg.temp_scale
    n_obs = x_d.shape[0]

    src_params = [p for n, p in model.named_parameters() if n in SOURCE_PARAM_NAMES]
    net_params = [p for n, p in model.named_parameters() if n not in SOURCE_PARAM_NAMES]
    optimizer = torch.optim.Adam(
        [
            {"params": net_params, "lr": args.lr},
            {"params": src_params, "lr": args.lr_source},
        ]
    )

    steps_per_epoch = math.ceil(n_obs / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    floor = 0.01

    def lr_factor(step):  # cosine decay to 1% of the base lr
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    print(f"device={device}  observations={n_obs}  t_max={cfg.t_max:g}  sources={n_src}")
    print("initial source guess:")
    for j in range(n_src):
        print(f"  [{j}] x={xs0[j]:.4f} y={ys0[j]:.4f} q={q0[j]:.4g}")

    # one (xs, ys, q) column triple per source: xs_0, ys_0, q_0, xs_1, ...
    src_keys = [f"{n}_{j}" for j in range(n_src) for n in ("xs", "ys", "q")]
    history = {k: [] for k in ["total", "data", "pde", *src_keys]}

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_obs, device=device)
        acc = np.zeros(3)

        for i in range(0, n_obs, args.batch_size):
            idx = perm[i:i + args.batch_size]

            optimizer.zero_grad(set_to_none=True)

            th_pred = model.theta(x_d[idx], y_d[idx], t_d[idx])
            data_loss = ((th_pred - th_obs[idx]) / sig[idx]).square().mean()

            xc, yc, tc = sample_collocation(args.n_colloc, model, args, device)
            pde_loss = pde_residual(model, xc, yc, tc).square().mean()

            loss = data_loss + args.lambda_pde * pde_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net_params, args.grad_clip)
            torch.nn.utils.clip_grad_norm_(src_params, args.grad_clip)

            optimizer.step()
            scheduler.step()

            acc += [loss.item(), data_loss.item(), pde_loss.item()]

        acc /= steps_per_epoch
        est = model.source_estimate()
        row = [*acc]
        for j in range(n_src):
            row += [est["x"][j], est["y"][j], est["q"][j]]
        for key, val in zip(history, row):
            history[key].append(float(val))

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d}/{args.epochs} | total={acc[0]:.4e} data={acc[1]:.4e} "
                f"pde={acc[2]:.4e} | {format_sources(est)}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "source_estimate": model.source_estimate(),
            "alpha": float(ALPHA),
            "Ti": float(TI),
        },
        args.out,
    )
    pd.DataFrame(history).to_csv(args.history, index=False)

    est = model.source_estimate()
    print("\nEstimated sources:")
    for j in range(n_src):
        print(f"  [{j}] x={est['x'][j]:.5f}  y={est['y'][j]:.5f}  q={est['q'][j]:.6g}")
    print(f"Saved {Path(args.out).resolve()} and {Path(args.history).resolve()}")


if __name__ == "__main__":
    main()