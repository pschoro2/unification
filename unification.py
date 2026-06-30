"""
Minimal self-unification implementation:

- recurrent generative model over external observations
- self-model observations over model state and statistics
- single objective: joint variational free-energy over data + self-model loop
- no external rewards or auxiliary losses
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


def gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL[q||p] for diagonal Gaussians."""
    return 0.5 * torch.sum(
        logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1.0,
        dim=-1,
    )


def clamp_logvar(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=-12.0, max=8.0)


@dataclass
class UnificationConfig:
    # data
    dataset: str = "synthetic"
    data_path: str | None = None
    obs_dim: int = 4
    seq_len: int = 24
    batch_size: int = 64
    shuffle: bool = True

    # model
    z_dim: int = 12
    hidden_dim: int = 48
    self_dim: int = 16
    repr_hid: int = 96

    # optimization
    steps: int = 4000
    lr: float = 2e-3
    beta_kl: float = 1.0
    sigma_x: float = 0.25
    sigma_m: float = 0.20
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"

    # fixed point convergence
    fixed_point_patience: int = 12
    fixed_point_tol: float = 1e-4
    min_steps_for_early_stop: int = 250

    # misc
    dataset_length: int = 4096
    print_every: int = 200
    eval_steps: int = 200


class RecurrentUnificationModel(nn.Module):
    """
    Single recurrent generative model with latent dynamics and self-observation head.
    """

    def __init__(self, cfg: UnificationConfig):
        super().__init__()
        self.obs_dim = cfg.obs_dim
        self.z_dim = cfg.z_dim
        self.h_dim = cfg.hidden_dim
        self.self_dim = cfg.self_dim
        self.cfg = cfg

        prior_in = self.z_dim + self.h_dim
        self.prior_net = build_mlp(prior_in, cfg.repr_hid, self.z_dim * 2)

        enc_in = self.obs_dim + self.self_dim + self.h_dim
        self.encoder = build_mlp(enc_in, cfg.repr_hid, self.z_dim * 2)

        # decode external obs and mean/logvar for self-observations
        dec_in = self.z_dim + self.h_dim
        dec_out = self.obs_dim + 2 * self.self_dim
        self.decoder = build_mlp(dec_in, cfg.repr_hid, dec_out)

        recur_in = self.z_dim + self.obs_dim + self.h_dim + self.self_dim
        self.recur = nn.GRUCell(recur_in, self.h_dim)

        # learned initial recurrent state
        self.h_init = nn.Parameter(torch.zeros(1, self.h_dim))

    @staticmethod
    def _split(stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return stats.chunk(2, dim=-1)

    def prior(self, z_prev: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_p, logvar_p = self._split(self.prior_net(torch.cat([z_prev, h], dim=-1)))
        return mu_p, clamp_logvar(logvar_p)

    def encode(self, x_t: torch.Tensor, m_t: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_q, logvar_q = self._split(self.encoder(torch.cat([x_t, m_t, h], dim=-1)))
        logvar_q = clamp_logvar(logvar_q)
        eps = torch.randn_like(logvar_q)
        z = mu_q + torch.exp(0.5 * logvar_q) * eps
        return z, mu_q, logvar_q

    def decode(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.decoder(torch.cat([z, h], dim=-1))
        x_hat = d[:, : self.obs_dim]
        m_mu = d[:, self.obs_dim : self.obs_dim + self.self_dim]
        m_logvar = clamp_logvar(d[:, self.obs_dim + self.self_dim :])
        return x_hat, m_mu, m_logvar

    def step(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
        m_t_obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mu_p, logvar_p = self.prior(z_t, h_t)
        z_new, mu_q, logvar_q = self.encode(x_t, m_t_obs, h_t)
        x_hat, m_hat_mu, m_hat_logvar = self.decode(z_new, h_t)
        h_next = self.recur(torch.cat([z_new, x_t, h_t, m_t_obs], dim=-1), h_t)

        return {
            "x_hat": x_hat,
            "m_hat_mu": m_hat_mu,
            "m_hat_logvar": m_hat_logvar,
            "z_new": z_new,
            "h_next": h_next,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
        }


class SequenceDataset(Dataset):
    """
    Generic sequence dataset: returns [T, obs_dim] windows.
    """

    def __init__(self, observations: torch.Tensor, seq_len: int):
        super().__init__()
        assert observations.dim() == 2, "Observations must be [T, obs_dim]"
        assert observations.size(1) >= 1

        self.observations = observations.float()
        self.seq_len = seq_len
        self.windows = observations.size(0) - seq_len
        if self.windows <= 0:
            raise ValueError("Sequence length longer than dataset length.")

    def __len__(self) -> int:
        return self.windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.observations[idx : idx + self.seq_len]


def build_synthetic_data(length: int, obs_dim: int, seed: int) -> torch.Tensor:
    """
    Nonlinear oscillatory trajectories; sufficient for recurrent unification dynamics.
    """
    gen = torch.Generator().manual_seed(seed)
    state = torch.rand(length, obs_dim, generator=gen) * 2 * math.pi
    noise = torch.randn(length, obs_dim, generator=gen) * 0.03

    for t in range(2, length):
        d = 0.9 * torch.sin(state[t - 1]) - 0.2 * torch.sin(3 * state[t - 1]) + 0.05 * state[t - 2]
        state[t] = (state[t - 1] + d) % (2 * math.pi)

    state = torch.sin(state) + noise
    return state.float()


def load_data_from_path(data_path: str | None, obs_dim: int, seq_len: int, seed: int, length: int) -> Tuple[torch.Tensor, int]:
    """
    Load [T, obs_dim] observations from npy / npz / csv / pt.
    """
    if data_path is None:
        data = build_synthetic_data(length=length, obs_dim=obs_dim, seed=seed)
        return data, obs_dim

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        if suffix == ".npy":
            arr = np.load(path)
        else:
            npz = np.load(path, allow_pickle=True)
            if "data" in npz.files:
                arr = npz["data"]
            else:
                first = npz.files[0]
                arr = npz[first]
    elif suffix in {".csv", ".txt"}:
        arr = np.loadtxt(path, delimiter=",")
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path)
        if isinstance(obj, torch.Tensor):
            arr = obj.cpu().numpy()
        elif isinstance(obj, dict) and "data" in obj:
            obj_data = obj["data"]
            arr = obj_data.cpu().numpy() if isinstance(obj_data, torch.Tensor) else np.asarray(obj_data)
        else:
            arr = np.asarray(obj)
    else:
        raise ValueError(f"Unsupported data format: {suffix}")

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Loaded data must be 1-D or 2-D.")
    if arr.shape[1] != obs_dim:
        obs_dim = arr.shape[1]

    data = torch.tensor(arr, dtype=torch.float32)
    if data.size(0) <= seq_len:
        raise ValueError("Loaded data shorter than seq_len")
    return data, obs_dim


class SelfModelStateBuilder:
    """
    Constructs m_t from model, activations, and short-memory error channels.
    """

    def __init__(self, model: RecurrentUnificationModel):
        self.model = model
        self.grad_norm: float = 0.0
        self.ema_ext_error: float = 0.0
        self.ema_self_error: float = 0.0

    def build(self, x_t: torch.Tensor, h_t: torch.Tensor, z_t: torch.Tensor, pred_errs: Tuple[float, float]) -> torch.Tensor:
        ext_err, self_err = pred_errs
        alpha = 0.95
        self.ema_ext_error = alpha * self.ema_ext_error + (1 - alpha) * ext_err
        self.ema_self_error = alpha * self.ema_self_error + (1 - alpha) * self_err

        feature_blocks = []

        # parameter level summaries (scale-invariant-ish)
        for p in self.model.parameters():
            if p.numel() == 0:
                continue
            ap = p.abs().flatten()
            feature_blocks.extend(
                [
                    ap.mean(),
                    ap.std(unbiased=False),
                    ap.pow(2).mean(),
                    ap.max(),
                    ap.min(),
                ]
            )

        # latent and activation summaries
        feature_blocks.extend(
            [
                h_t.mean(),
                h_t.std(unbiased=False),
                h_t.abs().mean(),
                z_t.mean(),
                z_t.std(unbiased=False),
                z_t.abs().mean(),
            ]
        )

        # current observation and recent error context
        feature_blocks.extend([x_t.mean(), x_t.std(unbiased=False), x_t.abs().mean()])
        feature_blocks.extend([torch.tensor(self.ema_ext_error, device=x_t.device), torch.tensor(self.ema_self_error, device=x_t.device), torch.tensor(self.grad_norm, device=x_t.device)])

        stats = torch.stack(feature_blocks).to(x_t.device)
        self_dim = self.model.self_dim

        if stats.numel() < self_dim:
            pad = self_dim - stats.numel()
            stats = torch.cat([stats, torch.zeros(pad, device=stats.device, dtype=stats.dtype)])
        else:
            stats = stats[:self_dim]

        return stats.expand(x_t.shape[0], -1).contiguous()

    def set_grad_norm(self, grad_norm: float) -> None:
        self.grad_norm = grad_norm

    def update(self) -> None:
        pass


def joint_variational_free_energy(
    x_hat: torch.Tensor,
    x_t: torch.Tensor,
    m_hat_mu: torch.Tensor,
    m_hat_logvar: torch.Tensor,
    m_t: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    cfg: UnificationConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext_error = (x_hat - x_t).pow(2).mean(dim=-1)
    F_ext = ext_error.mean() / (2 * (cfg.sigma_x ** 2))

    m_var = torch.exp(m_hat_logvar)
    F_int = 0.5 * (((m_hat_mu - m_t).pow(2) / m_var) + m_hat_logvar).mean()

    kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p).mean()
    fixed_point_error = (m_hat_mu - m_t).abs().mean()

    return F_ext, F_int, kl, fixed_point_error


def evaluate(model: RecurrentUnificationModel, loader: DataLoader, cfg: UnificationConfig, device: str) -> Dict[str, float]:
    model.eval()
    total = {"F_ext": 0.0, "F_int": 0.0, "KL": 0.0, "fixed": 0.0}
    builder = SelfModelStateBuilder(model)
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            bsz, seq_len, _ = batch.shape

            h_t = model.h_init.expand(bsz, -1).clone()
            z_t = torch.zeros(bsz, model.z_dim, device=device)
            ext_acc = torch.tensor(0.0, device=device)
            self_acc = torch.tensor(0.0, device=device)
            kl_acc = torch.tensor(0.0, device=device)
            fp_acc = torch.tensor(0.0, device=device)

            ext_ma = 0.0
            self_ma = 0.0

            for t in range(seq_len):
                x_t = batch[:, t]
                m_t = builder.build(x_t, h_t, z_t, (ext_ma, self_ma))
                out = model.step(x_t, h_t, z_t, m_t)
                z_t, h_t = out["z_new"], out["h_next"]

                f_ext, f_int, kl, fp = joint_variational_free_energy(
                    out["x_hat"],
                    x_t,
                    out["m_hat_mu"],
                    out["m_hat_logvar"],
                    m_t,
                    out["mu_q"],
                    out["logvar_q"],
                    out["mu_p"],
                    out["logvar_p"],
                    cfg,
                )

                ext_acc += f_ext
                self_acc += f_int
                kl_acc += kl
                fp_acc += fp

                ext_ma = float(f_ext.item())
                self_ma = float(f_int.item())

            total["F_ext"] += float((ext_acc / seq_len).item())
            total["F_int"] += float((self_acc / seq_len).item())
            total["KL"] += float((kl_acc / seq_len).item())
            total["fixed"] += float((fp_acc / seq_len).item())
            n_batches += 1

    model.train()
    return {k: v / max(1, n_batches) for k, v in total.items()}


def train(cfg: UnificationConfig) -> Dict[str, float]:
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    data, obs_dim = load_data_from_path(
        data_path=cfg.data_path,
        obs_dim=cfg.obs_dim,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        length=cfg.dataset_length,
    )
    cfg.obs_dim = obs_dim
    dataset = SequenceDataset(data, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, drop_last=True)

    model = RecurrentUnificationModel(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    builder = SelfModelStateBuilder(model)

    best_fixed_point = float("inf")
    best_step = 0
    no_improve = 0

    last: Dict[str, float] = {}

    for step in range(cfg.steps):
        accum = {"F_ext": 0.0, "F_int": 0.0, "KL": 0.0, "fixed": 0.0, "loss": 0.0}

        for batch in loader:
            batch = batch.to(cfg.device)
            bsz, seq_len, _ = batch.shape

            h_t = model.h_init.expand(bsz, -1).clone()
            z_t = torch.zeros(bsz, model.z_dim, device=cfg.device)

            ext_acc = torch.tensor(0.0, device=cfg.device)
            int_acc = torch.tensor(0.0, device=cfg.device)
            kl_acc = torch.tensor(0.0, device=cfg.device)
            fp_acc = torch.tensor(0.0, device=cfg.device)

            ext_ma = 0.0
            self_ma = 0.0

            for t in range(seq_len):
                x_t = batch[:, t]
                m_t = builder.build(x_t, h_t, z_t, (ext_ma, self_ma))
                out = model.step(x_t, h_t, z_t, m_t)
                z_t, h_t = out["z_new"], out["h_next"]

                f_ext, f_int, kl, fp = joint_variational_free_energy(
                    out["x_hat"],
                    x_t,
                    out["m_hat_mu"],
                    out["m_hat_logvar"],
                    m_t,
                    out["mu_q"],
                    out["logvar_q"],
                    out["mu_p"],
                    out["logvar_p"],
                    cfg,
                )

                ext_acc += f_ext
                int_acc += f_int
                kl_acc += kl
                fp_acc += fp
                ext_ma = float(f_ext.item())
                self_ma = float(f_int.item())

            F_ext = ext_acc / seq_len
            F_int = int_acc / seq_len
            KL = kl_acc / seq_len
            fixed_error = fp_acc / seq_len
            complexity = cfg.weight_decay * torch.stack([p.pow(2).sum() for p in model.parameters()]).sum()
            loss = F_ext + F_int + cfg.beta_kl * KL + complexity

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            if grad_norm is not None and torch.isfinite(grad_norm):
                builder.set_grad_norm(float(grad_norm.item()))

            accum["F_ext"] += float(F_ext.item())
            accum["F_int"] += float(F_int.item())
            accum["KL"] += float(KL.item())
            accum["fixed"] += float(fixed_error.item())
            accum["loss"] += float(loss.item())

        # batch mean
        for k in accum:
            accum[k] /= max(1, len(loader))

        # moving fixed-point criterion for convergence
        if accum["fixed"] < best_fixed_point - cfg.fixed_point_tol:
            best_fixed_point = accum["fixed"]
            best_step = step
            no_improve = 0
        else:
            no_improve += 1

        if step % cfg.print_every == 0 or step == cfg.steps - 1:
            print(
                f"step={step:04d} loss={accum['loss']:.6f} "
                f"F_ext={accum['F_ext']:.6f} F_int={accum['F_int']:.6f} "
                f"KL={accum['KL']:.6f} fixed={accum['fixed']:.6f}"
            )

        if cfg.eval_steps and (step + 1) % cfg.eval_steps == 0:
            eval_metrics = evaluate(model, loader, cfg, cfg.device)
            if eval_metrics["fixed"] <= accum["fixed"]:
                status = "self-loop improved"
            else:
                status = "self-loop worsened"
            print(
                f"eval|step={step+1:04d} "
                f"F_ext={eval_metrics['F_ext']:.6f} "
                f"F_int={eval_metrics['F_int']:.6f} "
                f"KL={eval_metrics['KL']:.6f} "
                f"fixed={eval_metrics['fixed']:.6f} ({status})"
            )

        if (step + 1) >= cfg.min_steps_for_early_stop and no_improve >= cfg.fixed_point_patience:
            print(f"early stop: fixed-point objective not improved for {cfg.fixed_point_patience} checks")
            break

        last = {
            "step": step,
            "loss": accum["loss"],
            "F_ext": accum["F_ext"],
            "F_int": accum["F_int"],
            "KL": accum["KL"],
            "fixed_point": accum["fixed"],
            "best_step": best_step,
            "stagnation": no_improve,
        }

    if not last:
        last = {
            "step": 0,
            "loss": float("nan"),
            "F_ext": float("nan"),
            "F_int": float("nan"),
            "KL": float("nan"),
            "fixed_point": float("nan"),
            "best_step": 0,
            "stagnation": 0,
        }
    last["best_fixed_point_step"] = best_step
    last["best_fixed_point_value"] = best_fixed_point
    return last


def parse_args() -> UnificationConfig:
    parser = argparse.ArgumentParser(description="Self-unification recurrent model")
    parser.add_argument("--dataset", choices=["synthetic", "custom"], default="synthetic")
    parser.add_argument("--data-path", type=str, default=None, help="Path to .npy/.npz/.csv/.pt data.")
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--self-dim", type=int, default=16)
    parser.add_argument("--repr-hid", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--sigma-x", type=float, default=0.25)
    parser.add_argument("--sigma-m", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fixed-point-patience", type=int, default=12)
    parser.add_argument("--fixed-point-tol", type=float, default=1e-4)
    parser.add_argument("--min-steps-for-early-stop", type=int, default=250)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--dataset-length", type=int, default=4096)

    args = parser.parse_args()
    if args.dataset == "custom" and args.data_path is None:
        raise ValueError("--data-path is required when --dataset=custom.")
    if args.dataset == "synthetic":
        args.data_path = None

    return UnificationConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        obs_dim=args.obs_dim,
        seq_len=args.seq_len,
        steps=args.steps,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        self_dim=args.self_dim,
        repr_hid=args.repr_hid,
        lr=args.lr,
        beta_kl=args.beta_kl,
        sigma_x=args.sigma_x,
        sigma_m=args.sigma_m,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        fixed_point_patience=args.fixed_point_patience,
        fixed_point_tol=args.fixed_point_tol,
        min_steps_for_early_stop=args.min_steps_for_early_stop,
        print_every=args.print_every,
        eval_steps=args.eval_steps,
        dataset_length=args.dataset_length,
    )


if __name__ == "__main__":
    cfg = parse_args() if __import__("sys").argv.__len__() > 1 else UnificationConfig()
    result = train(cfg)
    print("done:", result)
