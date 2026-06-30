"""
Benchmark harness for the unification architecture.

Compares:
- the unification model (joint external+self VFE loop)
- a baseline recurrent predictor (GRU next-step forecast)

Focus is on:
- external reconstruction error
- one-step predictive error (baseline)
- fixed-point residual (unification)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unification import (
    RecurrentUnificationModel,
    SelfModelStateBuilder,
    SequenceDataset,
    UnificationConfig,
    evaluate as evaluate_unification,
    gaussian_kl,
    joint_variational_free_energy,
    build_synthetic_data,
)


@dataclass
class BaselineConfig:
    hidden_dim: int = 64
    steps: int = 500
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    batch_size: int = 64
    seq_len: int = 24
    seed: int = 42
    device: str = "auto"


@dataclass
class NoSelfConfig:
    hidden_dim: int = 64
    steps: int = 500
    lr: float = 2e-3
    beta_kl: float = 1.0
    sigma_x: float = 0.25
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    batch_size: int = 64
    seq_len: int = 24
    seed: int = 42
    device: str = "auto"


def build_chaotic_data(length: int, obs_dim: int, seed: int) -> torch.Tensor:
    """
    A slightly more chaotic synthetic trajectory for stress-testing
    the joint self-model and baseline forecasting.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(length, obs_dim, generator=gen) * 2 * math.pi
    noise = torch.randn(length, obs_dim, generator=gen) * 0.06

    for t in range(2, length):
        d = 1.05 * torch.sin(x[t - 1]) - 0.45 * torch.sin(3 * x[t - 2]) + 0.15 * torch.cos(x[t - 3] if t > 2 else x[t - 2])
        x[t] = (x[t - 1] + d) % (2 * math.pi)

    return (torch.sin(x) + noise).float()


class BaselinePredictor(nn.Module):
    """
    Plain recurrent one-step predictor used as the baseline.
    Predicts x_{t+1} from x_{1:t}.
    """

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] ; predict for each t
        h, _ = self.rnn(x)
        return self.proj(h)


class NoSelfUnificationModel(nn.Module):
    """
    Same recurrent latent core as RecurrentUnificationModel but without self-observation input.
    This is the strict ablation baseline for apples-to-apples comparison.
    """

    def __init__(self, obs_dim: int, z_dim: int, hidden_dim: int, repr_hid: int, beta_kl: float = 1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.h_dim = hidden_dim
        self.beta_kl = beta_kl

        prior_in = self.z_dim + self.h_dim
        self.prior_net = nn.Sequential(
            nn.Linear(prior_in, repr_hid),
            nn.Tanh(),
            nn.Linear(repr_hid, repr_hid),
            nn.Tanh(),
            nn.Linear(repr_hid, self.z_dim * 2),
        )

        enc_in = self.obs_dim + self.h_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, repr_hid),
            nn.Tanh(),
            nn.Linear(repr_hid, repr_hid),
            nn.Tanh(),
            nn.Linear(repr_hid, self.z_dim * 2),
        )

        dec_in = self.z_dim + self.h_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, repr_hid),
            nn.Tanh(),
            nn.Linear(repr_hid, repr_hid),
            nn.Tanh(),
            nn.Linear(repr_hid, self.obs_dim),
        )

        recur_in = self.z_dim + self.obs_dim + self.h_dim
        self.recur = nn.GRUCell(recur_in, self.h_dim)
        self.h_init = nn.Parameter(torch.zeros(1, self.h_dim))

    @staticmethod
    def _split(stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, min=-12.0, max=8.0)

    def prior(self, z_prev: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_p, logvar_p = self._split(self.prior_net(torch.cat([z_prev, h], dim=-1)))
        return mu_p, logvar_p

    def encode(self, x_t: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_q, logvar_q = self._split(self.encoder(torch.cat([x_t, h], dim=-1)))
        logvar_q = torch.clamp(logvar_q, min=-12.0, max=8.0)
        eps = torch.randn_like(logvar_q)
        z = mu_q + torch.exp(0.5 * logvar_q) * eps
        return z, mu_q, logvar_q

    def decode(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, h], dim=-1))

    def step(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mu_p, logvar_p = self.prior(z_t, h_t)
        z_new, mu_q, logvar_q = self.encode(x_t, h_t)
        x_hat = self.decode(z_new, h_t)
        h_next = self.recur(torch.cat([z_new, x_t, h_t], dim=-1), h_t)

        return {
            "x_hat": x_hat,
            "z_new": z_new,
            "h_next": h_next,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
        }


def run_unification(
    observations: torch.Tensor,
    cfg: UnificationConfig,
    *,
    print_every: int = 0,
) -> Dict[str, float]:
    dataset = SequenceDataset(observations, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    model = RecurrentUnificationModel(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    builder = SelfModelStateBuilder(model)

    best_fixed_point = float("inf")
    no_improve = 0

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

        for k in accum:
            accum[k] /= max(1, len(loader))

        if accum["fixed"] < best_fixed_point - cfg.fixed_point_tol:
            best_fixed_point = accum["fixed"]
            no_improve = 0
        else:
            no_improve += 1

        if print_every and ((step + 1) % print_every == 0 or step == cfg.steps - 1):
            print(
                f"[unification][step={step+1}] loss={accum['loss']:.6f} "
                f"fixed={accum['fixed']:.6f} best_fixed={best_fixed_point:.6f}"
            )

        if (step + 1) >= cfg.min_steps_for_early_stop and no_improve >= cfg.fixed_point_patience:
            break

    eval_metrics = evaluate_unification(model, loader, cfg, cfg.device)
    return {
        "model": "unification",
        "train_steps": step + 1 if "step" in locals() else 0,
        "final_loss": accum["loss"],
        "train_F_ext": accum["F_ext"],
        "train_F_int": accum["F_int"],
        "train_KL": accum["KL"],
        "train_fixed_point": accum["fixed"],
        "eval_F_ext": eval_metrics["F_ext"],
        "eval_F_int": eval_metrics["F_int"],
        "eval_KL": eval_metrics["KL"],
        "eval_fixed_point": eval_metrics["fixed"],
        "eval_mse_from_F_ext": float((2 * (cfg.sigma_x ** 2)) * eval_metrics["F_ext"]),
        "best_fixed_point": best_fixed_point,
    }


def run_baseline(
    observations: torch.Tensor,
    cfg: BaselineConfig,
) -> Dict[str, float]:
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    dataset = SequenceDataset(observations, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = BaselinePredictor(obs_dim=observations.shape[1], hidden_dim=cfg.hidden_dim).to(cfg.device)
    optim_b = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    last_mse = float("inf")
    train_steps = 0

    for step in range(cfg.steps):
        mse_acc = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(cfg.device)
            # predict next step from prefix
            inp = batch[:, :-1, :]
            target = batch[:, 1:, :]

            pred = model(inp)
            pred = pred[:, : target.size(1), :]
            mse = ((pred - target).pow(2).mean())
            complexity = cfg.weight_decay * torch.stack([p.pow(2).sum() for p in model.parameters()]).sum()
            loss = mse + complexity

            optim_b.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim_b.step()

            mse_acc += float(loss.detach().item())
            n_batches += 1

        train_steps = step + 1
        last_mse = mse_acc / max(1, n_batches)

    # evaluation
    with torch.no_grad():
        eval_mse = 0.0
        eval_batches = 0
        for batch in loader:
            batch = batch.to(cfg.device)
            inp = batch[:, :-1, :]
            target = batch[:, 1:, :]
            pred = model(inp)[:, : target.size(1), :]
            eval_mse += ((pred - target).pow(2).mean()).item()
            eval_batches += 1

    return {
        "model": "baseline_rnn",
        "train_steps": train_steps,
        "final_mse": last_mse,
        "eval_mse": eval_mse / max(1, eval_batches),
    }


def run_noself_unification(
    observations: torch.Tensor,
    cfg: NoSelfConfig,
) -> Dict[str, float]:
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    dataset = SequenceDataset(observations, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = NoSelfUnificationModel(
        obs_dim=observations.shape[1],
        z_dim=12,
        hidden_dim=cfg.hidden_dim,
        repr_hid=96,
    ).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    last: Dict[str, float] = {}
    for step in range(cfg.steps):
        accum = {"F_ext": 0.0, "KL": 0.0, "loss": 0.0}
        for batch in loader:
            batch = batch.to(cfg.device)
            bsz, seq_len, _ = batch.shape

            h_t = model.h_init.expand(bsz, -1).clone()
            z_t = torch.zeros(bsz, model.z_dim, device=cfg.device)
            ext_acc = torch.tensor(0.0, device=cfg.device)
            kl_acc = torch.tensor(0.0, device=cfg.device)

            for t in range(seq_len):
                x_t = batch[:, t]
                out = model.step(x_t, h_t, z_t)
                h_t, z_t = out["h_next"], out["z_new"]

                f_ext = ((out["x_hat"] - x_t).pow(2).mean()) / (2 * (cfg.sigma_x ** 2))
                kl = gaussian_kl(out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"]).mean()
                ext_acc += f_ext
                kl_acc += kl

            F_ext = ext_acc / seq_len
            KL = kl_acc / seq_len
            complexity = cfg.weight_decay * torch.stack([p.pow(2).sum() for p in model.parameters()]).sum()
            loss = F_ext + cfg.beta_kl * KL + complexity

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            accum["F_ext"] += float(F_ext.item())
            accum["KL"] += float(KL.item())
            accum["loss"] += float(loss.item())

        for k in accum:
            accum[k] /= max(1, len(loader))
        last = accum

    # evaluation
    eval_F_ext = 0.0
    eval_KL = 0.0
    with torch.no_grad():
        eval_batches = 0
        for batch in loader:
            batch = batch.to(cfg.device)
            bsz, seq_len, _ = batch.shape
            h_t = model.h_init.expand(bsz, -1).clone()
            z_t = torch.zeros(bsz, model.z_dim, device=cfg.device)
            ext_acc = torch.tensor(0.0, device=cfg.device)
            kl_acc = torch.tensor(0.0, device=cfg.device)

            for t in range(seq_len):
                x_t = batch[:, t]
                out = model.step(x_t, h_t, z_t)
                h_t, z_t = out["h_next"], out["z_new"]
                f_ext = ((out["x_hat"] - x_t).pow(2).mean()) / (2 * (cfg.sigma_x ** 2))
                kl = gaussian_kl(out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"]).mean()
                ext_acc += f_ext
                kl_acc += kl

            eval_F_ext += float((ext_acc / seq_len).item())
            eval_KL += float((kl_acc / seq_len).item())
            eval_batches += 1

    return {
        "model": "noself_baseline",
        "train_steps": cfg.steps,
        "final_loss": last.get("loss", float("nan")),
        "train_F_ext": last.get("F_ext", float("nan")),
        "train_KL": last.get("KL", float("nan")),
        "eval_F_ext": eval_F_ext / max(1, eval_batches),
        "eval_KL": eval_KL / max(1, eval_batches),
        "eval_fixed_point": float("nan"),
        "eval_mse_from_F_ext": float((2 * (cfg.sigma_x ** 2)) * (eval_F_ext / max(1, eval_batches))),
    }


def make_benchmark_dataset(task: str, length: int, obs_dim: int, seed: int) -> Tuple[torch.Tensor, str]:
    if task == "smooth":
        return build_synthetic_data(length=length, obs_dim=obs_dim, seed=seed), "smooth"
    if task == "chaotic":
        return build_chaotic_data(length=length, obs_dim=obs_dim, seed=seed), "chaotic"
    raise ValueError(f"unknown task: {task}")


def compute_summary(rows: List[Dict[str, float]], output_path: str | None) -> Dict[str, Dict[str, float]]:
    # aggregate by model
    summary: Dict[str, Dict[str, List[float] | int]] = {}
    for row in rows:
        model = row["model"]
        if model not in summary:
            summary[model] = {
                "eval_mse": [],
                "eval_mse_from_F_ext": [],
                "eval_F_ext": [],
                "eval_F_int": [],
                "eval_fixed_point": [],
                "seeds": 0,
            }
        if "eval_mse" in row:
            summary[model]["eval_mse"].append(row["eval_mse"])  # type: ignore[index]
        if "eval_mse_from_F_ext" in row:
            summary[model]["eval_mse_from_F_ext"].append(row["eval_mse_from_F_ext"])  # type: ignore[index]
        if "eval_F_ext" in row:
            summary[model]["eval_F_ext"].append(row["eval_F_ext"])  # type: ignore[index]
        if "eval_F_int" in row:
            summary[model]["eval_F_int"].append(row["eval_F_int"])  # type: ignore[index]
        if "eval_fixed_point" in row and isinstance(row["eval_fixed_point"], (int, float)) and not isinstance(row["eval_fixed_point"], bool):
            summary[model]["eval_fixed_point"].append(row["eval_fixed_point"])  # type: ignore[index]
        summary[model].setdefault("seeds", 0)
        summary[model]["seeds"] += 1

    # reduce into readable values
    def safe_mean(values: List[float]) -> float:
        return sum([float(x) for x in values]) / len(values) if len(values) > 0 else float("nan")

    compact: Dict[str, Dict[str, float]] = {}
    for model, vals in summary.items():
        mse_values = vals["eval_mse"]  # type: ignore[index]
        mse_from_fext_values = vals["eval_mse_from_F_ext"]  # type: ignore[index]
        fext_values = vals["eval_F_ext"]  # type: ignore[index]
        fint_values = vals["eval_F_int"]  # type: ignore[index]
        fp_values = vals["eval_fixed_point"]  # type: ignore[index]
        seeds = vals["seeds"]  # type: ignore[assignment]

        compact[model] = {
            "n": float(seeds),  # keep JSON-friendly
            "eval_mse_from_F_ext_mean": safe_mean(mse_from_fext_values),
            "eval_F_ext_mean": safe_mean(fext_values),
            "eval_F_int_mean": safe_mean(fint_values),
            "eval_fixed_point_mean": safe_mean(fp_values),
            "eval_mse_mean": safe_mean(mse_values),
        }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"runs": rows, "summary": compact}, f, indent=2)

    return compact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run minimal benchmarks against unification")
    p.add_argument("--unification-steps", type=int, default=220)
    p.add_argument("--baseline-steps", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--dataset-length", type=int, default=2048)
    p.add_argument("--obs-dim", type=int, default=4)
    p.add_argument("--seeds", type=str, default="42,43")
    p.add_argument("--tasks", type=str, default="smooth,chaotic")
    p.add_argument("--noself-steps", type=int, default=None, help="No-self baseline steps; defaults to unification-steps when unset.")
    p.add_argument("--include-rnn-baseline", action="store_true", help="Include GRU one-step predictor baseline.")
    p.add_argument("--output", type=str, default="benchmark_results.json")
    p.add_argument("--print-every", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    rows: List[Dict[str, float]] = []
    for task in tasks:
        for seed in seeds:
            observations, task_name = make_benchmark_dataset(task=task, length=args.dataset_length, obs_dim=args.obs_dim, seed=seed)
            # ensure at least task-level identity
            print(f"task={task_name} seed={seed} start")

            uni_cfg = UnificationConfig(
                dataset="custom",
                data_path=None,
                obs_dim=args.obs_dim,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                hidden_dim=64,
                steps=args.unification_steps,
                lr=2e-3,
                beta_kl=1.0,
                sigma_x=0.25,
                sigma_m=0.2,
                weight_decay=1e-5,
                grad_clip=1.0,
                seed=seed,
                device=args.device,
                fixed_point_patience=8,
                fixed_point_tol=1e-4,
                min_steps_for_early_stop=120,
                print_every=args.print_every,
                eval_steps=0,
            )
            uni_res = run_unification(observations, uni_cfg, print_every=args.print_every)
            uni_res["task"] = task_name
            uni_res["seed"] = seed
            rows.append(uni_res)

            noself_cfg = NoSelfConfig(
                hidden_dim=64,
                steps=args.unification_steps if args.noself_steps is None else args.noself_steps,
                lr=2e-3,
                beta_kl=1.0,
                sigma_x=0.25,
                weight_decay=1e-5,
                grad_clip=1.0,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=seed,
                device=args.device,
            )
            noself_res = run_noself_unification(observations, noself_cfg)
            noself_res["task"] = task_name
            noself_res["seed"] = seed
            rows.append(noself_res)

            if args.include_rnn_baseline:
                base_cfg = BaselineConfig(
                    hidden_dim=64,
                    steps=args.baseline_steps,
                    lr=2e-3,
                    weight_decay=1e-5,
                    grad_clip=1.0,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    seed=seed,
                    device=args.device,
                )
                base_res = run_baseline(observations, base_cfg)
                base_res["task"] = task_name
                base_res["seed"] = seed
                rows.append(base_res)

    summary = compute_summary(rows, args.output)
    print(f"benchmark saved to {args.output}")
    print("summary:")
    for model, vals in summary.items():
        print(f"{model}: {vals}")


if __name__ == "__main__":
    main()
