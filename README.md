# Unification

This project implements a minimal recurrent self-referential inference system:

- external observations `x_t` are predicted from a recurrent latent state
- self-observations `m_t` are predicted from the same latent state + hidden state
- weights/activations and short error memory are treated as self-observed signals
- a single objective minimizes joint variational free energy:
  - external reconstruction energy
  - self-reconstruction energy
  - KL between prior and posterior latents
  - explicit parameter complexity term
- no reward/task loss

The model is aligned to the fixed-point intuition:
coherence grows when the self-loop residual (`|m̂_t - m_t|`) is small and stops improving.

## Files

- `unification.py` — core architecture, training loop, and dataset loading

## Run

```bash
python unification.py --steps 2000 --print-every 100
```

### Synthetic only

```bash
python unification.py
```

### Custom data

```bash
python unification.py --dataset custom --data-path ./data/my_series.npy
```

Supported formats:

- `*.npy`
- `*.npz` (uses key `data` when present, otherwise first array)
- `*.csv` / `*.txt`
- `*.pt` / `*.pth`

## Main outputs

- `loss`: full objective value
- `F_ext`: external free-energy term
- `F_int`: self-model free-energy term
- `KL`: latent KL term
- `fixed_point`: self-loop residual term (`mean|m̂_t - m_t|`)

Early stopping is based on fixed-point stability:
if `fixed_point` does not improve by `--fixed-point-tol`
for `--fixed-point-patience` checks after `--min-steps-for-early-stop`, training stops.

## Benchmarking

Use the benchmark harness to compare unification against a baseline recurrent predictor:

```bash
python3 benchmark_unification.py \
  --tasks smooth,chaotic \
  --seeds 11,12 \
  --unification-steps 40 \
  --baseline-steps 40 \
  --dataset-length 512 \
  --batch-size 32 \
  --seq-len 20 \
  --print-every 20 \
  --output benchmark_results.json
```

Benchmark output:

- prints per-task diagnostics
- writes per-run JSON at `--output` containing all raw run records and summary
- includes `eval_F_ext`, `eval_F_int`, `eval_fixed_point` for unification and `eval_mse` for baseline

See [BENCHMARK_REPORT.md](/home/psc/repo/ideas/A.I/unification/BENCHMARK_REPORT.md) for a sample evidence set.
