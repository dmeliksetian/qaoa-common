# qaoa-common

Shared utilities and problem instances for
[`qaoa-noise-landscape`](https://github.com/dmeliksetian/qaoa-noise-landscape):
QUBO construction and brute-force solving, QAOA circuit construction and
parameter binding, circuit folding for zero-noise extrapolation, and plotting.

## Install

```bash
pip install -e .
```

Requires Python 3.12 or later, `numpy`, `qiskit` and `matplotlib`.

## Instances

`instances/` holds cardinality-constrained mean–variance portfolio QUBOs built
from daily prices, January 2019 – December 2023. Each `.npz` file contains the
QUBO matrix `Q`, cardinality `k`, asset names, the brute-force optimum `e_opt`
and `x_opt`, and the random-sampling feasibility baseline; a `.json` sidecar,
where present, records how the instance was built.

Files used in arXiv:2604.19426 v2:

| Instance | File | E* |
|----------|------|----|
| 6 low-vol   | `6asset_lowvol_hires_unnorm.npz` | −16.885 |
| 8 low-vol   | `8asset_lowvol_hires.npz`        | −46.265 |
| 8 TSLA      | `8asset_tsla_hires.npz`          | −56.563 |
| 12 low-vol  | `12asset_lowvol_hires.npz`       | −156.361 |
| 12 high-vol | `12asset_highvol_hires.npz`      | −264.648 |
| 16 low-vol  | `16asset_lowvol_hires.npz`       | −434.338 |
| 16 high-vol | `16asset_highvol_hires.npz`      | −1134.467 |

`6asset_lowvol_hires.npz`, `8asset_lowvol.npz` and `8asset_tsla.npz` are
normalized variants (max |Q| = 1). `6asset_lowvol_hires_rerun.npz` is identical
in content to `6asset_lowvol_hires_unnorm.npz`.

## License

MIT (see `LICENSE`).
