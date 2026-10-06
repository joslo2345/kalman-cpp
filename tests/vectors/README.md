# Benchmark scenarios S1–S5

Frozen inputs shared by every implementation of this library (C, C++, Python, Rust). Every library reads exactly these files, so the results can be compared across libraries and across implementations.

**Do not regenerate these files after results have been collected.** Changing a scenario after seeing which library wins makes the numbers meaningless. To check that the files are intact, run `python scripts/make_vectors.py --check`.

| ID | Scenario | N / M | Data | Notes |
|---|---|---|---|---|
| S1 | 1D constant velocity | 2 / 1 | 10,000 steps | |
| S2 | 2D constant velocity | 4 / 2 | 10,000 steps | |
| S3 | Range-bearing tracking (close pass) | 4 / 2 | 200 seeds × 500 steps | Nonlinear. Bearing residuals must be wrapped to (−π, π]. |
| S4 | Ill-conditioned problem | 4 / 2 | 1,000,000 steps, float32 | Prior variance 1e6, measurement variance 1e-6. Has no truth. |
| S5 | Linearized INS error state | 15 / 6 | 10,000 steps | GPS position and velocity measurements. |

## File format

Each scenario folder holds `.npy` arrays: little-endian, C order, float64 except S4 (float32).

| Array | Shape |
|---|---|
| `F`, `Q`, `P0` | N × N |
| `H` | M × N (not in S3, whose measurement model is nonlinear) |
| `R` | M × M |
| `x0` | N |
| `dt` | () |
| `truth` | (steps, N), or (seeds, steps, N) for S3 |
| `z` | (steps, M), or (seeds, steps, M) for S3 |

Measurement `z[k]` is taken after the (k+1)-th predict from `x0`, and `truth[k]` is the true state at that time.

## Models

The full models are in `scripts/make_vectors.py`.

- **S3 measurement:** range and bearing from a sensor at the origin, `[hypot(px, py), atan2(py, px)]`.
- **S3 motion:** constant velocity, with `F` given.
