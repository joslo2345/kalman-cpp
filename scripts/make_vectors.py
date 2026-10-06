"""Generate the frozen benchmark scenarios S1-S5 into tests/vectors/.

Usage:
    python scripts/make_vectors.py           # (re)generate files and SHA256SUMS
    python scripts/make_vectors.py --check   # verify files against SHA256SUMS

The files are the shared inputs for every implementation (C, C++, Python,
Rust), so do not regenerate them after results have been collected; changing a
scenario after seeing which library wins makes the numbers meaningless.

Each scenario is a folder of .npy arrays (little-endian, C order):
    F, H, Q, R, x0, P0   model matrices (S3 has no H: its measurement is nonlinear)
    dt                   time step, shape ()
    truth                true states, shape (steps, N) or (seeds, steps, N)
    z                    measurements, shape (steps, M) or (seeds, steps, M)
Measurement k is taken after the k-th predict from x0; truth[k] is the state
at that time. S4 is stored in float32 and has no truth (it measures
numerical stability only).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "tests" / "vectors"


def psd_sqrt(M):
    """Any A with A A^T = M, for symmetric positive semi-definite M."""
    w, V = np.linalg.eigh(M)
    return V * np.sqrt(np.clip(w, 0.0, None))


def cv_transition(dt, axes):
    F = np.eye(2 * axes)
    F[:axes, axes:] = dt * np.eye(axes)
    return F


def cv_process_noise(dt, accel_sigma, axes):
    q = accel_sigma**2
    block = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    Q = np.zeros((2 * axes, 2 * axes))
    for a in range(axes):
        idx = [a, a + axes]
        Q[np.ix_(idx, idx)] = block
    return Q


def simulate_linear(rng, F, H, Q, R, x0, P0, steps):
    sq, sr = psd_sqrt(Q), psd_sqrt(R)
    x = x0 + psd_sqrt(P0) @ rng.standard_normal(len(x0))
    truth, zs = [], []
    for _ in range(steps):
        x = F @ x + sq @ rng.standard_normal(len(x0))
        truth.append(x)
        zs.append(H @ x + sr @ rng.standard_normal(H.shape[0]))
    return np.array(truth), np.array(zs)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def s1():
    dt = 0.1
    F, Q = cv_transition(dt, 1), cv_process_noise(dt, 0.5, 1)
    H, R = np.array([[1.0, 0.0]]), np.array([[1.0]])
    x0, P0 = np.array([0.0, 1.0]), np.diag([10.0, 4.0])
    truth, z = simulate_linear(np.random.default_rng(101), F, H, Q, R, x0, P0, 10_000)
    return dict(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, dt=np.float64(dt), truth=truth, z=z)


def s2():
    dt = 0.1
    F, Q = cv_transition(dt, 2), cv_process_noise(dt, 0.5, 2)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = np.eye(2)
    x0, P0 = np.array([0.0, 0.0, 1.0, 0.5]), np.diag([10.0, 10.0, 4.0, 4.0])
    truth, z = simulate_linear(np.random.default_rng(102), F, H, Q, R, x0, P0, 10_000)
    return dict(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, dt=np.float64(dt), truth=truth, z=z)


def s3():
    # Close pass: the target moves along y = 2 past a sensor at the origin that
    # has an accurate range (0.2 m) but a poor bearing (0.3 rad).
    dt, seeds, steps = 0.1, 200, 500
    F, Q = cv_transition(dt, 2), cv_process_noise(dt, 0.5, 2)
    R = np.diag([0.2**2, 0.3**2])
    x0, P0 = np.array([-25.0, 2.0, 5.0, 0.0]), np.diag([25.0, 25.0, 4.0, 4.0])
    rng = np.random.default_rng(103)
    sq, sr, sp = psd_sqrt(Q), psd_sqrt(R), psd_sqrt(P0)
    truth = np.zeros((seeds, steps, 4))
    z = np.zeros((seeds, steps, 2))
    for s in range(seeds):
        x = x0 + sp @ rng.standard_normal(4)
        for k in range(steps):
            x = F @ x + sq @ rng.standard_normal(4)
            truth[s, k] = x
            meas = np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])]) + sr @ rng.standard_normal(2)
            meas[1] = wrap(meas[1])
            z[s, k] = meas
    return dict(F=F, Q=Q, R=R, x0=x0, P0=P0, dt=np.float64(dt), truth=truth, z=z)


def s4():
    # A vague prior (variance 1e6) against very precise measurements (variance
    # 1e-6), run in float32: stresses the covariance update numerically.
    dt, steps = 1.0, 1_000_000
    F, Q = cv_transition(dt, 2), cv_process_noise(dt, 1e-4, 2)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = 1e-6 * np.eye(2)
    x0, P0 = np.array([0.0, 0.0, 1.0, -1.0]), np.diag([1e6, 1e6, 1e4, 1e4])
    _, z = simulate_linear(np.random.default_rng(104), F, H, Q, R, x0, P0, steps)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return dict(F=f32(F), H=f32(H), Q=f32(Q), R=f32(R), x0=f32(x0), P0=f32(P0), dt=np.float32(dt), z=f32(z))


def s5():
    # Linearized INS error state, 15 states: position, velocity, attitude,
    # accelerometer bias, gyro bias (3 each), with GPS position + velocity
    # measurements. Level flight: specific force f = (0, 0, 9.81), body = nav.
    dt, steps = 0.01, 10_000
    I3, Z3 = np.eye(3), np.zeros((3, 3))
    f = np.array([0.0, 0.0, 9.81])
    skew = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    A = np.block([
        [Z3, I3, Z3, Z3, Z3],
        [Z3, Z3, -skew(f), -I3, Z3],
        [Z3, Z3, Z3, Z3, -I3],
        [Z3, Z3, Z3, Z3, Z3],
        [Z3, Z3, Z3, Z3, Z3],
    ])
    F = np.eye(15) + A * dt + A @ A * dt**2 / 2
    accel_noise, gyro_noise, accel_bias_rw, gyro_bias_rw = 0.05, 0.005, 1e-4, 1e-5
    Q = np.zeros((15, 15))
    Q[3:6, 3:6] = accel_noise**2 * dt * I3
    Q[6:9, 6:9] = gyro_noise**2 * dt * I3
    Q[9:12, 9:12] = accel_bias_rw**2 * dt * I3
    Q[12:15, 12:15] = gyro_bias_rw**2 * dt * I3
    H = np.hstack([np.eye(6), np.zeros((6, 9))])
    R = np.diag([1.0] * 3 + [0.01] * 3)
    x0 = np.zeros(15)
    P0 = np.diag([4.0] * 3 + [0.25] * 3 + [0.01] * 3 + [0.01] * 3 + [1e-4] * 3)
    truth, z = simulate_linear(np.random.default_rng(105), F, H, Q, R, x0, P0, steps)
    return dict(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, dt=np.float64(dt), truth=truth, z=z)


SCENARIOS = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_files():
    return sorted(p for p in ROOT.glob("S*/*.npy"))


def generate():
    for name, make in SCENARIOS.items():
        folder = ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.npy"):
            old.unlink()
        for key, value in make().items():
            np.save(folder / f"{key}.npy", np.ascontiguousarray(value), allow_pickle=False)
        print(f"wrote {name}")
    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in all_files()]
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote SHA256SUMS ({len(lines)} files)")


def check():
    expected = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        expected[rel] = digest
    bad = [rel for rel, digest in expected.items() if not (ROOT / rel).exists() or sha256(ROOT / rel) != digest]
    extra = [p.relative_to(ROOT).as_posix() for p in all_files() if p.relative_to(ROOT).as_posix() not in expected]
    for rel in bad:
        print(f"MISMATCH {rel}")
    for rel in extra:
        print(f"UNLISTED {rel}")
    print("ok" if not bad and not extra else "FAILED")
    return 0 if not bad and not extra else 1


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else generate())
