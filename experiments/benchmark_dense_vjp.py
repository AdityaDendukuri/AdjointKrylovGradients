"""
benchmark_dense_vjp.py

Dense matrix-exponential VJP benchmark comparing backends on CPU and GPU:
  - scipy   (block-matrix Fréchet reference, CPU)
  - JAX     (jax.scipy.linalg.expm + jax.grad, CPU loop / CPU scan+einsum / GPU scan+einsum)
  - PyTorch (torch.linalg.matrix_exp + autograd, CPU / GPU)

Run:
    python -m pip install -r requirements-benchmark.txt
    python experiments/benchmark_dense_vjp.py
"""

import csv
import os
import numpy as np
import scipy.linalg as sla
import time, sys

USE_JAX   = True
USE_TORCH = True

if USE_JAX:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.linalg import expm as jexpm
    # Check for GPU
    JAX_GPU = any(d.platform == "gpu" for d in jax.devices())
    print(f"JAX {jax.__version__} | devices: {jax.devices()}")

if USE_TORCH:
    import torch
    TORCH_GPU = torch.cuda.is_available()
    TORCH_DEV = torch.device("cuda") if TORCH_GPU else torch.device("cpu")
    if TORCH_GPU:
        print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"PyTorch {torch.__version__} | CPU only")


# ── FPE generator library (matches fpe.jl) ────────────────────────────────────
def fpe_library_1d(n, L=2.0, D=0.5, M=10):
    xs = np.linspace(-L, L, n)
    h  = 2*L / (n - 1)

    diag = np.full(n, -2*D/h**2)
    diag[0] = diag[-1] = -D/h**2
    off = np.full(n-1, D/h**2)
    A0  = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

    L_scale = L
    dirs = []
    for j in range(1, M+1):
        bv = (xs / L_scale)**j
        d  = np.zeros((n, n))
        for k in range(n):
            if k < n - 1:
                d[k+1, k] +=  bv[k] / (2*h)
            if k > 0:
                d[k-1, k] += -bv[k] / (2*h)
        d[0,   0  ] += -bv[0]   / (2*h)
        d[n-1, n-1] +=  bv[-1]  / (2*h)
        dirs.append(d)

    return xs, A0, dirs


def make_data(A_true, n, T_obs=8, dt=0.3):
    xs = np.linspace(-2, 2, n)
    p0 = np.exp(-0.5 * xs**2); p0 /= p0.sum()
    expA = sla.expm(A_true * dt)
    ps = [p0]
    for _ in range(T_obs):
        pn = np.maximum(expA @ ps[-1], 0); pn /= pn.sum()
        ps.append(pn)
    return ps, [dt] * T_obs


# ── scipy: block-matrix Fréchet (suboptimal — O((2n)^3)) ─────────────────────
def gradient_scipy(A, dirs, p_data, dts):
    n  = A.shape[0]
    M  = len(dirs)
    At = A.T
    Z  = np.zeros((n, n))
    grad = np.zeros(M)
    loss = 0.0
    for t, dt in enumerate(dts):
        expAdt = sla.expm(A * dt)
        p_pred = expAdt @ p_data[t]
        r      = p_data[t+1] - p_pred
        loss  += 0.5 * np.dot(r, r)
        if np.linalg.norm(r) < 1e-15:
            continue
        # The loss cotangent is p_pred - p_next = -r.
        X = -dt * np.outer(r, p_data[t])
        B = np.block([[At*dt, X], [Z, At*dt]])
        G = sla.expm(B)[:n, n:]
        for j in range(M):
            grad[j] += np.sum(G * dirs[j])
    return grad, loss


# ── JAX: dense autodiff through jax.scipy.linalg.expm ─────────────────────────
# Two variants:
#   gradient_jax     — original: JIT'd per-step, Python time-loop
#   gradient_jax_opt — optimised: lax.scan over time, vmap/einsum for M projections
if USE_JAX:
    # --- original (per-step JIT, Python loop) ---
    @jax.jit
    def _jax_step_grad(A, p_t, p_next, dt, dirs_stack):
        def loss_fn(A):
            return 0.5 * jnp.sum((p_next - jexpm(A * dt) @ p_t)**2)
        grad_A = jax.grad(loss_fn)(A)
        # vectorised projection: einsum over the M dirs at once
        return jnp.einsum('ij,kij->k', grad_A, dirs_stack)

    def gradient_jax(A, dirs, p_data, dts):
        An   = jnp.array(A)
        dn   = jnp.stack([jnp.array(d) for d in dirs])
        pn   = [jnp.array(p) for p in p_data]
        grad = jnp.zeros(len(dirs))
        for t, dt in enumerate(dts):
            grad = grad + _jax_step_grad(An, pn[t], pn[t+1], float(dt), dn)
        return np.array(grad), 0.0

    # --- optimised: single JIT call, lax.scan over T, einsum over M ---
    def _make_jax_opt(T_obs):
        """Factory so lax.scan length is a compile-time constant."""
        @jax.jit
        def _gradient_jax_opt(A, dirs_stack, p_stack, dts_array):
            # p_stack:   (T+1, n)
            # dirs_stack: (M, n, n)
            # dts_array:  (T,)
            def scan_fn(grad_acc, t):
                p_t    = p_stack[t]
                p_next = p_stack[t + 1]
                dt     = dts_array[t]
                def loss_fn(A_):
                    return 0.5 * jnp.sum((p_next - jexpm(A_ * dt) @ p_t) ** 2)
                grad_A = jax.grad(loss_fn)(A)
                step_g = jnp.einsum('ij,kij->k', grad_A, dirs_stack)
                return grad_acc + step_g, None
            grad, _ = jax.lax.scan(scan_fn, jnp.zeros(dirs_stack.shape[0]),
                                   jnp.arange(T_obs))
            return grad
        return _gradient_jax_opt

    # cache compiled versions keyed by T
    _jax_opt_cache = {}

    def gradient_jax_opt(A, dirs, p_data, dts):
        T = len(dts)
        if T not in _jax_opt_cache:
            _jax_opt_cache[T] = _make_jax_opt(T)
        fn = _jax_opt_cache[T]
        An  = jnp.array(A)
        dn  = jnp.stack([jnp.array(d) for d in dirs])
        pn  = jnp.stack([jnp.array(p) for p in p_data])      # (T+1, n)
        dts_arr = jnp.array(dts)
        grad = fn(An, dn, pn, dts_arr)
        return np.array(grad), 0.0


if USE_TORCH:
    def gradient_torch(A_np, dirs_np, p_data_np, dts):
        A    = torch.tensor(A_np, dtype=torch.float64, requires_grad=True)
        dirs = [torch.tensor(d,   dtype=torch.float64) for d in dirs_np]
        ps   = [torch.tensor(p,   dtype=torch.float64) for p in p_data_np]
        loss = sum(
            0.5 * torch.sum((ps[t+1] - torch.linalg.matrix_exp(A * dt) @ ps[t])**2)
            for t, dt in enumerate(dts)
        )
        loss.backward()
        grad_A = A.grad
        # vectorised projection with einsum
        dirs_t = torch.stack(dirs)   # (M, n, n)
        grad   = torch.einsum('ij,kij->k', grad_A, dirs_t)
        return grad.detach().numpy(), float(loss.detach())

    # GPU variant: move everything to CUDA, use float64 on GPU
    if TORCH_GPU:
        def gradient_torch_gpu(A_np, dirs_np, p_data_np, dts):
            A    = torch.tensor(A_np, dtype=torch.float64, device=TORCH_DEV,
                                requires_grad=True)
            dirs_t = torch.tensor(np.stack(dirs_np), dtype=torch.float64,
                                  device=TORCH_DEV)   # (M, n, n)
            ps   = [torch.tensor(p, dtype=torch.float64, device=TORCH_DEV)
                    for p in p_data_np]
            loss = sum(
                0.5 * torch.sum((ps[t+1] - torch.linalg.matrix_exp(A * dt) @ ps[t])**2)
                for t, dt in enumerate(dts)
            )
            loss.backward()
            grad = torch.einsum('ij,kij->k', A.grad, dirs_t)
            torch.cuda.synchronize()
            return grad.cpu().detach().numpy(), float(loss.detach().cpu())


# ── JAX GPU: per-step JIT loop (avoids lax.scan materialisation, works at large n) ──
if USE_JAX and JAX_GPU:
    @jax.jit
    def _jax_gpu_step(A, p_t, p_next, dt, dirs_stack):
        """Single time-step gradient on GPU."""
        def loss_fn(A_):
            return 0.5 * jnp.sum((p_next - jexpm(A_ * dt) @ p_t) ** 2)
        grad_A = jax.grad(loss_fn)(A)
        return jnp.einsum('ij,kij->k', grad_A, dirs_stack)

    def gradient_jax_gpu(A, dirs, p_data, dts):
        """GPU gradient via per-step JIT: O(n³) per step on GPU, no scan OOM."""
        gpu = jax.devices("gpu")[0]
        An  = jax.device_put(jnp.array(A), gpu)
        dn  = jax.device_put(jnp.stack([jnp.array(d) for d in dirs]), gpu)
        pn  = [jax.device_put(jnp.array(p), gpu) for p in p_data]
        grad = jnp.zeros(len(dirs))
        for t, dt in enumerate(dts):
            grad = grad + _jax_gpu_step(An, pn[t], pn[t+1], float(dt), dn)
        jax.block_until_ready(grad)
        return np.array(grad), 0.0


# ── Device-resident benchmark closures ───────────────────────────────────────
# Backend conversion and host/device transfer happen once in these preparation
# functions and are excluded from the steady-state backward-pass timing.

if USE_JAX:
    def prepare_jax_opt(A, dirs, p_data, dts):
        cpu = jax.devices("cpu")[0]
        T = len(dts)
        if T not in _jax_opt_cache:
            _jax_opt_cache[T] = _make_jax_opt(T)
        fn = _jax_opt_cache[T]
        An = jax.device_put(np.asarray(A, dtype=np.float64), cpu)
        dn = jax.device_put(np.stack(dirs).astype(np.float64, copy=False), cpu)
        pn = jax.device_put(np.stack(p_data).astype(np.float64, copy=False), cpu)
        dts_arr = jax.device_put(np.asarray(dts, dtype=np.float64), cpu)

        def run():
            grad = fn(An, dn, pn, dts_arr)
            jax.block_until_ready(grad)
            return grad

        return run


if USE_TORCH:
    def prepare_torch(A, dirs, p_data, dts, device):
        A_base = torch.as_tensor(A, dtype=torch.float64, device=device)
        dirs_t = torch.as_tensor(np.stack(dirs), dtype=torch.float64, device=device)
        ps = tuple(torch.as_tensor(p, dtype=torch.float64, device=device) for p in p_data)
        dts_fixed = tuple(float(dt) for dt in dts)

        def run():
            A_leaf = A_base.detach().requires_grad_(True)
            loss = sum(
                0.5 * torch.sum(
                    (ps[t + 1] - torch.linalg.matrix_exp(A_leaf * dt) @ ps[t]) ** 2
                )
                for t, dt in enumerate(dts_fixed)
            )
            grad_A, = torch.autograd.grad(loss, A_leaf)
            grad = torch.einsum("ij,kij->k", grad_A, dirs_t)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return grad

        return run

    def prepare_torch_cpu(A, dirs, p_data, dts):
        return prepare_torch(A, dirs, p_data, dts, torch.device("cpu"))

    if TORCH_GPU:
        def prepare_torch_gpu(A, dirs, p_data, dts):
            return prepare_torch(A, dirs, p_data, dts, TORCH_DEV)


if USE_JAX and JAX_GPU:
    def prepare_jax_gpu(A, dirs, p_data, dts):
        gpu = jax.devices("gpu")[0]
        An = jax.device_put(np.asarray(A, dtype=np.float64), gpu)
        dn = jax.device_put(np.stack(dirs).astype(np.float64, copy=False), gpu)
        pn = tuple(jax.device_put(np.asarray(p, dtype=np.float64), gpu) for p in p_data)
        dts_fixed = tuple(float(dt) for dt in dts)

        def run():
            grad = jax.device_put(np.zeros(len(dirs), dtype=np.float64), gpu)
            for t, dt in enumerate(dts_fixed):
                grad = grad + _jax_gpu_step(An, pn[t], pn[t + 1], dt, dn)
            jax.block_until_ready(grad)
            return grad

        return run


def timeit(fn, n_trials=5, n_warmup=5):
    for _ in range(n_warmup):  # extra warmup for PyTorch JIT compilation
        try: fn()
        except Exception: break
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


# ── Backend sets ─────────────────────────────────────────────────────────────
def build_backends_cpu():
    """CPU backends only — for moderate-n sweep where GPU overhead dominates."""
    backends = []
    if USE_JAX:
        backends.append(("JAX CPU (compiled scan)", prepare_jax_opt))
    if USE_TORCH:
        backends.append(("PyTorch CPU (eager)", prepare_torch_cpu))
    return backends


def build_backends_gpu():
    """GPU + CPU backends for large-n sweep — CPU included for completeness."""
    backends = []
    if USE_JAX:
        backends.append(("JAX CPU (compiled scan)", prepare_jax_opt))
        if JAX_GPU:
            backends.append(("JAX GPU", prepare_jax_gpu))
    if USE_TORCH:
        backends.append(("PyTorch CPU (eager)", prepare_torch_cpu))
        if TORCH_GPU:
            backends.append(("PyTorch GPU", prepare_torch_gpu))
    return backends


def build_backends_m():
    """M-sweep: CPU only (n fixed at 80, GPU gains negligible at small n)."""
    backends = []
    if USE_JAX:
        backends.append(("JAX CPU (compiled scan)", prepare_jax_opt))
    if USE_TORCH:
        backends.append(("PyTorch CPU (eager)", prepare_torch_cpu))
    return backends


def run_problem(n, M, T_obs=8, dt=0.3):
    xs, A0, dirs = fpe_library_1d(n, M=M)
    theta_true = np.zeros(M); theta_true[0] = -2*2.0
    if M >= 3: theta_true[2] = 0.5*2.0**3
    A_true = A0 + sum(theta_true[j]*dirs[j] for j in range(M))
    p_data, dts = make_data(A_true, n, T_obs=T_obs, dt=dt)
    return A_true, dirs, p_data, dts


def run_sweep(label, sweep_key, vals, backends, build_fn, n_trials=7, **kw):
    header = f"{sweep_key:>6}" + "".join(f"  {name:>22}" for name, _ in backends)
    print(f"\n=== {label} ===")
    print(header); print("-" * len(header))
    all_results = {name: [] for name, _ in backends}

    for v in vals:
        A_true, dirs, p_data, dts = build_fn(v, **kw)
        prepared = {}
        for name, prepare in backends:
            try:
                prepared[name] = prepare(A_true, dirs, p_data, dts)
                prepared[name](); prepared[name]()
            except Exception as e:
                prepared[name] = None
                print(f"  [{name}] preparation error at {sweep_key}={v}: {e}")

        row = f"{v:>6}"
        for name, _ in backends:
            try:
                if prepared[name] is None:
                    raise RuntimeError("backend preparation failed")
                t_ms = timeit(prepared[name], n_trials=n_trials)
            except Exception as e:
                t_ms = float("nan")
                print(f"  [{name}] error at {sweep_key}={v}: {e}")
            row += f"  {t_ms:>20.2f}ms"
            all_results[name].append((v, t_ms))
        print(row); sys.stdout.flush()

    return all_results


def run_n_sweep(n_vals, M_fixed=10, T_obs=8, dt=0.3, n_trials=7):
    """CPU backends at moderate n — shows O(n) vs O(n³) scaling."""
    return run_sweep(f"CPU n-sweep (M={M_fixed})", "n", n_vals,
                     build_backends_cpu(),
                     lambda n: run_problem(n, M_fixed, T_obs, dt),
                     n_trials=n_trials)


def run_n_sweep_gpu(n_vals, M_fixed=10, T_obs=8, dt=0.3, n_trials=5,
                   cpu_max_n=1000):
    """GPU + CPU backends at large n.
    CPU backends are skipped above cpu_max_n (too slow) and return NaN."""
    gpu_only = {"JAX GPU", "PyTorch GPU"}
    all_backends = build_backends_gpu()
    label = f"GPU n-sweep (M={M_fixed})"
    header = f"{'n':>6}" + "".join(f"  {name:>22}" for name, _ in all_backends)
    print(f"\n=== {label} ===")
    print(header); print("-" * len(header))
    all_results = {name: [] for name, _ in all_backends}

    for n in n_vals:
        A_true, dirs, p_data, dts = run_problem(n, M_fixed, T_obs, dt)
        prepared = {}
        for name, prepare in all_backends:
            try:
                prepared[name] = prepare(A_true, dirs, p_data, dts)
                prepared[name](); prepared[name]()
            except Exception as e:
                prepared[name] = None
                print(f"  [{name}] preparation error at n={n}: {e}")

        row = f"{n:>6}"
        for name, _ in all_backends:
            if name not in gpu_only and n > cpu_max_n:
                t_ms = float("nan")
            else:
                try:
                    if prepared[name] is None:
                        raise RuntimeError("backend preparation failed")
                    t_ms = timeit(prepared[name], n_trials=n_trials)
                except Exception as e:
                    t_ms = float("nan")
                    print(f"  [{name}] error at n={n}: {e}")
            row += f"  {t_ms:>20.2f}ms"
            all_results[name].append((n, t_ms))
        print(row); sys.stdout.flush()

    return all_results


def run_m_sweep(M_vals, n_fixed=80, T_obs=8, dt=0.3, n_trials=7):
    """CPU M-sweep at fixed n=80 — shows library-size scaling."""
    return run_sweep(f"CPU M-sweep (n={n_fixed})", "M", M_vals,
                     build_backends_m(),
                     lambda M: run_problem(n_fixed, M, T_obs, dt),
                     n_trials=n_trials)


def write_results_csv(path, sweep_key, results, column_map):
    keys = [label for label in column_map if label in results]
    if not keys:
        print(f"Warning: no matching keys found for {path}")
        return
    values = [pair[0] for pair in results[keys[0]]]
    rows = []
    for i, v in enumerate(values):
        row = {sweep_key: v}
        for label in keys:
            row[column_map[label]] = results[label][i][1]
        rows.append(row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        fieldnames = [sweep_key] + [column_map[label] for label in keys]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    # CPU: moderate n where O(n) vs O(n³) story is clear
    n_cpu   = [
        int(v) for v in os.environ.get(
            "DENSE_BENCH_N_CPU", "20,40,80,150,250,400"
        ).split(",")
    ]
    # GPU: large n where GPU hardware has a real chance against CPU
    n_gpu   = [
        int(v) for v in os.environ.get(
            "DENSE_BENCH_N_GPU", "200,500,1000,2000,4000"
        ).split(",")
    ]
    M_vals  = [
        int(v) for v in os.environ.get(
            "DENSE_BENCH_M", "10,25,50,100,150,200"
        ).split(",")
    ]
    n_trials = int(os.environ.get("DENSE_BENCH_TRIALS", "7"))
    run_gpu = os.environ.get("DENSE_BENCH_RUN_GPU", "1") == "1"

    cpu_results = run_n_sweep(n_cpu, M_fixed=10, n_trials=n_trials)
    m_results = run_m_sweep(M_vals, n_fixed=80, n_trials=n_trials)
    gpu_results = (
        run_n_sweep_gpu(n_gpu, M_fixed=10, n_trials=max(5, n_trials))
        if run_gpu
        else None
    )

    col_cpu = {
        "JAX CPU (compiled scan)": "jax_scan_ms",
        "PyTorch CPU (eager)": "pytorch_ms",
    }
    col_gpu = {
        "JAX CPU (compiled scan)": "jax_ms",
        "JAX GPU":        "jax_gpu_ms",
        "PyTorch CPU (eager)": "pytorch_ms",
        "PyTorch GPU":    "pytorch_gpu_ms",
    }
    col_m = {
        "JAX CPU (compiled scan)": "jax_scan_ms",
        "PyTorch CPU (eager)": "pytorch_ms",
    }

    out_dir = os.environ.get(
        "DENSE_BENCH_OUTPUT_DIR", os.path.join("paper", "figures")
    )
    write_results_csv(os.path.join(out_dir, "benchmark_python_n.csv"),     "n", cpu_results, col_cpu)
    write_results_csv(os.path.join(out_dir, "benchmark_python_m.csv"),     "M", m_results,   col_m)
    if gpu_results is not None:
        write_results_csv(
            os.path.join(out_dir, "benchmark_python_n_gpu.csv"),
            "n",
            gpu_results,
            col_gpu,
        )
