import os, time, statistics, subprocess, json, platform, shutil, sys
import numpy as np
import matplotlib.pyplot as plt

# Optional deps (guarded)
try:
    import cupy as cp
    CUPY_OK = True
except Exception:
    CUPY_OK = False

try:
    import numba as nb
    from numba import prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# =========================
# Config
# =========================
ITERATIONS = 30
SAVE_DIR = "results"
CACHE_PATH = os.path.join(SAVE_DIR, "cache.json")
os.makedirs(SAVE_DIR, exist_ok=True)

# Use ONE shared set of sizes everywhere (>=5 values, “mostly common”)
COMMON_SIZES = [256, 384, 512, 768, 1024]

# If one method is painfully slow (e.g., Naive Python), let it use a smaller subset.
# Comment the line below if you want literally everything on COMMON_SIZES.
PER_METHOD_SIZES = {
    "Naive Python":                     [64, 96, 128, 160, 192, 256],
    "NumPy":                            [768, 1024, 1280, 1536, 1792],
    "CuPy FP32":                        [1536, 2048, 2560, 3072, 3584],
    "CuPy FP64":                        [512, 768, 1024, 1536, 2048],
    "Naive C":                          [128, 160, 192, 256, 384],
    "Blocked NumPy":                    [256, 512, 768, 1024, 1280],
    "C Blocked (no Oflags)":            [128, 160, 192, 256, 384],
    "C OpenMP Tiled (-O3)":             [512, 768, 1024, 1536, 2048],
    "Numba Tiled (parallel)":           [512, 768, 1024, 1280, 1536],
    "Strassen (hybrid, cutoff=128)":    [512, 768, 1024, 1280, 1536],
    "NumPy (B transposed precomputed)": [768, 1024, 1280, 1536, 1792],
    "C Naive + B^T":                    [128, 160, 192, 256, 384],
    "CuPy RawKernel (tiled)":           [768, 1024, 1536, 2048, 2560]
}

# Pick one implementation to "dump/freeze" (won’t re-run if cached is available).
# Examples: "Naive Python", "CuPy FP64", "C OpenMP Tiled (-O3)"
FREEZE_METHOD = ["Naive Python", "NumPy", "CuPy FP32", "CuPy FP64", "Naive C", "Blocked NumPy", "C Blocked (no Oflags)", "C OpenMP Tiled (-O3)", "Numba Tiled (parallel)", "Strassen (hybrid, cutoff=128)", "NumPy (B transposed precomputed)", "C Naive + B^T", "CuPy RawKernel (tiled)"]

RESULTS = {}   # method -> { "sizes":[...], "mean":[...], "std":[...], "raw":[[...]*ITERATIONS per size] }

def sizes_for(method_name: str):
    return PER_METHOD_SIZES.get(method_name, COMMON_SIZES)

# =========================
# Cache helpers
# =========================
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

CACHE = load_cache()

def maybe_load_frozen(method_name, sizes):
    """Return cached per-size values if FREEZE_METHOD matches and data exists; else None."""
    if method_name not in FREEZE_METHOD:
        return None
    node = CACHE.get(method_name)
    if not node:
        return None
    cached_sizes = node.get("sizes", [])
    cached_raw = node.get("raw", [])
    if cached_sizes == sizes and len(cached_raw) == len(sizes) and all(len(v)==ITERATIONS for v in cached_raw):
        return cached_raw
    return None

def maybe_save_frozen(method_name, sizes, raw):
    """If this is the frozen method (first time), write to cache so future runs can skip it."""
    if method_name in FREEZE_METHOD:
        CACHE[method_name] = {"sizes": sizes, "raw": raw}
        save_cache(CACHE)

# =========================
# Common utilities
# =========================
def gflops(n, secs): return (2.0 * (n**3)) / (secs * 1e9)

def record_result(name, sizes, gvals_per_size):
    means = [statistics.mean(vs) for vs in gvals_per_size]
    stds  = [statistics.stdev(vs) if len(vs) > 1 else 0.0 for vs in gvals_per_size]
    RESULTS[name] = {"sizes": sizes, "mean": means, "std": stds, "raw": gvals_per_size}

def plot_one(name):
    d = RESULTS[name]
    sizes, mean, std = d["sizes"], d["mean"], d["std"]
    import matplotlib.pyplot as plt
    plt.errorbar(sizes, mean, yerr=std, capsize=5, marker='o')
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Performance (GFLOPS)")
    plt.title(f"{name} Performance")
    plt.grid(True)
    out = os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.clf()
    return out

def save_json_all():
    with open(os.path.join(SAVE_DIR, "all_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)

# =========================
# Tasks
# =========================
def matmul_python(a, b):
    n = len(a)
    c = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c

def bench_naive_python():
    name = "Naive Python"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen)
        print(f"[CACHE] Using cached results for {name}")
        return

    gvals_per_size = []
    for n in sizes:
        a = [[float(x) for x in np.random.rand(n)] for _ in range(n)]
        b = [[float(x) for x in np.random.rand(n)] for _ in range(n)]
        vals = []
        for _ in range(ITERATIONS):
            t0 = time.time(); _ = matmul_python(a, b); t1 = time.time()
            vals.append(gflops(n, t1-t0))
        gvals_per_size.append(vals)

    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def bench_numpy():
    name = "NumPy"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    gvals_per_size = []
    for n in sizes:
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        vals = []
        for _ in range(ITERATIONS):
            t0 = time.time(); _ = a @ b; t1 = time.time()
            vals.append(gflops(n, t1-t0))
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def bench_cupy(dtype, label):
    name = label
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    if not CUPY_OK:
        print(f"[WARN] {name}: CuPy not available; skipping.")
        zeros = [[0.0]*ITERATIONS for _ in sizes]
        record_result(name, sizes, zeros)
        maybe_save_frozen(name, sizes, zeros)
        return

    gvals_per_size = []
    for n in sizes:
        a = cp.random.random((n, n), dtype=dtype)
        b = cp.random.random((n, n), dtype=dtype)
        _ = a @ b; cp.cuda.Stream.null.synchronize()  # warmup
        vals = []
        for _ in range(ITERATIONS):
            cp.cuda.Stream.null.synchronize()
            t0 = time.time(); _ = a @ b; cp.cuda.Stream.null.synchronize(); t1 = time.time()
            vals.append(gflops(n, t1-t0))
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

# =========================
# C helpers
# =========================
def compile_c(basename, src, flags=None):
    cmd = ["gcc", src, "-o", basename, "-lm"]
    if flags: cmd = ["gcc"] + flags + [src, "-o", basename, "-lm"]

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_c_exe(exe, n):
    exe_path = exe if os.name != "nt" else (exe if exe.endswith(".exe") else exe + ".exe")
    out = subprocess.run([exe_path, str(n)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())

def bench_c_naive():
    name = "Naive C"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    compile_c("matmul", "matmul.c")
    gvals_per_size = []
    for n in sizes:
        vals = [run_c_exe("matmul", n) for _ in range(ITERATIONS)]
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

# =========================
# Bonuses
# =========================
def blocked_matmul_numpy(a, b, bs=64):
    n = a.shape[0]
    c = np.zeros((n, n), dtype=a.dtype)
    for i in range(0, n, bs):
        for k in range(0, n, bs):
            c[i:i+bs, :] += a[i:i+bs, k:k+bs] @ b[k:k+bs, :]
    return c

def bench_blocked_numpy():
    name = "Blocked NumPy"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    gvals_per_size = []
    for n in sizes:
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        vals = []
        for _ in range(ITERATIONS):
            t0 = time.time(); _ = blocked_matmul_numpy(a, b, 64); t1 = time.time()
            vals.append(gflops(n, t1-t0))
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def bench_c_blocked():
    name = "C Blocked (no Oflags)"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    compile_c("matmul_blocked", "matmul_blocked.c")
    gvals_per_size = []
    for n in sizes:
        vals = [run_c_exe("matmul_blocked", n) for _ in range(ITERATIONS)]
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def bench_c_openmp():
    name = "C OpenMP Tiled (-O3)"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    flags = ["-O3", "-fopenmp", "-march=native"]
    compile_c("matmul_omp", "matmul_omp.c", flags=flags)
    gvals_per_size = []
    for n in sizes:
        vals = [run_c_exe("matmul_omp", n) for _ in range(ITERATIONS)]
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

if NUMBA_OK:
    @nb.njit(parallel=True, fastmath=True)
    def numba_tiled(a, b):
        n = a.shape[0]
        TILE = 64  # compile-time constant
        c = np.zeros((n, n), dtype=a.dtype)

        ntiles = (n + TILE - 1) // TILE  # number of row tiles

        # prange must have step == 1
        for bi in prange(ntiles):
            ii = bi * TILE
            iimax = ii + TILE if ii + TILE < n else n

            for kk in range(0, n, TILE):
                kkmax = kk + TILE if kk + TILE < n else n

                for i in range(ii, iimax):
                    for k in range(kk, kkmax):
                        aik = a[i, k]
                        for j in range(n):
                            c[i, j] += aik * b[k, j]
        return c

def bench_numba_tiled():
    name = "Numba Tiled (parallel)"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    if not NUMBA_OK:
        print(f"[WARN] {name}: Numba not available; skipping.")
        zeros = [[0.0]*ITERATIONS for _ in sizes]
        record_result(name, sizes, zeros)
        maybe_save_frozen(name, sizes, zeros)
        return

    gvals_per_size = []
    for n in sizes:
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        _ = numba_tiled(a, b)  # JIT warmup
        vals = []
        for _ in range(ITERATIONS):
            t0 = time.time(); _ = numba_tiled(a, b); t1 = time.time()
            vals.append(gflops(n, t1 - t0))
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def next_pow2(x): return 1 if x == 0 else 1<<(x-1).bit_length()

def strassen(a, b, cutoff=128):
    n = a.shape[0]
    if n <= cutoff: return a @ b
    m = n // 2
    A11, A12, A21, A22 = a[:m,:m], a[:m,m:], a[m:,:m], a[m:,m:]
    B11, B12, B21, B22 = b[:m,:m], b[:m,m:], b[m:,:m], b[m:,m:]
    M1 = strassen(A11 + A22, B11 + B22, cutoff)
    M2 = strassen(A21 + A22, B11, cutoff)
    M3 = strassen(A11, B12 - B22, cutoff)
    M4 = strassen(A22, B21 - B11, cutoff)
    M5 = strassen(A11 + A12, B22, cutoff)
    M6 = strassen(A21 - A11, B11 + B12, cutoff)
    M7 = strassen(A12 - A22, B21 + B22, cutoff)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    C = np.empty_like(a)
    C[:m,:m] = C11; C[:m,m:] = C12; C[m:,:m] = C21; C[m:,m:] = C22
    return C

def bench_strassen():
    name = "Strassen (hybrid, cutoff=128)"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    gvals_per_size = []
    for n in sizes:
        p = next_pow2(n)
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        ap = np.zeros((p, p), dtype=np.float32); ap[:n,:n] = a
        bp = np.zeros((p, p), dtype=np.float32); bp[:n,:n] = b
        vals = []
        for _ in range(ITERATIONS):
            t0 = time.time(); _ = strassen(ap, bp, cutoff=128); t1 = time.time()
            vals.append(gflops(n, t1-t0))
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def bench_numpy_bt():
    name = "NumPy (B transposed precomputed)"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    gvals_per_size = []
    for n in sizes:
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        bt = b.T.copy()
        vals = []
        for _ in range(ITERATIONS):
            t0 = time.time(); _ = a @ bt.T; t1 = time.time()
            vals.append(gflops(n, t1 - t0))
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def bench_c_bt():
    name = "C Naive + B^T"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    compile_c("matmul_bt", "matmul_bt.c")
    gvals_per_size = []
    for n in sizes:
        vals = [run_c_exe("matmul_bt", n) for _ in range(ITERATIONS)]
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

def bench_cupy_rawkernel():
    name = "CuPy RawKernel (tiled)"
    sizes = sizes_for(name)
    frozen = maybe_load_frozen(name, sizes)
    if frozen is not None:
        record_result(name, sizes, frozen); print(f"[CACHE] Using cached results for {name}"); return

    if not CUPY_OK:
        print(f"[WARN] {name}: CuPy not available; skipping.")
        zeros = [[0.0]*ITERATIONS for _ in sizes]
        record_result(name, sizes, zeros)
        maybe_save_frozen(name, sizes, zeros)
        return

    kernel = r'''
    extern "C" __global__
    void matmul_tiled(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C,
                      int N) {
        const int TILE = 32;
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];
        int row = blockIdx.y * TILE + threadIdx.y;
        int col = blockIdx.x * TILE + threadIdx.x;
        float sum = 0.0f;
        for (int t = 0; t < N; t += TILE) {
            if (row < N && t + threadIdx.x < N)
                As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
            else
                As[threadIdx.y][threadIdx.x] = 0.0f;
            if (col < N && t + threadIdx.y < N)
                Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TILE; ++k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            __syncthreads();
        }
        if (row < N && col < N) C[row * N + col] = sum;
    }'''
    mod = cp.RawModule(code=kernel, options=('-std=c++11',), name_expressions=['matmul_tiled'])
    kern = mod.get_function('matmul_tiled')

    gvals_per_size = []
    for n in sizes:
        a = cp.random.random((n, n), dtype=cp.float32)
        b = cp.random.random((n, n), dtype=cp.float32)
        c = cp.zeros((n, n), dtype=cp.float32)
        TILE = 32
        grid = ((n + TILE - 1)//TILE, (n + TILE - 1)//TILE, 1)
        block = (TILE, TILE, 1)
        kern(grid, block, (a, b, c, np.int32(n)))  # warmup
        cp.cuda.Stream.null.synchronize()
        vals = []
        for _ in range(ITERATIONS):
            cp.cuda.Stream.null.synchronize()
            t0 = time.time()
            kern(grid, block, (a, b, c, np.int32(n)))
            cp.cuda.Stream.null.synchronize()
            t1 = time.time()
            vals.append(gflops(n, t1 - t0))
        gvals_per_size.append(vals)
    record_result(name, sizes, gvals_per_size)
    maybe_save_frozen(name, sizes, gvals_per_size)

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Core tasks (mostly common sizes)
    bench_naive_python()
    bench_numpy()
    bench_c_naive()
    bench_cupy(np.float32, "CuPy FP32")
    bench_cupy(np.float64, "CuPy FP64")

    # Bonuses
    bench_blocked_numpy()
    bench_c_blocked()
    bench_c_openmp()
    bench_numba_tiled()
    bench_strassen()
    bench_numpy_bt()
    bench_c_bt()
    bench_cupy_rawkernel()

    # Save outputs
    for name in RESULTS:
        plot_one(name)
    save_json_all()
    print(f"Done. Plots + data in: {SAVE_DIR}/")
