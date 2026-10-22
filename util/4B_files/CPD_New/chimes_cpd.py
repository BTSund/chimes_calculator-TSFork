#!/usr/bin/env python3
"""
chimes_cpd.py (CP-only version) -- CP decomposition (CPD) of every 3-body and
4-body interaction in a ChIMES parameter file, with automatic rank selection,
1D CP tables at a chosen density and a ready-to-run parameter file.

This version writes CP evaluation paths only (3B CP tables, 4B CP tables or
4B CP-direct factors).  It has no option for the exact factored evaluators,
and it removes any "3B/4B EVALUATOR:" line from the input file, so the output
never selects them.

    python3 chimes_cpd.py params.txt [--h3 0.002] [--h4 0.005] [options]

For each 3B and 4B cluster type the script

 1. reads the coefficient tensor and, per pair slot, the inner and outer
    cutoff (PAIRIDX S_MINIM / S_MAXIM, overridden by SPECIAL 3B/4B S_MINIM /
    S_MAXIM, ALL or SPECIFIC), the Morse lambda and the cutoff function;
 2. estimates the rank: a lower bound (the largest matricization rank at the
    requested tolerance; Eckart-Young) and a starting guess (the generic rank
    of a tensor of that shape, symmetric for homo-atomic 3B types);
 3. fits at the guess, then lowers the rank (if it converged) or raises it in
    geometric steps (if not) and bisects to the smallest rank that converges.
    Each fit is warm-started from its neighbour (one rank down: drop each of
    the --drop-tries weakest terms in turn; further down: drop the weakest
    terms; up: add terms fitted to the residual); --n-init random restarts
    are the fallback (--screen-keep and --time-limit bound the cost on large
    tensors).  The rank just below the final one is always retried with
    every drop-one start, so the search ends at a locally minimal rank.
    A rank counts as converged when
      * the coefficient relative Frobenius error is <= --tol,
      * the force error of the exact CP on sampled clusters is <= --ftol-cp,
      * the cancellation ratio  sum_q |rank-1 term q| / |X|  is <= --kappa-max
        (large ratios amplify table and round-off errors);
 4. tabulates the CP factor functions on 1D grids (cubic Hermite, step <= h,
    starting at the inner cutoff, with the Tersoff switch-on point and the
    outer cutoff on nodes; cp_fit.table_grid) and evaluates the tabulated
    model on the same clusters, reproducing the C++ interpolation exactly.
    --auto-h halves h until the tabulated force error is <= --ftol;
 5. writes the tables, the factors (.npz; --warm DIR reuses them), a JSON
    report, and one parameter file per requested 3B/4B mode.

Cluster checks sample every pair distance uniformly in [inner - 0.02 A,
outer], i.e. the whole box the tables must cover, and compare against the
dense coefficient tensor with chimesFF's own basis (including the damped
continuation below the inner cutoff).  Errors are relative RMS over the
samples: energy, and dE/dr for every pair (the force-like quantity).  This
is a worst case: short distances are as frequent as long ones, so errors on
real configurations are usually lower (Si: 1e-8..8e-8 relative force error
in MD snapshots for a box value of ~4e-7).

Homo-atomic 3B types with identical slots use symmetric CP
(X = sum_q a_q (x) a_q (x) a_q): one table serves all three slots.

Large tensors (dense LM system min(#entries, #params) > 2000, e.g. a 4B of
order >= 5: 5^6 = 15625 entries at ranks of several hundred).  Each LM step is
an exact Cholesky solve of that system (assembled from Gram matrices, no
Jacobian; 15625^2 doubles = 2 GB, ~1.3 Tflop per factorisation), so these
types run with multi-threaded BLAS on all cores (--threads) and within
--mem; above the memory budget a matrix-free CG step is used.  Defaults
for them: start 10% above the generic rank and step down 5% at a time,
1 random start (ALS then LM), 2 drop-one starts, 100 LM iterations, rank
resolution 1% of the rank; progress is logged every 5 LM iterations.  The search stops growing the rank when the error stops falling
with it (the optimiser, not the rank, is then the limit).

Modes
  --3b tables     symmetric/general CP 1D tables ("3B CP TABLES:")      [default]
  --3b keep       leave the 3B as in the input file
  --4b tables     CP 1D tables ("4B CP TABLES:")                        [default]
  --4b direct     exact CP factors, Chebyshev at run time ("4B CP DIRECT:")
  --4b keep       leave the 4B as in the input file
  --4b tables,direct     one parameter file per mode (CPD done once)
  --2b keep       keep 2B TABULATED tags (tables are copied next to the output)
  --2b direct     drop 2B TABULATED tags (polynomial 2B)

The output parameter files need a chimesFF built with -DTABULATION when any
table or CP file is used.  The 3B coefficients stay in the file (needed when
3B CP tables are used: a TABULATED tag on a 3B type is removed).

Examples
  python3 chimes_cpd.py published_params.Si.2+3+4b.HSE06.tabulated --outdir Si_HSE06_cpd
  python3 chimes_cpd.py params.txt --density 500 --4b tables,direct --workers 4
  python3 chimes_cpd.py params.txt --dry-run          # cutoffs and rank estimates only
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

# One BLAS thread per worker: the tensors are small, threaded BLAS is slower.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # util/tensor_tools when run from the repository
sys.path.insert(0, HERE)                       # cp_fit.py / chimes_params.py next to this file
from chimes_params import ChimesParams, slot_basis, fcut_and_deriv, eval_dense, INNER_SMOOTH_DISTANCE  # noqa: E402
from cp_fit import (cp_als, lm_polish, cp_full, balance, is_symmetric3, sym_cp_fit, table_grid,  # noqa: E402
                    write_cp_tab_3b, write_cp_tab_4b, write_cp_direct_4b)

SAMPLE_BELOW = 0.02          # A below the inner cutoff covered by the cluster checks
LARGE_N = 2000               # a type is "large" when its LM system (min(#entries, #params)) exceeds this


def lm_size(X, R):
    """Order of the dense LM system for rank R: min(#entries, #params)."""
    return min(X.size, sum(X.shape) * R)


def set_threads(n):
    """Limit BLAS threads in this process (threadpoolctl if installed)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=max(1, int(n)))
    except ImportError:
        pass


# ----------------------------------------------------------------------------- rank estimates
def numerical_matricization_rank(X, tol):
    """Largest rank any bipartition matricization needs to reach relative
    error tol (Eckart-Young): a lower bound on the CP rank at that error."""
    import itertools
    N = X.ndim
    nX2 = float(np.sum(X * X))
    best = 0
    for k in range(1, N // 2 + 1):
        for left in itertools.combinations(range(N), k):
            right = tuple(m for m in range(N) if m not in left)
            M = np.transpose(X, left + right).reshape(int(np.prod([X.shape[m] for m in left])), -1)
            s = np.linalg.svd(M, compute_uv=False)
            tail = np.concatenate([np.cumsum((s ** 2)[::-1])[::-1], [0.0]])   # tail[r] = sum_{i>=r} s_i^2
            r = int(np.argmax(tail <= (tol ** 2) * nX2))
            best = max(best, r)
    return best


def rank_estimates(X, symmetric, tol):
    d = X.ndim
    lb = max(1, numerical_matricization_rank(X, tol))
    if symmetric:
        n = X.shape[0]
        nuniq = math.comb(n + d - 1, d)
        guess, ub = math.ceil(nuniq / n), nuniq
    else:
        guess = math.ceil(X.size / (sum(X.shape) - d + 1))
        ub = X.size // max(X.shape)
    return lb, max(lb, guess), max(ub, lb)


# ----------------------------------------------------------------------------- CP helpers
def kappa_of(A, normX):
    """sum_q prod_m |A_m[:, q]|  /  |X|  (1 = no cancellation between terms)."""
    return float(np.prod([np.linalg.norm(a, axis=0) for a in A], axis=0).sum() / normX)


def shrink(A, R, sym):
    """Keep the R strongest rank-1 terms."""
    if sym:
        w = np.linalg.norm(A[0], axis=0) ** 3
    else:
        A = balance(A)
        w = np.prod([np.linalg.norm(a, axis=0) for a in A], axis=0)
    keep = np.sort(np.argsort(-w)[:R])
    return [a[:, keep] for a in A]


def drop_candidates(A, R, sym, tries):
    """Warm starts for rank R from a rank > R solution.  One rank down: drop
    each of the `tries` weakest terms in turn (the right term to drop is often
    not the weakest one); further down: drop the weakest terms."""
    if A[0].shape[1] - R != 1:
        return [shrink(A, R, sym)]
    if sym:
        w = np.linalg.norm(A[0], axis=0) ** 3
    else:
        A = balance(A)
        w = np.prod([np.linalg.norm(a, axis=0) for a in A], axis=0)
    return [[np.delete(a, j, axis=1) for a in A] for j in np.argsort(w)[:tries]]


def grow(X, A, R, sym, seed):
    """Add R - rank(A) terms fitted to the residual X - [[A]]."""
    k = R - A[0].shape[1]
    Res = X - cp_full(A)
    if sym:
        B, _ = sym_cp_fit(Res, k, n_init=1, max_iter=40, seed=seed)
        return [np.hstack([A[0], B])] * 3
    B, _, _ = cp_als(Res, k, n_iter=60, seed=seed)
    return [np.hstack([a, b]) for a, b in zip(A, B)]


# ----------------------------------------------------------------------------- cluster checks
def cp_energy_forces(Xs, Ds):
    """E = sum_q prod_c X_c[:, q];  dE/dr_c = sum_q D_c[:, q] prod_{c' != c} X_c'[:, q]."""
    d = len(Xs)
    E = np.prod(Xs, axis=0).sum(1)
    dE = np.empty((len(E), d))
    for c in range(d):
        P = Ds[c].copy()
        for k in range(d):
            if k != c:
                P *= Xs[k]
        dE[:, c] = P.sum(1)
    return E, dE


def rel_errors(E, dE, Er, dEr):
    rmsF = math.sqrt(float(np.mean(dEr ** 2)))
    return dict(E_rms=float(np.linalg.norm(E - Er) / np.linalg.norm(Er)),
                F_rms=float(np.linalg.norm(dE - dEr) / np.linalg.norm(dEr)),
                F_max=float(np.abs(dE - dEr).max() / rmsF))


def make_table(slot, A, h, fcut_type, fcut_var):
    g = table_grid(slot, h, fcut_type, fcut_var)
    G, D = slot_basis(g, slot, A.shape[0], fcut_type, fcut_var)
    return dict(r0=float(g[0]), dr=float(g[1] - g[0]), V=G @ A, Dv=D @ A)


def interp_table(T, r, rc, fcut_type, fcut_var):
    """chimesFF::interpolateCP1DLinear, vectorised (same branches and formulas)."""
    V, Dv, r0, dr = T["V"], T["Dv"], T["r0"], T["dr"]
    ngrid = len(V)
    s = (r - r0) * (1.0 / dr)
    below = s < 0.0
    i = np.where(below, 0, np.minimum(np.where(below, 0, s).astype(int), ngrid - 2))
    t = np.minimum(s - i, 1.0)[:, None]
    t2, t3 = t * t, t * t * t
    h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, (t3 - 2 * t2 + t) * dr, -2 * t3 + 3 * t2, (t3 - t2) * dr
    g00 = (6 * t2 - 6 * t) / dr
    g10, g01, g11 = 3 * t2 - 4 * t + 1, -g00, 3 * t2 - 2 * t
    y0, m0, y1, m1 = V[i], Dv[i], V[i + 1], Dv[i + 1]
    val = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
    der = g00 * y0 + g10 * m0 + g01 * y1 + g11 * m1
    if below.any():
        rb = r[below]
        sm = INNER_SMOOTH_DISTANCE
        damp = np.exp((rb - r0) / sm)[:, None]
        a = sm * (damp - 1.0)
        f0, fd0, f, fd = 1.0, 0.0, np.ones((len(rb), 1)), np.zeros((len(rb), 1))
        if rc > r0:
            f0, fd0 = (float(x[0]) for x in fcut_and_deriv(np.array([r0]), rc, fcut_type, fcut_var))
            ff, ffd = fcut_and_deriv(rb, rc, fcut_type, fcut_var)
            f, fd = ff[:, None], ffd[:, None]
        if f0 != 1.0 or fd0 != 0.0:
            p0 = V[0] / f0
            pd0 = (Dv[0] - fd0 * p0) / f0
            p = p0 + a * pd0
            val[below] = f * p
            der[below] = fd * p + f * damp * pd0
        else:
            val[below] = V[0] + a * Dv[0]
            der[below] = damp * Dv[0]
    return val, der


# ----------------------------------------------------------------------------- one cluster type
def run_type(job):
    """CPD of one 3B/4B type: adaptive rank search, tables, cluster checks."""
    cfg, X, slots, sym, tag = job["cfg"], job["X"], job["slots"], job["sym"], job["tag"]
    body = job["body"]
    fct, fcv = job["fcut_type"], job["fcut_var"]
    t_start = time.time()
    set_threads(job.get("threads", 1))
    seed0 = cfg["seed"] + 7919 * body + 31 * job["type"]
    rng = np.random.default_rng(seed0)
    hist = []

    def log(msg):
        print(f"[{tag}] {msg}", flush=True)

    d = X.ndim
    normX = float(np.linalg.norm(X))
    tol = cfg["tol"]
    lb, guess, ub = rank_estimates(X, sym, tol)
    r_max = min(ub, cfg["max_rank"]) if cfg["max_rank"] else ub
    # size-aware defaults (explicit options win): large tensors get fewer, longer-converged starts
    large = (not sym) and lm_size(X, guess) > LARGE_N
    # large: start 10% above the generic rank (fits right at the rank boundary converge slowly; one
    # converged fit then warm-starts every smaller rank) and step down in 5% steps
    start = cfg["rank"].get(body) or (int(math.ceil(1.1 * guess)) if large else guess)
    step_down = min(cfg["grow"], 1.05) if large else cfg["grow"]
    warm0 = None
    if job.get("warm") is not None:
        warm0 = job["warm"]
        start = warm0[0].shape[1]
    start = int(min(max(start, lb), r_max))
    log(f"dims {X.shape}  nnz {int(np.count_nonzero(X))}  |X| {normX:.4e}  "
        f"{'symmetric CP' if sym else 'CP'}  rank: lower bound {lb}, guess {guess}, start {start}, max {r_max}")
    pick = lambda key, small, big: cfg[key] if cfg[key] is not None else (big if large else small)
    n_init, drop_tries = pick("n_init", 3, 1), pick("drop_tries", 16, 2)
    lm_iter, stall_win = pick("lm_iter", 300, 100), pick("stall", 60, 20)
    res = cfg["rank_resolution"] or (max(1, int(math.ceil(0.01 * start))) if large else 1)
    if large:
        nd = lm_size(X, start)
        log(f"large tensor: LM system {nd}x{nd} ({nd * nd * 8 / 2**30:.1f} GB, {job.get('threads', 1)} BLAS threads); "
            f"n-init {n_init}, drop-tries {drop_tries}, lm-iter {lm_iter}, stall {stall_win}, rank resolution {res}")
    lm_log = log if large else None

    # sampled clusters, reference from the dense tensor with chimesFF's basis
    n = cfg["samples"]
    rs = np.stack([rng.uniform(s["rmin"] - SAMPLE_BELOW, s["rmax"], n) for s in slots], 1)
    GD = [slot_basis(rs[:, c], slots[c], X.shape[c], fct, fcv) for c in range(d)]
    G = [g for g, _ in GD]
    D = [dd for _, dd in GD]
    Er, dEr = eval_dense(X, G, D)
    ftol_cp = cfg["ftol_cp"]

    def accept(A, err):
        E, dE = cp_energy_forces([G[c] @ A[c] for c in range(d)], [D[c] @ A[c] for c in range(d)])
        e = rel_errors(E, dE, Er, dEr)
        kap = kappa_of(A, normX)
        ok = err <= tol and e["F_rms"] <= ftol_cp and kap <= cfg["kappa_max"]
        return ok, dict(coef_err=float(err), kappa=kap, **{"cp_" + k: v for k, v in e.items()})

    stall = (stall_win, 1.5)
    mem = job.get("mem_budget")

    def start_fit(R, A0, iters):
        s = int(rng.integers(1 << 30))
        if sym:
            a, err = sym_cp_fit(X, R, n_init=1, max_iter=iters, seed=s, A0=None if A0 is None else A0[0],
                                stall=stall)
            return [a, a, a], err
        if A0 is not None:                                  # warm: straight to LM (ALS swamps near exact fits)
            return lm_polish(X, [a.copy() for a in A0], max_iter=iters, stall=stall, mem_budget=mem, log=lm_log)
        t0 = time.time()
        A, err, nsw = cp_als(X, R, n_iter=cfg["als_iter"], seed=s, target=0.01 * tol)
        if large:
            log(f"rank {R}: ALS {nsw} sweeps, rel err {err:.2e} ({time.time() - t0:.0f}s)")
        if err > 0.01 * tol:
            A2, err2 = lm_polish(X, A, max_iter=iters, stall=stall, mem_budget=mem, log=lm_log)
            if err2 < err:
                A, err = A2, err2
        return A, err

    def fit(R, warms):
        """Warm starts, then up to n_init random starts; stop at the first accepted fit.
        More than --screen-keep warm starts are screened first: --screen-iter LM
        iterations each, and only the best --screen-keep are run to the end."""
        best = None
        warms = list(warms)
        labels = [f"warm{k}" for k in range(len(warms))]
        if cfg["screen_keep"] and len(warms) > cfg["screen_keep"]:
            scr = []
            for k, A0 in enumerate(warms):
                A, err = start_fit(R, A0, cfg["screen_iter"])
                ok, m = accept(A, err)
                if ok:
                    m.update(init=f"warm{k} (screen)", starts=k + 1)
                    return A, err, ok, m
                scr.append((err, k, A))
            scr.sort(key=lambda t: t[0])
            warms = [A for _, _, A in scr[:cfg["screen_keep"]]]
            labels = [f"warm{k}" for _, k, _ in scr[:cfg["screen_keep"]]]
        inits = warms + [None] * n_init
        for k, A0 in enumerate(inits):
            A, err = start_fit(R, A0, lm_iter)
            ok, m = accept(A, err)
            m["init"] = labels[k] if k < len(warms) else f"random{k - len(warms)}"
            m["starts"] = k + 1
            if best is None or (ok, -err) > (best[2], -best[1]):
                best = (A, err, ok, m)
            if ok:
                break
        return best

    tried = {}
    dropped = set()                                        # (R, from rank) pairs tried with drop-one starts

    def attempt(R, warms):
        t0 = time.time()
        A, err, ok, m = fit(R, warms)
        m.update(rank=R, converged=bool(ok), seconds=round(time.time() - t0, 2))
        tried[R] = (ok, A, err, m)
        hist.append(m)
        log(f"rank {R:4d}: coef err {err:.2e}  CP force err {m['cp_F_rms']:.2e}  kappa {m['kappa']:6.1f}  "
            f"{'converged' if ok else 'not converged'} ({m['init']}, {m['seconds']:.1f}s)")
        return ok

    def out_of_time():
        if cfg["time_limit"] and time.time() - t_start > cfg["time_limit"]:
            log(f"time limit {cfg['time_limit']:g}s reached; stopping the rank refinement")
            return True
        return False

    def down(R, hi):
        if hi - R == 1:
            dropped.add((R, hi))
        return attempt(R, drop_candidates(tried[hi][1], R, sym, drop_tries))

    w0 = [] if warm0 is None else [warm0]
    if cfg["fixed_rank"]:
        attempt(start, w0)
        lo, hi = None, start
    else:
        lo, hi = lb - 1, None
        if attempt(start, w0):
            hi = start
            while hi > lb and not out_of_time():            # geometric steps down
                Rn = max(lb, min(hi - 1, int(math.floor(hi / step_down))))
                if down(Rn, hi):
                    hi = Rn
                else:
                    lo = Rn
                    break
        else:
            lo = start
            slow = 0
            while lo < r_max:                               # geometric steps up
                Rn = min(r_max, max(lo + 1, int(math.ceil(lo * cfg["grow"]))))
                if attempt(Rn, [grow(X, tried[lo][1], Rn, sym, int(rng.integers(1 << 30)))]):
                    hi = Rn
                    break
                # a larger rank that barely lowers the error means the fit, not the rank, is the limit
                slow = slow + 1 if tried[Rn][2] > tried[lo][2] / 3 else 0
                lo = Rn
                if slow >= 2:
                    log(f"error stopped falling with rank ({tried[lo][2]:.2e} at rank {lo}); the optimiser, not "
                        f"the rank, limits the fit: stopping the search (try --lm-iter, or a looser --tol/--ftol-cp)")
                    break
        if hi is not None:                                  # bisection
            while hi - lo > res and not out_of_time():
                mid = (lo + hi) // 2
                if down(mid, hi):
                    hi = mid
                else:
                    lo = mid
            # boundary: rank hi-1 retried with every drop-one start from the final hi
            while res == 1 and hi > lb and (hi - 1, hi) not in dropped and not out_of_time():
                if down(hi - 1, hi):
                    hi -= 1
                else:
                    break
    if hi is None or not tried[hi][0]:
        R = min(tried, key=lambda r: tried[r][2]) if hi is None else hi
        log(f"WARNING: no rank <= {r_max} converged; using rank {R} (coef err {tried[R][2]:.2e})")
        converged = False
    else:
        R = hi
        converged = True
    ok, A, err, m = tried[R]
    A = [a.copy() for a in A] if not sym else [A[0].copy()] * 3

    # tables at the requested density, checked on the same clusters
    tables = None
    if job["tabulate"]:
        h = cfg["h"][body]
        while True:
            Ts = [make_table(slots[c], A[c], h, fct, fcv) for c in range(1 if sym else d)]
            if sym:
                Ts = Ts * 3
            Xs, Ds = zip(*[interp_table(Ts[c], rs[:, c], slots[c]["rmax"], fct, fcv) for c in range(d)])
            E, dE = cp_energy_forces(list(Xs), list(Ds))
            te = rel_errors(E, dE, Er, dEr)
            ngrid = [len(T["V"]) for T in Ts[:1 if sym else d]]
            steps = [T["dr"] for T in Ts[:1 if sym else d]]
            tables = dict(h=h, steps=steps, ngrid=ngrid, bytes=int(sum(ngrid) * 2 * R * 8),
                          **{"tab_" + k: v for k, v in te.items()})
            log(f"tables h {h:.4g} A (steps {min(steps):.4g}-{max(steps):.4g} A, {sum(ngrid)} nodes, "
                f"{tables['bytes'] / 1024:.0f} KiB): "
                f"E err {te['E_rms']:.2e}  F err {te['F_rms']:.2e}  F max {te['F_max']:.2e}")
            if not cfg["auto_h"] or te["F_rms"] <= cfg["ftol"] or h / 2 < cfg["h_min"]:
                break
            h /= 2
        if tables["tab_F_rms"] > cfg["ftol"]:
            log(f"note: tabulated force error {tables['tab_F_rms']:.2e} > --ftol {cfg['ftol']:.1e}"
                + ("" if cfg["auto_h"] else "; use a finer grid or --auto-h"))
    log(f"done: rank {R} in {time.time() - t_start:.1f}s")
    return dict(tag=tag, body=body, type=job["type"], symmetric=sym, rank=R, converged=converged,
                lower_bound=lb, guess=guess, start=start, A=A, metrics=m, tables=tables, history=hist,
                seconds=round(time.time() - t_start, 1))


# ----------------------------------------------------------------------------- parameter file
BLOCK_KEYS = ("3B CP TABLES:", "4B CP TABLES:", "4B CP DIRECT:", "4B COEFF TABLES:",
              "4B SVD3X3 TABLES:", "4B SVD4X2 TABLES:")


def _strip_tab_tag(line):
    it = line.split()
    if "TABULATED" not in it:
        return line
    k = it.index("TABULATED")
    lead = line[:len(line) - len(line.lstrip())]
    return lead + " ".join(it[:k] + it[k + 2:])


def write_param(src, dst, tabdir, blocks, m3, m4, mode2b, header):
    """Copy src to dst with: old table/evaluator blocks removed, TABULATED tags
    removed from replaced 3B/4B types, 2B tables copied into tabdir (or their
    tags dropped), and the new blocks inserted before ENDFILE."""
    src_dir = os.path.dirname(os.path.abspath(src))   # chimesFF: get_directory_path(paramfile)
    dst_dir = os.path.dirname(os.path.abspath(dst))
    lines = open(src, errors="ignore").read().splitlines()
    out = list(header)
    i, section = 0, None
    while i < len(lines):
        s = lines[i]
        st = s.strip()
        if st.startswith("TRIPLETTYPE PARAMS:"):
            section = 3
        elif st.startswith("QUADRUPLETYPE PARAMS:"):
            section = 4
        elif st.startswith("PAIR CHEBYSHEV PARAMS") or st.startswith("PAIRTYPE PARAMS:"):
            section = 2
        if any(st.startswith(k) for k in BLOCK_KEYS):
            i += 1 + int(st.split()[3])
            continue
        if st.startswith("3B EVALUATOR:") or st.startswith("4B EVALUATOR:"):
            print(f"  note: removed '{st}' from the output (CP-only version)")
            i += 1
            continue
        if st.startswith("ENDFILE"):
            out += blocks
            out.append(s)
            i += 1
            continue
        if st.startswith("PAIRTYPE PARAMS:") and "TABULATED" in st.split():
            it = st.split()
            path = it[it.index("TABULATED") + 1]
            if mode2b == "direct":
                s = _strip_tab_tag(s)
            else:
                base = os.path.basename(path)
                for ext in (".energy", ".force"):
                    f_in = src_dir + path + ext          # chimesFF concatenates dir + tag path
                    if not os.path.exists(f_in):
                        raise FileNotFoundError(f"2B table {f_in} (from {src}) not found")
                    shutil.copyfile(f_in, os.path.join(dst_dir, tabdir, base + ext))
                k = it.index("TABULATED")
                it[k + 1] = "/" + tabdir + "/" + base
                s = " ".join(it)
        elif st.startswith("INDEX:") and "TABULATED" in st.split() and \
                ((section == 3 and m3 != "keep") or (section == 4 and m4 != "keep")):
            s = _strip_tab_tag(s)
        out.append(s)
        i += 1
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("param", help="ChIMES parameter file (.txt or .tabulated)")
    g = ap.add_argument_group("tabulation density")
    g.add_argument("--h3", type=float, default=None, help="3B table step in A (default 0.002)")
    g.add_argument("--h4", type=float, default=None, help="4B table step in A (default 0.005)")
    g.add_argument("--density", type=float, default=None,
                   help="table nodes per A for both 3B and 4B (h = 1/density); --h3/--h4 override")
    g.add_argument("--auto-h", action="store_true", help="halve h until the tabulated force error <= --ftol")
    g.add_argument("--h-min", type=float, default=2.5e-4, help="smallest step --auto-h may use")
    g = ap.add_argument_group("CPD hyperparameters")
    g.add_argument("--tol", type=float, default=1e-10, help="coefficient relative Frobenius error target")
    g.add_argument("--ftol", type=float, default=1e-6, help="tabulated cluster force error target (relative RMS)")
    g.add_argument("--ftol-cp", type=float, default=None, help="exact-CP cluster force error target (default ftol/10)")
    g.add_argument("--kappa-max", type=float, default=1e3, help="largest accepted cancellation ratio")
    g.add_argument("--rank3", type=int, default=None, help="start the 3B search here (default: estimate)")
    g.add_argument("--rank4", type=int, default=None, help="start the 4B search here (default: estimate)")
    g.add_argument("--fixed-rank", action="store_true", help="fit only the start rank (no search)")
    g.add_argument("--max-rank", type=int, default=None)
    g.add_argument("--grow", type=float, default=1.15, help="geometric step of the rank search")
    g.add_argument("--rank-resolution", type=int, default=None,
                   help="stop bisecting at this rank gap (default 1; 1%% of the rank for large tensors)")
    g.add_argument("--n-init", type=int, default=None,
                   help="random restarts per rank after the warm starts (default 3; 1 for large tensors)")
    g.add_argument("--drop-tries", type=int, default=None,
                   help="one rank down from a converged fit, try dropping each of this many weakest terms "
                        "(default 16; 2 for large tensors)")
    g.add_argument("--screen-iter", type=int, default=25, help="LM iterations of the warm-start screening")
    g.add_argument("--screen-keep", type=int, default=0,
                   help="screen warm starts and keep this many (0: no screening; screening saves time on "
                        "large tensors but can miss the start that converges)")
    g.add_argument("--time-limit", type=float, default=None,
                   help="seconds per type after which the search stops refining (keeps the best converged rank)")
    g.add_argument("--als-iter", type=int, default=400, help="ALS sweeps per random start")
    g.add_argument("--lm-iter", type=int, default=None,
                   help="Levenberg-Marquardt iterations per start (default 300; 100 for large tensors)")
    g.add_argument("--stall", type=int, default=None, help="LM stops a start whose error has not dropped "
                   "1.5x in this many iterations and is > 1e-6 (default 60; 20 for large tensors)")
    g.add_argument("--samples", type=int, default=20000, help="clusters sampled per type for the checks")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-symmetric", action="store_true", help="general CP for homo-atomic 3B types too")
    g = ap.add_argument_group("what to write")
    g.add_argument("--3b", dest="m3", default="tables", choices=["tables", "keep"])
    g.add_argument("--4b", dest="m4", default="tables",
                   help="comma list of: tables, direct, keep (one parameter file each)")
    g.add_argument("--2b", dest="m2", default="keep", choices=["keep", "direct"])
    g.add_argument("--types3", default="all", help="3B type indices to decompose (comma list, all or none)")
    g.add_argument("--types4", default="all", help="4B type indices to decompose (comma list, all or none)")
    g.add_argument("--outdir", default=None, help="default: <param name>_cpd")
    g.add_argument("--tabdir", default=None, help="table directory inside outdir (default <param stem>.cptab)")
    g.add_argument("--out", default=None, help="name of the (first) output parameter file")
    g.add_argument("--warm", default=None, help="directory with *.factors.npz of an earlier run (warm starts)")
    g.add_argument("--workers", type=int, default=None, help="parallel worker processes (default: #CPU)")
    g.add_argument("--threads", type=int, default=None,
                   help="CPU cores to use (default: all). Small types run one per core; large types (dense "
                        "LM systems) run with multi-threaded BLAS, at least 8 cores each")
    g.add_argument("--mem", type=float, default=None,
                   help="GB of memory for the dense LM systems of all large types together "
                        "(default: half the available memory; above it, matrix-free CG steps are used)")
    g.add_argument("--dry-run", action="store_true", help="print cutoffs and rank estimates, write nothing")
    args = ap.parse_args()

    m4s = [m.strip() for m in args.m4.split(",") if m.strip()]
    for m in m4s:
        if m not in ("tables", "direct", "keep"):
            ap.error(f"--4b: unknown mode {m}")
    h3 = args.h3 or (1.0 / args.density if args.density else 0.002)
    h4 = args.h4 or (1.0 / args.density if args.density else 0.005)
    cfg = dict(tol=args.tol, ftol=args.ftol, ftol_cp=args.ftol_cp or args.ftol / 10, kappa_max=args.kappa_max,
               rank={3: args.rank3, 4: args.rank4}, fixed_rank=args.fixed_rank, max_rank=args.max_rank,
               grow=args.grow, rank_resolution=args.rank_resolution, n_init=args.n_init, drop_tries=args.drop_tries,
               screen_iter=args.screen_iter, screen_keep=args.screen_keep, time_limit=args.time_limit,
               als_iter=args.als_iter, lm_iter=args.lm_iter, stall=args.stall, samples=args.samples,
               seed=args.seed, h={3: h3, 4: h4}, auto_h=args.auto_h, h_min=args.h_min)

    P = ChimesParams(args.param)
    name = os.path.basename(args.param)
    stem, ext = name, ""
    for e in (".tabulated", ".txt", ".dat"):
        if name.endswith(e):
            stem, ext = name[:-len(e)], e
            break
    outdir = args.outdir or (stem + "_cpd")
    tabdir = args.tabdir or (stem + ".cptab")

    print(f"{args.param}: atoms {P.atom_types}, polynomial orders {P.poly_orders}, "
          f"cutoff function {P.fcut_type}{'' if P.fcut_var is None else ' ' + str(P.fcut_var)}")
    print(f"tables: 3B step {h3:g} A, 4B step {h4:g} A{' (auto-refined)' if args.auto_h else ''}; "
          f"tol {args.tol:g}, ftol {args.ftol:g}, ftol-cp {cfg['ftol_cp']:g}, kappa-max {args.kappa_max:g}")

    # ---- jobs
    want = {3: args.m3 == "tables", 4: any(m in ("tables", "direct") for m in m4s)}
    sel = {3: args.types3, 4: args.types4}
    jobs, skipped = [], []
    for body, table in ((3, P.trip), (4, P.quad)):
        for t in sorted(table):
            rec = table[t]
            tag = f"{body}B type {t} {'-'.join(rec['atoms'])}"
            slots = P.slot_params(body, t, canonical=True)
            if rec["excluded"] or not rec["coeffs"]:
                print(f"  {tag}: excluded / no coefficients")
                continue
            X, c2p = P.dense_tensor(body, t, canonical=True)
            pt_can = [rec["pair_types"][p] for p in c2p]
            cut = "  ".join(f"{s['pair']} [{s['rmin']:g}, {s['rmax']:g}] A, morse {s['morse']:g}" for s in slots)
            print(f"  {tag}: {len(rec['coeffs'])} coefficients, dims {X.shape}; slots (inner, outer): {cut}")
            if not want[body] or sel[body] == "none" or \
                    (sel[body] != "all" and t not in [int(x) for x in sel[body].split(",")]):
                continue
            if any(s["rmax"] <= s["rmin"] for s in slots):
                skipped.append(tag + " (a slot has outer <= inner cutoff)")
                continue
            if not np.any(X):
                skipped.append(tag + " (all coefficients zero)")
                continue
            sym = (body == 3 and not args.no_symmetric and is_symmetric3(X)
                   and all((s["rmin"], s["rmax"], s["morse"]) == (slots[0]["rmin"], slots[0]["rmax"], slots[0]["morse"])
                           for s in slots))
            warm = None
            if args.warm:
                fn = os.path.join(args.warm, f"chimes_{body}b_cp.type_{t}.factors.npz")
                if os.path.exists(fn):
                    z = np.load(fn)
                    warm = [z[f"A{c}"] for c in range(X.ndim)]
            jobs.append(dict(cfg=cfg, X=X, slots=slots, sym=sym, tag=tag, body=body, type=t, c2p=c2p,
                             pt_can=pt_can, pt_orig=rec["pair_types"], warm=warm,
                             tabulate=(body == 3 or "tables" in m4s), fcut_type=P.fcut_type, fcut_var=P.fcut_var))
    for s in skipped:
        print(f"  skipped: {s}")
    if args.dry_run:
        for j in jobs:
            lb, guess, ub = rank_estimates(j["X"], j["sym"], args.tol)
            print(f"  {j['tag']}: {'symmetric ' if j['sym'] else ''}CP rank lower bound {lb}, estimate {guess}, "
                  f"trivial upper bound {ub}")
        return

    # ---- run (largest first)
    t0 = time.time()
    jobs.sort(key=lambda j: -j["X"].size * (2 if j["body"] == 4 else 1))
    ncpu = args.threads or os.cpu_count() or 1
    est = lambda j: lm_size(j["X"], args.rank4 if j["body"] == 4 and args.rank4 else
                            rank_estimates(j["X"], j["sym"], args.tol)[1])
    is_big = [not j["sym"] and est(j) > LARGE_N for j in jobs]     # (no `in` on job dicts: they hold arrays)
    big = [j for j, b in zip(jobs, is_big) if b]
    small = [j for j, b in zip(jobs, is_big) if not b]
    mem_total = (args.mem * 2**30) if args.mem else None
    results = []

    def run_pool(js, nw):
        if nw == 1 or len(js) <= 1:
            return [run_type(j) for j in js]
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(nw, mp_context=mp.get_context("fork")) as ex:
            return list(ex.map(run_type, js))

    if small:                                     # one BLAS thread per type, one type per core
        for j in small:
            j["threads"] = 1
        results += run_pool(small, max(1, min(args.workers or ncpu, len(small))))
    if big:                                       # dense LM: share the cores and memory among few workers
        nwb = max(1, min(len(big), args.workers or len(big), max(1, ncpu // 8)))
        from cp_fit import _mem_available
        mem = (mem_total or 0.5 * _mem_available()) / nwb
        for j in big:
            j["threads"], j["mem_budget"] = max(1, ncpu // nwb), mem
        results += run_pool(big, nwb)
    jobs = small + big
    byjob = {(r["body"], r["type"]): (r, j) for r, j in zip(results, jobs)}

    # ---- write
    os.makedirs(os.path.join(outdir, tabdir), exist_ok=True)
    lines = {"3B CP TABLES": [], "4B CP TABLES": [], "4B CP DIRECT": []}
    report = dict(param=os.path.abspath(args.param), settings={k: v for k, v in cfg.items()}, types=[])
    for (body, t), (r, j) in sorted(byjob.items()):
        A, R = r["A"], r["rank"]
        err = r["metrics"]["coef_err"]
        td = os.path.join(outdir, tabdir)
        np.savez(os.path.join(td, f"chimes_{body}b_cp.type_{t}.factors.npz"),
                 **{f"A{c}": a for c, a in enumerate(A)}, canon_to_param=np.array(j["c2p"]), rel_err=err,
                 shared=r["symmetric"])
        if body == 3:
            fn = f"chimes_3b_cp.type_{t}.rank_{R}.dat"
            write_cp_tab_3b(os.path.join(td, fn), t, A, j["slots"], j["c2p"], j["pt_can"], err,
                            P.fcut_type, P.fcut_var, (None, None, r["tables"]["h"]), r["symmetric"])
            lines["3B CP TABLES"].append(f"{t} {tabdir}/{fn}")
        else:
            if "tables" in m4s:
                prefix = os.path.join(td, f"chimes_4b_cp.type_{t}.rank_{R}")
                files, meta = write_cp_tab_4b(prefix, t, A, j["slots"], j["c2p"], j["pt_orig"], j["pt_can"], err,
                                              P.fcut_type, P.fcut_var, None, None, r["tables"]["h"])
                lines["4B CP TABLES"].append(f"{t} " + " ".join(f"{tabdir}/{f}" for f in files + [meta]))
            if "direct" in m4s:
                fn = f"chimes_4b_cp_direct.type_{t}.rank_{R}.dat"
                write_cp_direct_4b(os.path.join(td, fn), t, A, j["c2p"], j["pt_orig"], j["pt_can"], err)
                lines["4B CP DIRECT"].append(f"{t} {tabdir}/{fn}")
        rr = {k: v for k, v in r.items() if k != "A"}
        report["types"].append(rr)

    written = []
    for k, m4 in enumerate(m4s):
        blocks = []
        if args.m3 == "tables" and lines["3B CP TABLES"]:
            blocks += [f"3B CP TABLES: {len(lines['3B CP TABLES'])}"] + lines["3B CP TABLES"]
        if m4 == "tables" and lines["4B CP TABLES"]:
            blocks += [f"4B CP TABLES: {len(lines['4B CP TABLES'])}"] + lines["4B CP TABLES"]
        elif m4 == "direct" and lines["4B CP DIRECT"]:
            blocks += [f"4B CP DIRECT: {len(lines['4B CP DIRECT'])}"] + lines["4B CP DIRECT"]
        tag3 = {"tables": "cp" if lines["3B CP TABLES"] else "orig", "keep": "orig"}[args.m3]
        tag4 = {"tables": "cp" if lines["4B CP TABLES"] else "orig", "direct": "cpdirect" if lines["4B CP DIRECT"]
                else "orig", "keep": "orig"}[m4]
        out = args.out if (args.out and k == 0) else f"{stem}.3b-{tag3}.4b-{tag4}{ext}"
        ranks = ", ".join(f"{r['body']}B type {r['type']} rank {r['rank']}" for r in results)
        header = [f"! Generated by util/tensor_tools/chimes_cpd.py from {name} ({time.strftime('%Y-%m-%d')})",
                  f"! 3B: {args.m3}, 4B: {m4}, 2B: {args.m2}; CP ranks: {ranks or 'none'}",
                  f"! table steps: 3B {h3:g} A, 4B {h4:g} A; needs chimesFF built with -DTABULATION",
                  "!"]
        dst = os.path.join(outdir, out)
        write_param(args.param, dst, tabdir, blocks, args.m3, m4, args.m2, header)
        written.append(dst)
    report["parameter_files"] = written
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(outdir, tabdir, "cpd_report.json"), "w"), indent=1)

    print("\nsummary")
    print(f"{'type':28s} {'rank':>5s} {'bound':>5s} {'guess':>5s} {'coef err':>9s} {'kappa':>6s} "
          f"{'CP F err':>9s} {'step':>7s} {'tab F err':>9s} {'tab F max':>9s} {'KiB':>7s} {'s':>6s}")
    for (body, t), (r, j) in sorted(byjob.items()):
        m, tb = r["metrics"], r["tables"] or {}
        print(f"{r['tag'] + (' (sym)' if r['symmetric'] else ''):28s} {r['rank']:5d} {r['lower_bound']:5d} "
              f"{r['guess']:5d} {m['coef_err']:9.2e} {m['kappa']:6.1f} {m['cp_F_rms']:9.2e} "
              f"{min(tb.get('steps', [float('nan')])):7.4f} {tb.get('tab_F_rms', float('nan')):9.2e} "
              f"{tb.get('tab_F_max', float('nan')):9.2e} {tb.get('bytes', 0) / 1024:7.0f} {r['seconds']:6.1f}"
              + ("" if r["converged"] else "  NOT CONVERGED"))
    print(f"\nwrote {', '.join(written)}\n      tables and cpd_report.json in {os.path.join(outdir, tabdir)}/ "
          f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()