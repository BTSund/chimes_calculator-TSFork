#!/usr/bin/env python3
"""
cp_fit.py -- fast CP (CANDECOMP/PARAFAC) fitting for ChIMES 3B/4B coefficient
tensors, plus writers for the chimesFF CP file formats.

Improvements over util/4B_files/Tab_CPD.py
-----------------------------------------
* The fit error is obtained from Gram matrices (O(R^2) per sweep) instead of
  re-assembling the full tensor term-by-term every iteration (O(R * n^6) with
  an einsum per rank-1 term), which was the dominant cost of Tab_CPD.py.
* ALS with Bro-style line-search extrapolation (typically 3-10x fewer sweeps).
* Optional Levenberg-Marquardt polish (scipy) to drive near-exact fits to
  round-off, which is what CP needs for ChIMES: the 4B energy is the result
  of strong cancellation between coefficients, so a 1e-6 relative
  coefficient error is ~1e-3 relative error in cluster energies.
* Rank search starts at the largest matricization rank (a hard lower bound
  on the CP rank) and bisects instead of scanning in fixed steps.
* Factors are balanced across modes so that float32 storage loses the least.
* Writes all the files chimesFF needs (CP-direct factor file, CP 1D slot
  tables + meta, and the .factors.npz that TabNPZ_to_Direct.py expected but
  Tab_CPD.py never wrote).

Library use:  A, err = cp_fit(X, R, ...)
CLI:          see --help
"""

import argparse
import math
import os
import time

import os
# One BLAS thread per process: the tensors are tiny and multi-threaded BLAS
# (or many worker processes each spawning BLAS threads) is 10-300x slower.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np


# --------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------

def unfold(X, mode):
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def khatri_rao(mats):
    R = mats[0].shape[1]
    out = mats[0]
    for B in mats[1:]:
        out = (out[:, None, :] * B[None, :, :]).reshape(-1, R)
    return out


def cp_full(A):
    shape = [a.shape[0] for a in A]
    return khatri_rao(A).sum(axis=1).reshape(shape)


def cp_rel_error(X, A, normX=None):
    normX = np.linalg.norm(X) if normX is None else normX
    return float(np.linalg.norm(X - cp_full(A)) / normX)


def balance(A):
    """Equalise column norms across modes (same tensor, better conditioned storage)."""
    N = len(A)
    nr = np.array([np.linalg.norm(a, axis=0) for a in A])
    nr[nr == 0] = 1.0
    g = np.prod(nr, axis=0) ** (1.0 / N)
    return [A[m] / nr[m] * g for m in range(N)]


def max_matricization_rank(X):
    """Lower bound on the CP rank: max over all bipartitions of the modes."""
    import itertools
    N = X.ndim
    best = 0
    tol = 1e-12 * np.linalg.norm(X)
    for k in range(1, N // 2 + 1):
        for left in itertools.combinations(range(N), k):
            right = tuple(m for m in range(N) if m not in left)
            M = np.transpose(X, left + right).reshape(int(np.prod([X.shape[m] for m in left])), -1)
            best = max(best, np.linalg.matrix_rank(M, tol=tol))
    return int(best)


# --------------------------------------------------------------------------
# ALS with line search
# --------------------------------------------------------------------------

def _sweep(Xunf, A, R, ridge):
    N = len(A)
    grams = [a.T @ a for a in A]
    last_mt = None
    for m in range(N):
        V = np.ones((R, R))
        for k in range(N):
            if k != m:
                V *= grams[k]
        KR = khatri_rao([A[k] for k in range(N) if k != m])
        MT = Xunf[m] @ KR
        Vr = V + ridge * (np.trace(V) / R + 1e-300) * np.eye(R)
        try:
            A[m] = np.linalg.solve(Vr, MT.T).T
        except np.linalg.LinAlgError:
            A[m] = MT @ np.linalg.pinv(Vr)
        grams[m] = A[m].T @ A[m]
        last_mt = (MT, V)
    # ||X - Xhat||^2 = ||X||^2 - 2 <MT_N, A_N> + sum(V_N * G_N)
    MT, V = last_mt
    inner = float(np.sum(MT * A[N - 1]))
    nrm2 = float(np.sum(V * grams[N - 1]))
    return A, inner, nrm2


def cp_als(X, R, n_iter=5000, tol=1e-15, seed=0, ridge=1e-15, A0=None,
           line_search=True, target=0.0, verbose=False):
    rng = np.random.default_rng(seed)
    N = X.ndim
    normX2 = float(np.sum(X * X))
    Xunf = [unfold(X, m) for m in range(N)]
    if A0 is None:
        A = []
        for m in range(N):
            U, s, _ = np.linalg.svd(Xunf[m], full_matrices=False)
            F = rng.normal(size=(X.shape[m], R))
            k = min(R, U.shape[1])
            F[:, :k] += 2.0 * U[:, :k] * np.sqrt(s[:k] / s[0])
            A.append(F)
    else:
        A = [a.copy() for a in A0]
    A = balance(A)
    err_prev = None
    err = 1.0
    hist = []
    for it in range(n_iter):
        A_old = [a.copy() for a in A]
        A, inner, nrm2 = _sweep(Xunf, A, R, ridge)
        err = math.sqrt(max(normX2 - 2 * inner + nrm2, 0.0) / normX2)
        if line_search and it > 2:
            s = it ** (1.0 / 3.0)
            A_try = [a + s * (a - ao) for a, ao in zip(A, A_old)]
            e_try = cp_rel_error(X, A_try, math.sqrt(normX2))
            if e_try < err:
                A, err = A_try, e_try
        if it % 20 == 0:
            A = balance(A)
            # Gram-based error loses accuracy near round-off: recompute exactly
            if err < 1e-6:
                err = cp_rel_error(X, A, math.sqrt(normX2))
        hist.append(err)
        if verbose and it % 200 == 0:
            print(f"      ALS it {it:5d}  rel err {err:.3e}", flush=True)
        if err <= target:
            break
        if err_prev is not None and abs(err_prev - err) <= tol * max(err, 1e-300) and it > 50:
            break
        err_prev = err
    A = balance(A)
    return A, cp_rel_error(X, A, math.sqrt(normX2)), len(hist)


def _cp_jacobian(A, shape):
    """Dense Jacobian d vec(Xhat) / d [vec(A_0), ..., vec(A_{N-1})]."""
    N = len(A)
    R = A[0].shape[1]
    size = int(np.prod(shape))
    cols = []
    for m in range(N):
        others = [A[k] for k in range(N) if k != m]
        KRt = khatri_rao(others).reshape([shape[k] for k in range(N) if k != m] + [R])
        J = np.zeros(list(shape) + [shape[m], R])
        for i in range(shape[m]):
            idx = [slice(None)] * N
            idx[m] = i
            J[tuple(idx) + (i, slice(None))] = KRt
        cols.append(J.reshape(size, -1))
    return np.concatenate(cols, axis=1)


def _cp_gn_system(A, Res, out=None):
    """Gauss-Newton matrix H = J^T J and gradient g = J^T vec(Res) of a CP
    model, assembled from the factor Gram matrices without forming J:
        H_mm[(i,r),(j,s)] = d_ij  prod_{k!=m} G_k[r,s]
        H_mn[(i,r),(j,s)] = A_m[i,s] A_n[j,r] prod_{k!=m,n} G_k[r,s]   (m != n)
        g_m = Res_(m) @ khatri_rao(A_k, k != m)
    O(P^2) work for P parameters instead of O(P^2 * #entries) for J^T J."""
    N = len(A)
    R = A[0].shape[1]
    G = [a.T @ a for a in A]
    off = np.concatenate([[0], np.cumsum([a.size for a in A])])
    H = np.empty((off[-1], off[-1])) if out is None else out
    for m in range(N):
        for n in range(m, N):
            W = np.ones((R, R))
            for k in range(N):
                if k != m and k != n:
                    W *= G[k]
            if m == n:
                blk = np.kron(np.eye(A[m].shape[0]), W)
            else:
                blk = np.einsum("is,jr,rs->irjs", A[m], A[n], W).reshape(A[m].size, A[n].size)
                H[off[n]:off[n + 1], off[m]:off[m + 1]] = blk.T
            H[off[m]:off[m + 1], off[n]:off[n + 1]] = blk
    if Res is None:
        return H, None
    return H, _mttkrp_all(A, Res)


def _cp_gn_pcg(A, g, lam, max_cg=250, rtol=1e-10):
    """(J^T J + lam I)^{-1} g without forming J^T J: preconditioned conjugate
    gradients with the Gram-matrix product
        (H v)_m = V_m W_mm + sum_{n!=m} A_m (W_mn * (A_n^T V_n))^T,
        W_mn = prod_{k!=m,n} A_k^T A_k,
    O(N^2 n R^2) per product, and the exact inverse of the block diagonal
    (W_mm + lam I per mode) as preconditioner.  For large tensors (4B order
    >= 4 at ranks of a few hundred) this is far cheaper than any dense form."""
    N = len(A)
    R = A[0].shape[1]
    G = [a.T @ a for a in A]
    W = [[None] * N for _ in range(N)]
    for m in range(N):
        for n in range(m, N):
            w = np.ones((R, R))
            for k in range(N):
                if k != m and k != n:
                    w *= G[k]
            W[m][n] = W[n][m] = w
    sizes = [a.size for a in A]
    off = np.concatenate([[0], np.cumsum(sizes)])
    split = lambda v: [v[off[m]:off[m + 1]].reshape(A[m].shape) for m in range(N)]
    Pinv = [np.linalg.inv(W[m][m] + lam * np.eye(R)) for m in range(N)]

    def Hv(v):
        V = split(v)
        C = [A[n].T @ V[n] for n in range(N)]
        out = []
        for m in range(N):
            acc = V[m] @ W[m][m] + lam * V[m]
            for n in range(N):
                if n != m:
                    acc += A[m] @ (W[m][n] * C[n]).T
            out.append(acc.ravel())
        return np.concatenate(out)

    prec = lambda v: np.concatenate([(Vm @ Pinv[m]).ravel() for m, Vm in enumerate(split(v))])
    x = prec(g)
    r = g - Hv(x)
    z = prec(r)
    p = z.copy()
    rz = float(r @ z)
    ng = float(np.linalg.norm(g)) + 1e-300
    for _ in range(max_cg):
        if np.linalg.norm(r) <= rtol * ng:
            break
        Hp = Hv(p)
        alpha = rz / float(p @ Hp)
        x += alpha * p
        r -= alpha * Hp
        z = prec(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _spd_solve(M, lam, b):
    """(M + lam I)^{-1} b by Cholesky (raises LinAlgError if not positive definite)."""
    L = np.linalg.cholesky(M + lam * np.eye(M.shape[0]))
    try:
        from scipy.linalg import solve_triangular
        return solve_triangular(L, solve_triangular(L, b, lower=True), lower=True, trans="T")
    except ImportError:
        return np.linalg.solve(L.T, np.linalg.solve(L, b))


def _mttkrp_all(A, T):
    """J^T vec(T) for a CP model: the MTTKRP of tensor T with every mode."""
    N = len(A)
    return np.concatenate([(unfold(T, m) @ khatri_rao([A[k] for k in range(N) if k != m])).ravel()
                           for m in range(N)])


def _cp_residual_gram(A, out):
    """K = J J^T (#entries x #entries) of a CP model without forming J:
        K[e, f] = sum_m delta(e_m, f_m) G_m[e_-m, f_-m],
        G_m = M_m M_m^T,  M_m = khatri_rao(A_k, k != m),
    written into `out`.  Costs sum_m (E/n_m)^2 R flops, far below the
    E^2 * #params of J J^T."""
    N = len(A)
    shape = [a.shape[0] for a in A]
    out.fill(0.0)
    K = out.reshape(shape + shape)
    for m in range(N):
        M = khatri_rao([A[k] for k in range(N) if k != m])
        other = [shape[k] for k in range(N) if k != m]
        G = (M @ M.T).reshape(other + other)
        for i in range(shape[m]):
            idx = [slice(None)] * N
            idx[m] = i
            K[tuple(idx + idx)] += G
    return out


def _mem_available():
    """Bytes of memory available (Linux /proc/meminfo; 4 GB if unknown)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 4 << 30


def _chol_solve_inplace(W, b):
    """Solve W y = b for SPD W, factorising W in place (W is destroyed)."""
    try:
        from scipy.linalg import cho_factor, cho_solve
        c = cho_factor(W, lower=True, overwrite_a=True, check_finite=False)
        return cho_solve(c, b, check_finite=False)
    except ImportError:
        L = np.linalg.cholesky(W)
        return np.linalg.solve(L.T, np.linalg.solve(L, b))


def lm_polish(X, A, max_iter=60, verbose=False, stall=None, solver="auto", mem_budget=None, log=None):
    """Levenberg-Marquardt refinement of all CP factors.

    Step dx = (J^T J + lam I)^{-1} J^T r = J^T (J J^T + lam I)^{-1} r, solved
    exactly by Cholesky in the smaller of the two spaces, with neither J nor
    a dense product of J formed:
      * residual space (#entries <= #params, the usual case for near-exact
        ChIMES fits): K = J J^T from Khatri-Rao Grams (_cp_residual_gram);
      * parameter space: J^T J from factor Grams (_cp_gn_system).
    The matrix is re-assembled for every damping trial into one buffer, so
    memory is min(#entries, #params)^2 doubles.  If that exceeds mem_budget
    (default: half the available memory) a matrix-free preconditioned CG
    step is used instead (_cp_gn_pcg; cheaper per step, but converges
    linearly near round-off).  Damping follows Nielsen's gain-ratio rule.
    stall: optional (window, factor), as in sym_cp_fit.
    solver: "auto", "dense" or "pcg".  log: optional callable for progress."""
    shapes = [a.shape for a in A]
    sizes = [a.size for a in A]
    N = len(A)
    normX = np.linalg.norm(X)

    def unpack(x):
        out, o = [], 0
        for sh, sz in zip(shapes, sizes):
            out.append(x[o:o + sz].reshape(sh))
            o += sz
        return out

    E, Pn = X.size, sum(sizes)
    nd = min(E, Pn)
    if mem_budget is None:
        mem_budget = 0.5 * _mem_available()
    if solver == "auto":
        solver = "dense" if nd * nd * 8 <= mem_budget else "pcg"
    # Fortran-ordered buffer, filled through its C-ordered transpose (the matrix is symmetric):
    # LAPACK then factorises it in place, without a second n^2 copy.
    W = np.empty((nd, nd), order="F") if solver == "dense" else None
    Wc = W.T if W is not None else None
    resid_space = E <= Pn

    x = np.concatenate([a.ravel() for a in A])
    r = cp_full(A).ravel() - X.ravel()
    f = float(r @ r)
    lam, nu = None, 2.0
    fhist = []
    t0 = time.time()
    for it in range(max_iter):
        fhist.append(f)
        if stall is not None and len(fhist) > stall[0] and math.sqrt(f) / normX > 1e-6:
            if f > fhist[-1 - stall[0]] / stall[1] ** 2:
                break
            # hopeless start: at the rate of the last `window` steps, the error would not even reach
            # 1e-6 (where LM turns quadratic) within the remaining iterations
            if len(fhist) > 2 * stall[0]:
                rate = (fhist[-1 - stall[0]] / f) ** (0.5 / stall[0])
                need = math.log(math.sqrt(f) / normX / 1e-6) / math.log(rate) if rate > 1 else float("inf")
                if need > max_iter - it:
                    if log is not None:
                        log(f"LM stopped at it {it}: rel err {math.sqrt(f) / normX:.2e} falls {rate:.3f}x per step, "
                            f"needs ~{need:.0f} more steps to reach 1e-6")
                    break
        Ac = unpack(x)
        g = _mttkrp_all(Ac, r.reshape(X.shape))                   # J^T r
        if solver == "pcg":
            if lam is None:
                lam = 1e-3 * f / max(len(x), 1) + 1e-30
            step_fn = lambda l: _cp_gn_pcg(Ac, g, l)
        elif resid_space:
            def step_fn(l):
                _cp_residual_gram(Ac, Wc)
                Wc.flat[::nd + 1] += l
                y = _chol_solve_inplace(W, r)
                return _mttkrp_all(Ac, y.reshape(X.shape))
        else:
            def step_fn(l):
                _cp_gn_system(Ac, None, out=Wc)
                Wc.flat[::nd + 1] += l
                return _chol_solve_inplace(W, g)
        if lam is None:                                           # Nielsen: tau * max diag(J^T J)
            n2 = [np.sum(a * a, axis=0) for a in Ac]
            dmax = max(float(np.max(np.prod([n2[k] for k in range(N) if k != m], axis=0))) for m in range(N))
            lam = 1e-3 * dmax
        improved = False
        for _ in range(16):
            try:
                dx = step_fn(lam)
            except np.linalg.LinAlgError:
                lam *= nu
                nu *= 2.0
                continue
            xn = x - dx
            rn = cp_full(unpack(xn)).ravel() - X.ravel()
            fn = float(rn @ rn)
            pred = float(dx @ g) + lam * float(dx @ dx)            # model decrease of ||r||^2
            rho = (f - fn) / pred if pred > 0 else -1.0
            if fn < f:
                x, r, f = xn, rn, fn
                lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * min(rho, 1.0) - 1.0) ** 3), 1e-300)
                nu = 2.0
                improved = True
                break
            lam *= nu
            nu *= 2.0
        err = math.sqrt(f) / normX
        if verbose:
            print(f"      LM it {it} rel err {err:.3e} lam {lam:.1e}", flush=True)
        if log is not None and (it % 5 == 0 or err < 1e-12):
            log(f"LM ({solver}, {nd}x{nd}) it {it}: rel err {err:.2e}  ({time.time() - t0:.0f}s)")
        if not improved or err < 1e-15:
            break
    A = balance(unpack(x))
    return A, cp_rel_error(X, A)


def cp_fit(X, R, n_init=2, n_iter=5000, target=0.0, polish=True, seed=0, verbose=False):
    best = None
    for k in range(n_init):
        A, err, nit = cp_als(X, R, n_iter=n_iter, seed=seed + 1000 * k, target=target, verbose=verbose)
        if verbose:
            print(f"    init {k}: ALS err {err:.3e} after {nit} sweeps", flush=True)
        if best is None or err < best[1]:
            best = (A, err)
        if err <= target:
            break
    A, err = best
    if polish and err > target and X.size * sum(a.size for a in A) < 4e8:
        A2, err2 = lm_polish(X, A, verbose=verbose)
        if verbose:
            print(f"    LM polish: {err:.3e} -> {err2:.3e}", flush=True)
        if err2 < err:
            A, err = A2, err2
    return A, err


def cp_rank_search(X, target, r_lo=None, r_hi=None, verbose=False, **kw):
    """Smallest rank (bisection) whose fit reaches `target` relative error."""
    r_lo = max_matricization_rank(X) if r_lo is None else r_lo
    r_hi = int(np.prod(X.shape) // max(X.shape)) if r_hi is None else r_hi
    found = None
    cache = {}
    # grow until success
    R = r_lo
    while True:
        A, err = cp_fit(X, R, target=target, verbose=verbose, **kw)
        cache[R] = (A, err)
        if verbose:
            print(f"  rank {R}: rel err {err:.3e}", flush=True)
        if err <= target:
            found = R
            break
        if R >= r_hi:
            break
        R = min(r_hi, int(math.ceil(R * 1.25)))
    if found is None:
        R = max(cache, key=lambda r: -cache[r][1])
        return R, cache[R][0], cache[R][1]
    lo = max(k for k in cache if k < found) if any(k < found for k in cache) else r_lo - 1
    hi = found
    while hi - lo > max(2, hi // 20):
        mid = (lo + hi) // 2
        A, err = cp_fit(X, mid, target=target, verbose=False, **kw)
        cache[mid] = (A, err)
        if verbose:
            print(f"  rank {mid}: rel err {err:.3e}", flush=True)
        if err <= target:
            hi = mid
        else:
            lo = mid
    return hi, cache[hi][0], cache[hi][1]



# --------------------------------------------------------------------------
# Symmetric CP for fully symmetric 3-way tensors (homo-atomic 3B types)
#     X = sum_q a_q (x) a_q (x) a_q        (odd order: no weights/signs needed)
# One factor matrix -> one shared 1D table for all three pair slots.
# --------------------------------------------------------------------------

def is_symmetric3(X, tol=1e-12):
    if X.ndim != 3 or len(set(X.shape)) != 1:
        return False
    s = np.linalg.norm(X)
    return all(np.linalg.norm(X - X.transpose(p)) <= tol * s
               for p in [(1, 0, 2), (0, 2, 1), (2, 1, 0)])


def sym_cp_fit(X, R, n_init=6, max_iter=400, seed=0, verbose=False, A0=None, stall=None, target=1e-13):
    """LM fit of a symmetric rank-R CP model on the unique entries of X.

    A0: optional (n, R) warm start, used for the first try (the remaining
    n_init - 1 tries start from random factors).  stall: optional (window,
    factor); a try stops early once its error is still above 1e-6 and has not
    dropped by `factor` over the last `window` iterations (saves time on
    ranks that cannot converge).  target: no further tries once a try ends
    below this relative error."""
    n = X.shape[0]
    iu = np.array([(a, b, c) for a in range(n) for b in range(a, n) for c in range(b, n)])
    mult = np.array([6 if len({a, b, c}) == 3 else (3 if len({a, b, c}) == 2 else 1) for a, b, c in iu], float)
    w = np.sqrt(mult)
    ia, ib, ic = iu.T
    y = X[ia, ib, ic] * w
    ny = np.linalg.norm(y)

    def resid(A):
        return (A[ia] * A[ib] * A[ic]).sum(1) * w - y

    def jac(A):
        J = np.zeros((len(iu), n, R))
        np.add.at(J, (np.arange(len(iu)), ia), A[ib] * A[ic])
        np.add.at(J, (np.arange(len(iu)), ib), A[ia] * A[ic])
        np.add.at(J, (np.arange(len(iu)), ic), A[ia] * A[ib])
        return (J * w[:, None, None]).reshape(len(iu), -1)

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_init):
        if k == 0 and A0 is not None:
            x = np.array(A0, float).reshape(n, R).ravel()
        else:
            x = (rng.normal(size=(n, R)) * (np.linalg.norm(X) / R) ** (1 / 3) / math.sqrt(n)).ravel()
        r = resid(x.reshape(n, R))
        f = float(r @ r)
        lam = 1e-3 * f
        fhist = []
        for it in range(max_iter):
            fhist.append(f)
            if stall is not None and len(fhist) > stall[0] and math.sqrt(f) / ny > 1e-6 \
                    and f > fhist[-1 - stall[0]] / stall[1] ** 2:
                break
            J = jac(x.reshape(n, R))
            K = J @ J.T if J.shape[0] <= J.shape[1] else None
            ok = False
            for _ in range(15):
                try:
                    if K is not None:
                        dx = J.T @ np.linalg.solve(K + lam * np.eye(K.shape[0]), r)
                    else:
                        dx = np.linalg.solve(J.T @ J + lam * np.eye(J.shape[1]), J.T @ r)
                except np.linalg.LinAlgError:       # lam underflowed on a rank-deficient J
                    lam = max(lam * 8, 1e-300)
                    continue
                xn = x - dx
                rn = resid(xn.reshape(n, R))
                fn = float(rn @ rn)
                if fn < f:
                    x, r, f, lam, ok = xn, rn, fn, lam / 5, True
                    break
                lam *= 8
            if not ok or math.sqrt(f) / ny < 1e-15:
                break
        err = math.sqrt(f) / ny
        if verbose:
            print(f"    sym init {k}: rel err {err:.3e}", flush=True)
        if best is None or err < best[1]:
            best = (x.reshape(n, R).copy(), err)
        if err < target:
            break
    A, err = best
    full = np.einsum("ar,br,cr->abc", A, A, A)
    return A, float(np.linalg.norm(full - X) / np.linalg.norm(X))


def sym_cp_rank_search(X, target, r_lo=None, verbose=False):
    n = X.shape[0]
    r_lo = r_lo or max(n, int(math.ceil(n * (n + 1) * (n + 2) / 6 / n)) - 2)
    R = r_lo
    while True:
        A, err = sym_cp_fit(X, R, verbose=False)
        if verbose:
            print(f"  symmetric rank {R}: rel err {err:.3e}", flush=True)
        if err <= target or R > 4 * n * n:
            return R, A, err
        R += max(1, R // 10)


def write_cp_tab_3b(fname, trip_type, A, slots_canon, canon_to_param, pair_types_canonical, rel_err,
                    fcut_type, fcut_var, grid, shared):
    """3B CP table file read by chimesFF::read_3B_cp_file."""
    from chimes_params import slot_basis
    R = A[0].shape[1]
    r0, r1, dr = grid
    with open(fname, "w") as f:
        f.write("CHIMES_3B_CP_TABLE\n")
        f.write(f"trip_type {trip_type}\nrank {R}\nshared {1 if shared else 0}\n")
        f.write("canon_to_param " + " ".join(map(str, canon_to_param)) + "\n")
        f.write("pair_types_canonical " + " ".join(pair_types_canonical) + "\n")
        f.write(f"coefficient_relative_frobenius_error {rel_err:.6e}\n")
        for c in range(1 if shared else 3):
            g = table_grid(slots_canon[c], dr, fcut_type, fcut_var, r0, r1)
            G, D = slot_basis(g, slots_canon[c], A[c].shape[0], fcut_type, fcut_var)
            V, Dv = G @ A[c], D @ A[c]
            f.write(f"slot {c} {len(g)} {g[0]:.17e} {g[1]-g[0]:.17e}\n")
            for i in range(len(g)):
                f.write(" ".join(f"{x:.17e}" for x in V[i]) + " " + " ".join(f"{x:.17e}" for x in Dv[i]) + "\n")
        f.write("END_CHIMES_3B_CP_TABLE\n")

# --------------------------------------------------------------------------
# Writers (formats read by chimesFF)
# --------------------------------------------------------------------------


def table_grid(slot, dr, fcut_type, fcut_var, lo=None, hi=None):
    """Uniform grid (step <= dr) for a 1D CP slot table.

    Cubic Hermite interpolation is O(h^4) only where the tabulated function is
    smooth, so the grid puts nodes on the points where it is not:
      * rmin, the inner cutoff: the grid starts there, because chimesFF
        continues a table below its first node exactly as it continues the
        polynomials below rmin (and the function has a curvature jump there);
      * rmax(1 - fcut_var), where the Tersoff cutoff switches on, if it lies
        above rmin, and rmax.
    A node falling between two of these costs orders of magnitude in accuracy
    (O(h^2) locally).  When rmin, r_t and rmax cannot all be nodes of a grid
    with step >= dr/8, r_t and rmax are kept as nodes and the grid starts at
    the last node <= rmin (the earlier behaviour).
    lo/hi: explicit range [lo, hi] with step dr (no alignment)."""
    if lo is not None or hi is not None:
        rmin = slot["rmin"] if lo is None else lo
        rmax = slot["rmax"] if hi is None else hi
        n = int(math.ceil((rmax - rmin) / dr - 1e-12)) + 1
        return rmin + dr * np.arange(n)
    rmin, rmax = slot["rmin"], slot["rmax"]
    rt = rmax * (1.0 - fcut_var) if (fcut_type == "TERSOFF" and fcut_var is not None) else None
    if rt is None or rt <= rmin + 1e-12:
        m = max(1, int(math.ceil((rmax - rmin) / dr - 1e-9)))
        return rmin + (rmax - rmin) / m * np.arange(m + 1)
    # rmin, r_t and rmax all on the grid: step = common divisor of (r_t - rmin) and (rmax - r_t)
    a, b, u = rt - rmin, rmax - rt, 1e-6
    A, B = int(round(a / u)), int(round(b / u))
    if abs(A * u - a) < 1e-9 and abs(B * u - b) < 1e-9:
        g = math.gcd(A, B) * u
        h = g / max(1, int(math.ceil(g / dr - 1e-9)))
        if h >= dr / 8:
            return rmin + h * np.arange(int(round((rmax - rmin) / h)) + 1)
    m = max(1, int(math.ceil((rmax - rt) / dr)))
    h = (rmax - rt) / m
    k0 = int(math.floor((rmin - rt) / h + 1e-12))         # negative: nodes below r_t
    start = rt + k0 * h
    n = int(round((rmax - start) / h)) + 1
    return start + h * np.arange(n)

def write_cp_direct_4b(fname, quad_type, A, canon_to_param, pair_types_original, pair_types_canonical, rel_err):
    R = A[0].shape[1]
    with open(fname, "w") as f:
        f.write("CHIMES_4B_CP_DIRECT\n")
        f.write(f"quad_type {quad_type}\n")
        f.write(f"rank {R}\n")
        f.write("canon_to_param " + " ".join(map(str, canon_to_param)) + "\n")
        f.write("max_orders " + " ".join(str(a.shape[0] - 1) for a in A) + "\n")
        f.write("tensor_dims " + " ".join(str(a.shape[0]) for a in A) + "\n")
        f.write(f"coefficient_relative_frobenius_error {rel_err:.16e}\n")
        f.write("pair_types_original " + " ".join(pair_types_original) + "\n")
        f.write("pair_types_canonical " + " ".join(pair_types_canonical) + "\n")
        for c, a in enumerate(A):
            f.write(f"factor {c} {a.shape[0]} {R}\n")
            for p in range(a.shape[0]):
                f.write(f"{p} " + " ".join(f"{v:.17e}" for v in a[p]) + "\n")
        f.write("END_CHIMES_4B_CP_DIRECT\n")


def write_cp_tab_4b(prefix, quad_type, A, slots_canon, canon_to_param, pair_types_original, pair_types_canonical,
                    rel_err, fcut_type, fcut_var, r0, r1, dr):
    from chimes_params import slot_basis
    R = A[0].shape[1]
    files = []
    for c, a in enumerate(A):
        # per-slot grid: [rmin, rmax] of the slot unless an explicit range is given
        grid = table_grid(slots_canon[c], dr, fcut_type, fcut_var, r0, r1)
        G, D = slot_basis(grid, slots_canon[c], a.shape[0], fcut_type, fcut_var)
        V, Dv = G @ a, D @ a
        fn = f"{prefix}.slot{c}.dat"
        with open(fn, "w") as f:
            f.write(f"{len(grid)}\n")
            for i, r in enumerate(grid):
                f.write(f"{r:.17f} " + " ".join(f"{x:.17e}" for x in V[i]) + " "
                        + " ".join(f"{x:.17e}" for x in Dv[i]) + "\n")
        files.append(os.path.basename(fn))
    meta = f"{prefix}.meta"
    with open(meta, "w") as f:
        f.write(f"quad_type {quad_type}\nmethod cp_factorized_1d\n")
        f.write("pair_types_original " + " ".join(pair_types_original) + "\n")
        f.write("pair_types_canonical " + " ".join(pair_types_canonical) + "\n")
        f.write("canon_to_param " + " ".join(map(str, canon_to_param)) + "\n")
        f.write("max_orders " + " ".join(str(a.shape[0] - 1) for a in A) + "\n")
        f.write(f"rank {R}\ncoefficient_relative_frobenius_error {rel_err:.16e}\n")
        f.write(f"grid_step {dr:.16e}\nslot_files\n")
        for c, fn in enumerate(files):
            f.write(f"{c} {fn}\n")
    return files, os.path.basename(meta)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    from chimes_params import ChimesParams, canonical_slot_order
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("param")
    ap.add_argument("--body", type=int, default=4)
    ap.add_argument("--types", default="all", help="comma list of cluster type indices or 'all'")
    ap.add_argument("--rank", type=int, default=None, help="fixed rank (default: bisection search)")
    ap.add_argument("--rank-min", type=int, default=None, help="start of the rank search (default: max matricization rank)")
    ap.add_argument("--target", type=float, default=1e-12, help="target relative Frobenius error")
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=5000)
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--from-npz", default=None, help="reuse factors saved by an earlier run (directory)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--direct", action="store_true", help="write CP-direct factor files")
    ap.add_argument("--tables", action="store_true", help="write CP 1D slot tables (4B) or 3B CP table file")
    ap.add_argument("--grid", default=None,
                    help="table grid: 'dr' (per-slot rmin..rmax, recommended) or 'r0,r1,dr'; "
                         "default dr = 0.005 (4B) / 0.01 (3B)")
    ap.add_argument("--symmetric", action="store_true", help="3B: symmetric CP for homo-atomic triplets")
    args = ap.parse_args()

    P = ChimesParams(args.param)
    table = P.quad if args.body == 4 else P.trip
    types = sorted(table) if args.types == "all" else [int(t) for t in args.types.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    param_lines = []
    for t in types:
        rec = table[t]
        if rec["excluded"] or not rec["coeffs"]:
            continue
        X, c2p = P.dense_tensor(args.body, t, canonical=True)
        slots = P.slot_params(args.body, t, canonical=True)
        pt_can = [rec["pair_types"][p] for p in c2p]
        if any(sl["rmax"] <= sl["rmin"] for sl in slots):
            print(f"type {t}: skipped (a slot has rmax <= rmin, so the cluster is never inside the cutoff)")
            continue
        if not np.any(X):
            print(f"type {t}: skipped (all coefficients are zero)")
            continue
        t0 = time.time()
        shared = False
        npz_in = os.path.join(args.from_npz, f"chimes_{args.body}b_cp.type_{t}.factors.npz") if args.from_npz else None
        if npz_in:
            z = np.load(npz_in)
            A = [z[f"A{c}"] for c in range(X.ndim)]
            R = A[0].shape[1]
            err = cp_rel_error(X, A)
            shared = bool(z["shared"]) if "shared" in z else False
        elif args.body == 3 and args.symmetric and is_symmetric3(X) \
                and all(sl == slots[0] for sl in slots):
            if args.rank:
                Asym, err = sym_cp_fit(X, args.rank, verbose=True)
                R = args.rank
            else:
                R, Asym, err = sym_cp_rank_search(X, args.target, verbose=True)
            A = [Asym, Asym, Asym]
            shared = True
        elif args.rank:
            A, err = cp_fit(X, args.rank, n_init=args.n_init, n_iter=args.n_iter, target=args.target,
                            polish=not args.no_polish, verbose=True)
            R = args.rank
        else:
            R, A, err = cp_rank_search(X, args.target, r_lo=args.rank_min, verbose=True, n_init=args.n_init,
                                       n_iter=args.n_iter, polish=not args.no_polish)
        print(f"type {t}: rank {R} rel err {err:.3e}{' (symmetric, shared table)' if shared else ''}"
              f"  ({time.time()-t0:.1f}s)", flush=True)
        if args.body == 3 and args.tables:
            dr = 0.01
            r0 = r1 = None
            if args.grid:
                g = args.grid.split(",")
                if len(g) == 1:
                    dr = float(g[0])
                else:
                    r0, r1, dr = [float(x) for x in g]
            fn = os.path.join(args.outdir, f"chimes_3b_cp.type_{t}.rank_{R}.dat")
            write_cp_tab_3b(fn, t, A, slots, c2p, pt_can, err, P.fcut_type, P.fcut_var, (r0, r1, dr), shared)
            param_lines.append(("3B CP TABLES", f"{t} {os.path.basename(fn)}"))
        np.savez(os.path.join(args.outdir, f"chimes_{args.body}b_cp.type_{t}.factors.npz"),
                 **{f"A{c}": a for c, a in enumerate(A)}, canon_to_param=np.array(c2p), rel_err=err,
                 shared=shared)
        if args.body == 4 and args.direct:
            fn = os.path.join(args.outdir, f"chimes_4b_cp_direct.type_{t}.dat")
            write_cp_direct_4b(fn, t, A, c2p, rec["pair_types"], pt_can, err)
            param_lines.append(("4B CP DIRECT", f"{t} {os.path.basename(fn)}"))
        if args.body == 4 and args.tables:
            r0 = r1 = None          # per-slot [rmin, rmax]
            dr = 0.005
            if args.grid:
                g = args.grid.split(",")
                if len(g) == 1:
                    dr = float(g[0])
                else:
                    r0, r1, dr = [float(x) for x in g]
            prefix = os.path.join(args.outdir, f"chimes_4b_cp.type_{t}.rank_{R}")
            files, meta = write_cp_tab_4b(prefix, t, A, slots, c2p, rec["pair_types"], pt_can, err,
                                          P.fcut_type, P.fcut_var, r0, r1, dr)
            param_lines.append(("4B CP TABLES", f"{t} " + " ".join(files) + f" {meta}"))
    for key in sorted(set(k for k, _ in param_lines)):
        rows = [v for k, v in param_lines if k == key]
        print(f"\nAdd to the parameter file (before ENDFILE):\n{key}: {len(rows)}")
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()