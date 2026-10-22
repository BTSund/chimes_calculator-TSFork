#!/usr/bin/env python3
"""
chimes_params.py

Light-weight, dependency-free (numpy only) parser for ChIMES parameter files
plus helpers that turn the 3-body and 4-body Chebyshev expansions into dense
coefficient tensors and evaluate the per-pair 1D basis

    g_n(r) = fcut(r) * T_n(x(r)),   x(r) = Morse-scaled distance in [-1, 1]

exactly the way chimesFF.cpp does (including the rmin clamp/extrapolation
convention used by the tabulation generators).

Everything else in util/tensor_tools builds on this module.

Slot conventions
----------------
3B: parameter slots follow the "PAIRS:" line of the TRIPLETTYPE block
    (runtime order ij, ik, jk after chimesFF's pair mapping).
4B: parameter slots follow the "PAIRS:" line of the QUADRUPLETYPE block
    (runtime order ij, ik, il, jk, jl, kl after chimesFF's pair mapping).

Canonical slot order (used by all tabulation files) sorts the parameter
slots by (pair-type string, slot index), identical to Tab_CPD.py /
SVD-*.py, and is stored in files as canon_to_param.
"""

import math
import re
import numpy as np


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def _split(line):
    if "!" in line:
        line = line[:line.find("!")]
    if "##" in line:
        line = line[:line.find("##")]
    return line.split()


class ChimesParams:
    """Parsed ChIMES parameter file."""

    def __init__(self, fname):
        self.fname = fname
        with open(fname, "r", errors="ignore") as f:
            self.lines = [l.replace("\xa0", " ").rstrip("\n") for l in f]
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self):
        L = self.lines
        self.poly_orders = None
        self.atom_types = []
        self.pairs = {}            # pair index -> dict
        self.pair_by_name = {}     # "SiSi" -> pair index
        self.fcut_type = "CUBIC"
        self.fcut_var = None
        self.trip = {}             # trip index -> dict
        self.quad = {}             # quad index -> dict
        self.special3_max = {}
        self.special3_min = {}
        self.special4_max = {}
        self.special4_min = {}
        self.tripmaps = {}
        self.quadmaps = {}

        i = 0
        n = len(L)
        while i < n:
            it = _split(L[i])
            s = L[i]
            if not it and "# PAIRIDX #" not in s:     # '!'-commented PAIRIDX header still counts (as in chimesFF)
                i += 1
                continue
            if not it:
                pass
            elif it[0] == "PAIRTYP:":
                self.poly_orders = [int(it[2]), int(it[3]), int(it[4])]
            elif s.strip().startswith("ATOM TYPES:"):
                nat = int(it[2])
                i += 1
                while len(self.atom_types) < nat and i < n:
                    jt = _split(L[i])
                    if jt and jt[0].isdigit():
                        self.atom_types.append(jt[1])
                    i += 1
                continue
            if "# PAIRIDX #" in s and "USEOVRP" not in s:
                i += 1
                while i < n:
                    jt = _split(L[i])
                    if not jt:
                        break
                    if not jt[0].isdigit():
                        break
                    idx = int(jt[0])
                    a, b = jt[1], jt[2]
                    smin, smax = float(jt[3]), float(jt[4])
                    if len(jt) == 8:
                        style, lam = jt[6], float(jt[7])
                    else:
                        style, lam = jt[5], float(jt[6]) if len(jt) > 6 else None
                    self.pairs[idx] = dict(a=a, b=b, smin=smin, smax=smax,
                                           style=style, morse=lam)
                    self.pair_by_name[a + b] = idx
                    i += 1
                continue
            elif s.strip().startswith("FCUT TYPE:"):
                self.fcut_type = it[2].upper()
                if self.fcut_type == "TERSOFF":
                    self.fcut_var = float(it[3])
            elif s.strip().startswith("SPECIAL 3B S_MAXIM:") or s.strip().startswith("SPECIAL 3B S_MINIM:") \
                    or s.strip().startswith("SPECIAL 4B S_MAXIM:") or s.strip().startswith("SPECIAL 4B S_MINIM:"):
                body = 3 if "3B" in s else 4
                kind = "max" if "S_MAXIM" in s else "min"
                target = getattr(self, f"special{body}_{kind}")
                if it[3] == "ALL":
                    target["__ALL__"] = float(it[4])
                else:
                    nent = int(it[4])
                    npair = 3 if body == 3 else 6
                    for _ in range(nent):
                        i += 1
                        jt = _split(L[i])
                        target[jt[0]] = (jt[1:1 + npair],
                                         [float(x) for x in jt[1 + npair:1 + 2 * npair]])
            elif s.strip().startswith("TRIPLETTYPE PARAMS:"):
                i = self._parse_cluster_block(i, 3)
                continue
            elif s.strip().startswith("QUADRUPLETYPE PARAMS:"):
                i = self._parse_cluster_block(i, 4)
                continue
            elif it[0] == "TRIPMAPS:":
                for _ in range(int(it[1])):
                    i += 1
                    jt = _split(L[i])
                    self.tripmaps[jt[1]] = int(jt[0])
            elif it[0] == "QUADMAPS:":
                for _ in range(int(it[1])):
                    i += 1
                    jt = _split(L[i])
                    self.quadmaps[jt[1]] = int(jt[0])
            i += 1

        self._resolve_cutoffs()

    # ------------------------------------------------------------------
    def _parse_cluster_block(self, i, body):
        L = self.lines
        npair = 3 if body == 3 else 6
        i += 1
        hdr = _split(L[i])
        idx = int(hdr[1])
        atoms = hdr[3:3 + body]
        tabfile = None
        if "TABULATED" in hdr:
            tabfile = hdr[hdr.index("TABULATED") + 1]
        i += 1
        pl = _split(L[i])
        pair_types = pl[1:1 + npair]
        rec = dict(index=idx, atoms=atoms, pair_types=pair_types, tabfile=tabfile,
                   excluded=False, powers=[], coeffs=[], equiv=[], pidx=[])
        if len(pl) > 1 + npair and pl[1 + npair] == "EXCLUDED:":
            rec["excluded"] = True
        else:
            ntot = int(pl[pl.index("TOTAL:") + 1])
            i += 2  # header lines
            for _ in range(ntot):
                i += 1
                jt = _split(L[i])
                rec["powers"].append(tuple(int(x) for x in jt[1:1 + npair]))
                rec["equiv"].append(int(jt[1 + npair]))
                rec["pidx"].append(int(jt[2 + npair]))
                rec["coeffs"].append(float(jt[3 + npair]))
        (self.trip if body == 3 else self.quad)[idx] = rec
        return i + 1

    # ------------------------------------------------------------------
    def _resolve_cutoffs(self):
        """Per-slot (rmin, rmax, morse) for every 3B/4B cluster type."""
        for body, table, smax, smin, maps in ((3, self.trip, self.special3_max, self.special3_min, self.tripmaps),
                                              (4, self.quad, self.special4_max, self.special4_min, self.quadmaps)):
            inv_maps = {}                      # type index -> every map name (any atom order) for it
            for k, v in maps.items():
                inv_maps.setdefault(v, []).append(k)
            for idx, rec in table.items():
                slots = []
                for pt in rec["pair_types"]:
                    pidx = self.pair_by_name[pt]
                    p = self.pairs[pidx]
                    slots.append(dict(pair=pt, pair_index=pidx, rmin=p["smin"], rmax=p["smax"], morse=p["morse"]))
                if "__ALL__" in smax:
                    for sl in slots:
                        sl["rmax"] = smax["__ALL__"]
                if "__ALL__" in smin:
                    for sl in slots:
                        sl["rmin"] = smin["__ALL__"]
                for src, key in ((smax, "rmax"), (smin, "rmin")):
                    # SPECIFIC entries may be keyed by any of the type's map names
                    name = next((nm for nm in inv_maps.get(idx, []) if nm in src), None)
                    if name is not None:
                        names, vals = src[name]
                        used = [False] * len(slots)
                        for nm, v in zip(names, vals):
                            for s, sl in enumerate(slots):
                                if not used[s] and sl["pair"] == nm:
                                    sl[key] = v
                                    used[s] = True
                                    break
                rec["slots"] = slots

    # ------------------------------------------------------------------
    # Dense tensors
    # ------------------------------------------------------------------
    def dense_tensor(self, body, idx, canonical=False):
        """Dense coefficient tensor C[n_0, ..., n_{npair-1}] in parameter
        (or canonical) slot order.  Dimension per mode = max power + 1."""
        rec = (self.trip if body == 3 else self.quad)[idx]
        npair = 3 if body == 3 else 6
        pw = np.array(rec["powers"], dtype=int)
        c = np.array(rec["coeffs"], dtype=float)
        perm = list(range(npair))
        if canonical:
            perm = canonical_slot_order(rec["pair_types"])[0]
            pw = pw[:, perm]
        dims = tuple(int(pw[:, m].max()) + 1 for m in range(npair))
        X = np.zeros(dims)
        for p, v in zip(pw, c):
            X[tuple(p)] += v
        return X, perm

    def slot_params(self, body, idx, canonical=False):
        rec = (self.trip if body == 3 else self.quad)[idx]
        slots = rec["slots"]
        if canonical:
            perm = canonical_slot_order(rec["pair_types"])[0]
            slots = [slots[p] for p in perm]
        return slots


def canonical_slot_order(pair_types):
    order = sorted(range(len(pair_types)), key=lambda s: (pair_types[s], s))
    canon_to_param = list(order)
    param_to_canon = [0] * len(order)
    for c, p in enumerate(canon_to_param):
        param_to_canon[p] = c
    return canon_to_param, param_to_canon


# --------------------------------------------------------------------------
# 1D basis  g_n(r) = fcut(r) T_n(x(r))  and derivative wrt r
# --------------------------------------------------------------------------

def fcut_and_deriv(r, rmax, fcut_type, fcut_var):
    r = np.asarray(r, dtype=float)
    if fcut_type == "CUBIC":
        f0 = 1.0 - r / rmax
        return f0 ** 3, -3.0 * f0 ** 2 / rmax
    thresh = rmax - fcut_var * rmax
    arg = (r - thresh) / (rmax - thresh) * math.pi + math.pi / 2.0
    darg = math.pi / (rmax - thresh)
    fc = np.where(r < thresh, 1.0, np.where(r > rmax, 0.0, 0.5 + 0.5 * np.sin(arg)))
    dfc = np.where((r < thresh) | (r > rmax), 0.0, 0.5 * np.cos(arg) * darg)
    return fc, dfc


INNER_SMOOTH_DISTANCE = 0.01   # chimesFF::inner_smooth_distance


def cheb_basis(r, rmin, rmax, morse, nmax, extrapolate=True):
    """T_n(x(r)) and dT_n/dr for n = 0..nmax-1 ; r array (N,) -> (N, nmax).

    extrapolate=True (default) reproduces chimesFF::set_polys_out_of_range
    exactly: below rmin, T_n(r) = T_n(rmin) + s (exp((r-rmin)/s) - 1) T'_n(rmin)
    and T'_n(r) = exp((r-rmin)/s) T'_n(rmin), with s = 0.01 A.
    extrapolate=False clamps to rmin (what the original util/4B_files
    generators did -- note their derivative is then inconsistent)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    xmin = math.exp(-rmin / morse)
    xmax = math.exp(-rmax / morse)
    xavg = 0.5 * (xmax + xmin)
    xdiff = -0.5 * (xmax - xmin)
    rr = np.maximum(r, rmin)
    e = np.exp(-rr / morse)
    x = (e - xavg) / xdiff
    dxdr = (-e / morse) / xdiff
    N = len(r)
    T = np.zeros((N, nmax))
    U = np.zeros((N, nmax))
    T[:, 0] = 1.0
    U[:, 0] = 1.0
    if nmax > 1:
        T[:, 1] = x
        U[:, 1] = 2 * x
    for k in range(2, nmax):
        T[:, k] = 2 * x * T[:, k - 1] - T[:, k - 2]
        U[:, k] = 2 * x * U[:, k - 1] - U[:, k - 2]
    dT = np.zeros_like(T)
    for k in range(1, nmax):
        dT[:, k] = k * dxdr * U[:, k - 1]
    if extrapolate:
        below = r < rmin
        if below.any():
            s = INNER_SMOOTH_DISTANCE
            damp = np.exp((r[below] - rmin) / s)[:, None]
            T[below] = T[below] + s * (damp - 1.0) * dT[below]
            dT[below] = dT[below] * damp
    return T, dT


def slot_basis(r, slot, nmax, fcut_type, fcut_var, extrapolate=True):
    """g_n(r) = fcut*T_n and dg_n/dr for one slot dict -> (N,nmax),(N,nmax)."""
    T, dT = cheb_basis(r, slot["rmin"], slot["rmax"], slot["morse"], nmax, extrapolate)
    fc, dfc = fcut_and_deriv(np.atleast_1d(r), slot["rmax"], fcut_type, fcut_var)
    G = fc[:, None] * T
    D = dfc[:, None] * T + fc[:, None] * dT
    return G, D


def eval_dense(X, Gs, Ds=None):
    """Energy (and per-slot dE/dr) of a dense coefficient tensor for a batch.
    Gs[m]: (N, dim_m) basis values for slot m.  Reference implementation."""
    nm = X.ndim
    letters = "abcdefgh"[:nm]
    spec = letters + "," + ",".join("z" + l for l in letters) + "->z"
    E = np.einsum(spec, X, *Gs, optimize=True)
    if Ds is None:
        return E
    dE = []
    for m in range(nm):
        ops = [Gs[k] if k != m else Ds[k] for k in range(nm)]
        dE.append(np.einsum(spec, X, *ops, optimize=True))
    return E, np.array(dE).T


if __name__ == "__main__":
    import sys
    P = ChimesParams(sys.argv[1])
    print("poly orders", P.poly_orders, "atoms", P.atom_types, "fcut", P.fcut_type, P.fcut_var)
    for idx, rec in P.trip.items():
        if rec["excluded"]:
            continue
        X, _ = P.dense_tensor(3, idx)
        print("3B", idx, rec["pair_types"], "nterms", len(rec["coeffs"]), "dims", X.shape,
              "slots", [(s["rmin"], s["rmax"]) for s in rec["slots"]])
    for idx, rec in P.quad.items():
        if rec["excluded"]:
            continue
        X, _ = P.dense_tensor(4, idx)
        print("4B", idx, rec["pair_types"], "nterms", len(rec["coeffs"]), "dims", X.shape,
              "nnz", int((X != 0).sum()), "slots", [(s["rmin"], s["rmax"]) for s in rec["slots"]])