#!/usr/bin/env python3
"""
SVD.py

Generate SVD-factorized 3|3 tabulated 4-body ChIMES tables.

For each 4-body quad type:
  1. Parse 4-body coefficient tensor c_{abcdef}
  2. Canonicalize six pair slots by pair type then original slot index
  3. Matricize as (a,b,c) | (d,e,f)
  4. Compute SVD:
         M ~= U S V^T
  5. Build factors:
         L = U sqrt(S)
         R = V sqrt(S)
  6. Tabulate:
         X_p(r0,r1,r2) = L_{abc,p} G_a(r0) G_b(r1) G_c(r2)
         Y_p(r3,r4,r5) = R_{def,p} G_d(r3) G_e(r4) G_f(r5)

Outputs per quad type:
  chimes_scan_4b_svd3x3.type_<quad_type>.meta
  chimes_scan_4b_svd3x3.type_<quad_type>.left.dat
  chimes_scan_4b_svd3x3.type_<quad_type>.right.dat

Table row layout:
  r0 r1 r2  val[0:R]  d0[0:R]  d1[0:R]  d2[0:R]

For the right table, r0/r1/r2 are the local coordinates corresponding
to canonical slots 3/4/5.

At runtime:
  E = dot(left_values, right_values)

Derivative examples:
  dE/dr_left_0  = dot(left_d0, right_values)
  dE/dr_right_0 = dot(left_values, right_d0)
"""

import os
import sys
import math
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.append(os.path.normpath(os.getcwd()))

if not os.path.exists("config.py"):
    print("Error: Cannot find config.py")
    sys.exit(1)

import config


# =============================================================================
# Basic parsing helpers
# =============================================================================

def split_clean(line):
    if "!" in line:
        line = line[:line.find("!")]
    if "##" in line:
        line = line[:line.find("##")]
    line = line.rstrip("\n")
    return line.split()


def get_file_lines(param_file):
    with open(param_file, "r") as f:
        return f.readlines()


def get_config_indexed_value(obj, idx):
    """
    Allows config values to be either scalar or per-quad lists.
    """
    if isinstance(obj, (list, tuple, np.ndarray)):
        return obj[idx]
    return obj


# =============================================================================
# Parameter-file parsing
# =============================================================================

def parse_fcut_type(param_file):
    lines = get_file_lines(param_file)
    fcut_type = "CUBIC"
    fcut_var = None

    for line in lines:
        items = split_clean(line)
        if len(items) >= 3 and items[0] == "FCUT" and items[1] == "TYPE:":
            fcut_type = items[2].upper()
            if fcut_type == "TERSOFF":
                if len(items) < 4:
                    raise RuntimeError("FCUT TYPE: TERSOFF requires parameter")
                fcut_var = float(items[3])
            return fcut_type, fcut_var

    return fcut_type, fcut_var


def parse_all_pair_metadata(param_file):
    lines = get_file_lines(param_file)

    no_pairs = None
    pair_meta_by_index = {}

    i = 0
    while i < len(lines):
        items = split_clean(lines[i])

        if not items:
            i += 1
            continue

        if len(items) >= 3 and items[0] == "ATOM" and items[1] == "PAIRS:":
            no_pairs = int(items[2])

        if len(items) >= 3 and items[0] == "#" and items[1] == "PAIRIDX" and items[2] == "#":
            raw = lines[i]
            if "# USEOVRP #" in raw:
                i += 1
                continue

            if no_pairs is None:
                raise RuntimeError("Found pair table header before ATOM PAIRS count")

            tmp_xform_style = None

            for _ in range(no_pairs):
                i += 1
                items = split_clean(lines[i])

                if len(items) not in (7, 8):
                    raise RuntimeError(
                        f"Incorrect pair specification line; expected 7 or 8 entries:\n"
                        f"{lines[i].rstrip()}"
                    )

                pair_idx = int(items[0])
                type1 = items[1]
                type2 = items[2]
                s_min = float(items[3])
                s_max = float(items[4])

                if len(items) == 8:
                    xform_style_idx = 6
                    morse_idx = 7
                else:
                    xform_style_idx = 5
                    morse_idx = 6

                dist_type = items[xform_style_idx]

                if tmp_xform_style is None:
                    tmp_xform_style = dist_type
                elif dist_type != tmp_xform_style:
                    raise RuntimeError("Distance transformation style differs across pair types")

                morse_lambda = None
                if dist_type.upper() == "MORSE":
                    morse_lambda = float(items[morse_idx])

                pair_meta_by_index[pair_idx] = {
                    "pair_index": pair_idx,
                    "type1": type1,
                    "type2": type2,
                    "s_min": s_min,
                    "s_max": s_max,
                    "dist_type": dist_type,
                    "morse_lambda": morse_lambda,
                }

        i += 1

    if not pair_meta_by_index:
        raise RuntimeError("Failed to parse any pair metadata")

    return pair_meta_by_index


def get_pair_meta_for_pair_name(pair_name, pair_meta_by_index):
    matches = []
    for _, meta in pair_meta_by_index.items():
        if meta["type1"] + meta["type2"] == pair_name:
            matches.append(meta)

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise RuntimeError(f"Could not resolve pair name '{pair_name}' against pair metadata")
    raise RuntimeError(f"Ambiguous pair name '{pair_name}' in pair metadata")


def parse_quad_pair_types(param_file, quad_index=0):
    lines = get_file_lines(param_file)

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if "QUADRUPLETYPE PARAMS:" in s:
            i += 1
            if i >= len(lines):
                break

            hdr = split_clean(lines[i])
            if len(hdr) < 7 or hdr[0] != "INDEX:":
                raise RuntimeError(
                    f"Malformed 4B header after QUADRUPLETYPE PARAMS at line {i+1}: "
                    f"{lines[i].rstrip()}"
                )

            this_idx = int(hdr[1])

            i += 1
            if i >= len(lines):
                break

            pairline = split_clean(lines[i])
            if len(pairline) < 11:
                raise RuntimeError(
                    f"Malformed 4B pair/type line for quad index {this_idx}: "
                    f"{lines[i].rstrip()}"
                )

            if this_idx == quad_index:
                atom_types = hdr[3:7]
                pair_types = pairline[1:7]
                excluded = pairline[7] == "EXCLUDED:"
                ncoeff = None if excluded else int(pairline[10])

                return {
                    "quad_index": this_idx,
                    "atom_types": atom_types,
                    "pair_types": pair_types,
                    "excluded": excluded,
                    "ncoeff": ncoeff,
                }

        i += 1

    raise RuntimeError(f"Could not find QUADRUPLETYPE PARAMS for quad_index={quad_index}")


def parse_quad_terms_all(param_file, quad_index=0):
    lines = get_file_lines(param_file)

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if "QUADRUPLETYPE PARAMS:" in s:
            i += 1
            if i >= len(lines):
                break

            hdr = split_clean(lines[i])
            if len(hdr) < 2 or hdr[0] != "INDEX:":
                raise RuntimeError(
                    f"Malformed 4B header after QUADRUPLETYPE PARAMS at line {i+1}: "
                    f"{lines[i].rstrip()}"
                )

            this_idx = int(hdr[1])

            i += 1
            if i >= len(lines):
                break

            pairline = split_clean(lines[i])
            if len(pairline) < 11:
                raise RuntimeError(
                    f"Malformed 4B pair/type line for quad index {this_idx}: "
                    f"{lines[i].rstrip()}"
                )

            if this_idx == quad_index:
                if pairline[7] == "EXCLUDED:":
                    return []

                ncoeff = int(pairline[10])

                i += 1
                i += 1

                terms = []
                for _ in range(ncoeff):
                    i += 1
                    if i >= len(lines):
                        raise RuntimeError("Unexpected EOF while reading 4B coefficient rows")

                    parts = split_clean(lines[i])
                    if len(parts) < 10:
                        raise RuntimeError(f"Malformed 4B coeff row: {lines[i].rstrip()}")

                    terms.append({
                        "row_index": int(parts[0]),
                        "powers": tuple(int(x) for x in parts[1:7]),
                        "equiv_index": int(parts[7]),
                        "param_index": int(parts[8]),
                        "coeff": float(parts[9]),
                        "raw": lines[i].strip(),
                    })

                return terms

        i += 1

    raise RuntimeError(f"No quadruplet terms found for quad_index={quad_index}")


def parse_quadmaps(param_file):
    lines = get_file_lines(param_file)

    atom_typ_quad_map = []
    atom_idx_quad_map = []

    i = 0
    while i < len(lines):
        items = split_clean(lines[i])

        if items and items[0] == "QUADMAPS:":
            if len(items) < 2:
                raise RuntimeError("Malformed QUADMAPS line")
            n_quad_maps = int(items[1])

            for _ in range(n_quad_maps):
                i += 1
                row = split_clean(lines[i])
                if len(row) < 2:
                    raise RuntimeError(f"Malformed QUADMAPS row: {lines[i].rstrip()}")
                atom_idx_quad_map.append(int(row[0]))
                atom_typ_quad_map.append(row[1])
            break

        i += 1

    return atom_typ_quad_map, atom_idx_quad_map


def parse_special_4b_cutoffs(param_file, atom_typ_quad_map, atom_idx_quad_map):
    lines = get_file_lines(param_file)

    slow_to_quadidx = {
        name: idx for name, idx in zip(atom_typ_quad_map, atom_idx_quad_map)
    }

    special_min = {}
    special_max = {}

    i = 0
    while i < len(lines):
        items = split_clean(lines[i])

        if not items:
            i += 1
            continue

        if (
            len(items) >= 4
            and items[0] == "SPECIAL"
            and items[1] == "4B"
            and items[2] == "S_MAXIM:"
        ):
            if items[3] == "ALL":
                val = float(items[4])
                special_max["__ALL__"] = val
            else:
                nentries = int(items[4])

                for _ in range(nentries):
                    i += 1
                    row = split_clean(lines[i])
                    if len(row) < 13:
                        raise RuntimeError(
                            f"Malformed SPECIAL 4B S_MAXIM entry: {lines[i].rstrip()}"
                        )

                    quad_slow_name = row[0]
                    quadidx = slow_to_quadidx[quad_slow_name]

                    pair_names = row[1:7]
                    cutoffvals = [float(x) for x in row[7:13]]

                    special_max[quadidx] = {
                        "pair_names": pair_names,
                        "values": cutoffvals,
                    }

        if (
            len(items) >= 4
            and items[0] == "SPECIAL"
            and items[1] == "4B"
            and items[2] == "S_MINIM:"
        ):
            if items[3] == "ALL":
                val = float(items[4])
                special_min["__ALL__"] = val
            else:
                nentries = int(items[4])

                for _ in range(nentries):
                    i += 1
                    row = split_clean(lines[i])
                    if len(row) < 13:
                        raise RuntimeError(
                            f"Malformed SPECIAL 4B S_MINIM entry: {lines[i].rstrip()}"
                        )

                    quad_slow_name = row[0]
                    quadidx = slow_to_quadidx[quad_slow_name]

                    pair_names = row[1:7]
                    cutoffvals = [float(x) for x in row[7:13]]

                    special_min[quadidx] = {
                        "pair_names": pair_names,
                        "values": cutoffvals,
                    }

        i += 1

    return special_min, special_max


def resolve_occurrence_mapped_values(pair_types_reference, pair_names_list, values_list):
    resolved = [None] * 6
    used = [False] * 6

    for name, val in zip(pair_names_list, values_list):
        found = False
        for i in range(6):
            if pair_types_reference[i] == name and not used[i]:
                resolved[i] = val
                used[i] = True
                found = True
                break

        if not found:
            raise RuntimeError(
                f"Could not map special 4B pair name '{name}' into reference pair list "
                f"{pair_types_reference}"
            )

    if any(v is None for v in resolved):
        raise RuntimeError(
            f"Failed to resolve all 6 special 4B values against reference pair list "
            f"{pair_types_reference}"
        )

    return resolved


def build_quad_slot_params(param_file, quad_type):
    pair_meta_by_index = parse_all_pair_metadata(param_file)
    quad_info = parse_quad_pair_types(param_file, quad_index=quad_type)

    if quad_info["excluded"]:
        return []

    atom_typ_quad_map, atom_idx_quad_map = parse_quadmaps(param_file)

    special_min, special_max = parse_special_4b_cutoffs(
        param_file,
        atom_typ_quad_map,
        atom_idx_quad_map,
    )

    pair_types = quad_info["pair_types"]

    slot_params = []
    for pair_name in pair_types:
        meta = get_pair_meta_for_pair_name(pair_name, pair_meta_by_index)

        if meta["dist_type"].upper() != "MORSE":
            raise RuntimeError(
                f"Only MORSE mapping is currently implemented, found "
                f"{meta['dist_type']} for pair '{pair_name}'"
            )

        slot_params.append({
            "pair_name": pair_name,
            "pair_index": meta["pair_index"],
            "type1": meta["type1"],
            "type2": meta["type2"],
            "rmin": meta["s_min"],
            "rmax": meta["s_max"],
            "morse": meta["morse_lambda"],
            "dist_type": meta["dist_type"],
        })

    if "__ALL__" in special_min:
        all_min = special_min["__ALL__"]
        for sp in slot_params:
            sp["rmin"] = all_min

    if quad_type in special_min:
        resolved = resolve_occurrence_mapped_values(
            pair_types_reference=pair_types,
            pair_names_list=special_min[quad_type]["pair_names"],
            values_list=special_min[quad_type]["values"],
        )
        for i in range(6):
            slot_params[i]["rmin"] = resolved[i]

    if "__ALL__" in special_max:
        all_max = special_max["__ALL__"]
        for sp in slot_params:
            sp["rmax"] = all_max

    if quad_type in special_max:
        resolved = resolve_occurrence_mapped_values(
            pair_types_reference=pair_types,
            pair_names_list=special_max[quad_type]["pair_names"],
            values_list=special_max[quad_type]["values"],
        )
        for i in range(6):
            slot_params[i]["rmax"] = resolved[i]

    return slot_params


# =============================================================================
# Canonical ordering
# =============================================================================

def canonical_slot_order(pair_types):
    """
    Sort parameter slots by:
      1. pair type string
      2. original slot index
    """
    pairs = [(pair_types[i], i) for i in range(6)]
    pairs.sort(key=lambda x: (x[0], x[1]))

    canon_to_param = [slot for _, slot in pairs]
    param_to_canon = [None] * 6

    for c, p in enumerate(canon_to_param):
        param_to_canon[p] = c

    return canon_to_param, param_to_canon


def reorder_terms_to_canonical(terms, param_to_canon):
    new_terms = []

    for powers, coeff in terms:
        cpows = [0] * 6

        for pslot in range(6):
            cslot = param_to_canon[pslot]
            cpows[cslot] = powers[pslot]

        new_terms.append((tuple(cpows), coeff))

    return new_terms


def infer_max_orders(terms):
    max_orders = [0] * 6

    for powers, _ in terms:
        for i in range(6):
            max_orders[i] = max(max_orders[i], powers[i])

    return max_orders


# =============================================================================
# Chebyshev/cutoff functions
# =============================================================================

def morse_transform_data(r, rmin, rmax, morse_param):
    x_min = math.exp(-rmin / morse_param)
    x_max = math.exp(-rmax / morse_param)
    x_avg = 0.5 * (x_max + x_min)
    x_diff = -0.5 * (x_max - x_min)

    exprlen = math.exp(-r / morse_param)
    x = (exprlen - x_avg) / x_diff
    dx_dr = (-exprlen / morse_param) / x_diff

    return x, dx_dr


def cheb_value_deriv_wrt_r(r, rmin, rmax, morse_param, order):
    r_eval = rmin if r < rmin else r
    x, dx_dr = morse_transform_data(r_eval, rmin, rmax, morse_param)

    Tn = np.zeros(order + 1, dtype=float)
    TndU = np.zeros(order + 1, dtype=float)
    Tnd = np.zeros(order + 1, dtype=float)

    Tn[0] = 1.0
    TndU[0] = 1.0

    if order >= 1:
        Tn[1] = x
        TndU[1] = 2.0 * x

    for i in range(2, order + 1):
        Tn[i] = 2.0 * x * Tn[i - 1] - Tn[i - 2]
        TndU[i] = 2.0 * x * TndU[i - 1] - TndU[i - 2]

    Tnd[0] = 0.0
    for i in range(1, order + 1):
        Tnd[i] = i * dx_dr * TndU[i - 1]

    return Tn, Tnd


def get_fcut(r, outer_cutoff, fcut_type, fcut_var=None):
    if fcut_type == "CUBIC":
        fcut0 = 1.0 - r / outer_cutoff
        fcut = fcut0 ** 3
        fcutderiv = -3.0 * (fcut0 ** 2) / outer_cutoff
        return fcut, fcutderiv

    if fcut_type == "TERSOFF":
        if fcut_var is None:
            raise RuntimeError("TERSOFF cutoff requires fcut_var")

        thresh = outer_cutoff - fcut_var * outer_cutoff

        if r < thresh:
            return 1.0, 0.0
        if r > outer_cutoff:
            return 0.0, 0.0

        arg = (r - thresh) / (outer_cutoff - thresh) * math.pi + math.pi / 2.0
        arg_deriv = math.pi / (outer_cutoff - thresh)

        fcut = 0.5 + 0.5 * math.sin(arg)
        fcutderiv = 0.5 * math.cos(arg) * arg_deriv

        return fcut, fcutderiv

    raise RuntimeError(f"Unknown cutoff type: {fcut_type}")


def build_GT_table_for_slot(grid, slot_param, max_order, fcut_type, fcut_var):
    ngrid = len(grid)

    G = np.zeros((ngrid, max_order + 1), dtype=float)
    D = np.zeros((ngrid, max_order + 1), dtype=float)

    rmin = slot_param["rmin"]
    rmax = slot_param["rmax"]
    morse = slot_param["morse"]

    for i, r in enumerate(grid):
        Tn, Tnd = cheb_value_deriv_wrt_r(r, rmin, rmax, morse, max_order)
        fcut, fcutderiv = get_fcut(r, rmax, fcut_type, fcut_var)

        G[i, :] = fcut * Tn
        D[i, :] = fcutderiv * Tn + fcut * Tnd

    return G, D


# =============================================================================
# SVD factorization
# =============================================================================

def flat_index_variable(indices, dims):
    idx = 0
    for x, dim in zip(indices, dims):
        idx = idx * dim + x
    return idx


def build_3x3_matrix(terms, max_orders, left_dims=(0, 1, 3), right_dims=(2, 4, 5)):
    left_shape = tuple(max_orders[d] + 1 for d in left_dims)
    right_shape = tuple(max_orders[d] + 1 for d in right_dims)

    left_size = int(np.prod(left_shape))
    right_size = int(np.prod(right_shape))

    M = np.zeros((left_size, right_size), dtype=float)

    for powers, coeff in terms:
        li = flat_index_variable(
            tuple(powers[d] for d in left_dims),
            left_shape,
        )
        ri = flat_index_variable(
            tuple(powers[d] for d in right_dims),
            right_shape,
        )

        M[li, ri] += coeff

    return M, left_shape, right_shape


def choose_svd_rank(svals, tol):
    """
    Choose smallest R such that relative Frobenius loss is <= tol.

    loss(R) = sqrt(sum_{i>R} s_i^2 / sum_i s_i^2)
    """
    svals = np.asarray(svals, dtype=float)
    s2 = svals ** 2
    total = np.sum(s2)

    if total == 0.0:
        return 0, 0.0

    cumulative = np.cumsum(s2) / total
    loss = np.sqrt(np.maximum(0.0, 1.0 - cumulative))

    idx = np.where(loss <= tol)[0]
    if len(idx) == 0:
        R = len(svals)
    else:
        R = int(idx[0] + 1)

    final_loss = float(loss[R - 1]) if R > 0 else 0.0

    return R, final_loss


def get_forced_rank_for_quad(quad_type, quad_idx, default_rank=None):
    """
    Optional config support.

    Supported forms:
      SVD_RANK = 26
      SVD_RANKS = {0: 26, 1: 30}
      SVD_RANKS = [26, 30, ...]
    """
    if hasattr(config, "SVD_RANK"):
        return int(config.SVD_RANK)

    if hasattr(config, "SVD_RANKS"):
        ranks = config.SVD_RANKS

        if isinstance(ranks, dict):
            if quad_type in ranks:
                return int(ranks[quad_type])
            return default_rank

        if isinstance(ranks, (list, tuple, np.ndarray)):
            return int(ranks[quad_idx])

        return int(ranks)

    return default_rank


def svd_factorize_3x3(terms, max_orders, tol, forced_rank=None):
    left_dims = (0, 1, 3)
    right_dims = (2, 4, 5)

    M, left_shape, right_shape = build_3x3_matrix(
        terms,
        max_orders,
        left_dims=left_dims,
        right_dims=right_dims,
    )

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    auto_rank, rel_loss = choose_svd_rank(S, tol)

    if forced_rank is None:
        R = auto_rank
    else:
        R = int(forced_rank)
        if R < 0:
            raise RuntimeError("Forced SVD rank must be nonnegative")
        if R > len(S):
            raise RuntimeError(
                f"Forced SVD rank {R} exceeds available singular values {len(S)}"
            )

        if R == 0:
            rel_loss = 1.0 if np.sum(S ** 2) > 0.0 else 0.0
        else:
            s2 = S ** 2
            cumulative = np.cumsum(s2) / np.sum(s2)
            rel_loss = float(np.sqrt(max(0.0, 1.0 - cumulative[R - 1])))

    if R == 0:
        left_factor = np.zeros(left_shape + (0,), dtype=float)
        right_factor = np.zeros(right_shape + (0,), dtype=float)
        return left_factor, right_factor, S, R, rel_loss, left_shape, right_shape

    sqrtS = np.sqrt(S[:R])

    left_mat = U[:, :R] * sqrtS[None, :]
    right_mat = Vh[:R, :].T * sqrtS[None, :]

    left_factor = left_mat.reshape(left_shape + (R,))
    right_factor = right_mat.reshape(right_shape + (R,))

    return left_factor, right_factor, S, R, rel_loss, left_shape, right_shape


# =============================================================================
# 3D factor table generation
# =============================================================================

def iter_grid_index_chunks(ngrid, k, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    iterator = itertools.product(range(ngrid), repeat=k)

    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def compute_factor_table_chunk(args):
    """
    Compute rows for one 3D table.

    factor shape:
      (n0, n1, n2, R)

    G_arrays/D_arrays:
      list of three arrays, each shape (ngrid, n_i)
    """
    idx_chunk, grid, factor, G_arrays, D_arrays = args

    out_lines = []
    R = factor.shape[-1]

    for i0, i1, i2 in idx_chunk:
        g0 = G_arrays[0][i0]
        g1 = G_arrays[1][i1]
        g2 = G_arrays[2][i2]

        d0 = D_arrays[0][i0]
        d1 = D_arrays[1][i1]
        d2 = D_arrays[2][i2]

        vals = np.einsum("abcp,a,b,c->p", factor, g0, g1, g2, optimize=True)
        der0 = np.einsum("abcp,a,b,c->p", factor, d0, g1, g2, optimize=True)
        der1 = np.einsum("abcp,a,b,c->p", factor, g0, d1, g2, optimize=True)
        der2 = np.einsum("abcp,a,b,c->p", factor, g0, g1, d2, optimize=True)

        parts = [
            f"{grid[i0]:.12f}",
            f"{grid[i1]:.12f}",
            f"{grid[i2]:.12f}",
        ]

        parts.extend(f"{x:.16e}" for x in vals)
        parts.extend(f"{x:.16e}" for x in der0)
        parts.extend(f"{x:.16e}" for x in der1)
        parts.extend(f"{x:.16e}" for x in der2)

        out_lines.append(" ".join(parts) + "\n")

    return out_lines


def write_factor_table(
    outfile,
    grid,
    factor,
    slot_params_local,
    max_orders_local,
    fcut_type,
    fcut_var,
    nprocs=None,
    chunk_size=200,
):
    ngrid = len(grid)
    nrows = ngrid ** 3
    R = factor.shape[-1]

    G_arrays = []
    D_arrays = []

    for j in range(3):
        G, D = build_GT_table_for_slot(
            grid=grid,
            slot_param=slot_params_local[j],
            max_order=max_orders_local[j],
            fcut_type=fcut_type,
            fcut_var=fcut_var,
        )
        G_arrays.append(G)
        D_arrays.append(D)

    values_per_row = 3 + 4 * R

    print(f"Writing table: {outfile}", flush=True)
    print(f"  rows             = {nrows}", flush=True)
    print(f"  rank R           = {R}", flush=True)
    print(f"  values per row   = {values_per_row}", flush=True)
    print(f"  chunk_size       = {chunk_size}", flush=True)
    print(f"  nprocs           = {nprocs}", flush=True)

    chunk_iter = iter_grid_index_chunks(ngrid, 3, chunk_size)

    with open(outfile, "w") as f:
        f.write(f"{nrows}\n")

        if nprocs is None or nprocs <= 1:
            row = 0
            report_every = max(chunk_size * 10, 1000)
            next_report = report_every

            for chunk in chunk_iter:
                out_lines = compute_factor_table_chunk(
                    (chunk, grid, factor, G_arrays, D_arrays)
                )
                f.writelines(out_lines)

                row += len(out_lines)
                if row >= next_report or row == nrows:
                    print(f"  wrote {row} / {nrows} rows", flush=True)
                    while next_report <= row:
                        next_report += report_every

        else:
            row = 0
            report_every = max(chunk_size * 10, 1000)
            next_report = report_every

            with ProcessPoolExecutor(max_workers=nprocs) as ex:
                args_iter = (
                    (chunk, grid, factor, G_arrays, D_arrays)
                    for chunk in chunk_iter
                )

                for out_lines in ex.map(compute_factor_table_chunk, args_iter, chunksize=1):
                    f.writelines(out_lines)

                    row += len(out_lines)
                    if row >= next_report or row == nrows:
                        print(f"  wrote {row} / {nrows} rows", flush=True)
                        while next_report <= row:
                            next_report += report_every


# =============================================================================
# Metadata output
# =============================================================================

def write_meta_file(
    outfile,
    quad_type,
    pair_types_original,
    pair_types_canonical,
    canon_to_param,
    left_dims,
    right_dims,
    max_orders,
    left_shape,
    right_shape,
    singular_values,
    rank,
    svd_tol,
    rel_loss,
    grid_start,
    grid_stop,
    grid_step,
    ngrid,
):
    with open(outfile, "w") as f:
        f.write(f"quad_type {quad_type}\n")
        f.write("method svd_factorized_3x3\n")
        f.write("canonical_order pairtype_then_origslot\n")

        f.write("pair_types_original " + " ".join(pair_types_original) + "\n")
        f.write("pair_types_canonical " + " ".join(pair_types_canonical) + "\n")
        f.write("canon_to_param " + " ".join(str(x) for x in canon_to_param) + "\n")

        f.write("left_dims " + " ".join(str(x) for x in left_dims) + "\n")
        f.write("right_dims " + " ".join(str(x) for x in right_dims) + "\n")

        f.write("max_orders " + " ".join(str(x) for x in max_orders) + "\n")
        f.write("left_shape " + " ".join(str(x) for x in left_shape) + "\n")
        f.write("right_shape " + " ".join(str(x) for x in right_shape) + "\n")

        f.write(f"svd_tol {svd_tol:.16e}\n")
        f.write(f"rank {rank}\n")
        f.write(f"relative_frobenius_loss {rel_loss:.16e}\n")

        f.write(f"grid_start {grid_start:.16e}\n")
        f.write(f"grid_stop {grid_stop:.16e}\n")
        f.write(f"grid_step {grid_step:.16e}\n")
        f.write(f"ngrid {ngrid}\n")

        f.write("table_layout r0 r1 r2 values[rank] d0[rank] d1[rank] d2[rank]\n")
        f.write("singular_values\n")
        for sv in singular_values:
            f.write(f"{sv:.16e}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    quad_types = getattr(config, "QUADTYPES", [])
    if not quad_types:
        print("No QUADTYPES defined in config.py")
        return

    required_attrs = [
        "PARAM_FILE",
        "QUADSTART",
        "QUADSTOP",
        "QUADSTEP",
    ]

    for attr in required_attrs:
        if not hasattr(config, attr):
            raise RuntimeError(f"config.py is missing required attribute: {attr}")

    svd_tol = float(getattr(config, "SVD_TRUNC_TOL", 1.0e-8))
    nprocs = getattr(config, "NPROCS", None)
    chunk_size = int(getattr(config, "QUAD_CHUNK_SIZE", 200))

    fcut_type, fcut_var = parse_fcut_type(config.PARAM_FILE)

    print(f"Parameter file : {config.PARAM_FILE}")
    print(f"FCUT type      : {fcut_type}" + (f" {fcut_var}" if fcut_var is not None else ""))
    print(f"SVD tolerance  : {svd_tol:.3e}")
    print()

    for quad_idx, quad_type in enumerate(quad_types):
        print("=" * 72)
        print(f"Processing 4B quad type {quad_type}", flush=True)

        quad_info = parse_quad_pair_types(config.PARAM_FILE, quad_index=quad_type)
        if quad_info["excluded"]:
            print(f"Quad type {quad_type} is excluded; skipping.", flush=True)
            continue

        termdicts = parse_quad_terms_all(config.PARAM_FILE, quad_index=quad_type)
        if not termdicts:
            print(f"Quad type {quad_type} has no terms; skipping.", flush=True)
            continue

        raw_terms = [(t["powers"], t["coeff"]) for t in termdicts]
        raw_slot_params = build_quad_slot_params(config.PARAM_FILE, quad_type)

        pair_types_original = quad_info["pair_types"]
        canon_to_param, param_to_canon = canonical_slot_order(pair_types_original)

        pair_types_canonical = [pair_types_original[p] for p in canon_to_param]
        slot_params = [raw_slot_params[p] for p in canon_to_param]

        terms = reorder_terms_to_canonical(raw_terms, param_to_canon)
        max_orders = infer_max_orders(terms)

        forced_rank = get_forced_rank_for_quad(
            quad_type=quad_type,
            quad_idx=quad_idx,
            default_rank=None,
        )

        left_factor, right_factor, singular_values, rank, rel_loss, left_shape, right_shape = (
            svd_factorize_3x3(
                terms=terms,
                max_orders=max_orders,
                tol=svd_tol,
                forced_rank=forced_rank,
            )
        )

        left_dims = (0, 1, 3)
        right_dims = (2, 4, 5)

        grid_start = float(get_config_indexed_value(config.QUADSTART, quad_idx))
        grid_stop = float(get_config_indexed_value(config.QUADSTOP, quad_idx))
        grid_step = float(get_config_indexed_value(config.QUADSTEP, quad_idx))

        grid = np.arange(grid_start, grid_stop, grid_step, dtype=float)
        grid = np.append(grid, grid_stop)

        if grid.size == 0:
            raise RuntimeError(f"Empty grid for quad type {quad_type}")

        metafile = f"chimes_scan_4b_svd3x3.type_{quad_type}.meta"
        leftfile = f"chimes_scan_4b_svd3x3.type_{quad_type}.left.dat"
        rightfile = f"chimes_scan_4b_svd3x3.type_{quad_type}.right.dat"

        print("3|3 SVD summary:", flush=True)
        print(f"  pair_types original   = {pair_types_original}", flush=True)
        print(f"  pair_types canonical  = {pair_types_canonical}", flush=True)
        print(f"  canon_to_param        = {canon_to_param}", flush=True)
        print(f"  nterms                = {len(terms)}", flush=True)
        print(f"  max_orders            = {max_orders}", flush=True)
        print(f"  left_shape            = {left_shape}", flush=True)
        print(f"  right_shape           = {right_shape}", flush=True)
        print(f"  singular value count  = {len(singular_values)}", flush=True)
        print(f"  chosen rank R         = {rank}", flush=True)
        print(f"  relative loss         = {rel_loss:.6e}", flush=True)
        print(f"  grid points per dim   = {grid.size}", flush=True)
        print(f"  rows per table        = {grid.size ** 3}", flush=True)

        write_meta_file(
            outfile=metafile,
            quad_type=quad_type,
            pair_types_original=pair_types_original,
            pair_types_canonical=pair_types_canonical,
            canon_to_param=canon_to_param,
            left_dims=left_dims,
            right_dims=right_dims,
            max_orders=max_orders,
            left_shape=left_shape,
            right_shape=right_shape,
            singular_values=singular_values,
            rank=rank,
            svd_tol=svd_tol,
            rel_loss=rel_loss,
            grid_start=grid_start,
            grid_stop=grid_stop,
            grid_step=grid_step,
            ngrid=grid.size,
        )

        left_slot_params = [slot_params[d] for d in left_dims]
        right_slot_params = [slot_params[d] for d in right_dims]

        left_max_orders = [max_orders[d] for d in left_dims]
        right_max_orders = [max_orders[d] for d in right_dims]

        write_factor_table(
            outfile=leftfile,
            grid=grid,
            factor=left_factor,
            slot_params_local=left_slot_params,
            max_orders_local=left_max_orders,
            fcut_type=fcut_type,
            fcut_var=fcut_var,
            nprocs=nprocs,
            chunk_size=chunk_size,
        )

        write_factor_table(
            outfile=rightfile,
            grid=grid,
            factor=right_factor,
            slot_params_local=right_slot_params,
            max_orders_local=right_max_orders,
            fcut_type=fcut_type,
            fcut_var=fcut_var,
            nprocs=nprocs,
            chunk_size=chunk_size,
        )

        print(f"Wrote metadata : {metafile}", flush=True)
        print(f"Wrote left     : {leftfile}", flush=True)
        print(f"Wrote right    : {rightfile}", flush=True)

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()