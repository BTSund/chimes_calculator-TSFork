#!/usr/bin/env python3
"""
SVD4x2.py

Generate SVD-factorized 4|2 tabulated 4-body ChIMES tables.

Coefficient factorization:

    c[a,b,c,d,e,f] ~= sum_p L[a,b,c,d,p] R[e,f,p]

Runtime:

    E4 ~= sum_p X_p(r0,r1,r2,r3) Y_p(r4,r5)

The singular values are absorbed as sqrt(S) into both factors, so runtime
does not multiply by singular values.

Requires config.py with at least:
    PARAM_FILE
    QUADTYPES
    QUADSTART
    QUADSTOP
    QUADSTEP

Optional:
    SVD_TRUNC_TOL = 1e-12
    SVD_RANK = integer
    SVD_RANKS = {quad_type: rank}
    NPROCS
    QUAD_CHUNK_SIZE
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
    if isinstance(obj, (list, tuple, np.ndarray)):
        return obj[idx]
    return obj


def parse_fcut_type(param_file):
    lines = get_file_lines(param_file)
    fcut_type = "CUBIC"
    fcut_var = None

    for line in lines:
        items = split_clean(line)
        if len(items) >= 3 and items[0] == "FCUT" and items[1] == "TYPE:":
            fcut_type = items[2].upper()
            if fcut_type == "TERSOFF":
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

        if len(items) >= 3 and items[0] == "ATOM" and items[1] == "PAIRS:":
            no_pairs = int(items[2])

        if len(items) >= 3 and items[0] == "#" and items[1] == "PAIRIDX" and items[2] == "#":
            raw = lines[i]
            if "# USEOVRP #" in raw:
                i += 1
                continue

            if no_pairs is None:
                raise RuntimeError("Found pair table header before ATOM PAIRS count")

            for _ in range(no_pairs):
                i += 1
                items = split_clean(lines[i])

                if len(items) not in (7, 8):
                    raise RuntimeError(f"Bad pair metadata line: {lines[i].rstrip()}")

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
        raise RuntimeError("Failed to parse pair metadata")

    return pair_meta_by_index


def get_pair_meta_for_pair_name(pair_name, pair_meta_by_index):
    matches = []
    for _, meta in pair_meta_by_index.items():
        if meta["type1"] + meta["type2"] == pair_name:
            matches.append(meta)

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise RuntimeError(f"Could not resolve pair name {pair_name}")
    raise RuntimeError(f"Ambiguous pair name {pair_name}")


def parse_quad_pair_types(param_file, quad_index=0):
    lines = get_file_lines(param_file)

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if "QUADRUPLETYPE PARAMS:" in s:
            i += 1
            hdr = split_clean(lines[i])
            this_idx = int(hdr[1])

            i += 1
            pairline = split_clean(lines[i])

            if this_idx == quad_index:
                return {
                    "quad_index": this_idx,
                    "atom_types": hdr[3:7],
                    "pair_types": pairline[1:7],
                    "excluded": pairline[7] == "EXCLUDED:",
                    "ncoeff": None if pairline[7] == "EXCLUDED:" else int(pairline[10]),
                }

        i += 1

    raise RuntimeError(f"Could not find quad index {quad_index}")


def parse_quad_terms_all(param_file, quad_index=0):
    lines = get_file_lines(param_file)

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if "QUADRUPLETYPE PARAMS:" in s:
            i += 1
            hdr = split_clean(lines[i])
            this_idx = int(hdr[1])

            i += 1
            pairline = split_clean(lines[i])

            if this_idx == quad_index:
                if pairline[7] == "EXCLUDED:":
                    return []

                ncoeff = int(pairline[10])

                i += 1
                i += 1

                terms = []
                for _ in range(ncoeff):
                    i += 1
                    parts = split_clean(lines[i])
                    terms.append({
                        "powers": tuple(int(x) for x in parts[1:7]),
                        "coeff": float(parts[9]),
                    })

                return terms

        i += 1

    raise RuntimeError(f"No 4B terms found for quad index {quad_index}")


def parse_quadmaps(param_file):
    lines = get_file_lines(param_file)
    atom_typ_quad_map = []
    atom_idx_quad_map = []

    i = 0
    while i < len(lines):
        items = split_clean(lines[i])
        if items and items[0] == "QUADMAPS:":
            n = int(items[1])
            for _ in range(n):
                i += 1
                row = split_clean(lines[i])
                atom_idx_quad_map.append(int(row[0]))
                atom_typ_quad_map.append(row[1])
            break
        i += 1

    return atom_typ_quad_map, atom_idx_quad_map


def parse_special_4b_cutoffs(param_file, atom_typ_quad_map, atom_idx_quad_map):
    lines = get_file_lines(param_file)
    slow_to_quadidx = {name: idx for name, idx in zip(atom_typ_quad_map, atom_idx_quad_map)}

    special_min = {}
    special_max = {}

    i = 0
    while i < len(lines):
        items = split_clean(lines[i])
        if not items:
            i += 1
            continue

        if len(items) >= 4 and items[0] == "SPECIAL" and items[1] == "4B" and items[2] == "S_MAXIM:":
            if items[3] == "ALL":
                special_max["__ALL__"] = float(items[4])
            else:
                nentries = int(items[4])
                for _ in range(nentries):
                    i += 1
                    row = split_clean(lines[i])
                    quadidx = slow_to_quadidx[row[0]]
                    special_max[quadidx] = {
                        "pair_names": row[1:7],
                        "values": [float(x) for x in row[7:13]],
                    }

        if len(items) >= 4 and items[0] == "SPECIAL" and items[1] == "4B" and items[2] == "S_MINIM:":
            if items[3] == "ALL":
                special_min["__ALL__"] = float(items[4])
            else:
                nentries = int(items[4])
                for _ in range(nentries):
                    i += 1
                    row = split_clean(lines[i])
                    quadidx = slow_to_quadidx[row[0]]
                    special_min[quadidx] = {
                        "pair_names": row[1:7],
                        "values": [float(x) for x in row[7:13]],
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
            raise RuntimeError(f"Could not resolve special cutoff pair {name}")

    return resolved


def build_quad_slot_params(param_file, quad_type):
    pair_meta = parse_all_pair_metadata(param_file)
    quad_info = parse_quad_pair_types(param_file, quad_index=quad_type)

    if quad_info["excluded"]:
        return []

    atom_typ_quad_map, atom_idx_quad_map = parse_quadmaps(param_file)
    special_min, special_max = parse_special_4b_cutoffs(param_file, atom_typ_quad_map, atom_idx_quad_map)

    pair_types = quad_info["pair_types"]

    slot_params = []
    for pair_name in pair_types:
        meta = get_pair_meta_for_pair_name(pair_name, pair_meta)
        if meta["dist_type"].upper() != "MORSE":
            raise RuntimeError("Only MORSE mapping currently implemented")

        slot_params.append({
            "pair_name": pair_name,
            "pair_index": meta["pair_index"],
            "rmin": meta["s_min"],
            "rmax": meta["s_max"],
            "morse": meta["morse_lambda"],
        })

    if "__ALL__" in special_min:
        for sp in slot_params:
            sp["rmin"] = special_min["__ALL__"]

    if quad_type in special_min:
        vals = resolve_occurrence_mapped_values(
            pair_types,
            special_min[quad_type]["pair_names"],
            special_min[quad_type]["values"],
        )
        for i in range(6):
            slot_params[i]["rmin"] = vals[i]

    if "__ALL__" in special_max:
        for sp in slot_params:
            sp["rmax"] = special_max["__ALL__"]

    if quad_type in special_max:
        vals = resolve_occurrence_mapped_values(
            pair_types,
            special_max[quad_type]["pair_names"],
            special_max[quad_type]["values"],
        )
        for i in range(6):
            slot_params[i]["rmax"] = vals[i]

    return slot_params


# =============================================================================
# Canonical ordering
# =============================================================================

def canonical_slot_order(pair_types):
    pairs = [(pair_types[i], i) for i in range(6)]
    pairs.sort(key=lambda x: (x[0], x[1]))

    canon_to_param = [slot for _, slot in pairs]
    param_to_canon = [None] * 6

    for c, p in enumerate(canon_to_param):
        param_to_canon[p] = c

    return canon_to_param, param_to_canon


def reorder_terms_to_canonical(raw_terms, param_to_canon):
    out = []
    for powers, coeff in raw_terms:
        cpows = [0] * 6
        for pslot in range(6):
            cpows[param_to_canon[pslot]] = powers[pslot]
        out.append((tuple(cpows), coeff))
    return out


def infer_max_orders(terms):
    max_orders = [0] * 6
    for powers, _ in terms:
        for i in range(6):
            max_orders[i] = max(max_orders[i], powers[i])
    return max_orders


# =============================================================================
# Basis functions
# =============================================================================

def morse_transform_data(r, rmin, rmax, morse):
    x_min = math.exp(-rmin / morse)
    x_max = math.exp(-rmax / morse)
    x_avg = 0.5 * (x_max + x_min)
    x_diff = -0.5 * (x_max - x_min)

    exprlen = math.exp(-r / morse)
    x = (exprlen - x_avg) / x_diff
    dx_dr = (-exprlen / morse) / x_diff
    return x, dx_dr


def cheb_value_deriv_wrt_r(r, rmin, rmax, morse, order):
    r_eval = rmin if r < rmin else r
    x, dx_dr = morse_transform_data(r_eval, rmin, rmax, morse)

    Tn = np.zeros(order + 1)
    U = np.zeros(order + 1)
    Tnd = np.zeros(order + 1)

    Tn[0] = 1.0
    U[0] = 1.0

    if order >= 1:
        Tn[1] = x
        U[1] = 2.0 * x

    for i in range(2, order + 1):
        Tn[i] = 2.0 * x * Tn[i - 1] - Tn[i - 2]
        U[i] = 2.0 * x * U[i - 1] - U[i - 2]

    Tnd[0] = 0.0
    for i in range(1, order + 1):
        Tnd[i] = i * dx_dr * U[i - 1]

    return Tn, Tnd


def get_fcut(r, outer_cutoff, fcut_type, fcut_var=None):
    if fcut_type == "CUBIC":
        f0 = 1.0 - r / outer_cutoff
        return f0**3, -3.0 * f0**2 / outer_cutoff

    if fcut_type == "TERSOFF":
        thresh = outer_cutoff - fcut_var * outer_cutoff
        if r < thresh:
            return 1.0, 0.0
        if r > outer_cutoff:
            return 0.0, 0.0

        arg = (r - thresh) / (outer_cutoff - thresh) * math.pi + math.pi / 2.0
        arg_deriv = math.pi / (outer_cutoff - thresh)
        return 0.5 + 0.5 * math.sin(arg), 0.5 * math.cos(arg) * arg_deriv

    raise RuntimeError(f"Unknown cutoff {fcut_type}")


def build_GT_table_for_slot(grid, slot_param, max_order, fcut_type, fcut_var):
    G = np.zeros((len(grid), max_order + 1))
    D = np.zeros((len(grid), max_order + 1))

    for i, r in enumerate(grid):
        Tn, Tnd = cheb_value_deriv_wrt_r(
            r,
            slot_param["rmin"],
            slot_param["rmax"],
            slot_param["morse"],
            max_order,
        )
        fc, fcd = get_fcut(r, slot_param["rmax"], fcut_type, fcut_var)
        G[i, :] = fc * Tn
        D[i, :] = fcd * Tn + fc * Tnd

    return G, D


# =============================================================================
# SVD factorization
# =============================================================================

def flat_index_variable(indices, dims):
    idx = 0
    for x, dim in zip(indices, dims):
        idx = idx * dim + x
    return idx


def build_4x2_matrix(terms, max_orders, left_dims=(0,1,2,3), right_dims=(4,5)):
    left_shape = tuple(max_orders[d] + 1 for d in left_dims)
    right_shape = tuple(max_orders[d] + 1 for d in right_dims)

    M = np.zeros((int(np.prod(left_shape)), int(np.prod(right_shape))))

    for powers, coeff in terms:
        li = flat_index_variable(tuple(powers[d] for d in left_dims), left_shape)
        ri = flat_index_variable(tuple(powers[d] for d in right_dims), right_shape)
        M[li, ri] += coeff

    return M, left_shape, right_shape


def choose_svd_rank(S, tol):
    s2 = S*S
    total = np.sum(s2)
    if total == 0:
        return 0, 0.0

    capture = np.cumsum(s2) / total
    loss = np.sqrt(np.maximum(0.0, 1.0 - capture))
    idx = np.where(loss <= tol)[0]
    R = int(idx[0] + 1) if len(idx) else len(S)
    return R, float(loss[R-1])


def get_forced_rank_for_quad(quad_type, quad_idx):
    if hasattr(config, "SVD_RANK"):
        return int(config.SVD_RANK)

    if hasattr(config, "SVD_RANKS"):
        ranks = config.SVD_RANKS
        if isinstance(ranks, dict):
            return int(ranks[quad_type])
        if isinstance(ranks, (list, tuple, np.ndarray)):
            return int(ranks[quad_idx])
        return int(ranks)

    return None


def svd_factorize_4x2(terms, max_orders, tol, forced_rank=None):
    M, left_shape, right_shape = build_4x2_matrix(terms, max_orders)

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    auto_R, rel_loss = choose_svd_rank(S, tol)
    R = auto_R if forced_rank is None else int(forced_rank)

    if R > len(S):
        raise RuntimeError(f"Requested rank {R}, but max rank is {len(S)}")

    if R == 0:
        left_factor = np.zeros(left_shape + (0,))
        right_factor = np.zeros(right_shape + (0,))
        return left_factor, right_factor, S, R, rel_loss, left_shape, right_shape

    if forced_rank is not None:
        s2 = S*S
        cap = np.cumsum(s2)/np.sum(s2)
        rel_loss = float(np.sqrt(max(0.0, 1.0 - cap[R-1])))

    sqrtS = np.sqrt(S[:R])

    left_mat = U[:, :R] * sqrtS[None, :]
    right_mat = Vh[:R, :].T * sqrtS[None, :]

    left_factor = left_mat.reshape(left_shape + (R,))
    right_factor = right_mat.reshape(right_shape + (R,))

    return left_factor, right_factor, S, R, rel_loss, left_shape, right_shape


# =============================================================================
# Table writing
# =============================================================================

def iter_grid_index_chunks(ngrid, ndim, chunk_size):
    iterator = itertools.product(range(ngrid), repeat=ndim)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def compute_factor_chunk(args):
    idx_chunk, grid, factor, G_arrays, D_arrays = args

    ndim = len(G_arrays)
    R = factor.shape[-1]
    out_lines = []

    for pt in idx_chunk:
        gs = [G_arrays[d][pt[d]] for d in range(ndim)]
        ds = [D_arrays[d][pt[d]] for d in range(ndim)]

        if ndim == 4:
            vals = np.einsum("abcdp,a,b,c,d->p", factor, gs[0], gs[1], gs[2], gs[3], optimize=True)
            der0 = np.einsum("abcdp,a,b,c,d->p", factor, ds[0], gs[1], gs[2], gs[3], optimize=True)
            der1 = np.einsum("abcdp,a,b,c,d->p", factor, gs[0], ds[1], gs[2], gs[3], optimize=True)
            der2 = np.einsum("abcdp,a,b,c,d->p", factor, gs[0], gs[1], ds[2], gs[3], optimize=True)
            der3 = np.einsum("abcdp,a,b,c,d->p", factor, gs[0], gs[1], gs[2], ds[3], optimize=True)
            ders = [der0, der1, der2, der3]
        elif ndim == 2:
            vals = np.einsum("abp,a,b->p", factor, gs[0], gs[1], optimize=True)
            der0 = np.einsum("abp,a,b->p", factor, ds[0], gs[1], optimize=True)
            der1 = np.einsum("abp,a,b->p", factor, gs[0], ds[1], optimize=True)
            ders = [der0, der1]
        else:
            raise RuntimeError("Only ndim=4 or ndim=2 supported")

        parts = [f"{grid[i]:.12f}" for i in pt]
        parts.extend(f"{x:.16e}" for x in vals)
        for der in ders:
            parts.extend(f"{x:.16e}" for x in der)

        out_lines.append(" ".join(parts) + "\n")

    return out_lines


def write_factor_table(outfile, grid, factor, slot_params, max_orders, fcut_type, fcut_var,
                       nprocs=None, chunk_size=200):
    ndim = len(slot_params)
    ngrid = len(grid)
    nrows = ngrid ** ndim
    R = factor.shape[-1]

    G_arrays = []
    D_arrays = []
    for d in range(ndim):
        G, D = build_GT_table_for_slot(grid, slot_params[d], max_orders[d], fcut_type, fcut_var)
        G_arrays.append(G)
        D_arrays.append(D)

    print(f"Writing {outfile}")
    print(f"  ndim={ndim}, rows={nrows}, rank={R}, values/row={ndim + (ndim+1)*R}")

    with open(outfile, "w") as f:
        f.write(f"{nrows}\n")
        chunks = iter_grid_index_chunks(ngrid, ndim, chunk_size)

        if nprocs is None or nprocs <= 1:
            row = 0
            for chunk in chunks:
                lines = compute_factor_chunk((chunk, grid, factor, G_arrays, D_arrays))
                f.writelines(lines)
                row += len(lines)
                if row % max(1000, 10*chunk_size) == 0 or row == nrows:
                    print(f"  wrote {row}/{nrows}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=nprocs) as ex:
                args_iter = ((chunk, grid, factor, G_arrays, D_arrays) for chunk in chunks)
                row = 0
                for lines in ex.map(compute_factor_chunk, args_iter, chunksize=1):
                    f.writelines(lines)
                    row += len(lines)
                    if row % max(1000, 10*chunk_size) == 0 or row == nrows:
                        print(f"  wrote {row}/{nrows}", flush=True)


def write_meta(outfile, quad_type, pair_types_original, pair_types_canonical, canon_to_param,
               max_orders, left_shape, right_shape, singular_values, rank, tol, rel_loss,
               grid_start, grid_stop, grid_step, ngrid):
    with open(outfile, "w") as f:
        f.write(f"quad_type {quad_type}\n")
        f.write("method svd_factorized_4x2\n")
        f.write("canonical_order pairtype_then_origslot\n")
        f.write("pair_types_original " + " ".join(pair_types_original) + "\n")
        f.write("pair_types_canonical " + " ".join(pair_types_canonical) + "\n")
        f.write("canon_to_param " + " ".join(str(x) for x in canon_to_param) + "\n")
        f.write("left_dims 0 1 2 3\n")
        f.write("right_dims 4 5\n")
        f.write("max_orders " + " ".join(str(x) for x in max_orders) + "\n")
        f.write("left_shape " + " ".join(str(x) for x in left_shape) + "\n")
        f.write("right_shape " + " ".join(str(x) for x in right_shape) + "\n")
        f.write(f"svd_tol {tol:.16e}\n")
        f.write(f"rank {rank}\n")
        f.write(f"relative_frobenius_loss {rel_loss:.16e}\n")
        f.write(f"grid_start {grid_start:.16e}\n")
        f.write(f"grid_stop {grid_stop:.16e}\n")
        f.write(f"grid_step {grid_step:.16e}\n")
        f.write(f"ngrid {ngrid}\n")
        f.write("left_table_layout r0 r1 r2 r3 values[rank] d0[rank] d1[rank] d2[rank] d3[rank]\n")
        f.write("right_table_layout r4 r5 values[rank] d0[rank] d1[rank]\n")
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

    svd_tol = float(getattr(config, "SVD_TRUNC_TOL", 1e-12))
    nprocs = getattr(config, "NPROCS", None)
    chunk_size = int(getattr(config, "QUAD_CHUNK_SIZE", 200))

    fcut_type, fcut_var = parse_fcut_type(config.PARAM_FILE)

    print(f"Parameter file: {config.PARAM_FILE}")
    print(f"SVD tolerance : {svd_tol}")
    print(f"FCUT          : {fcut_type}" + (f" {fcut_var}" if fcut_var is not None else ""))

    for qidx, quad_type in enumerate(quad_types):
        print("=" * 80)
        print(f"Processing quad type {quad_type}")

        quad_info = parse_quad_pair_types(config.PARAM_FILE, quad_type)
        if quad_info["excluded"]:
            print("Excluded; skipping")
            continue

        raw_terms_dict = parse_quad_terms_all(config.PARAM_FILE, quad_type)
        raw_terms = [(t["powers"], t["coeff"]) for t in raw_terms_dict]

        raw_slot_params = build_quad_slot_params(config.PARAM_FILE, quad_type)

        pair_types = quad_info["pair_types"]
        canon_to_param, param_to_canon = canonical_slot_order(pair_types)
        pair_types_canon = [pair_types[p] for p in canon_to_param]

        terms = reorder_terms_to_canonical(raw_terms, param_to_canon)
        slot_params = [raw_slot_params[p] for p in canon_to_param]
        max_orders = infer_max_orders(terms)

        forced_rank = get_forced_rank_for_quad(quad_type, qidx)

        left_factor, right_factor, S, R, rel_loss, left_shape, right_shape = svd_factorize_4x2(
            terms, max_orders, svd_tol, forced_rank=forced_rank
        )

        grid_start = float(get_config_indexed_value(config.QUADSTART, qidx))
        grid_stop  = float(get_config_indexed_value(config.QUADSTOP, qidx))
        grid_step  = float(get_config_indexed_value(config.QUADSTEP, qidx))

        grid = np.arange(grid_start, grid_stop, grid_step, dtype=float)
        grid = np.append(grid, grid_stop)

        metafile = f"chimes_scan_4b_svd4x2.type_{quad_type}.meta"
        leftfile = f"chimes_scan_4b_svd4x2.type_{quad_type}.left.dat"
        rightfile = f"chimes_scan_4b_svd4x2.type_{quad_type}.right.dat"

        print("4|2 SVD summary:")
        print(f"  pair_types original  = {pair_types}")
        print(f"  pair_types canonical = {pair_types_canon}")
        print(f"  canon_to_param       = {canon_to_param}")
        print(f"  max_orders           = {max_orders}")
        print(f"  left_shape           = {left_shape}")
        print(f"  right_shape          = {right_shape}")
        print(f"  rank R               = {R}")
        print(f"  relative loss        = {rel_loss:.6e}")
        print(f"  grid points per dim  = {grid.size}")
        print(f"  left rows            = {grid.size**4}")
        print(f"  right rows           = {grid.size**2}")

        write_meta(
            metafile,
            quad_type,
            pair_types,
            pair_types_canon,
            canon_to_param,
            max_orders,
            left_shape,
            right_shape,
            S,
            R,
            svd_tol,
            rel_loss,
            grid_start,
            grid_stop,
            grid_step,
            grid.size,
        )

        write_factor_table(
            leftfile,
            grid,
            left_factor,
            [slot_params[i] for i in (0,1,2,3)],
            [max_orders[i] for i in (0,1,2,3)],
            fcut_type,
            fcut_var,
            nprocs=nprocs,
            chunk_size=chunk_size,
        )

        write_factor_table(
            rightfile,
            grid,
            right_factor,
            [slot_params[i] for i in (4,5)],
            [max_orders[i] for i in (4,5)],
            fcut_type,
            fcut_var,
            nprocs=nprocs,
            chunk_size=chunk_size,
        )

        print(f"Wrote metadata: {metafile}")
        print(f"Wrote left    : {leftfile}")
        print(f"Wrote right   : {rightfile}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()