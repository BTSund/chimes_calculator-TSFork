#!/usr/bin/env python3
"""
Tab_CP.py

Generate CP-factorized 1D tabulated 4-body ChIMES tables using worker-level
parallelization per (quad_type, rank, init) combination with early dynamic pruning
and real-time file generation.
"""

import os
import sys
import math
import itertools
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

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

            if this_idx == quad_index:
                if pairline[7] == "EXCLUDED:":
                    return []

                ncoeff = int(pairline[10])
                i += 2  # Skip headers

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
            raise RuntimeError(f"Could not map special 4B pair name '{name}' into {pair_types_reference}")

    return resolved


def build_quad_slot_params(param_file, quad_type):
    pair_meta_by_index = parse_all_pair_metadata(param_file)
    quad_info = parse_quad_pair_types(param_file, quad_index=quad_type)

    if quad_info["excluded"]:
        return []

    atom_typ_quad_map, atom_idx_quad_map = parse_quadmaps(param_file)
    special_min, special_max = parse_special_4b_cutoffs(param_file, atom_typ_quad_map, atom_idx_quad_map)
    pair_types = quad_info["pair_types"]

    slot_params = []
    for pair_name in pair_types:
        meta = get_pair_meta_for_pair_name(pair_name, pair_meta_by_index)
        if meta["dist_type"].upper() != "MORSE":
            raise RuntimeError(f"Only MORSE mapping implemented, found {meta['dist_type']}")

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
        for sp in slot_params:
            sp["rmin"] = special_min["__ALL__"]

    if quad_type in special_min:
        resolved = resolve_occurrence_mapped_values(
            pair_types, special_min[quad_type]["pair_names"], special_min[quad_type]["values"]
        )
        for i in range(6):
            slot_params[i]["rmin"] = resolved[i]

    if "__ALL__" in special_max:
        for sp in slot_params:
            sp["rmax"] = special_max["__ALL__"]

    if quad_type in special_max:
        resolved = resolve_occurrence_mapped_values(
            pair_types, special_max[quad_type]["pair_names"], special_max[quad_type]["values"]
        )
        for i in range(6):
            slot_params[i]["rmax"] = resolved[i]

    return slot_params


# =============================================================================
# Canonical ordering & Tensor Math
# =============================================================================

def canonical_slot_order(pair_types):
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


def build_dense_tensor(terms, dims):
    X = np.zeros(dims, dtype=float)
    for powers, coeff in terms:
        X[powers] += coeff
    return X


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
    U = np.zeros(order + 1, dtype=float)
    Tnd = np.zeros(order + 1, dtype=float)

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
    rmin, rmax, morse = slot_param["rmin"], slot_param["rmax"], slot_param["morse"]

    for i, r in enumerate(grid):
        Tn, Tnd = cheb_value_deriv_wrt_r(r, rmin, rmax, morse, max_order)
        fcut, fcutderiv = get_fcut(r, rmax, fcut_type, fcut_var)
        G[i, :] = fcut * Tn
        D[i, :] = fcutderiv * Tn + fcut * Tnd

    return G, D


# =============================================================================
# CP-ALS implementation
# =============================================================================

def unfold_tensor(X, mode):
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))


def khatri_rao_for_mode(factors, skip_mode):
    modes = [m for m in range(len(factors)) if m != skip_mode]
    KR = factors[modes[0]]
    for m in modes[1:]:
        A = KR
        B = factors[m]
        KR = np.einsum("ir,jr->ijr", A, B, optimize=True).reshape(
            A.shape[0] * B.shape[0], A.shape[1]
        )
    return KR


def normalize_cp_factors_store_scale_in_first(factors, eps=1.0e-300):
    nmodes = len(factors)
    R = factors[0].shape[1]
    weights = np.ones(R, dtype=float)

    for m in range(nmodes):
        norms = np.linalg.norm(factors[m], axis=0)
        norms = np.maximum(norms, eps)
        factors[m] = factors[m] / norms[None, :]
        weights *= norms

    factors[0] = factors[0] * weights[None, :]
    return factors


def cp_to_tensor(factors, shape):
    R = factors[0].shape[1]
    Xhat = np.zeros(shape, dtype=float)
    for q in range(R):
        Xhat += np.einsum(
            "a,b,c,d,e,f->abcdef",
            factors[0][:, q], factors[1][:, q], factors[2][:, q],
            factors[3][:, q], factors[4][:, q], factors[5][:, q],
            optimize=True,
        )
    return Xhat


def cp_relative_error(X, factors, normX=None):
    if normX is None:
        normX = np.linalg.norm(X)
    if normX == 0.0:
        return 0.0
    Xhat = cp_to_tensor(factors, X.shape)
    return float(np.linalg.norm(X - Xhat) / normX)


def cp_als(X, rank, n_iter=500, tol=1.0e-8, ridge=1.0e-10, seed=12345, verbose=False,
           target_rel_error=None, quad_type=None, shared_state=None):
    rng = np.random.default_rng(seed)
    shape = X.shape
    nmodes = len(shape)

    factors = []
    for n in range(nmodes):
        A = rng.normal(size=(shape[n], rank))
        A /= np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1.0e-300)
        factors.append(A)

    factors = normalize_cp_factors_store_scale_in_first(factors)
    normX = np.linalg.norm(X)
    prev_err = None
    errors = []
    aborted = False

    for it in range(n_iter):
        # Active cancellation check: terminate if equal or lower rank solved this quad type
        if shared_state is not None and quad_type is not None:
            solved_r = shared_state.get(f"solved_rank_{quad_type}", None)
            if solved_r is not None:
                if rank > solved_r or (rank == solved_r and shared_state.get(f"solved_done_{quad_type}_{rank}", False)):
                    aborted = True
                    break

        for mode in range(nmodes):
            Xn = unfold_tensor(X, mode)
            KR = khatri_rao_for_mode(factors, mode)
            V = np.ones((rank, rank), dtype=float)
            for m in range(nmodes):
                if m == mode:
                    continue
                V *= factors[m].T @ factors[m]
            if ridge > 0.0:
                V = V + ridge * np.eye(rank)

            MTTKRP = Xn @ KR
            try:
                Anew_T = np.linalg.solve(V.T, MTTKRP.T)
                Anew = Anew_T.T
            except np.linalg.LinAlgError:
                Anew = MTTKRP @ np.linalg.pinv(V)

            factors[mode] = Anew

        factors = normalize_cp_factors_store_scale_in_first(factors)
        err = cp_relative_error(X, factors, normX=normX)
        errors.append(err)

        if verbose:
            print(f"      iter {it+1:5d}: rel_err = {err:.12e}", flush=True)

        # Early stop criteria: stop iterations instantly if hitting goal error
        if target_rel_error is not None and err <= target_rel_error:
            if shared_state is not None and quad_type is not None:
                curr_solved = shared_state.get(f"solved_rank_{quad_type}", float("inf"))
                if rank <= curr_solved:
                    shared_state[f"solved_rank_{quad_type}"] = rank
                    shared_state[f"solved_done_{quad_type}_{rank}"] = True
            break

        if prev_err is not None and abs(prev_err - err) < tol * max(1.0, prev_err):
            break
        prev_err = err

    return {
        "factors": factors,
        "rel_error": errors[-1] if errors else None,
        "errors": errors,
        "n_iter_done": len(errors),
        "aborted": aborted,
    }


# =============================================================================
# Worker Function for Single Task (quad_type, rank, init)
# =============================================================================

def process_single_task(task):
    """
    Worker function handling exactly one (quad_type, rank, init_idx) combination.
    """
    quad_idx = task["quad_idx"]
    quad_type = task["quad_type"]
    rank = task["rank"]
    init_idx = task["init_idx"]

    cp_n_iter = task["cp_n_iter"]
    cp_tol = task["cp_tol"]
    cp_ridge = task["cp_ridge"]
    cp_seed = task["cp_seed"]
    cp_verbose = task["cp_verbose"]
    target_rel_error = task.get("target_rel_error", None)
    shared_state = task.get("shared_state", None)

    # Check before starting work if this rank/init is already superseded by a target-meeting rank
    if shared_state is not None:
        solved_r = shared_state.get(f"solved_rank_{quad_type}", None)
        if solved_r is not None:
            if rank > solved_r or (rank == solved_r and shared_state.get(f"solved_done_{quad_type}_{rank}", False)):
                return {
                    "quad_idx": quad_idx, "quad_type": quad_type, "rank": rank, "init_idx": init_idx,
                    "rel_error": None, "n_iter_done": 0, "status": "pruned",
                    "message": "Pruned prior to execution", "factors": None
                }

    init_seed = cp_seed + 1000003 * init_idx + 9176 * rank + 10000019 * int(quad_type)

    try:
        quad_info = parse_quad_pair_types(config.PARAM_FILE, quad_index=quad_type)

        if quad_info["excluded"]:
            return {
                "quad_idx": quad_idx, "quad_type": quad_type, "rank": rank, "init_idx": init_idx,
                "rel_error": None, "n_iter_done": 0, "status": "skipped", "message": "Excluded", "factors": None
            }

        termdicts = parse_quad_terms_all(config.PARAM_FILE, quad_index=quad_type)
        if not termdicts:
            return {
                "quad_idx": quad_idx, "quad_type": quad_type, "rank": rank, "init_idx": init_idx,
                "rel_error": None, "n_iter_done": 0, "status": "skipped", "message": "No terms", "factors": None
            }

        raw_terms = [(t["powers"], t["coeff"]) for t in termdicts]
        pair_types_original = quad_info["pair_types"]
        canon_to_param, param_to_canon = canonical_slot_order(pair_types_original)

        terms = reorder_terms_to_canonical(raw_terms, param_to_canon)
        max_orders = infer_max_orders(terms)
        tensor_dims = tuple(m + 1 for m in max_orders)
        X = build_dense_tensor(terms, tensor_dims)

        res = cp_als(
            X, rank=rank, n_iter=cp_n_iter, tol=cp_tol, ridge=cp_ridge, seed=init_seed,
            verbose=cp_verbose, target_rel_error=target_rel_error, quad_type=quad_type,
            shared_state=shared_state
        )

        if res.get("aborted"):
            return {
                "quad_idx": quad_idx, "quad_type": quad_type, "rank": rank, "init_idx": init_idx,
                "rel_error": res["rel_error"], "n_iter_done": res["n_iter_done"],
                "status": "pruned", "message": "Aborted during execution due to target met elsewhere",
                "factors": None
            }

        rel_err = res["rel_error"]

        if target_rel_error is not None and rel_err is not None and rel_err <= target_rel_error:
            if shared_state is not None:
                curr_min = shared_state.get(f"solved_rank_{quad_type}", float("inf"))
                if rank <= curr_min:
                    shared_state[f"solved_rank_{quad_type}"] = rank
                    shared_state[f"solved_done_{quad_type}_{rank}"] = True

        return {
            "quad_idx": quad_idx,
            "quad_type": quad_type,
            "rank": rank,
            "init_idx": init_idx,
            "rel_error": rel_err,
            "n_iter_done": res["n_iter_done"],
            "status": "ok",
            "message": "",
            "factors": res["factors"],
            "tensor_dims": tensor_dims,
            "max_orders": max_orders,
            "canon_to_param": canon_to_param,
            "pair_types_original": pair_types_original,
            "atom_types": quad_info["atom_types"],
        }

    except Exception as exc:
        return {
            "quad_idx": quad_idx, "quad_type": quad_type, "rank": rank, "init_idx": init_idx,
            "rel_error": None, "n_iter_done": 0, "status": "error", "message": repr(exc), "factors": None
        }


# =============================================================================
# Rank configuration and summary helpers
# =============================================================================

def get_config_value_for_quad(attr_name, quad_type, quad_idx, default=None):
    if not hasattr(config, attr_name):
        return default
    obj = getattr(config, attr_name)
    if isinstance(obj, dict):
        return obj[quad_type]
    if isinstance(obj, (list, tuple, np.ndarray)):
        return obj[quad_idx]
    return obj


def get_task_quad_settings(quad_type, quad_idx):
    target_rel_error = float(get_config_value_for_quad("CP_TARGET_REL_ERROR", quad_type, quad_idx, default=1.0e-4))
    
    if hasattr(config, "CP_RANK_SCAN"):
        ranks = [int(r) for r in config.CP_RANK_SCAN]
    elif hasattr(config, "CP_RANK_SCANS"):
        scans = config.CP_RANK_SCANS
        ranks = [int(r) for r in (scans[quad_type] if isinstance(scans, dict) else scans[quad_idx])]
    else:
        rank_start = int(get_config_value_for_quad("CP_RANK_START", quad_type, quad_idx, default=10))
        rank_step = int(get_config_value_for_quad("CP_RANK_STEP", quad_type, quad_idx, default=10))
        rank_max = int(get_config_value_for_quad("CP_RANK_MAX", quad_type, quad_idx, default=100))
        ranks = list(range(rank_start, rank_max + 1, rank_step))

    return {"target_rel_error": target_rel_error, "ranks": sorted(ranks)}


def write_cp_slot_table(outfile, grid, factor, slot_param, max_order, fcut_type, fcut_var):
    Q = factor.shape[1]
    G, D = build_GT_table_for_slot(grid, slot_param, max_order, fcut_type, fcut_var)
    with open(outfile, "w") as f:
        f.write(f"{len(grid)}\n")
        for i, r in enumerate(grid):
            vals = G[i, :] @ factor
            ders = D[i, :] @ factor
            parts = [f"{r:.17f}"] + [f"{x:.17e}" for x in vals] + [f"{x:.17e}" for x in ders]
            f.write(" ".join(parts) + "\n")


def write_meta_file(outfile, task_res, grid_start, grid_stop, grid_step, ngrid, slot_files, cp_n_init, cp_tol, cp_ridge, cp_seed):
    with open(outfile, "w") as f:
        f.write(f"quad_type {task_res['quad_type']}\n")
        f.write("method cp_factorized_1d\n")
        f.write("atom_types " + " ".join(task_res["atom_types"]) + "\n")
        f.write("pair_types_original " + " ".join(task_res["pair_types_original"]) + "\n")
        f.write("canon_to_param " + " ".join(str(x) for x in task_res["canon_to_param"]) + "\n")
        f.write("max_orders " + " ".join(str(x) for x in task_res["max_orders"]) + "\n")
        f.write("tensor_dims " + " ".join(str(x) for x in task_res["tensor_dims"]) + "\n")
        f.write(f"rank {task_res['rank']}\n")
        f.write(f"coefficient_relative_frobenius_error {task_res['rel_error']:.16e}\n")
        f.write(f"cp_n_iter_done {task_res['n_iter_done']}\n")
        f.write(f"cp_n_init {cp_n_init}\n")
        f.write(f"grid_start {grid_start:.16e}\n")
        f.write(f"grid_stop {grid_stop:.16e}\n")
        f.write(f"grid_step {grid_step:.16e}\n")
        f.write(f"ngrid {ngrid}\n")
        f.write("slot_files\n")
        for s, fn in enumerate(slot_files):
            f.write(f"{s} {fn}\n")


def generate_task_files(task_res, fcut_type, fcut_var, cp_n_init, cp_tol, cp_ridge, cp_seed):
    q_type = task_res["quad_type"]
    q_idx = task_res["quad_idx"]
    rank = task_res["rank"]
    factors = task_res["factors"]

    grid_start = float(get_config_indexed_value(config.QUADSTART, q_idx))
    grid_stop = float(get_config_indexed_value(config.QUADSTOP, q_idx))
    grid_step = float(get_config_indexed_value(config.QUADSTEP, q_idx))
    grid = np.arange(grid_start, grid_stop + 0.5 * grid_step, grid_step, dtype=float)

    raw_slot_params = build_quad_slot_params(config.PARAM_FILE, q_type)
    slot_params = [raw_slot_params[p] for p in task_res["canon_to_param"]]

    prefix = f"chimes_scan_4b_cp.type_{q_type}.rank_{rank}"
    slot_files = []

    for s in range(6):
        slot_file = f"{prefix}.slot{s}.dat"
        slot_files.append(slot_file)
        write_cp_slot_table(
            slot_file, grid, factors[s], slot_params[s],
            task_res["max_orders"][s], fcut_type, fcut_var
        )

    meta_file = f"{prefix}.meta"
    write_meta_file(
        meta_file, task_res, grid_start, grid_stop, grid_step, len(grid),
        slot_files, cp_n_init, cp_tol, cp_ridge, cp_seed
    )
    return prefix


# =============================================================================
# Main Parallel Controller
# =============================================================================

def main():
    quad_types = getattr(config, "QUADTYPES", [])
    if not quad_types:
        print("No QUADTYPES defined in config.py")
        return

    cp_n_iter = int(getattr(config, "CP_N_ITER", 500))
    cp_n_init = int(getattr(config, "CP_N_INIT", 3))
    cp_tol = float(getattr(config, "CP_TOL", 1.0e-8))
    cp_ridge = float(getattr(config, "CP_RIDGE", 1.0e-10))
    cp_seed = int(getattr(config, "CP_SEED", 12345))
    cp_verbose = bool(getattr(config, "CP_VERBOSE", False))
    cp_n_workers = int(getattr(config, "CP_N_WORKERS", 4))

    fcut_type, fcut_var = parse_fcut_type(config.PARAM_FILE)

    quad_settings = {}
    max_rank_steps = 0
    for q_idx, q_type in enumerate(quad_types):
        settings = get_task_quad_settings(q_type, q_idx)
        quad_settings[q_type] = settings
        max_rank_steps = max(max_rank_steps, len(settings["ranks"]))

    # Inter-process shared state dictionary for active termination
    manager = mp.Manager()
    shared_state = manager.dict()

    # Priority Task Queue Generation: Lowest Rank -> Inits -> Quad Types
    queued_tasks = []
    for step_idx in range(max_rank_steps):
        for init_idx in range(cp_n_init):
            for q_idx, q_type in enumerate(quad_types):
                ranks = quad_settings[q_type]["ranks"]
                if step_idx < len(ranks):
                    queued_tasks.append({
                        "quad_idx": q_idx,
                        "quad_type": q_type,
                        "rank": ranks[step_idx],
                        "init_idx": init_idx,
                        "cp_n_iter": cp_n_iter,
                        "cp_tol": cp_tol,
                        "cp_ridge": cp_ridge,
                        "cp_seed": cp_seed,
                        "cp_verbose": cp_verbose,
                        "target_rel_error": quad_settings[q_type]["target_rel_error"],
                        "shared_state": shared_state,
                    })

    print(f"Total tasks generated: {len(queued_tasks)} across {cp_n_workers} worker processes.")

    # Tracking states
    solved_threshold_rank = {}  # quad_type -> min solved rank meeting target
    best_error_per_quad_rank = {}  # (quad_type, rank) -> best rel_error achieved so far
    completed_results = []

    with ProcessPoolExecutor(max_workers=cp_n_workers) as executor:
        future_to_task = {}

        # Fill executor with queued tasks
        for task in queued_tasks:
            fut = executor.submit(process_single_task, task)
            future_to_task[fut] = task

        print("All tasks submitted to priority queue. Processing...", flush=True)

        for future in as_completed(future_to_task):
            task_info = future_to_task[future]
            q_type = task_info["quad_type"]
            rank = task_info["rank"]
            target_err = quad_settings[q_type]["target_rel_error"]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "quad_idx": task_info["quad_idx"],
                    "quad_type": q_type,
                    "rank": rank,
                    "init_idx": task_info["init_idx"],
                    "rel_error": None,
                    "status": "error",
                    "message": repr(exc),
                }

            completed_results.append(result)

            if result.get("status") == "ok" and result.get("rel_error") is not None:
                rel_err = result["rel_error"]

                print(
                    f"[Finished] Quad={q_type:2d} | Rank={rank:3d} | Init={result['init_idx']} | "
                    f"RelErr={rel_err:.6e} | Target={target_err:.3e}", flush=True
                )

                # Requirement: Always make resulting files as worker finishes, replacing if rel error is lower
                key = (q_type, rank)
                prev_best = best_error_per_quad_rank.get(key, float("inf"))

                if rel_err < prev_best:
                    best_error_per_quad_rank[key] = rel_err
                    prefix = generate_task_files(
                        result, fcut_type, fcut_var, cp_n_init, cp_tol, cp_ridge, cp_seed
                    )
                    print(
                        f"  ==> Produced/Updated files for Quad={q_type}, Rank={rank} "
                        f"with RelErr={rel_err:.6e} ({prefix})", flush=True
                    )

                # Check if target met
                if rel_err <= target_err:
                    prev_solved_rank = solved_threshold_rank.get(q_type, float("inf"))

                    if rank < prev_solved_rank:
                        solved_threshold_rank[q_type] = rank

                        print(
                            f"  ==> TARGET REACHED for Quad={q_type} at Rank={rank} "
                            f"(Err={rel_err:.6e} <= {target_err:.3e}). Actively terminating equal/higher rank tasks!",
                            flush=True
                        )

                        # Update shared dict so active running tasks abort instantly
                        curr_min = shared_state.get(f"solved_rank_{q_type}", float("inf"))
                        if rank <= curr_min:
                            shared_state[f"solved_rank_{q_type}"] = rank
                            shared_state[f"solved_done_{q_type}_{rank}"] = True

                        # Cancel any queued (unstarted) tasks of equal or higher rank
                        for fut_ref, t_spec in list(future_to_task.items()):
                            if t_spec["quad_type"] == q_type:
                                should_cancel = (
                                    t_spec["rank"] > rank or 
                                    (t_spec["rank"] == rank and t_spec["init_idx"] != result["init_idx"])
                                )
                                if should_cancel:
                                    canceled = fut_ref.cancel()
                                    if canceled:
                                        completed_results.append({
                                            "quad_idx": t_spec["quad_idx"],
                                            "quad_type": q_type,
                                            "rank": t_spec["rank"],
                                            "init_idx": t_spec["init_idx"],
                                            "rel_error": None,
                                            "status": "pruned",
                                            "message": f"Pruned due to solved rank {rank}",
                                        })

    print("\nProcess execution completed successfully.")


if __name__ == "__main__":
    main()