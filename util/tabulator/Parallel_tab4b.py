#!/usr/bin/env python3
"""
4b_tab.py

Generate contracted 4-body ChIMES tables for 2D/3D/4D contraction.

For each quad type, contract k pair dimensions where k in {2,3,4}.
The remaining (6-k) dimensions are retained explicitly at runtime.

For each retained power tuple P and contracted grid point rc:
  CE(P; rc)     = sum_q c_q prod_{d in C} G_d
  Cder_i(P; rc) = sum_q c_q D_{C[i]} prod_{d in C, d!=C[i]} G_d

where
  G_d = fcut_d(r) * T_n(s(r))
  D_d = fcut'_d(r) * T_n(s(r)) + fcut_d(r) * dT_n/dr

Output:
  chimes_scan_4b_partial.type_<quad_type>.meta
  chimes_scan_4b_partial.type_<quad_type>.dat

Data row format:
  r0 r1 ... r(k-1)
  E0 ... EN
  d0_0 ... d0_N
  d1_0 ... d1_N
  ...
  d(k-1)_0 ... d(k-1)_N

Flattened on one line.

Parallelized version:
- Uses ProcessPoolExecutor to compute table rows in chunks
- Uses array-based G/D caches indexed by grid index rather than float keys
- Writes output in deterministic order
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


# ----------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# parameter parsing
# ----------------------------------------------------------------------

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
                        f"Incorrect pair specification line; expected 7 or 8 entries:\n{lines[i].rstrip()}"
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
                    f"Malformed 4B header after QUADRUPLETYPE PARAMS at line {i+1}: {lines[i].rstrip()}"
                )

            this_idx = int(hdr[1])

            i += 1
            if i >= len(lines):
                break

            pairline = split_clean(lines[i])
            if len(pairline) < 11:
                raise RuntimeError(
                    f"Malformed 4B pair/type line for quad index {this_idx}: {lines[i].rstrip()}"
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
                    f"Malformed 4B header after QUADRUPLETYPE PARAMS at line {i+1}: {lines[i].rstrip()}"
                )

            this_idx = int(hdr[1])

            i += 1
            if i >= len(lines):
                break

            pairline = split_clean(lines[i])
            if len(pairline) < 11:
                raise RuntimeError(
                    f"Malformed 4B pair/type line for quad index {this_idx}: {lines[i].rstrip()}"
                )

            if this_idx == quad_index:
                atom_types = hdr[3:7]
                pair_types = pairline[1:7]
                excluded = (pairline[7] == "EXCLUDED:")
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
                if len(items) < 5:
                    raise RuntimeError("Malformed SPECIAL 4B S_MAXIM: ALL line")
                val = float(items[4])
                special_max["__ALL__"] = val
            else:
                if len(items) < 5:
                    raise RuntimeError("Malformed SPECIAL 4B S_MAXIM line")
                nentries = int(items[4])

                for _ in range(nentries):
                    i += 1
                    row = split_clean(lines[i])
                    if len(row) < 13:
                        raise RuntimeError(f"Malformed SPECIAL 4B S_MAXIM entry: {lines[i].rstrip()}")

                    quad_slow_name = row[0]
                    quadidx = slow_to_quadidx[quad_slow_name]

                    pair_names = row[1:7]
                    cutoffvals = [float(x) for x in row[7:13]]

                    special_max[quadidx] = {
                        "pair_names": pair_names,
                        "values": cutoffvals,
                    }

        if len(items) >= 4 and items[0] == "SPECIAL" and items[1] == "4B" and items[2] == "S_MINIM:":
            if items[3] == "ALL":
                if len(items) < 5:
                    raise RuntimeError("Malformed SPECIAL 4B S_MINIM: ALL line")
                val = float(items[4])
                special_min["__ALL__"] = val
            else:
                if len(items) < 5:
                    raise RuntimeError("Malformed SPECIAL 4B S_MINIM line")
                nentries = int(items[4])

                for _ in range(nentries):
                    i += 1
                    row = split_clean(lines[i])
                    if len(row) < 13:
                        raise RuntimeError(f"Malformed SPECIAL 4B S_MINIM entry: {lines[i].rstrip()}")

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
                f"Could not map special 4B pair name '{name}' into reference pair list {pair_types_reference}"
            )

    if any(v is None for v in resolved):
        raise RuntimeError(
            f"Failed to resolve all 6 special 4B values against reference pair list {pair_types_reference}"
        )

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
            raise RuntimeError(
                f"Only MORSE mapping is currently implemented, found {meta['dist_type']} for pair '{pair_name}'"
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


# ----------------------------------------------------------------------
# math
# ----------------------------------------------------------------------

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
    if r < rmin:
        r_eval = rmin
    else:
        r_eval = r

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

    elif fcut_type == "TERSOFF":
        if fcut_var is None:
            raise RuntimeError("TERSOFF cutoff requires fcut_var")

        thresh = outer_cutoff - fcut_var * outer_cutoff

        if r < thresh:
            return 1.0, 0.0
        elif r > outer_cutoff:
            return 0.0, 0.0
        else:
            arg = (r - thresh) / (outer_cutoff - thresh) * math.pi + math.pi / 2.0
            arg_deriv = math.pi / (outer_cutoff - thresh)
            fcut = 0.5 + 0.5 * math.sin(arg)
            fcutderiv = 0.5 * math.cos(arg) * arg_deriv
            return fcut, fcutderiv

    else:
        raise RuntimeError(f"Unknown cutoff type: {fcut_type}")


# ----------------------------------------------------------------------
# term helpers
# ----------------------------------------------------------------------

def convert_termdicts_to_simple_terms(termdicts):
    return [(t["powers"], t["coeff"]) for t in termdicts]


def infer_max_orders(terms):
    max_orders = [0] * 6
    for powers, coeff in terms:
        for i in range(6):
            if powers[i] > max_orders[i]:
                max_orders[i] = powers[i]
    return max_orders


def infer_retained_power_list(terms, retained_dims):
    power_set = set()
    for powers, coeff in terms:
        power_set.add(tuple(powers[d] for d in retained_dims))
    return sorted(power_set)


def get_retained_power_list(quad_type, terms, retained_dims):
    if hasattr(config, "QUAD_RETAINED_POWER_LIST"):
        power_map = config.QUAD_RETAINED_POWER_LIST
        if quad_type in power_map:
            return [tuple(x) for x in power_map[quad_type]]
    return infer_retained_power_list(terms, retained_dims)


# ----------------------------------------------------------------------
# cache builders
# ----------------------------------------------------------------------

def build_GT_tables_for_slot(grid, slot_param, max_order, fcut_type, fcut_var):
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


# ----------------------------------------------------------------------
# contraction
# ----------------------------------------------------------------------

def contract_terms_general(
    terms,
    contracted_dims,
    retained_dims,
    contracted_G,
    contracted_D,
    retained_power_list
):
    """
    Returns:
      coeffE : shape [nret]
      coeffDeriv : list length k, each shape [nret]
    """
    k = len(contracted_dims)
    coeff_map = {p: i for i, p in enumerate(retained_power_list)}
    nret = len(retained_power_list)

    coeffE = np.zeros(nret, dtype=float)
    coeffDeriv = [np.zeros(nret, dtype=float) for _ in range(k)]

    for powers, coeff in terms:
        retained_p = tuple(powers[d] for d in retained_dims)
        idx = coeff_map.get(retained_p, None)
        if idx is None:
            continue

        gvals = [contracted_G[j][powers[contracted_dims[j]]] for j in range(k)]
        dvals = [contracted_D[j][powers[contracted_dims[j]]] for j in range(k)]

        prodG = 1.0
        for val in gvals:
            prodG *= val

        coeffE[idx] += coeff * prodG

        for j in range(k):
            prod = coeff * dvals[j]
            for m in range(k):
                if m != j:
                    prod *= gvals[m]
            coeffDeriv[j][idx] += prod

    return coeffE, coeffDeriv


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

def validate_internal_contraction(
    terms,
    contracted_dims,
    retained_dims,
    retained_power_list,
    rvals,
    slot_params,
    max_orders,
    fcut_type,
    fcut_var
):
    full_G = []
    full_D = []

    for d in range(6):
        sp = slot_params[d]
        Tn, Tnd = cheb_value_deriv_wrt_r(rvals[d], sp["rmin"], sp["rmax"], sp["morse"], max_orders[d])
        fcut, fcutderiv = get_fcut(rvals[d], sp["rmax"], fcut_type, fcut_var)
        full_G.append(fcut * Tn)
        full_D.append(fcutderiv * Tn + fcut * Tnd)

    exact_E = 0.0
    exact_dE = np.zeros(6, dtype=float)

    for powers, coeff in terms:
        eprod = coeff
        for d in range(6):
            eprod *= full_G[d][powers[d]]
        exact_E += eprod

        for d in range(6):
            dprod = coeff * full_D[d][powers[d]]
            for m in range(6):
                if m != d:
                    dprod *= full_G[m][powers[m]]
            exact_dE[d] += dprod

    coeffE, coeffDeriv = contract_terms_general(
        terms=terms,
        contracted_dims=contracted_dims,
        retained_dims=retained_dims,
        contracted_G=[full_G[d] for d in contracted_dims],
        contracted_D=[full_D[d] for d in contracted_dims],
        retained_power_list=retained_power_list
    )

    recon_E = 0.0
    recon_dE = np.zeros(6, dtype=float)

    retained_data = {}
    for d in retained_dims:
        sp = slot_params[d]
        Tn, Tnd = cheb_value_deriv_wrt_r(rvals[d], sp["rmin"], sp["rmax"], sp["morse"], max_orders[d])
        fcut, fcutderiv = get_fcut(rvals[d], sp["rmax"], fcut_type, fcut_var)
        retained_data[d] = (Tn, Tnd, fcut, fcutderiv)

    for q, ptuple in enumerate(retained_power_list):
        retained_prod = 1.0
        for loc, d in enumerate(retained_dims):
            Tn, Tnd, fcut, fcutderiv = retained_data[d]
            retained_prod *= fcut * Tn[ptuple[loc]]

        recon_E += coeffE[q] * retained_prod

        for j, d in enumerate(contracted_dims):
            recon_dE[d] += coeffDeriv[j][q] * retained_prod

        for loc, d in enumerate(retained_dims):
            part = coeffE[q]
            for loc2, d2 in enumerate(retained_dims):
                Tn, Tnd, fcut, fcutderiv = retained_data[d2]
                if d2 == d:
                    part *= fcutderiv * Tn[ptuple[loc2]] + fcut * Tnd[ptuple[loc2]]
                else:
                    part *= fcut * Tn[ptuple[loc2]]
            recon_dE[d] += part

    return exact_E, recon_E, exact_dE, recon_dE


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------

def write_meta_file(outfile, quad_type, contracted_dims, retained_dims, retained_power_list):
    with open(outfile, "w") as f:
        f.write(f"quad_type {quad_type}\n")
        f.write(f"ncontracted {len(contracted_dims)}\n")
        f.write("contracted_dims " + " ".join(str(x) for x in contracted_dims) + "\n")
        f.write(f"nretained {len(retained_dims)}\n")
        f.write("retained_dims " + " ".join(str(x) for x in retained_dims) + "\n")
        f.write(f"ncoeff {len(retained_power_list)}\n")
        f.write("table_layout energy")
        for i in range(len(contracted_dims)):
            f.write(f" d{i}")
        f.write("\n")
        f.write("coeff_powers\n")
        for p in retained_power_list:
            f.write(" ".join(str(x) for x in p) + "\n")


def iter_grid_index_points(ngrid, k):
    for pt in itertools.product(range(ngrid), repeat=k):
        yield pt


def compute_row_chunk(args):
    (
        idx_chunk,
        grid,
        terms,
        contracted_dims,
        retained_dims,
        retained_power_list,
        G_arrays,
        D_arrays,
    ) = args

    k = len(contracted_dims)
    out_lines = []

    for pt_idx in idx_chunk:
        contracted_G = []
        contracted_D = []

        for loc, d in enumerate(contracted_dims):
            gi = pt_idx[loc]
            contracted_G.append(G_arrays[d][gi])
            contracted_D.append(D_arrays[d][gi])

        coeffE, coeffDeriv = contract_terms_general(
            terms=terms,
            contracted_dims=contracted_dims,
            retained_dims=retained_dims,
            contracted_G=contracted_G,
            contracted_D=contracted_D,
            retained_power_list=retained_power_list
        )

        pt_vals = [grid[i] for i in pt_idx]

        parts = [" ".join(f"{x:.12f}" for x in pt_vals)]
        parts.extend(f"{x:.16e}" for x in coeffE)

        for j in range(k):
            parts.extend(f"{x:.16e}" for x in coeffDeriv[j])

        out_lines.append(" ".join(parts) + "\n")

    return out_lines


def write_coeff_table(
    outfile,
    grid,
    terms,
    contracted_dims,
    retained_dims,
    retained_power_list,
    slot_params,
    max_orders,
    fcut_type,
    fcut_var,
    nprocs=None,
    chunk_size=200
):
    k = len(contracted_dims)
    ngrid = len(grid)
    nrows = ngrid ** k

    G_arrays = {}
    D_arrays = {}

    for d in contracted_dims:
        G_arrays[d], D_arrays[d] = build_GT_tables_for_slot(
            grid, slot_params[d], max_orders[d], fcut_type, fcut_var
        )

    all_index_pts = list(iter_grid_index_points(ngrid, k))
    chunks = [
        all_index_pts[i:i + chunk_size]
        for i in range(0, len(all_index_pts), chunk_size)
    ]

    with open(outfile, "w") as f:
        f.write(f"{nrows}\n")

        row = 0

        if nprocs is None or nprocs <= 1:
            for chunk in chunks:
                out_lines = compute_row_chunk((
                    chunk,
                    grid,
                    terms,
                    contracted_dims,
                    retained_dims,
                    retained_power_list,
                    G_arrays,
                    D_arrays,
                ))
                f.writelines(out_lines)
                row += len(out_lines)
                if row % 100 == 0 or row == nrows:
                    print(f"Wrote {row} / {nrows} rows", flush=True)
        else:
            args_iter = (
                (
                    chunk,
                    grid,
                    terms,
                    contracted_dims,
                    retained_dims,
                    retained_power_list,
                    G_arrays,
                    D_arrays,
                )
                for chunk in chunks
            )

            with ProcessPoolExecutor(max_workers=nprocs) as ex:
                for out_lines in ex.map(compute_row_chunk, args_iter):
                    f.writelines(out_lines)
                    row += len(out_lines)
                    if row % 100 == 0 or row == nrows:
                        print(f"Wrote {row} / {nrows} rows", flush=True)


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------

def print_slot_params(slot_params):
    print("Per-slot 4B transform parameters:")
    for i, sp in enumerate(slot_params):
        print(
            f"  dim {i}: pair={sp['pair_name']:<8s} pair_index={sp['pair_index']:>3d} "
            f"rmin={sp['rmin']:.8f} rmax={sp['rmax']:.8f} morse={sp['morse']:.8f}"
        )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    quad_types = getattr(config, "QUADTYPES", [])
    if not quad_types:
        print("No QUADTYPES defined in config.py")
        return

    required_attrs = [
        "QUAD_CONTRACT_DIMS",
        "QUADSTART",
        "QUADSTOP",
        "QUADSTEP",
    ]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise RuntimeError(f"config.py is missing required attribute: {attr}")

    nprocs = getattr(config, "NPROCS", None)
    chunk_size = getattr(config, "QUAD_CHUNK_SIZE", 200)

    fcut_type, fcut_var = parse_fcut_type(config.PARAM_FILE)
    print(f"Using cutoff type: {fcut_type}" + (f" {fcut_var}" if fcut_var is not None else ""))

    if nprocs is None:
        print("Parallel row generation: auto/default")
    else:
        print(f"Parallel row generation: NPROCS={nprocs}")
    print(f"Chunk size: {chunk_size}")

    for idx, quad_type in enumerate(quad_types):
        print("=" * 72)
        print(f"Processing 4B quad type {quad_type}")

        contracted_dims = tuple(config.QUAD_CONTRACT_DIMS[idx])
        if len(contracted_dims) not in (2, 3, 4):
            raise RuntimeError("This script supports only 2, 3, or 4 contracted dimensions")

        retained_dims = tuple(d for d in range(6) if d not in contracted_dims)

        termdicts = parse_quad_terms_all(config.PARAM_FILE, quad_index=quad_type)
        if not termdicts:
            print(f"Quad type {quad_type} is excluded or has no terms; skipping.")
            continue

        terms = convert_termdicts_to_simple_terms(termdicts)
        max_orders = infer_max_orders(terms)
        retained_power_list = get_retained_power_list(quad_type, terms, retained_dims)
        slot_params = build_quad_slot_params(config.PARAM_FILE, quad_type)

        grid = np.arange(
            config.QUADSTART[idx],
            config.QUADSTOP[idx],
            config.QUADSTEP[idx],
            dtype=float
        )
        grid = np.append(grid, config.QUADSTOP[idx])
        if grid.size == 0:
            raise RuntimeError(f"Empty grid for quad type {quad_type}")

        datafile = f"chimes_scan_4b_partial.type_{quad_type}.dat"
        metafile = f"chimes_scan_4b_partial.type_{quad_type}.meta"

        print("4B term summary:")
        print(f"  number of 4B terms         = {len(terms)}")
        print(f"  max_orders                 = {max_orders}")
        print(f"  contracted_dims            = {contracted_dims}")
        print(f"  retained_dims              = {retained_dims}")
        print(f"  retained coefficient count = {len(retained_power_list)}")
        print(f"  grid points per dim        = {grid.size}")
        print(f"  total rows                 = {grid.size ** len(contracted_dims)}")

        print_slot_params(slot_params)

        test_r = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
        exact_E, recon_E, exact_dE, recon_dE = validate_internal_contraction(
            terms=terms,
            contracted_dims=contracted_dims,
            retained_dims=retained_dims,
            retained_power_list=retained_power_list,
            rvals=test_r,
            slot_params=slot_params,
            max_orders=max_orders,
            fcut_type=fcut_type,
            fcut_var=fcut_var
        )

        print("Validation at test geometry:")
        print("  exact energy      =", f"{exact_E:.16e}")
        print("  recon energy      =", f"{recon_E:.16e}")
        print("  energy abs err    =", f"{abs(exact_E-recon_E):.16e}")
        print("  exact dE/dr       =", " ".join(f"{x:.6e}" for x in exact_dE))
        print("  recon dE/dr       =", " ".join(f"{x:.6e}" for x in recon_dE))
        print("  dE/dr abs err     =", " ".join(f"{abs(a-b):.3e}" for a, b in zip(exact_dE, recon_dE)))

        write_meta_file(
            outfile=metafile,
            quad_type=quad_type,
            contracted_dims=contracted_dims,
            retained_dims=retained_dims,
            retained_power_list=retained_power_list
        )

        write_coeff_table(
            outfile=datafile,
            grid=grid,
            terms=terms,
            contracted_dims=contracted_dims,
            retained_dims=retained_dims,
            retained_power_list=retained_power_list,
            slot_params=slot_params,
            max_orders=max_orders,
            fcut_type=fcut_type,
            fcut_var=fcut_var,
            nprocs=nprocs,
            chunk_size=chunk_size
        )

        print(f"Wrote metadata: {metafile}")
        print(f"Wrote data    : {datafile}")

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()