#!/usr/bin/env python3
"""
4b_tab.py

Generate contracted 4-body ChIMES coefficient tables with:
  - contracted-dimension cutoff INCLUDED in the tabulated coefficients
  - contracted-dimension force-chain-rule coefficients also tabulated

For each retained basis function P, tabulate on a 3D grid over contracted dims:
  CE(P; rA,rB,rC)   = sum_q c_q G_A G_B G_C
  CDA(P; rA,rB,rC)  = sum_q c_q D_A G_B G_C
  CDB(P; rA,rB,rC)  = sum_q c_q G_A D_B G_C
  CDC(P; rA,rB,rC)  = sum_q c_q G_A G_B D_C

where
  G_d = fcut_d(r) * T_n(s(r))
  D_d = fcut'_d(r) * T_n(s(r)) + fcut_d(r) * dT_n/dr

Retained dimensions remain explicit at runtime.

Output:
  chimes_scan_4b_partial.type_<quad_type>.meta
  chimes_scan_4b_partial.type_<quad_type>.dat

Data row format:
  rA rB rC
  E0 ... EN
  dA0 ... dAN
  dB0 ... dBN
  dC0 ... dCN
(all flattened on one line)
"""

import os
import sys
import math
import numpy as np

sys.path.append(os.path.normpath(os.getcwd()))

if not os.path.exists("config.py"):
    print("Error: Cannot find config.py")
    sys.exit(1)

import config


# ----------------------------------------------------------------------
# helpers
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
    if len(pair_types_reference) != 6 or len(pair_names_list) != 6 or len(values_list) != 6:
        raise RuntimeError("Expected length-6 lists in resolve_occurrence_mapped_values")

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
    rd0, rd1, rd2 = retained_dims

    for powers, coeff in terms:
        power_set.add((powers[rd0], powers[rd1], powers[rd2]))

    return sorted(power_set)


def get_retained_power_list(quad_type, terms, retained_dims):
    if hasattr(config, "QUAD_RETAINED_POWER_LIST"):
        power_map = config.QUAD_RETAINED_POWER_LIST
        if quad_type in power_map:
            return [tuple(x) for x in power_map[quad_type]]

    return infer_retained_power_list(terms, retained_dims)


# ----------------------------------------------------------------------
# math helpers: Morse transform, Chebyshev, derivatives, cutoff
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
    """
    Mirror chimesFF::set_cheby_polys for in-range usage.
    Returns:
      Tn[0..order]
      Tnd[0..order]   derivative wrt r
    """
    if r < rmin:
        # For tabulation you should generally stay inside domain.
        # Clamp to avoid nonsense if user chooses otherwise.
        r_eval = rmin
    else:
        r_eval = r

    x, dx_dr = morse_transform_data(r_eval, rmin, rmax, morse_param)

    Tn = np.zeros(order + 1, dtype=float)
    TndU = np.zeros(order + 1, dtype=float)   # temporary U-like recursion
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


def build_GT_tables_for_slot(grid, slot_param, max_order, fcut_type, fcut_var):
    """
    For each r on grid build:
      G[n] = fcut * Tn[n]
      D[n] = fcut' * Tn[n] + fcut * Tnd[n]
    """
    G = {}
    D = {}

    rmin = slot_param["rmin"]
    rmax = slot_param["rmax"]
    morse = slot_param["morse"]

    for r in grid:
        Tn, Tnd = cheb_value_deriv_wrt_r(r, rmin, rmax, morse, max_order)
        fcut, fcutderiv = get_fcut(r, rmax, fcut_type, fcut_var)

        G[r] = fcut * Tn
        D[r] = fcutderiv * Tn + fcut * Tnd

    return G, D


# ----------------------------------------------------------------------
# contraction
# ----------------------------------------------------------------------

def contract_6d_to_selected_coeffs_with_cutoff(
    terms,
    contracted_dims,
    retained_dims,
    contracted_G,
    contracted_D,
    retained_power_list
):
    coeff_map = {p: i for i, p in enumerate(retained_power_list)}

    nret = len(retained_power_list)
    coeffE = np.zeros(nret, dtype=float)
    coeffDA = np.zeros(nret, dtype=float)
    coeffDB = np.zeros(nret, dtype=float)
    coeffDC = np.zeros(nret, dtype=float)

    cd0, cd1, cd2 = contracted_dims
    rd0, rd1, rd2 = retained_dims

    GA, GB, GC = contracted_G
    DA, DB, DC = contracted_D

    for powers, coeff in terms:
        retained_p = (powers[rd0], powers[rd1], powers[rd2])
        idx = coeff_map.get(retained_p, None)
        if idx is None:
            continue

        pA = powers[cd0]
        pB = powers[cd1]
        pC = powers[cd2]

        e  = coeff * GA[pA] * GB[pB] * GC[pC]
        da = coeff * DA[pA] * GB[pB] * GC[pC]
        db = coeff * GA[pA] * DB[pB] * GC[pC]
        dc = coeff * GA[pA] * GB[pB] * DC[pC]

        coeffE[idx]  += e
        coeffDA[idx] += da
        coeffDB[idx] += db
        coeffDC[idx] += dc

    return coeffE, coeffDA, coeffDB, coeffDC


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

def build_runtime_retained_basis(rvals, retained_dims, retained_power_list, slot_params, max_orders, fcut_type, fcut_var):
    """
    Construct retained basis pieces for explicit runtime reconstruction.
    """
    rd0, rd1, rd2 = retained_dims

    data = []
    for dim in [rd0, rd1, rd2]:
        sp = slot_params[dim]
        Tn, Tnd = cheb_value_deriv_wrt_r(rvals[dim], sp["rmin"], sp["rmax"], sp["morse"], max_orders[dim])
        fcut, fcutderiv = get_fcut(rvals[dim], sp["rmax"], fcut_type, fcut_var)
        data.append((Tn, Tnd, fcut, fcutderiv))

    return data


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
    """
    Compare exact direct 6D evaluation against contracted form including contracted cutoffs.
    """
    # exact direct
    exact_E = 0.0
    exact_dE = np.zeros(6, dtype=float)

    full_G = []
    full_D = []

    for d in range(6):
        sp = slot_params[d]
        Tn, Tnd = cheb_value_deriv_wrt_r(rvals[d], sp["rmin"], sp["rmax"], sp["morse"], max_orders[d])
        fcut, fcutderiv = get_fcut(rvals[d], sp["rmax"], fcut_type, fcut_var)

        full_G.append(fcut * Tn)
        full_D.append(fcutderiv * Tn + fcut * Tnd)

    for powers, coeff in terms:
        term_g = 1.0
        for d in range(6):
            term_g *= full_G[d][powers[d]]
        exact_E += coeff * term_g

        for d in range(6):
            term_d = coeff * full_D[d][powers[d]]
            for k in range(6):
                if k != d:
                    term_d *= full_G[k][powers[k]]
            exact_dE[d] += term_d

    cd0, cd1, cd2 = contracted_dims
    rd0, rd1, rd2 = retained_dims

    coeffE, coeffDA, coeffDB, coeffDC = contract_6d_to_selected_coeffs_with_cutoff(
        terms,
        contracted_dims,
        retained_dims,
        contracted_G=[full_G[cd0], full_G[cd1], full_G[cd2]],
        contracted_D=[full_D[cd0], full_D[cd1], full_D[cd2]],
        retained_power_list=retained_power_list
    )

    retained_runtime = build_runtime_retained_basis(
        rvals, retained_dims, retained_power_list, slot_params, max_orders, fcut_type, fcut_var
    )

    (Tu, Tud, fu, fud) = retained_runtime[0]
    (Tv, Tvd, fv, fvd) = retained_runtime[1]
    (Tw, Twd, fw, fwd) = retained_runtime[2]

    recon_E = 0.0
    recon_dE = np.zeros(6, dtype=float)

    for coeff_idx, p in enumerate(retained_power_list):
        pu, pv, pw = p

        bu = Tu[pu]
        bv = Tv[pv]
        bw = Tw[pw]

        recon_E += coeffE[coeff_idx] * (fu * bu) * (fv * bv) * (fw * bw)

        recon_dE[cd0] += coeffDA[coeff_idx] * (fu * bu) * (fv * bv) * (fw * bw)
        recon_dE[cd1] += coeffDB[coeff_idx] * (fu * bu) * (fv * bv) * (fw * bw)
        recon_dE[cd2] += coeffDC[coeff_idx] * (fu * bu) * (fv * bv) * (fw * bw)

        recon_dE[rd0] += coeffE[coeff_idx] * (fud * bu + fu * Tud[pu]) * (fv * bv) * (fw * bw)
        recon_dE[rd1] += coeffE[coeff_idx] * (fu * bu) * (fvd * bv + fv * Tvd[pv]) * (fw * bw)
        recon_dE[rd2] += coeffE[coeff_idx] * (fu * bu) * (fv * bv) * (fwd * bw + fw * Twd[pw])

    return exact_E, recon_E, exact_dE, recon_dE


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------

def write_meta_file(outfile, quad_type, contracted_dims, retained_dims, retained_power_list):
    with open(outfile, "w") as f:
        f.write(f"quad_type {quad_type}\n")
        f.write("contracted_dims " + " ".join(str(x) for x in contracted_dims) + "\n")
        f.write("retained_dims " + " ".join(str(x) for x in retained_dims) + "\n")
        f.write(f"ncoeff {len(retained_power_list)}\n")
        f.write("table_layout energy dA dB dC\n")
        f.write("coeff_powers\n")
        for p in retained_power_list:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def write_large_coeff_table(
    outfile,
    grid,
    terms,
    contracted_dims,
    retained_dims,
    retained_power_list,
    slot_params,
    max_orders,
    fcut_type,
    fcut_var
):
    nrows = len(grid) ** 3
    cd0, cd1, cd2 = contracted_dims

    GA_cache, DA_cache = build_GT_tables_for_slot(grid, slot_params[cd0], max_orders[cd0], fcut_type, fcut_var)
    GB_cache, DB_cache = build_GT_tables_for_slot(grid, slot_params[cd1], max_orders[cd1], fcut_type, fcut_var)
    GC_cache, DC_cache = build_GT_tables_for_slot(grid, slot_params[cd2], max_orders[cd2], fcut_type, fcut_var)

    with open(outfile, "w") as f:
        f.write(f"{nrows}\n")

        row = 0
        for ra in grid:
            GA = GA_cache[ra]
            DA = DA_cache[ra]
            for rb in grid:
                GB = GB_cache[rb]
                DB = DB_cache[rb]
                for rc in grid:
                    GC = GC_cache[rc]
                    DC = DC_cache[rc]

                    coeffE, coeffDA, coeffDB, coeffDC = contract_6d_to_selected_coeffs_with_cutoff(
                        terms=terms,
                        contracted_dims=contracted_dims,
                        retained_dims=retained_dims,
                        contracted_G=[GA, GB, GC],
                        contracted_D=[DA, DB, DC],
                        retained_power_list=retained_power_list
                    )

                    f.write(f"{ra:.12f} {rb:.12f} {rc:.12f}")

                    for arr in [coeffE, coeffDA, coeffDB, coeffDC]:
                        for c in arr:
                            f.write(f" {c:.16e}")

                    f.write("\n")

                    row += 1
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
        "QUAD_RETAIN_DIMS",
        "QUADSTART",
        "QUADSTOP",
        "QUADSTEP",
    ]

    for attr in required_attrs:
        if not hasattr(config, attr):
            raise RuntimeError(f"config.py is missing required attribute: {attr}")

    fcut_type, fcut_var = parse_fcut_type(config.PARAM_FILE)
    print(f"Using cutoff type: {fcut_type}" + (f" {fcut_var}" if fcut_var is not None else ""))

    for idx, quad_type in enumerate(quad_types):
        print("=" * 72)
        print(f"Processing 4B quad type {quad_type}")

        contracted_dims = tuple(config.QUAD_CONTRACT_DIMS[idx])
        retained_dims = tuple(config.QUAD_RETAIN_DIMS[idx])

        if len(contracted_dims) != 3 or len(retained_dims) != 3:
            raise RuntimeError("Each QUAD_CONTRACT_DIMS and QUAD_RETAIN_DIMS entry must have length 3")

        all_dims = list(contracted_dims) + list(retained_dims)
        if sorted(all_dims) != [0, 1, 2, 3, 4, 5]:
            raise RuntimeError("Contracted + retained dims must be a permutation of [0,1,2,3,4,5]")

        termdicts = parse_quad_terms_all(config.PARAM_FILE, quad_index=quad_type)
        if not termdicts:
            print(f"Quad type {quad_type} is excluded or has no terms; skipping.")
            continue

        terms = convert_termdicts_to_simple_terms(termdicts)
        max_orders = infer_max_orders(terms)
        retained_power_list = get_retained_power_list(quad_type, terms, retained_dims)

        slot_params = build_quad_slot_params(config.PARAM_FILE, quad_type)
        if len(slot_params) != 6:
            raise RuntimeError(f"Expected 6 slot params for quad type {quad_type}")

        grid = np.arange(
            config.QUADSTART[idx],
            config.QUADSTOP[idx],
            config.QUADSTEP[idx],
            dtype=float
        )
        grid=np.append(grid, config.QUADSTOP[idx])
        if grid.size == 0:
            raise RuntimeError(f"Empty grid for quad type index {idx}")

        datafile = f"chimes_scan_4b_partial.type_{quad_type}.dat"
        metafile = f"chimes_scan_4b_partial.type_{quad_type}.meta"

        print("4B term summary:")
        print(f"  number of 4B terms         = {len(terms)}")
        print(f"  max_orders                 = {max_orders}")
        print(f"  contracted_dims            = {contracted_dims}")
        print(f"  retained_dims              = {retained_dims}")
        print(f"  retained coefficient count = {len(retained_power_list)}")
        print(f"  grid points per dim        = {grid.size}")
        print(f"  total rows                 = {grid.size ** 3}")

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

        write_large_coeff_table(
            outfile=datafile,
            grid=grid,
            terms=terms,
            contracted_dims=contracted_dims,
            retained_dims=retained_dims,
            retained_power_list=retained_power_list,
            slot_params=slot_params,
            max_orders=max_orders,
            fcut_type=fcut_type,
            fcut_var=fcut_var
        )

        print(f"Wrote metadata: {metafile}")
        print(f"Wrote data    : {datafile}")

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()