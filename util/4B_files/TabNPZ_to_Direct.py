#!/usr/bin/env python3
"""
Make_CP_Direct.py

Convert CP factors from Tab_CP.py's .factors.npz output into a text file
for exact/on-the-fly CP evaluation in chimesFF.

Input:
    chimes_scan_4b_cp.type_<quad>.meta
    chimes_scan_4b_cp.type_<quad>.factors.npz

Output:
    chimes_scan_4b_cp_direct.type_<quad>.dat

The output stores:
    rank
    canonical slot mapping
    max Chebyshev order per canonical slot
    six CP factor matrices A_m[p,q]

Recommended workflow:
    1. In config.py for Tab_CP.py:
           CP_WRITE_FACTORS_NPZ = True

    2. Run:
           python3 Tab_CP.py

    3. Convert:
           python3 Make_CP_Direct.py \
               --meta chimes_scan_4b_cp.type_0.meta \
               --npz chimes_scan_4b_cp.type_0.factors.npz

    4. Add to params file:
           4B CP DIRECT: 1
           0 chimes_scan_4b_cp_direct.type_0.dat
"""

import argparse
import os
import numpy as np


def split_clean(line):
    if "!" in line:
        line = line[:line.find("!")]
    if "##" in line:
        line = line[:line.find("##")]
    return line.rstrip("\n").split()


def read_lines(path):
    with open(path, "r") as f:
        return f.readlines()


def parse_cp_meta(meta_file):
    meta = {
        "slot_files": {},
    }

    in_slot_files = False

    for line in read_lines(meta_file):
        items = split_clean(line)
        if not items:
            continue

        if items[0] == "slot_files":
            in_slot_files = True
            continue

        if in_slot_files:
            if len(items) >= 2 and items[0].isdigit():
                meta["slot_files"][int(items[0])] = items[1]
                continue
            else:
                in_slot_files = False

        key = items[0]

        if key == "quad_type":
            meta["quad_type"] = int(items[1])
        elif key == "rank":
            meta["rank"] = int(items[1])
        elif key == "atom_types":
            meta["atom_types"] = items[1:]
        elif key == "pair_types_original":
            meta["pair_types_original"] = items[1:]
        elif key == "pair_types_canonical":
            meta["pair_types_canonical"] = items[1:]
        elif key == "canon_to_param":
            meta["canon_to_param"] = [int(x) for x in items[1:]]
        elif key == "max_orders":
            meta["max_orders"] = [int(x) for x in items[1:]]
        elif key == "tensor_dims":
            meta["tensor_dims"] = [int(x) for x in items[1:]]
        elif key == "coefficient_relative_frobenius_error":
            meta["coeff_rel_error"] = float(items[1])

    required = [
        "quad_type",
        "rank",
        "canon_to_param",
        "max_orders",
        "tensor_dims",
        "pair_types_original",
        "pair_types_canonical",
    ]

    for key in required:
        if key not in meta:
            raise RuntimeError(f"Missing required metadata key: {key}")

    if len(meta["canon_to_param"]) != 6:
        raise RuntimeError("canon_to_param must have length 6")

    if len(meta["max_orders"]) != 6:
        raise RuntimeError("max_orders must have length 6")

    return meta


def default_npz_name(meta_file, quad_type):
    dirname = os.path.dirname(os.path.abspath(meta_file))
    return os.path.join(dirname, f"chimes_scan_4b_cp.type_{quad_type}.factors.npz")


def default_out_name(meta_file, quad_type):
    dirname = os.path.dirname(os.path.abspath(meta_file))
    return os.path.join(dirname, f"chimes_scan_4b_cp_direct.type_{quad_type}.dat")


def write_cp_direct_file(outfile, meta, factors):
    quad_type = meta["quad_type"]
    rank = meta["rank"]
    canon_to_param = meta["canon_to_param"]
    max_orders = meta["max_orders"]
    tensor_dims = meta["tensor_dims"]

    with open(outfile, "w") as f:
        f.write("CHIMES_4B_CP_DIRECT\n")
        f.write(f"quad_type {quad_type}\n")
        f.write(f"rank {rank}\n")
        f.write("canon_to_param " + " ".join(str(x) for x in canon_to_param) + "\n")
        f.write("max_orders " + " ".join(str(x) for x in max_orders) + "\n")
        f.write("tensor_dims " + " ".join(str(x) for x in tensor_dims) + "\n")

        if "coeff_rel_error" in meta:
            f.write(f"coefficient_relative_frobenius_error {meta['coeff_rel_error']:.16e}\n")

        f.write("pair_types_original " + " ".join(meta["pair_types_original"]) + "\n")
        f.write("pair_types_canonical " + " ".join(meta["pair_types_canonical"]) + "\n")

        for c in range(6):
            A = factors[c]
            nrows, ncols = A.shape

            expected_rows = max_orders[c] + 1
            if nrows != expected_rows:
                raise RuntimeError(
                    f"Factor A{c} row mismatch: got {nrows}, expected {expected_rows}"
                )

            if ncols != rank:
                raise RuntimeError(
                    f"Factor A{c} rank mismatch: got {ncols}, expected {rank}"
                )

            f.write(f"factor {c} {nrows} {ncols}\n")

            for p in range(nrows):
                vals = " ".join(f"{A[p, q]:.17e}" for q in range(ncols))
                f.write(f"{p} {vals}\n")

        f.write("END_CHIMES_4B_CP_DIRECT\n")


def main():
    ap = argparse.ArgumentParser(
        description="Export exact CP 4B factor file from Tab_CP.py .factors.npz"
    )

    ap.add_argument("--meta", required=True, help="CP metadata file")
    ap.add_argument("--npz", default=None, help="CP factors .npz file")
    ap.add_argument("--out", default=None, help="Output CP-direct text file")

    args = ap.parse_args()

    meta = parse_cp_meta(args.meta)
    quad_type = meta["quad_type"]

    npz_file = args.npz
    if npz_file is None:
        npz_file = default_npz_name(args.meta, quad_type)

    out_file = args.out
    if out_file is None:
        out_file = default_out_name(args.meta, quad_type)

    if not os.path.exists(npz_file):
        raise RuntimeError(f"Cannot find factors npz file: {npz_file}")

    z = np.load(npz_file)

    factors = []
    for c in range(6):
        key = f"A{c}"
        if key not in z:
            raise RuntimeError(f"Missing {key} in {npz_file}")
        factors.append(np.asarray(z[key], dtype=float))

    print("Writing exact CP factor file")
    print(f"  meta       : {args.meta}")
    print(f"  npz        : {npz_file}")
    print(f"  output     : {out_file}")
    print(f"  quad_type  : {quad_type}")
    print(f"  rank       : {meta['rank']}")
    print(f"  max_orders : {meta['max_orders']}")
    print(f"  canon_to_param : {meta['canon_to_param']}")

    write_cp_direct_file(out_file, meta, factors)

    print()
    print("Add this block to the ChIMES parameter file:")
    print("4B CP DIRECT: 1")
    print(f"{quad_type} {os.path.basename(out_file)}")


if __name__ == "__main__":
    main()