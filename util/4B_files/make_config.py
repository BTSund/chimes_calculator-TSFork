import re
from itertools import combinations_with_replacement

def parse_params(filename="params.txt"):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        # Replace non-breaking spaces (\xa0) with standard spaces
        cleaned_lines = [line.replace('\xa0', ' ').strip() for line in f]

    # 1. Parse ATOM TYPES
    atom_types = []
    in_atom_types = False
    for line in cleaned_lines:
        if "ATOM TYPES:" in line.upper():
            in_atom_types = True
            continue
        if in_atom_types:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].isdigit():
                atom_types.append(parts[1])
            else:
                in_atom_types = False

    # 2. Parse ATOM PAIRS (S_MINIM mapping)
    pair_s_minim = {}
    in_atom_pairs = False
    for line in cleaned_lines:
        if "ATOM PAIRS:" in line.upper():
            in_atom_pairs = True
            continue
        if in_atom_pairs:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].isdigit() and len(parts) >= 4:
                a1, a2, s_min = parts[1], parts[2], float(parts[3])
                pair_s_minim[f"{a1}{a2}"] = s_min
                pair_s_minim[f"{a2}{a1}"] = s_min
            elif not line.startswith("#"):
                in_atom_pairs = False

    # 3. Build pair combination -> quadruplet index mapping
    pair_str_to_idx = {}
    for i, line in enumerate(cleaned_lines):
        if "QUADRUPLETYPE PARAMS:" in line.upper():
            idx, pairs = None, None
            for offset in range(1, 4):
                if i + offset < len(cleaned_lines):
                    sub = cleaned_lines[i + offset]
                    if "INDEX:" in sub.upper():
                        m = re.search(r'INDEX:\s*(\d+)', sub, re.IGNORECASE)
                        if m:
                            idx = int(m.group(1))
                    if "PAIRS:" in sub.upper():
                        pairs = " ".join(sub.split("PAIRS:")[1].strip().split()[:6])
            if idx is not None and pairs is not None:
                pair_str_to_idx[pairs] = idx

    # Fallback for implicit 4B canonical indices
    all_quads = list(combinations_with_replacement(atom_types, 4))
    for idx, (a0, a1, a2, a3) in enumerate(all_quads):
        pairs_key = f"{a0}{a1} {a0}{a2} {a0}{a3} {a1}{a2} {a1}{a3} {a2}{a3}"
        if pairs_key not in pair_str_to_idx:
            pair_str_to_idx[pairs_key] = idx

    # 4. Extract 4B Outer Cutoffs
    quad_types = []
    quad_start_entries = []
    quad_stop_values = []

    in_4b_section = False
    for line in cleaned_lines:
        u_line = line.upper()
        if "4B OUTER CUTOFFS:" in u_line or "SPECIAL 4B S_MAXIM:" in u_line:
            in_4b_section = True
            continue

        if in_4b_section:
            if u_line.startswith("!") or u_line.startswith("EXAMPLE") or u_line.startswith("QUADRUPLETYPE"):
                in_4b_section = False
                continue

            parts = line.split()
            if len(parts) >= 7 and not parts[0].endswith(":"):
                pairs = parts[1:7]
                if all(p in pair_s_minim for p in pairs):
                    label = parts[0]
                    pair_key = " ".join(pairs)
                    q_idx = pair_str_to_idx.get(pair_key)
                    if q_idx is not None:
                        min_s = min(pair_s_minim[p] for p in pairs)

                        # Parse outer cutoff distance (parts 7-12)
                        if len(parts) >= 13:
                            max_cutoff = max(float(x) for x in parts[7:13])
                        else:
                            max_cutoff = 4.0

                        stop_val = int(max_cutoff) if max_cutoff.is_integer() else max_cutoff

                        quad_types.append(q_idx)
                        quad_start_entries.append((min_s, f"{label}: {' '.join(pairs)}"))
                        quad_stop_values.append(stop_val)

    # Output formatting
    print(f"QUADTYPES   = {quad_types} # Triplet type index for scans, i.e. number after \"TRIPLETTYPE PARAMS:\" in parameter file")
    print("QUADSTART = [")
    for s_val, comment in quad_start_entries:
        print(f"    {s_val:.3f},  # {comment}")
    print("]")
    print(f"QUADSTOP   = {quad_stop_values} # Largest distance for scan")

if __name__ == "__main__":
    parse_params("params.txt")