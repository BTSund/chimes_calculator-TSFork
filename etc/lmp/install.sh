#!/bin/bash
#
# Builds LAMMPS executables with the ChIMES pair style (pair_style chimesFF).
#
# ---------------------------------------------------------------------------
# USAGE
#
#   [export hosttype=<LLNL-LC|UM-ARC|JHU-ARCH|UT-TACC>;] ./install.sh [options] [VARIANT ...]
#
# A VARIANT is one or more FEATURES joined with '+'. Each variant becomes its
# own executable in ./exe/. Several variants can be built in one call; after the
# first, only the few ChIMES-dependent objects are recompiled per variant (fast).
#
#   FEATURE      aliases              compile flag(s)
#   -----------  -------------------  -------------------------------------------
#   BASE         NONE, PLAIN          (none)
#   TAB          TABULATION           -DTABULATION
#   REUSE        SUBGRAPH, SGR        -DCHIMES_SUBGRAPH_REUSE
#   PROF         PROFILE              -DCHIMES_PROFILE -DCHIMES_PROFILE_EVERY=<N>
#   FP           FINGERPRINT          -DFINGERPRINT   (also builds the FP histogram tool)
#
#   Rules: FP cannot be combined with TAB. BASE cannot be combined with anything.
#
#   PRESETS (expand to several variants)
#     BOTH          -> TAB  TAB+PROF                       (legacy behaviour)
#     TAB_ALL       -> TAB  TAB+PROF  TAB+REUSE  TAB+REUSE+PROF
#     REUSE_ALL     -> REUSE  REUSE+PROF
#     ALL           -> BASE REUSE TAB TAB+REUSE TAB+REUSE+PROF FP
#
#   Legacy single-word options still work: TABULATION, TABULATION_PROF, FINGERPRINT.
#
# Executable names: exe/lmp_mpi_chimes[_tabulation][_reuse][_prof][_fingerprint]
#   e.g. TAB+REUSE+PROF -> exe/lmp_mpi_chimes_tabulation_reuse_prof
#        BASE           -> exe/lmp_mpi_chimes
#
# OPTIONS
#   -y, --fresh            Delete and re-clone LAMMPS without asking
#   -k, --keep             Reuse the existing LAMMPS build tree without asking
#   -j, --jobs N           Parallel make jobs (default: number of cores, max 16)
#   -e, --profile-every N  Profiling report interval in steps (default: 10)
#       --cxx COMPILER     Override the MPI C++ compiler in the Makefile
#                          (e.g. mpicxx, mpiicpx). Default: Makefile setting.
#       --list             Print the variants that would be built and exit
#   -h, --help             Show this help
#
# EXAMPLES
#   ./install.sh                          # basic ChIMES build
#   ./install.sh TAB+REUSE                # tabulation + subgraph reuse
#   ./install.sh -k TAB+REUSE TAB+REUSE+PROF
#   ./install.sh -y -j 8 TAB_ALL
#   ./install.sh --cxx mpicxx REUSE       # non-Intel MPI wrapper
# ---------------------------------------------------------------------------

set -o pipefail

lammps="stable_29Aug2024_update1"
MACHINE="mpi_chimes"

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

die() { echo ""; echo "ERROR: $*"; echo "Run './install.sh --help' for usage."; echo ""; exit 1; }

# ********** ARGUMENT PARSING **********

FRESH=""            # "", "yes", "no"
JOBS=""
PROF_EVERY=10
CXX_OVERRIDE=""
LIST_ONLY=0
RAW_VARIANTS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)           usage; exit 0 ;;
        -y|--fresh|--yes)    FRESH="yes" ;;
        -k|--keep)           FRESH="no" ;;
        -j|--jobs)           shift; JOBS="$1" ;;
        -j*)                 JOBS="${1#-j}" ;;
        -e|--profile-every)  shift; PROF_EVERY="$1" ;;
        --cxx)               shift; CXX_OVERRIDE="$1" ;;
        --list)              LIST_ONLY=1 ;;
        -*)                  die "Unknown option '$1'" ;;
        *)                   RAW_VARIANTS+=("$1") ;;
    esac
    shift
done

[[ "$PROF_EVERY" =~ ^[0-9]+$ ]] && [ "$PROF_EVERY" -gt 0 ] || die "--profile-every expects a positive integer"

if [ -z "$JOBS" ]; then
    JOBS=$( (nproc || sysctl -n hw.ncpu || echo 4) 2>/dev/null | head -1 )
    [ "$JOBS" -gt 16 ] && JOBS=16
fi
[[ "$JOBS" =~ ^[0-9]+$ ]] || die "--jobs expects an integer"

[ ${#RAW_VARIANTS[@]} -eq 0 ] && RAW_VARIANTS=("BASE")

# Expand presets / legacy names

VARIANTS=()
for v in "${RAW_VARIANTS[@]}"; do
    case "${v^^}" in
        BOTH)            VARIANTS+=("TAB" "TAB+PROF") ;;
        TAB_ALL)         VARIANTS+=("TAB" "TAB+PROF" "TAB+REUSE" "TAB+REUSE+PROF") ;;
        REUSE_ALL)       VARIANTS+=("REUSE" "REUSE+PROF") ;;
        ALL)             VARIANTS+=("BASE" "REUSE" "TAB" "TAB+REUSE" "TAB+REUSE+PROF" "FP") ;;
        TABULATION_PROF) VARIANTS+=("TAB+PROF") ;;
        *)               VARIANTS+=("$v") ;;
    esac
done

# Parse each variant -> flags + exe name (validated before anything is deleted)

V_FLAGS=()
V_EXE=()
V_DESC=()
ANY_FP=0

for v in "${VARIANTS[@]}"; do
    tab=0; reuse=0; prof=0; fp=0; base=0
    IFS='+' read -r -a feats <<< "${v^^}"
    [ ${#feats[@]} -eq 0 ] && die "Empty variant"
    for f in "${feats[@]}"; do
        case "$f" in
            BASE|NONE|PLAIN)        base=1 ;;
            TAB|TABULATION)         tab=1 ;;
            REUSE|SUBGRAPH|SGR|SUBGRAPH_REUSE) reuse=1 ;;
            PROF|PROFILE)           prof=1 ;;
            FP|FINGERPRINT)         fp=1 ;;
            *) die "Unknown feature '$f' in variant '$v'. Valid: BASE TAB REUSE PROF FP" ;;
        esac
    done

    [ $base -eq 1 ] && [ $((tab+reuse+prof+fp)) -gt 0 ] && die "BASE cannot be combined with other features ('$v')"
    [ $fp -eq 1 ] && [ $tab -eq 1 ] && die "FINGERPRINT and TABULATION cannot be combined ('$v')"

    flags=""; name="lmp_${MACHINE}"; desc=""
    [ $tab   -eq 1 ] && { flags+=" -DTABULATION";            name+="_tabulation";  desc+="TABULATION "; }
    [ $reuse -eq 1 ] && { flags+=" -DCHIMES_SUBGRAPH_REUSE"; name+="_reuse";       desc+="SUBGRAPH_REUSE "; }
    [ $prof  -eq 1 ] && { flags+=" -DCHIMES_PROFILE -DCHIMES_PROFILE_EVERY=${PROF_EVERY}"; name+="_prof"; desc+="PROFILE(every ${PROF_EVERY}) "; }
    [ $fp    -eq 1 ] && { flags+=" -DFINGERPRINT";           name+="_fingerprint"; desc+="FINGERPRINT "; ANY_FP=1; }
    [ -z "$desc" ] && desc="basic (no extra flags)"

    # skip duplicates
    dup=0
    for e in "${V_EXE[@]}"; do [ "$e" = "$name" ] && dup=1; done
    [ $dup -eq 1 ] && continue

    V_FLAGS+=("$flags")
    V_EXE+=("$name")
    V_DESC+=("$desc")
done

echo ""
echo "ChIMES/LAMMPS build plan (LAMMPS ${lammps}, make -j ${JOBS}):"
for n in "${!V_EXE[@]}"; do
    printf "  %d. exe/%-45s %s\n" $((n+1)) "${V_EXE[$n]}" "${V_DESC[$n]}"
done
echo ""

[ $LIST_ONLY -eq 1 ] && exit 0

echo "Note: This install script assumes: "
echo "1. Availability of Intel C++ compilers with c++11 support (or use --cxx)"
echo "2. Availability of Intel MPI compilers"
echo "...Intel oneapi compilers are now freely available"
echo ""

# ********** HOST MODULES **********

if [ -z "$hosttype" ] ; then
    echo "WARNING: No hosttype specified"
    echo "Be sure to load modules/configure compilers by hand before running this script!"
    echo ""
elif [ -f "modfiles/${hosttype}.mod" ] ; then
    source "modfiles/${hosttype}.mod"
    echo "Detected hosttype: $hosttype"
    [ -n "$hasmod" ] && module list
else
    echo "ERROR: Unknown hosttype ($hosttype) specified"
    echo ""
    echo "Valid options are:"
    for i in modfiles/*.mod; do i=${i##*/}; echo "   ${i%.mod}"; done
    echo ""
    echo "Please run again with: export hosttype=<host type>; ./install.sh"
    echo "Or manually load modules and run with: ./install.sh"
    exit 1
fi

MAKEFILE_SRC="etc/Makefile.mpi_chimes"
[ "$hosttype" = "UT-TACC" ] && MAKEFILE_SRC="etc/Makefile.mpi_chimes.UT-TACC"
[ -f "$MAKEFILE_SRC" ] || die "Cannot find $MAKEFILE_SRC"

# ********** LAMMPS SOURCE **********

SRC="build/${lammps}/src"

if [ -z "$FRESH" ]; then
    if [ -d "$SRC" ]; then
        read -r -p "Delete and re-clone the existing LAMMPS build? (y/N) " ans
        [[ "$ans" =~ ^[Yy] ]] && FRESH="yes" || FRESH="no"
    else
        FRESH="yes"
    fi
fi

if [ "$FRESH" = "yes" ] || [ ! -d "$SRC" ]; then
    echo "Fresh clone of LAMMPS ${lammps}"
    ./uninstall.sh
    mkdir -p build/${lammps}
    git clone --depth 1 --branch ${lammps} https://github.com/lammps/lammps.git build/${lammps} || die "git clone failed"
else
    echo "Using pre-existing LAMMPS build in ${SRC}"
fi

# ********** COPY CHIMES FILES AND INSTALL PACKAGES **********

cp ../../chimesFF/src/chimesFF.{h,cpp}  ${SRC}/MANYBODY/ || die "Could not copy chimesFF sources"
cp src/pair_chimes.{h,cpp}              ${SRC}/MANYBODY/ || die "Could not copy pair_chimes sources"
cp etc/pair.{h,cpp}                     ${SRC}/          || die "Could not copy pair.{h,cpp}"

( cd ${SRC} && make yes-manybody > /dev/null && make yes-extra-pair > /dev/null ) || die "Package install failed"

# Packages that are already installed are not re-copied by 'make yes-*', so
# make sure the active copies in src/ are the current ones.
cp ${SRC}/MANYBODY/chimesFF.{h,cpp} ${SRC}/MANYBODY/pair_chimes.{h,cpp} ${SRC}/

mkdir -p exe

# ********** BUILD EACH VARIANT **********
#
# The ChIMES compile flags only affect sources that see chimesFF.h, i.e.
# chimesFF.cpp, pair_chimes.cpp and the files including style_pair.h (which
# instantiate PairCHIMES, whose memory layout depends on the flags). Only those
# objects are removed and rebuilt between variants; the rest of LAMMPS is
# compiled once.

OBJDIR="${SRC}/Obj_${MACHINE}"
CHIMES_DEP_OBJS=$(cd ${SRC} && grep -l -E '#include[[:space:]]*"(style_pair|pair_chimes|chimesFF)\.h"' *.cpp | sed 's/\.cpp$/.o/')
NV=${#V_EXE[@]}
FAILED=()

for n in "${!V_EXE[@]}"; do
    flags="${V_FLAGS[$n]}"
    exe="${V_EXE[$n]}"

    echo ""
    echo "=========================================================="
    echo "Compiling $((n+1))/${NV}: ${exe}"
    echo "  flags:${flags:- (none)}"
    echo "=========================================================="

    sed -e "s|^CCFLAGS\(.*\)|CCFLAGS\1${flags}|" "$MAKEFILE_SRC" > ${SRC}/MAKE/Makefile.${MACHINE}
    if [ -n "$CXX_OVERRIDE" ]; then
        sed -i -e "s|^CC[[:space:]]*=.*|CC =		${CXX_OVERRIDE}|" \
               -e "s|^LINK[[:space:]]*=.*|LINK =		${CXX_OVERRIDE}|" ${SRC}/MAKE/Makefile.${MACHINE}
    fi

    for o in ${CHIMES_DEP_OBJS}; do rm -f "${OBJDIR}/${o}"; done
    rm -f ${OBJDIR}/Makefile ${SRC}/lmp_${MACHINE}

    if ( cd ${SRC} && make -j ${JOBS} ${MACHINE} ) && [ -f ${SRC}/lmp_${MACHINE} ]; then
        mv ${SRC}/lmp_${MACHINE} exe/${exe}
    else
        echo "ERROR: build of ${exe} failed"
        FAILED+=("${exe}")
    fi
done

# ********** FINGERPRINT HISTOGRAM TOOL **********

if [ $ANY_FP -eq 1 ]; then
    echo ""
    echo "Compiling histogram executable for ChIMES fingerprints"
    FP_CXX="${CXX_OVERRIDE:-mpiicc}"
    ${FP_CXX} -O3 -o ../../chimesFF/src/FP/histogram \
        ../../chimesFF/src/FP/multi_calc_histogram.cpp ../../chimesFF/src/FP/chimesFF.cpp \
        || FAILED+=("FP histogram")
fi

# ********** SUMMARY **********

loc=$(pwd)
echo ""
echo "Compilation complete."
echo "Generated ChIMES executables:"
for n in "${!V_EXE[@]}"; do
    if [ -f "exe/${V_EXE[$n]}" ]; then
        printf "  %-60s %s\n" "${loc}/exe/${V_EXE[$n]}" "${V_DESC[$n]}"
    fi
done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILED: ${FAILED[*]}"
    echo ""
    exit 1
fi
echo "See ${loc}/tests for usage examples"
echo ""