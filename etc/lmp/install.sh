#!/bin/bash

# Builds all relevant chimes_calculator executables/library files 
#
# If working on a machine with a corresponding .mod file in the modfiles folder
# (e.g., modfiles/LLNL-LC.mod), execute with, e.g.:
#
#   export hosttype=LLNL-LC; ./install.sh
# 
# Otherwise, load necessary modules manually and execute with 
# 
# ./install.sh 
# 
# Note that additional arguments can be specified:
#   ./install.sh [TABULATION | TABULATION_PROF | BOTH | FINGERPRINT]

echo ""
echo "Note: This install script assumes: "
echo "1. Availability of Intel C++ compilers with c++11 support"
echo "2. Availability of Intel MPI compilers"
echo "...Intel oneapi compilers are now freely available"
echo ""

# ********** FLAG HANDLING **********
TAB_FLAG=""
FINGERPRINT_FLAG=""
DO_BOTH=0

if [ "$1" = "FINGERPRINT" ]; then
    FINGERPRINT_FLAG="-DFINGERPRINT"
    echo "Enabling FINGERPRINT compilation flag for ChIMES files"
elif [ "$1" = "TABULATION" ]; then
    TAB_FLAG="-DTABULATION"
    echo "Enabling TABULATION compilation flag for ChIMES files"
elif [ "$1" = "TABULATION_PROF" ]; then
    TAB_FLAG="-DTABULATION -DCHIMES_PROFILE -DCHIMES_PROFILE_EVERY=10"
    echo "Enabling TABULATION compilation flag for ChIMES files"
elif [ "$1" = "BOTH" ]; then
    DO_BOTH=1
    echo "Enabling compilation for BOTH TABULATION and TABULATION + PROFILING executables"
elif [ -n "$1" ]; then
    echo "ERROR: Invalid option '$1'. Use one of: TABULATION, TABULATION_PROF, BOTH, or FINGERPRINT"
    exit 1
fi

# Cleanup any previous installation

echo "Lammps directory will be deleted and re-cloned/installed. Proceed? (y/n)"
lammps="stable_29Aug2024_update1"
read fresh
if [ "$fresh" = "n" ] ; then
    echo 'Will use pre-existing lammps build'
else
    echo 'Fresh compiling lammps'
    ./uninstall.sh
    mkdir -p build/${lammps}
    git clone --depth 1 --branch ${lammps} https://github.com/lammps/lammps.git build/${lammps}
fi

# Copy ChIMES files to correct locations

cp ../../chimesFF/src/chimesFF.{h,cpp}  build/${lammps}/src/MANYBODY/
cp src/pair_chimes.{h,cpp}              build/${lammps}/src/MANYBODY/
cp etc/pair.{h,cpp}                     build/${lammps}/src

MAKEFILE_SRC="etc/Makefile.mpi_chimes"
if [ "$hosttype" = "UT-TACC" ]; then
    MAKEFILE_SRC="etc/Makefile.mpi_chimes.UT-TACC"
fi

# ********** MODIFIED MAKEFILE HANDLING **********
if [ "$DO_BOTH" -eq 1 ]; then
    : # Will be handled during the compile step
elif [ -n "$TAB_FLAG" ]; then
    sed -e "s/^CCFLAGS.*/& $TAB_FLAG/" "$MAKEFILE_SRC" > build/${lammps}/src/MAKE/Makefile.mpi_chimes
elif [ -n "$FINGERPRINT_FLAG" ]; then
    sed -e "s/^CCFLAGS.*/& $FINGERPRINT_FLAG/" "$MAKEFILE_SRC" > build/${lammps}/src/MAKE/Makefile.mpi_chimes
else
    cp "$MAKEFILE_SRC" build/${lammps}/src/MAKE/Makefile.mpi_chimes
fi

# Load module files and configure compilers

if [ -z "$hosttype" ] ; then
    echo ""
    echo "WARNING: No hosttype specified"
    echo "Be sure to load modules/configure compilers by hand before running this script!"
    echo ""
elif [ "$hosttype" = "LLNL-LC" ] ; then
    source modfiles/LLNL-LC.mod
elif [ "$hosttype" = "UM-ARC" ] ; then
    source modfiles/UM-ARC.mod
elif [ "$hosttype" = "JHU-ARCH" ] ; then
    source modfiles/JHU-ARCH.mod
    ICC=`which icc`
    MPI=`which mpicxx`
elif [ "$hosttype" = "UT-TACC" ] ; then
    source modfiles/UT-TACC.mod
else
    echo ""
    echo "ERROR: Unknown hosttype ($hosttype) specified"
    echo ""
    echo "Valid options are:"
    for i in `ls modfiles`; do echo "   ${i%.mod}"; done
    echo ""
    echo "Please run again with: export hosttype=<host type>; ./install.sh"
    echo "Or manually load modules and run with: ./install.sh"
    exit 0
fi

echo "Detected hosttype: $hosttype"
if [ ! -z "$hasmod" ] ; then
    module list
fi

# Prepare directories and install packages
mkdir -p exe

cd build/${lammps}/src
make yes-manybody
make yes-extra-pair
cd -

# Compile

if [ "$DO_BOTH" -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo "Compiling 1/2: lmp_mpi_chimes_tabulation"
    echo "=========================================="
    sed -e "s/^CCFLAGS.*/& -DTABULATION/" "$MAKEFILE_SRC" > build/${lammps}/src/MAKE/Makefile.mpi_chimes
    cd build/${lammps}/src
    make clean-mpi
    make -j 4 mpi_chimes
    cd -
    mv build/${lammps}/src/lmp_mpi_chimes exe/lmp_mpi_chimes_tabulation

    echo ""
    echo "=========================================="
    echo "Compiling 2/2: lmp_mpi_chimes_tabulation_prof"
    echo "=========================================="
    sed -e "s/^CCFLAGS.*/& -DTABULATION -DCHIMES_PROFILE -DCHIMES_PROFILE_EVERY=10/" "$MAKEFILE_SRC" > build/${lammps}/src/MAKE/Makefile.mpi_chimes
    cd build/${lammps}/src
    make clean-mpi
    make -j 4 mpi_chimes
    cd -
    mv build/${lammps}/src/lmp_mpi_chimes exe/lmp_mpi_chimes_tabulation_prof
else
    cd build/${lammps}/src
    make -j 4 mpi_chimes
    cd -

    if [ -n "$FINGERPRINT_FLAG" ]; then
        echo ""
        echo "Compiling histogram executable for ChIMES fingerprints"
        mpiicc -O3 -o ../../chimesFF/src/FP/histogram ../../chimesFF/src/FP/multi_calc_histogram.cpp ../../chimesFF/src/FP/chimesFF.cpp
    fi

    mv build/${lammps}/src/lmp_mpi_chimes exe/
fi

# Finish

loc=`pwd`
echo ""
echo "Compilation complete. "
if [ "$DO_BOTH" -eq 1 ]; then
    echo "Generated ChIMES executables:"
    echo "1. ${loc}/exe/lmp_mpi_chimes_tabulation"
    echo "2. ${loc}/exe/lmp_mpi_chimes_tabulation_prof"
elif [ -n "$TAB_FLAG" ]; then
    echo "Generated LAMMPS executable with ChIMES TABULATION support:"
    echo "${loc}/exe/lmp_mpi_chimes"
elif [ -n "$FINGERPRINT_FLAG" ]; then
    echo "Generated LAMMPS executable with ChIMES FINGERPRINT support:"
    echo "${loc}/exe/lmp_mpi_chimes"
else
    echo "Generated LAMMPS executable with basic ChIMES support (no extra flags):"
    echo "${loc}/exe/lmp_mpi_chimes"
fi
echo "See ${loc}/tests for usage examples"
echo ""