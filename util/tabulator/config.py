"""

    Configuration file for the ChIMES potential energy surface generator (pes_generator.py)

	Don't forget to set PAIR CHEBYSHEV PENALTY SCALING to zero in the parameter file.

"""

CHMS_REPO  = "/work2/09981/btsund/stampede3/codes/New_Lammps-test/"

PARAM_FILE = "/scratch/09981/btsund/VAST/Model_Dev/Funct_test/HSE06_25/params.txt"

PAIRTYPES  = [0] # Pair type index for scans, i.e. number after "PAIRTYPE PARAMS:" in parameter file
PAIRSTART  = [1.55] # Smallest distance for scan
PAIRSTOP   = [7.0] # Largest distance for scan
PAIRSTEP   = [0.001] # Step size for scan
TRIPTYPES  = [0] # Triplet type index for scans, i.e. number after "TRIPLETTYPE PARAMS:" in parameter file
TRIPSTART  = [1.55] # Smallest distance for scan:
TRIPSTOP   = [6] # Largest distance for scan
TRIPSTEP   = [0.05] # Step size for scan
QUADTYPES  = [0] # Triplet type index for scans, i.e. number after "TRIPLETTYPE PARAMS:" in parameter file
QUADSTART  = [1.6] # Smallest distance for scan:
QUADSTOP   = [5.0] # Largest distance for scan
QUADSTEP   = [0.1] # Step size for scan
QUAD_CONTRACT_DIMS = [(0,1,2)]
QUAD_RETAIN_DIMS   = [(3,4,5)]