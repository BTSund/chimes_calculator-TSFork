"""

    Configuration file for the ChIMES potential energy surface generator (pes_generator.py)

 Don't forget to set PAIR CHEBYSHEV PENALTY SCALING to zero in the parameter file.

"""
CP_N_WORKERS=112

PARAM_FILE = "/scratch/09982/aalmohri/SetupTab/params.txt.reduced"

QUADTYPES  = [3,4,7,8,11,12] # Triplet type index for scans, i.e. number after "TRIPLETTYPE PARAMS:" in parameter file
QUADSTART = [
    0.750,  # HHHOHOHOHOOO: HH HO HO HO HO OO
    0.750,  # HHHOHSiHOHSiOSi: HH HO HSi HO HSi OSi
    0.750,  # HOHOHSiOOOSiOSi: HO HO HSi OO OSi OSi
    0.750,  # HOHSiHSiOSiOSiSiSi: HO HSi HSi OSi OSi SiSi
    1.380,  # OOOOOSiOOOSiOSi: OO OO OSi OO OSi OSi
    1.380,  # OOOSiOSiOSiOSiSiSi: OO OSi OSi OSi OSi SiSi
]
QUADSTOP   = [4,4,4,4,4,4] # Largest distance for scan
QUADSTEP   = [0.001]*6 # Step size for scan

CP_N_ITER = 250000
CP_N_INIT = 2
CP_TOL = 1.0e-21
CP_RIDGE = 0
CP_SEED = 12346
CP_VERBOSE = True
CP_WRITE_FACTORS_NPZ = True

CP_TARGET_REL_ERROR = 1.0e-14

CP_RANK_START = 100
CP_RANK_STEP = 50
CP_RANK_MAX = 500
 