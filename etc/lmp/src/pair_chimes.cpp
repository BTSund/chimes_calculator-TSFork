/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Rebecca K. Lindsey (LLNL)
------------------------------------------------------------------------- */


#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "my_page.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "pair_chimes.h"
#include "group.h"
#include "update.h" // Needed for mb neighlist updates and info printing for fitting
#include "output.h" // Needed for infor printing for fitting -- dump 1 must be the "main" dump file used for fitting
#include  "utils.h" // Needed for infor printing for fitting
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   Optional profiling (-DCHIMES_PROFILE [-DCHIMES_PROFILE_EVERY=N])

   Exclusive "lap" timing: a single stopwatch runs from the first line of
   compute() to its last line. Every CHIMES_PROF_LAP(id) charges the time
   elapsed since the previous lap to section `id` and restarts the watch.
   Sections therefore never overlap and never leave gaps, so they add up to
   exactly 100% of the time spent in compute() (loop control, type checks,
   copies, allocations and destructors are charged to the section whose
   lap follows them). The report prints the independently measured
   compute() wall time and the (zero) unaccounted remainder as a check.

   Percentages use the rank-averaged times, which sum to the averaged
   total. max_rank and the imbalance column (max/avg) show load balance.
------------------------------------------------------------------------- */

#ifdef CHIMES_PROFILE

#ifndef CHIMES_PROFILE_EVERY
#define CHIMES_PROFILE_EVERY 100
#endif

enum ChimesProfID
{
    PROF_SETUP = 0,          // ev_setup, locals, scratch allocation
    PROF_MB_BUILD,           // many-body cluster graph build (+ its log output)

    PROF_1B,
    PROF_2B_DIST,            // 2B neighbor loop control + distance
    PROF_2B_EVAL,
    PROF_2B_FORCE_TALLY,
    PROF_IO,                 // fingerprint / badness file output

    PROF_3B_DIST,            // 3-mer loop control + distances
    PROF_3B_EVAL,
    PROF_3B_FORCE_TALLY,

    PROF_4B_TRIP_DIST,       // reuse only: shared-triplet distances, once per 4-mer group
    PROF_4B_DIST,            // 4-mer loop control + distances (reuse: il, jl, kl only)
    PROF_4B_PREPARE,         // subgraph reuse preparation (CHIMES_SUBGRAPH_REUSE)
    PROF_4B_EVAL_REUSE,
    PROF_4B_EVAL_CP_DIRECT,
    PROF_4B_EVAL_CP_TAB,
    PROF_4B_EVAL_SVD4,
    PROF_4B_EVAL_SVD3,
    PROF_4B_EVAL_DIRECT,
    PROF_4B_FORCE_TALLY,

    PROF_FDOTR,
    PROF_OTHER,              // function exit: scratch destructors, etc.

    PROF_N
};

static const char *chimes_prof_name[PROF_N] =
{
    "setup",
    "mb_build",

    "1B_energy",
    "2B_distance",
    "2B_eval",
    "2B_force_tally",
    "io_fingerprint_badness",

    "3B_distance",
    "3B_eval",
    "3B_force_tally",

    "4B_triplet_dist",
    "4B_distance",
    "4B_prepare_reuse",
    "4B_eval_reuse",
    "4B_eval_cp_direct",
    "4B_eval_cp_tab",
    "4B_eval_svd4",
    "4B_eval_svd3",
    "4B_eval_direct",
    "4B_force_tally",

    "virial_fdotr",
    "other_exit"
};

// Group each section for the summary block
enum ChimesProfGroup { PG_OVERHEAD = 0, PG_NEIGH, PG_1B2B, PG_3B, PG_4B, PG_N };

static const char *chimes_prof_group_name[PG_N] =
{
    "overhead (setup/io/virial/exit)",
    "many-body neighbor list build",
    "1-body + 2-body",
    "3-body",
    "4-body"
};

static const int chimes_prof_group[PROF_N] =
{
    PG_OVERHEAD, PG_NEIGH,
    PG_1B2B, PG_1B2B, PG_1B2B, PG_1B2B, PG_OVERHEAD,
    PG_3B, PG_3B, PG_3B,
    PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B,
    PG_OVERHEAD, PG_OVERHEAD
};

// 1 = evaluation kernel (the actual ChIMES energy/force math), 0 = bookkeeping
static const int chimes_prof_kernel[PROF_N] =
{
    0, 0,
    1, 0, 1, 0, 0,
    0, 1, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0
};

struct ChimesProfData
{
    double t[PROF_N];
    double n[PROF_N];

    double wall;    // independently measured compute() wall time
    double ncalls;  // compute() calls since the last report
    double nlaps;   // number of timer reads since the last report
    double last;    // stopwatch
    double start;

    ChimesProfData()
    {
        reset();
        last = start = 0.0;
    }

    void reset()
    {
        for (int i = 0; i < PROF_N; i++)
        {
            t[i] = 0.0;
            n[i] = 0.0;
        }
        wall   = 0.0;
        ncalls = 0.0;
        nlaps  = 0.0;
    }

    inline void begin()
    {
        start = last = MPI_Wtime();
    }

    inline void lap(int id)
    {
        const double now = MPI_Wtime();
        t[id] += now - last;
        n[id] += 1.0;
        nlaps += 1.0;
        last   = now;
    }

    inline void end(int id)
    {
        lap(id);
        wall   += last - start;
        ncalls += 1.0;
    }
};

static ChimesProfData chimes_prof;

#define CHIMES_PROF_BEGIN()   chimes_prof.begin()
#define CHIMES_PROF_LAP(id)   chimes_prof.lap(id)
#define CHIMES_PROF_END(id)   chimes_prof.end(id)

static void chimes_profile_report(
    MPI_Comm world,
    int me,
    long long timestep
)
{
    int nprocs = 1;
    MPI_Comm_size(world, &nprocs);

    // Measure the cost of one timer read on this rank (used for the overhead estimate)
    double tick_cost;
    {
        const int NT = 20000;
        double t0 = MPI_Wtime(), dummy = 0.0;
        for (int it = 0; it < NT; it++)
            dummy += MPI_Wtime();
        tick_cost = (MPI_Wtime() - t0) / NT;
        if (dummy < 0.0) tick_cost = 0.0;   // keep the loop from being optimized away
    }

    // Pack: [0,PROF_N) times, [PROF_N] wall, [PROF_N+1] estimated timer overhead
    double local_t[PROF_N+2];
    double local_n[PROF_N];

    for (int i = 0; i < PROF_N; i++)
    {
        local_t[i] = chimes_prof.t[i];
        local_n[i] = chimes_prof.n[i];
    }
    local_t[PROF_N]   = chimes_prof.wall;
    local_t[PROF_N+1] = chimes_prof.nlaps * tick_cost;

    double sum_t[PROF_N+2];
    double max_t[PROF_N+2];
    double sum_n[PROF_N];

    MPI_Reduce(local_t, sum_t, PROF_N+2, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(local_t, max_t, PROF_N+2, MPI_DOUBLE, MPI_MAX, 0, world);
    MPI_Reduce(local_n, sum_n, PROF_N,   MPI_DOUBLE, MPI_SUM, 0, world);

    if (me == 0)
    {
        const double avg_wall = sum_t[PROF_N] / (double)nprocs;
        const double max_wall = max_t[PROF_N];

        double avg_sum = 0.0;
        for (int i = 0; i < PROF_N; i++)
            avg_sum += sum_t[i] / (double)nprocs;

        const double denom = (avg_sum > 0.0) ? avg_sum : 1.0;

        double group_avg[PG_N] = {0.0, 0.0, 0.0, 0.0, 0.0};
        double group_ker[PG_N] = {0.0, 0.0, 0.0, 0.0, 0.0};
        double kernel_avg = 0.0, kernel_max_sum = 0.0;

        std::ios_base::fmtflags old_flags = std::cout.flags();
        std::streamsize         old_prec  = std::cout.precision();

        const std::string rule(104, '-');
        const std::string dbl (104, '=');

        std::cout << std::endl << dbl << std::endl;
        std::cout << "CHIMES_PROFILE timestep " << timestep
                  << " : exclusive timing of PairCHIMES::compute() over the last "
                  << (long long) chimes_prof.ncalls << " call(s), " << nprocs << " rank(s)" << std::endl;
        std::cout << "Sections do not overlap and sum to 100% of compute(). % uses rank-averaged time;"
                  << " imbal = max_rank/avg_rank." << std::endl;
        std::cout << "Rows marked * are evaluation kernels (ChIMES energy/force math); all others are bookkeeping." << std::endl;
        std::cout << rule << std::endl;

        std::cout
            << std::left  << std::setw(26) << "section" << std::right
            << std::setw(15) << "avg_rank(s)"
            << std::setw(15) << "max_rank(s)"
            << std::setw(10) << "%total"
            << std::setw(8)  << "imbal"
            << std::setw(15) << "calls_sum"
            << std::setw(15) << "us_per_call"
            << std::endl;

        std::cout << rule << std::endl;

        for (int i = 0; i < PROF_N; i++)
        {
            const double avg_rank = sum_t[i] / (double)nprocs;

            group_avg[chimes_prof_group[i]] += avg_rank;
            if (chimes_prof_kernel[i])
            {
                group_ker[chimes_prof_group[i]] += avg_rank;
                kernel_avg     += avg_rank;
                kernel_max_sum += max_t[i];
            }

            if (sum_n[i] <= 0.0 && sum_t[i] <= 0.0)
                continue;

            const double pct   = 100.0 * avg_rank / denom;
            const double imbal = (avg_rank > 0.0) ? max_t[i] / avg_rank : 0.0;
            const double us_per_call = (sum_n[i] > 0.0) ? 1.0e6 * sum_t[i] / sum_n[i] : 0.0;

            std::cout
                << std::left  << std::setw(26) << (std::string(chimes_prof_kernel[i] ? "* " : "  ") + chimes_prof_name[i]) << std::right
                << std::setw(15) << std::scientific << std::setprecision(5) << avg_rank
                << std::setw(15) << std::scientific << std::setprecision(5) << max_t[i]
                << std::setw(10) << std::fixed      << std::setprecision(2) << pct
                << std::setw(8)  << std::fixed      << std::setprecision(2) << imbal
                << std::setw(15) << std::fixed      << std::setprecision(0) << sum_n[i]
                << std::setw(15) << std::scientific << std::setprecision(4) << us_per_call
                << std::endl;
        }

        std::cout << rule << std::endl;
        std::cout
            << std::left  << std::setw(26) << "SUM of sections" << std::right
            << std::setw(15) << std::scientific << std::setprecision(5) << avg_sum
            << std::setw(15) << " "
            << std::setw(10) << std::fixed << std::setprecision(2) << 100.0 * avg_sum / denom
            << std::endl;
        std::cout
            << std::left  << std::setw(26) << "compute() wall" << std::right
            << std::setw(15) << std::scientific << std::setprecision(5) << avg_wall
            << std::setw(15) << std::scientific << std::setprecision(5) << max_wall
            << std::endl;
        std::cout
            << std::left  << std::setw(26) << "unaccounted" << std::right
            << std::setw(15) << std::scientific << std::setprecision(5) << (avg_wall - avg_sum)
            << std::endl;

        std::cout << rule << std::endl;
        std::cout << "Group summary (avg_rank s, % of compute)    total                * eval kernels         bookkeeping" << std::endl;

        for (int g = 0; g < PG_N; g++)
        {
            std::cout
                << "  " << std::left << std::setw(34) << chimes_prof_group_name[g] << std::right
                << std::setw(13) << std::scientific << std::setprecision(4) << group_avg[g]
                << std::setw(8)  << std::fixed      << std::setprecision(2) << 100.0 * group_avg[g] / denom << "%"
                << std::setw(13) << std::scientific << std::setprecision(4) << group_ker[g]
                << std::setw(8)  << std::fixed      << std::setprecision(2) << 100.0 * group_ker[g] / denom << "%"
                << std::setw(13) << std::scientific << std::setprecision(4) << (group_avg[g] - group_ker[g])
                << std::setw(8)  << std::fixed      << std::setprecision(2) << 100.0 * (group_avg[g] - group_ker[g]) / denom << "%"
                << std::endl;
        }

        std::cout << rule << std::endl;
        std::cout
            << "  " << std::left << std::setw(34) << "ALL * eval kernels (compute)" << std::right
            << std::setw(13) << std::scientific << std::setprecision(4) << kernel_avg
            << std::setw(8)  << std::fixed      << std::setprecision(2) << 100.0 * kernel_avg / denom << "%"
            << std::endl;
        std::cout
            << "  " << std::left << std::setw(34) << "ALL bookkeeping" << std::right
            << std::setw(13) << std::scientific << std::setprecision(4) << (avg_sum - kernel_avg)
            << std::setw(8)  << std::fixed      << std::setprecision(2) << 100.0 * (avg_sum - kernel_avg) / denom << "%"
            << std::endl;

        const double avg_ovh = sum_t[PROF_N+1] / (double)nprocs;
        std::cout
            << "  " << std::left << std::setw(34) << "est. profiler timer overhead" << std::right
            << std::setw(13) << std::scientific << std::setprecision(4) << avg_ovh
            << std::setw(8)  << std::fixed      << std::setprecision(2) << 100.0 * avg_ovh / denom << "%"
            << "   (already inside the rows above, mostly the fine bookkeeping rows)"
            << std::endl;

        std::cout << dbl << std::endl << std::endl;

        std::cout.flags(old_flags);
        std::cout.precision(old_prec);
    }

    chimes_prof.reset();
}

#else

#define CHIMES_PROF_BEGIN()   do { } while (0)
#define CHIMES_PROF_LAP(id)   do { } while (0)
#define CHIMES_PROF_END(id)   do { } while (0)

#endif

/*	Functions required by LAMMPS:


settings 	(done)		reads the input script line with arguments defined here
coeff		(done)		set coefficients for one i,j pair type
compute		(done)		workhorse routine that computes pairwise interactions
init_one	(done)		perform initalization for one i,j type pair
init_style 	(done)		initialization specific to this pair style

write_restart			write i,j pair coeffs to restart file
read_restart			read i,j pair coeffs from restart file
write_restart_settings	write global settings to restart file
read_restart_settings	read global settings from restart file
single				    force and energy fo a single pairwise interaction between two atoms
*/




PairCHIMES::PairCHIMES(LAMMPS *lmp) : Pair(lmp)
{
	restartinfo = 0;

	int me = comm->me;
	MPI_Comm_rank(world,&me);

	chimes_calculator.init(me);
    for_fitting = false;
    fingerprint = false;
    IO_freq     = 1;

    maxcut_3b = 0.0;
    maxcut_4b = 0.0;

#ifdef CHIMES_SUBGRAPH_REUSE
    reuse_epoch = 0;
#endif

	// 2, 3, and 4-body vars for chimesFF access

	dr     .resize(CHDIM);
	dr_3b  .resize(3*CHDIM);
	dr_4b  .resize(6*CHDIM);

	dist_3b.resize(3);
	dist_4b.resize(6);

	// CHDIM is the number of spatial dimensions (usually 3).
	force_2b.resize(2*CHDIM);
	force_3b.resize(3*CHDIM) ;
	force_4b.resize(4*CHDIM) ;

	typ_idxs_2b.resize(2);
	typ_idxs_3b.resize(3);
	typ_idxs_4b.resize(4);

	// Vars for neighlist construction

	tmp_3mer.resize(3);
	tmp_4mer.resize(4);

	if (chimes_calculator.rank == 0)
	{
		std::cout << std::endl;
		std::cout << "************************* WARNING (pair_style chimesFF) ************************" << std::endl;
		std::cout << "Assuming n-body interactions have longer cutoffs than all (n+1)-body interactions" << std::endl;
		std::cout << "************************* WARNING (pair_style chimesFF) ************************" << std::endl;
		std::cout << std::endl;
#ifdef CHIMES_SUBGRAPH_REUSE
		std::cout << "pair_style chimesFF: 4-body subgraph reuse ENABLED (CHIMES_SUBGRAPH_REUSE)" << std::endl;
#else
		std::cout << "pair_style chimesFF: 4-body subgraph reuse disabled (standard 4-body evaluation)" << std::endl;
#endif
	}

}

PairCHIMES::~PairCHIMES()
{
	if (allocated)
	{
	    memory->destroy(setflag);
	    memory->destroy(cutsq);
	}

    if (badness_stream.is_open())
        badness_stream.close();
}

void PairCHIMES::settings(int narg, char **arg)
{
	if (narg > 3)
		error -> all(FLERR,"Illegal pair_style command. Expects no more than three arguments (string: fitting or fingerprint and iofreq)");

    if (narg == 1)
    {
        if (utils::strmatch(arg[0],"fitting"))
        {
            for_fitting   = true;
            stringstream ss;
            ss << chimes_calculator.rank;
            badness_stream.open("rank-" + ss.str() + ".badness.log");
        }
    }
	#ifdef FINGERPRINT
    if (narg == 2)
    {
        if (utils::strmatch(arg[0],"fingerprint"))
        {

            fingerprint   = true;
			IO_freq = std::stoi(arg[1]);;
        }
    }

    if (narg == 3)
    {
        if (utils::strmatch(arg[0],"fingerprint"))
        {

            fingerprint   = true;
			IO_freq = std::stoi(arg[1]);;
        }
        if (utils::strmatch(arg[1],"fingerprint"))
        {

            fingerprint   = true;
			IO_freq = std::stoi(arg[2]);;
        }
        if (utils::strmatch(arg[0],"fitting") || utils::strmatch(arg[1],"fitting") || utils::strmatch(arg[2],"fitting"))
        {
            for_fitting   = true;
            stringstream ss;
            ss << chimes_calculator.rank;
            badness_stream.open("rank-" + ss.str() + ".badness.log");
        }
    }
	#endif

	return;
}

void PairCHIMES::coeff(int narg, char **arg)
{
	// Expect: pair_coeff * * <parameter file name>

	if (narg != 3)
		error -> all(FLERR,"Illegal pair_style command. Expects \"pair_coeff * * <parameter file name>\" ");

	chimesFF_paramfile = arg[2];

	chimes_calculator.read_parameters(chimesFF_paramfile);

	set_chimes_type();

    //chimes_calculator.set_atomtypes(chimes_type);
    chimes_calculator.build_pair_int_trip_map() ;
    chimes_calculator.build_pair_int_quad_map() ;

	// Set special LAMMPS flags/cutoffs

	if (!allocated)
		allocate();

	vector<vector<double> > cutoff_2b;
	chimes_calculator.get_cutoff_2B(cutoff_2b);

	for(int i=1; i<=atom->ntypes; i++)
	{
		for(int j=i; j<=atom->ntypes; j++)
		{
			setflag[i][j] = 1;
			setflag[j][i] = 1;

			if (chimes_type[i-1] < 0 || chimes_type[j-1] < 0)
			{
				cutsq[i][j] = 0.0;
				if (i!=j)
					cutsq[j][i] = 0.0;
				continue;
			}

			cutsq[i][j]  = cutoff_2b[ chimes_calculator.get_atom_pair_index( chimes_type[i-1]*chimes_calculator.natmtyps + chimes_type[j-1] ) ][1];
			cutsq[i][j] *= cutsq[i][j];

			if (i!=j)
			{
				cutsq[j][i]  = cutoff_2b[ chimes_calculator.get_atom_pair_index( chimes_type[j-1]*chimes_calculator.natmtyps + chimes_type[i-1]) ][1];
				cutsq[j][i] *= cutsq[j][i];
			}
		}
	}

	maxcut_3b = chimes_calculator.max_cutoff_3B();
	maxcut_4b = chimes_calculator.max_cutoff_4B();

	if (maxcut_3b > 0.0 && maxcut_4b > maxcut_3b && comm->me == 0)
		std::cout << "pair_style chimesFF: NOTE: 4B cutoff exceeds 3B cutoff; the many-body cluster list handles this case." << std::endl;

#ifdef CHIMES_SUBGRAPH_REUSE
	const int nct = chimes_calculator.natmtyps;
	reuse_pool .assign(nct, chimesFF::chimes4BTripletReuseTmp());
	reuse_stamp.assign(nct, -1);
	reuse_state.assign(nct, 0);
	reuse_epoch = 0;
#endif
}

void writeClusterDataComp(const string& filename, const vector<vector<double>>& data)
{
    ofstream ofs(filename);
    if (!ofs) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    ostringstream buffer;

    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); j++) {
            buffer << row[j];
            if (j < row.size() - 1) buffer << " ";  // Add space between elements
        }
        buffer << "\n";
    }

    ofs << buffer.str(); // Single large write operation
    ofs.close();
}

void PairCHIMES::allocate()
{
	allocated = 1;

	memory->create(setflag,atom->ntypes+1,atom->ntypes+1,"pair:setflag");

	for(int i=1; i<=atom->ntypes; i++)
		for(int j=i; j<=atom->ntypes; j++)
			setflag[i][j] = 0;

	memory->create(cutsq,atom->ntypes+1,atom->ntypes+1,"pair:cutsq");
}

void PairCHIMES::init_style()
{
	if (atom -> tag_enable == 0)
		error -> all(FLERR,"Pair style ChIMES requires atom IDs");

	if (force->newton_pair == 0)
		error->all(FLERR,"Pair style ChIMES requires newton pair on");

	// Need a full neighbor list, including neighbors of ghosts (as in pair_airebo)

    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
}

double PairCHIMES::init_one(int i, int j)
{
	// Sets the cutoff for each pair interaction.
	// The maximum of the returned values are used to set outer cutoff for neighbor lists
	// WARNING: This means linking won't work properly if 2-b interactions do not have larger cutoffs than all other
	// higher bodied interactions!!

	if (setflag[i][j] == 0)
		error->all(FLERR,"All pair coeffs are not set");

	return sqrt(cutsq[i][j]);
}

inline double PairCHIMES::get_dist(int i, int j, double *dr)
{
	double 	**x    = atom -> x;	// Access to system coordinates

	dr[0] = x[j][0] - x[i][0];
	dr[1] = x[j][1] - x[i][1];
	dr[2] = x[j][2] - x[i][2];

	return sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
}

inline double PairCHIMES::get_dist(int i, int j)
{
	double dummy_dr[3] ;

	return get_dist(i,j, dummy_dr);
}

/* ----------------------------------------------------------------------
   Build the ChIMES many-body neighbor lists.

   This is the original ChIMES/LAMMPS construction, unchanged in logic:
     * i loops over local atoms, j and k over i's full LAMMPS neighbor list,
       with the tag ordering  jtag >= itag, ktag >= itag, ktag >= jtag,
     * a 3-mer is stored when all three distances are < maxcut_3b + skin,
     * a 4-mer (i,j,k,l) is stored when all six distances are < maxcut_4b + skin,
       with l scanned from i's full neighbor list and ltag >= itag,jtag,ktag.

   The only addition is bookkeeping: because the l-loop is nested inside the
   (i,j,k) loop, all 4-mers sharing a triplet are already contiguous in
   neighborlist_4mers. Their [begin,end) range is recorded in
   neighborlist_4mer_groups so that subgraph reuse can process one shared
   triplet at a time. Nothing about which clusters are found is changed.
------------------------------------------------------------------------- */

void PairCHIMES::build_mb_neighlists()
{
	neighborlist_3mers.clear();
	neighborlist_4mers.clear();
	neighborlist_4mer_groups.clear();

	if ( (chimes_calculator.poly_orders[1] == 0) &&  (chimes_calculator.poly_orders[2] == 0)
		#ifdef FINGERPRINT
			&& (!fingerprint)
		#endif
		)
		return;

	// List gets built based on atoms owned by calling proc.

	int i,j,k,l,inum,jnum,knum,lnum, ii, jj, kk, ll;		 // Local iterator vars
	int *ilist,*jlist,*klist,*llist, *numneigh,**firstneigh; // Local neighborlist vars
	tagint 	*tag   = atom -> tag;					         // Access to global atom indices
	tagint  itag, jtag, ktag, ltag;					         // holds tags
	int     *type  = atom -> type;					         // Access to system atom types

	double maxcut_3b_padded = maxcut_3b + neighbor-> skin;
	double maxcut_4b_padded = maxcut_4b + neighbor-> skin;

	double dist_ij, dist_ik, dist_il, dist_jk, dist_jl, dist_kl;

	////////////////////////////////////////
	// Access to neighbor list vars
	////////////////////////////////////////

	inum       = list -> inum; 		 // length of the list
	ilist      = list -> ilist; 	 // list of i atoms for which neighbor list exists
	numneigh   = list -> numneigh;	 // length of each of the ilist neighbor lists
	firstneigh = list -> firstneigh; // point to the list of neighbors of i

	for (ii = 0; ii < inum; ii++) // Loop over real atoms (ai)
	{
		i     = ilist[ii];
		itag  = tag[i];
		jlist = firstneigh[i];
		jnum  = numneigh[i];

		if (chimes_type[type[i]-1] < 0)
			continue;

		for (jj = 0; jj < jnum; jj++)
		{
			j     = jlist[jj];
			j    &= NEIGHMASK;
			jtag  = tag[j];

			if (chimes_type[type[j]-1] < 0)
				continue;

			if (j == i)
				continue;
			if (jtag < itag)
				continue;

			// Check ij distance

			dist_ij = get_dist(i,j);

			if ( (dist_ij >= maxcut_3b_padded) && (dist_ij >= maxcut_4b_padded) )
				continue;

			klist = firstneigh[i];	// ChIMES assumes all atoms must be within cutoff of eachother for a valid interaction
			knum  = numneigh[i];

			for (kk = 0; kk < knum; kk++)
			{
				k     = klist[kk];
				k    &= NEIGHMASK;
				ktag  = tag[k];

				if (chimes_type[type[k]-1] < 0)
					continue;

				if ( (k==i) || (k==j) )
					continue;
				if ( (ktag < itag) || (ktag < jtag) )
					continue;

				// Check ik distance

				dist_ik = get_dist(i,k);

				if ( (dist_ik >= maxcut_3b_padded) && (dist_ik >= maxcut_4b_padded) )
					continue;

				// Check jk distance

				dist_jk = get_dist(j,k);

				if( (dist_ij < maxcut_3b_padded) &&  (dist_ik < maxcut_3b_padded) && (dist_jk < maxcut_3b_padded) )
				{
					// If we're here and valid_3mer == true, then add the triplet to the chimes neigh list

					tmp_3mer[0] = i;
					tmp_3mer[1] = j;
					tmp_3mer[2] = k;

					neighborlist_3mers.push_back(tmp_3mer);
				}

				if ((dist_ij >= maxcut_4b_padded) || (dist_ik >= maxcut_4b_padded) || (dist_jk >= maxcut_4b_padded) )
					continue;

				// Now decide if we should continue on to 4-body neighbor list construction

				if (chimes_calculator.poly_orders[2] == 0
					#ifdef FINGERPRINT
						&& !fingerprint
					#endif
					)
					continue;

				const int group_begin = (int) neighborlist_4mers.size();

				llist = firstneigh[i];
				lnum  = numneigh[i];

				for (ll = 0; ll < lnum; ll++)
				{
					l     = llist[ll];
					l    &= NEIGHMASK;
					ltag  = tag[l];

					if (chimes_type[type[l]-1] < 0)
						continue;

					if ( (l==i) || (l==j) || (l==k))
						continue;
					if ((ltag < itag) ||(ltag < jtag)||(ltag < ktag))
						continue;

					// Check il distance

					dist_il = get_dist(i,l);

					if (dist_il >= maxcut_4b_padded)
						continue;

					// Check jl distance

					dist_jl = get_dist(j,l);

					if (dist_jl >= maxcut_4b_padded)
						continue;

					// Check kl distance

					dist_kl = get_dist(k,l);

					if (dist_kl >= maxcut_4b_padded)
						continue;

					// If we're here and valid_4mer == true, then add the quadruplet to the chimes neigh list

					tmp_4mer[0] = i;
					tmp_4mer[1] = j;
					tmp_4mer[2] = k;
					tmp_4mer[3] = l;

					neighborlist_4mers.push_back(tmp_4mer);
				}

				// Record the shared-triplet group (bookkeeping only)
				const int group_end = (int) neighborlist_4mers.size();

				if (group_end > group_begin)
					neighborlist_4mer_groups.push_back(std::make_pair(group_begin, group_end));
			}
		}
	}
}

/* ----------------------------------------------------------------------
   Standard (non-reuse) 4-body evaluator chain. Operates on the member
   dist_4b / dr_4b / typ_idxs_4b / force_4b buffers.

   Precedence: cp_direct > cp > svd4x2 > svd3x3 > direct polynomial
------------------------------------------------------------------------- */

void PairCHIMES::eval_4b_standard(std::vector<double> & stensor, chimes4BTmp & tmp, double & energy)
{
#ifdef TABULATION
	if (chimes_calculator.tabulate_4B_cp_direct)
	{
		chimes_calculator.compute_4B_cp_direct(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, tmp);
		CHIMES_PROF_LAP(PROF_4B_EVAL_CP_DIRECT);
		return;
	}

	if (chimes_calculator.tabulate_4B_cp)
	{
		chimes_calculator.compute_4B_cp_tab(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, tmp);
		CHIMES_PROF_LAP(PROF_4B_EVAL_CP_TAB);
		return;
	}

	if (chimes_calculator.tabulate_4B_svd4x2)
	{
		chimes_calculator.compute_4B_svd4x2_tab(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, tmp);
		CHIMES_PROF_LAP(PROF_4B_EVAL_SVD4);
		return;
	}

	if (chimes_calculator.tabulate_4B_svd3x3)
	{
		chimes_calculator.compute_4B_svd3x3_tab(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, tmp);
		CHIMES_PROF_LAP(PROF_4B_EVAL_SVD3);
		return;
	}
#endif

	{
		chimes_calculator.compute_4B(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, tmp);
		CHIMES_PROF_LAP(PROF_4B_EVAL_DIRECT);
	}
}

/* ----------------------------------------------------------------------
   compute(): thin wrapper so that profiling brackets *everything*, including
   the destruction of the scratch objects created inside compute_body().
------------------------------------------------------------------------- */

void PairCHIMES::compute(int eflag, int vflag)
{
	CHIMES_PROF_BEGIN();

	compute_body(eflag, vflag);

#ifdef CHIMES_PROFILE
	CHIMES_PROF_END(PROF_OTHER);   // charges function exit / destructors, closes the stopwatch

	if (update->ntimestep % CHIMES_PROFILE_EVERY == 0)
	{
		chimes_profile_report(
			world,
			comm->me,
			static_cast<long long>(update->ntimestep)
		);
	}
#endif
}

void PairCHIMES::compute_body(int eflag, int vflag)
{
	// Vars for access to chimesFF compute_XB functions

	std::vector  <double>  stensor(6);	// pointers to system stress tensor

	// Temp vars to hold chimes output for passing to ev_tally function

    int                  atmidxlst[6][2];

	// General LAMMPS compute vars

	int 	i,j,k,l,inum,jnum, ii, jj;	// Local iterator vars
	int 	*ilist,*jlist, *numneigh,**firstneigh;	// Local neighborlist vars
	int     idx;

	double 	**f    = atom -> f;		    // Access to system forces

	int 	*type  = atom -> type;		// Acces to system atom types (countng starts from 1, chimesFF class expects counting from 0!)
	tagint 	*tag   = atom -> tag;		// Access to global atom indices (sort of like "parent" indices)
	tagint  itag, jtag;			        // holds tags
	double  energy;				        // pair energy

	int me = comm->me;
	MPI_Comm_rank(world,&me);

	// Set up vars controlling if energy/pressure (virial) contributions are computed

	{

		if (eflag || vflag)
		{
			ev_setup(eflag, vflag);
		}
		else
		{
			evflag      = 0;
			vflag_fdotr = 0;
			vflag_atom  = 0;
		}
	}

	// Compile if fingerprinting desired
#ifdef FINGERPRINT
	std::vector<std::vector<double>> tmp_dist_2b;
    std::vector<std::vector<double>> tmp_dist_3b;
    std::vector<std::vector<double>> tmp_dist_4b;
	bool 				 tmp_FP = false;
	bool 				 valid_order;
	if (fingerprint){
		if(update->ntimestep % IO_freq == 0){tmp_FP = true;}
	}else{tmp_FP = false;}
#endif

	////////////////////////////////////////
	// Access to (2-body) neighbor list vars
	////////////////////////////////////////

	inum       = list -> inum; 		 // length of the list
	ilist      = list -> ilist; 	 // list of i atoms for which neighbor list exists
	numneigh   = list -> numneigh;	 // length of each of the ilist neighbor lists
	firstneigh = list -> firstneigh; // point to the list of neighbors of i


    chimes2BTmp chimes_2btmp(chimes_calculator.poly_orders[0]) ;
    chimes3BTmp chimes_3btmp(chimes_calculator.poly_orders[1]) ;
    chimes4BTmp chimes_4btmp(chimes_calculator.poly_orders[2]) ;

	CHIMES_PROF_LAP(PROF_SETUP);

	// Build the ChIMES many-body cluster graph.. only do so when LAMMPS neighborlist has been updated

	if (neighbor->ago == 0)
	{
		build_mb_neighlists();

		if (chimes_calculator.rank == 0)
		{
			std::cout << "Updating chimesFF neighbor lists..." << std::endl;
			std::cout << "  Rank " << me << " 3-body list size: " << neighborlist_3mers.size() << std::endl;
			std::cout << "  Rank " << me << " 4-body list size: " << neighborlist_4mers.size()
			          << " (" << neighborlist_4mer_groups.size() << " shared-triplet groups)" << std::endl;
			std::cout << "  ...update complete" << std::endl;
		}

		CHIMES_PROF_LAP(PROF_MB_BUILD);
	}

    // Prepare the badness variable

    chimes_calculator.reset_badness();

	////////////////////////////////////////
	// Compute 1- and 2-body interactions
	////////////////////////////////////////
	for (ii = 0; ii < inum; ii++)		// Loop over the atoms owned by the current process
	{
		i     = ilist[ii];				// Index of the current atom
		itag  = tag[i];					// Get i's global atom index (sort of like its "parent")

		if (chimes_type[type[i]-1] < 0)
			continue;

		jlist = firstneigh[i];			// Neighborlist for atom i
		jnum  = numneigh[i];			// Number of neighbors of atom i

		// First, get the single-atom energy contribution

		energy = 0.0;
		{
			chimes_calculator.compute_1B(chimes_type[type[i]-1], energy);
			CHIMES_PROF_LAP(PROF_1B);
		}

        atmidxlst[0][0] = i;

		if(evflag)
			ev_tally_mb(1, 0, atmidxlst, energy, stensor);

		// Now move on to two-body force, stress, and energy

		for (jj = 0; jj < jnum; jj++) // Loop over neighbors of i
		{
			j     = jlist[jj];			// Index of the jj atom
			j    &= NEIGHMASK;			// Strip possible extra bits of j
			jtag  = tag[j];				// Get j's global atom index (sort of like its "parent")

			if (chimes_type[type[j]-1] < 0)
				continue;


			if (jtag <= itag) // only allow calculation for j<i, since we've requested a full neighbor list
				continue;

			// Get distance using ghost atoms... don't need MIC since we're using ghost atoms
			{
				dist = get_dist(i, j, &dr[0]);
				CHIMES_PROF_LAP(PROF_2B_DIST);
			}

			typ_idxs_2b[0] = chimes_type[type[i]-1]; // Type (index) of the current atom... subtract 1 to account for chimesFF vs LAMMPS numbering convention
			typ_idxs_2b[1] = chimes_type[type[j]-1];

			std::fill(force_2b.begin(), force_2b.end(), 0.0) ;
			std::fill(stensor.begin(), stensor.end(), 0.0) ;

			energy = 0.0;

			{
#ifdef TABULATION
				if (chimes_calculator.tabulate_2B)
				{
					chimes_calculator.compute_2B_tab(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp);
				}
				else
#endif
				{
#ifdef FINGERPRINT
					valid_order = (i < j);

					if (tmp_FP && valid_order)
					{
						double tmp_force_scalar;

						chimes_calculator.compute_2B(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp,
						                             tmp_force_scalar, tmp_dist_2b, tmp_FP && valid_order);
					}
					else
#endif
					{
						chimes_calculator.compute_2B(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp);
					}
				}
				CHIMES_PROF_LAP(PROF_2B_EVAL);
			}

			{

				for (idx=0; idx<3; idx++)
				{
					f[i][idx] += force_2b[0*CHDIM+idx];
					f[j][idx] += force_2b[1*CHDIM+idx];
				}

				if (vflag_atom)
				{
					atmidxlst[0][0] = i;
					atmidxlst[0][1] = j;
				}

				if (evflag)
					ev_tally_mb(2, 1, atmidxlst, energy, stensor);
				CHIMES_PROF_LAP(PROF_2B_FORCE_TALLY);
			}
		} // end loop over j neighbors
	}     // end loop over i atoms

#ifdef FINGERPRINT
	std::string ts = std::to_string(update->ntimestep);
	if (tmp_FP)
	{
		std::stringstream filename;
		filename << ts << "." << std::to_string(chimes_calculator.rank) <<".2b_clusters.txt";
		writeClusterDataComp(filename.str(), tmp_dist_2b);
	}
#endif

	// Document badness for configuration: current timestep, current rank, worst badness seen by rank
	if (for_fitting && (update->ntimestep % output->every_dump[0] == 0))
		badness_stream << update->ntimestep << " " <<  chimes_calculator.get_badness() << endl;

	CHIMES_PROF_LAP(PROF_IO);

	////////////////////////////////////////
	// Compute 3-body interactions
	////////////////////////////////////////

	if (chimes_calculator.poly_orders[1] > 0)
	{
		const int n3 = (int) neighborlist_3mers.size();

		for (ii = 0; ii < n3; ii++)
		{
			i     = neighborlist_3mers[ii][0];
			j     = neighborlist_3mers[ii][1];
			k     = neighborlist_3mers[ii][2];

			dist_3b[0] = get_dist(i,j,&dr_3b[0*CHDIM]);
			dist_3b[1] = get_dist(i,k,&dr_3b[1*CHDIM]);
			dist_3b[2] = get_dist(j,k,&dr_3b[2*CHDIM]);

			CHIMES_PROF_LAP(PROF_3B_DIST);

			typ_idxs_3b[0] = chimes_type[type[i]-1];
			typ_idxs_3b[1] = chimes_type[type[j]-1];
			typ_idxs_3b[2] = chimes_type[type[k]-1];

			std::fill(force_3b.begin(), force_3b.end(), 0.0) ;
			std::fill(stensor.begin(), stensor.end(), 0.0) ;

			energy = 0.0 ;

#ifdef TABULATION
			if (chimes_calculator.tabulate_3B)
			{
				chimes_calculator.compute_3B_tab( dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);
			}
			else
#endif
			{
#ifdef FINGERPRINT
				valid_order = (tag[i] < tag[j] && tag[i] < tag[k] && tag[j] < tag[k]);
				if (tmp_FP && valid_order)
				{
					vector<double> tmp_force_scalar_3b(3);
					chimes_calculator.compute_3B( dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp, tmp_force_scalar_3b, tmp_dist_3b, tmp_FP && valid_order);
				}
				else
#endif
				{
					chimes_calculator.compute_3B( dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);
				}
			}

			CHIMES_PROF_LAP(PROF_3B_EVAL);

			for (idx=0; idx<3; idx++)
			{
				f[i][idx] += force_3b[0*CHDIM+idx] ;
				f[j][idx] += force_3b[1*CHDIM+idx] ;
				f[k][idx] += force_3b[2*CHDIM+idx] ;
			}

			if (vflag_atom)
			{
				atmidxlst[0][0] = i; atmidxlst[0][1] = j;
				atmidxlst[1][0] = i; atmidxlst[1][1] = k;
				atmidxlst[2][0] = j; atmidxlst[2][1] = k;
			}

			if (evflag)
				ev_tally_mb(3, 3, atmidxlst, energy, stensor);

			CHIMES_PROF_LAP(PROF_3B_FORCE_TALLY);
		}
	}

#ifdef FINGERPRINT
	if (tmp_FP)
	{
		std::stringstream filename_3b;
		filename_3b << ts << "." << std::to_string(chimes_calculator.rank) <<".3b_clusters.txt";
		writeClusterDataComp(filename_3b.str(), tmp_dist_3b);
	}
	CHIMES_PROF_LAP(PROF_IO);
#endif

	////////////////////////////////////////
	// Compute 4-body interactions
	////////////////////////////////////////

	if (chimes_calculator.poly_orders[2] > 0)
	{
#ifdef CHIMES_SUBGRAPH_REUSE
		/*
		 * Subgraph reuse: iterate over the shared-triplet groups recorded by
		 * build_mb_neighlists(). The triplet distances (ij, ik, jk) are computed
		 * once per group, the triplet-side factor is prepared once per fourth-atom
		 * type, and only the il, jl, kl distances are computed per quadruplet.
		 */
		std::vector<double> trip_dx(3);

		const int ngroups = (int) neighborlist_4mer_groups.size();

		for (int g = 0; g < ngroups; g++)
		{
			const int q_begin = neighborlist_4mer_groups[g].first;
			const int q_end   = neighborlist_4mer_groups[g].second;

			i = neighborlist_4mers[q_begin][0];
			j = neighborlist_4mers[q_begin][1];
			k = neighborlist_4mers[q_begin][2];

			// Triplet side -> runtime pair slots 0 = ij, 1 = ik, 3 = jk
			dist_4b[0] = get_dist(i,j,&dr_4b[0*CHDIM]);
			dist_4b[1] = get_dist(i,k,&dr_4b[1*CHDIM]);
			dist_4b[3] = get_dist(j,k,&dr_4b[3*CHDIM]);

			trip_dx[0] = dist_4b[0];
			trip_dx[1] = dist_4b[1];
			trip_dx[2] = dist_4b[3];

			typ_idxs_4b[0] = chimes_type[type[i]-1];
			typ_idxs_4b[1] = chimes_type[type[j]-1];
			typ_idxs_4b[2] = chimes_type[type[k]-1];

			// New triplet: invalidate every per-type reuse entry in O(1).
			reuse_epoch++;

			CHIMES_PROF_LAP(PROF_4B_TRIP_DIST);

			for (int q = q_begin; q < q_end; q++)
			{
				l = neighborlist_4mers[q][3];

				dist_4b[2] = get_dist(i,l,&dr_4b[2*CHDIM]);
				dist_4b[4] = get_dist(j,l,&dr_4b[4*CHDIM]);
				dist_4b[5] = get_dist(k,l,&dr_4b[5*CHDIM]);

				const int type_l = chimes_type[type[l]-1];
				typ_idxs_4b[3] = type_l;

				CHIMES_PROF_LAP(PROF_4B_DIST);

				/*
				 * Reuse entry for (this triplet, type_l). Prepared on first use;
				 * state: 0 = reuse, 1 = no interaction possible (skip), 2 = use
				 * the standard evaluator (reuse unsupported for this mode).
				 */
				chimesFF::chimes4BTripletReuseTmp & reuse = reuse_pool[type_l];

				if (reuse_stamp[type_l] != reuse_epoch)
				{
					reuse_stamp[type_l] = reuse_epoch;

					const bool ok = chimes_calculator.prepare_4B_triplet_reuse(trip_dx, typ_idxs_4b, chimes_4btmp, reuse);

					if (ok && reuse.valid)
						reuse_state[type_l] = 0;
					else if (reuse.triplet_out_of_range)
						reuse_state[type_l] = 1;
					else
						reuse_state[type_l] = 2;

					CHIMES_PROF_LAP(PROF_4B_PREPARE);
				}

				if (reuse_state[type_l] == 1)
					continue;   // triplet side outside the 4B cutoff: zero contribution

				std::fill(force_4b.begin(), force_4b.end(), 0.0);
				std::fill(stensor.begin(), stensor.end(), 0.0);

				energy = 0.0;

				if (reuse_state[type_l] == 0)
				{
					chimes_calculator.compute_4B_from_triplet_reuse(dist_4b, dr_4b, typ_idxs_4b, reuse,
					                                                force_4b, stensor, energy, chimes_4btmp);
					CHIMES_PROF_LAP(PROF_4B_EVAL_REUSE);
				}
				else
				{
					eval_4b_standard(stensor, chimes_4btmp, energy);   // laps its own evaluator row
				}

				for (idx = 0; idx < CHDIM; idx++)
				{
					f[i][idx] += force_4b[0*CHDIM + idx];
					f[j][idx] += force_4b[1*CHDIM + idx];
					f[k][idx] += force_4b[2*CHDIM + idx];
					f[l][idx] += force_4b[3*CHDIM + idx];
				}

				if (vflag_atom)
				{
					atmidxlst[0][0] = i; atmidxlst[0][1] = j;
					atmidxlst[1][0] = i; atmidxlst[1][1] = k;
					atmidxlst[2][0] = i; atmidxlst[2][1] = l;
					atmidxlst[3][0] = j; atmidxlst[3][1] = k;
					atmidxlst[4][0] = j; atmidxlst[4][1] = l;
					atmidxlst[5][0] = k; atmidxlst[5][1] = l;
				}

				if (evflag)
					ev_tally_mb(4, 6, atmidxlst, energy, stensor);

				CHIMES_PROF_LAP(PROF_4B_FORCE_TALLY);
			} // end loop over quadruplets of this triplet group
		}     // end loop over triplet groups
#else
		/*
		 * Standard path: identical loop to the original ChIMES implementation,
		 * one independent evaluation per stored 4-mer.
		 */
		const int n4 = (int) neighborlist_4mers.size();

		for (ii = 0; ii < n4; ii++)
		{
			i     = neighborlist_4mers[ii][0];
			j     = neighborlist_4mers[ii][1];
			k     = neighborlist_4mers[ii][2];
			l     = neighborlist_4mers[ii][3];

			dist_4b[0] = get_dist(i,j,&dr_4b[0*CHDIM]);
			dist_4b[1] = get_dist(i,k,&dr_4b[1*CHDIM]);
			dist_4b[2] = get_dist(i,l,&dr_4b[2*CHDIM]);
			dist_4b[3] = get_dist(j,k,&dr_4b[3*CHDIM]);
			dist_4b[4] = get_dist(j,l,&dr_4b[4*CHDIM]);
			dist_4b[5] = get_dist(k,l,&dr_4b[5*CHDIM]);

			typ_idxs_4b[0] = chimes_type[type[i]-1];
			typ_idxs_4b[1] = chimes_type[type[j]-1];
			typ_idxs_4b[2] = chimes_type[type[k]-1];
			typ_idxs_4b[3] = chimes_type[type[l]-1];

			CHIMES_PROF_LAP(PROF_4B_DIST);

			std::fill(force_4b.begin(), force_4b.end(), 0.0) ;
			std::fill(stensor.begin(), stensor.end(), 0.0) ;

			energy = 0.0 ;

#ifdef FINGERPRINT
			valid_order = (tag[i] < tag[j] && tag[j] < tag[k] && tag[k] < tag[l]);
			if (tmp_FP && valid_order)
			{
				vector<double> tmp_force_scalar_4b(6);
				chimes_calculator.compute_4B( dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp, tmp_force_scalar_4b, tmp_dist_4b, tmp_FP && valid_order);
				CHIMES_PROF_LAP(PROF_4B_EVAL_DIRECT);
			}
			else
#endif
			{
				eval_4b_standard(stensor, chimes_4btmp, energy);   // laps its own evaluator row
			}

			for (idx=0; idx<3; idx++)
			{
				f[i][idx] += force_4b[0*CHDIM+idx] ;
				f[j][idx] += force_4b[1*CHDIM+idx] ;
				f[k][idx] += force_4b[2*CHDIM+idx] ;
				f[l][idx] += force_4b[3*CHDIM+idx] ;
			}

			if (vflag_atom)
			{
				atmidxlst[0][0] = i; atmidxlst[0][1] = j;
				atmidxlst[1][0] = i; atmidxlst[1][1] = k;
				atmidxlst[2][0] = i; atmidxlst[2][1] = l;
				atmidxlst[3][0] = j; atmidxlst[3][1] = k;
				atmidxlst[4][0] = j; atmidxlst[4][1] = l;
				atmidxlst[5][0] = k; atmidxlst[5][1] = l;
			}

			if (evflag)
				ev_tally_mb(4, 6, atmidxlst, energy, stensor);

			CHIMES_PROF_LAP(PROF_4B_FORCE_TALLY);
		}
#endif
	}

#ifdef FINGERPRINT
	if (tmp_FP)
	{
		std::stringstream filename_4b;
		filename_4b << ts << "." << std::to_string(chimes_calculator.rank) <<".4b_clusters.txt";
		writeClusterDataComp(filename_4b.str(), tmp_dist_4b);
	}
	CHIMES_PROF_LAP(PROF_IO);
#endif

	if (vflag_fdotr)
		virial_fdotr_compute();

	CHIMES_PROF_LAP(PROF_FDOTR);   // also charges the (tiny) exit of the many-body loops

	return;
}

void PairCHIMES::set_chimes_type()
{
	if(comm->me == 0)
		std::cout << "Attempting to match LAMMPS and ChIMES atom types by comparing masses. Looking for matches within a tolerance of 1e-3." << std::endl;
	int nmatches = 0;

	chimes_type.assign(atom->ntypes, -1);

	for (int i=1; i<= atom->ntypes; i++) // Lammps indexing starts at 1
	{
		bool matched = false;
		for (int j=0; j<chimes_calculator.natmtyps; j++) // ChIMES indexing starts at 0
		{
			if (comm->me == 0)
				std::cout << "LAMMPS atom type idx: " << i << " LAMMPS mass: " << atom->mass[i] << " ; " << "ChIMES atom type idx: " << j << " ChIMES mass: " << chimes_calculator.masses[j] << " ... status: " << std::endl;
			if (abs(atom->mass[i] - chimes_calculator.masses[j]) < 1e-3) // Masses should match to at least 3 decimal places
			{
				chimes_type[i-1] = j;
				nmatches++;
				matched = true;
				if(comm->me == 0)
					std::cout << "Match" << std::endl;
				break;
			}
		}
		if (!matched && comm->me == 0)
			std::cout << "No match" << std::endl;
	}

	if (nmatches == 0 && comm->me == 0)
	{
		std::cout << "ERROR: No element matches found between LAMMPS data file and ChIMES parameter file!" << std::endl;
		std::cout << "       Either check your atom masses or do not request ChIMES interactions" << std::endl;
		exit(0);
	}
	else if (nmatches < atom->ntypes && comm->me == 0)
	{
		// Commented out to allow for hybrid/overlay pair style. For example, combining ChIMES description for carbon
		// with LJ description of Ar for simulations of carbon nanoparticles in an argon bath
		//std::cout << "ERROR: LAMMPS coordinate file has " << atom->ntypes << " atom type masses" << std::endl;
		//std::cout << "       but only found " << nmatches << " matches with the ChIMES parameter file." << std::endl;
		//exit(0);

		std::cout << "WARNING: LAMMPS coordinate file has " << atom->ntypes << " atom type masses" << std::endl;
		std::cout << "       but only found " << nmatches << " matches with the ChIMES parameter file." << std::endl;
		std::cout << "       Will not use ChIMES to evaluate interactions related to unmatched atom types!" << std::endl;
	}
}

void PairCHIMES::write_restart(){}
void PairCHIMES::read_restart(){}
void PairCHIMES::write_restart_settings(){}
void PairCHIMES::read_restart_settings(){}
void PairCHIMES::single(){}