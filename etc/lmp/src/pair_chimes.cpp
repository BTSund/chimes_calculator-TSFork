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

#ifdef CHIMES_PROFILE

#ifndef CHIMES_PROFILE_EVERY
#define CHIMES_PROFILE_EVERY 100
#endif

enum ChimesProfID
{
    PROF_EV_SETUP = 0,

    PROF_MB_BUILD,

    PROF_1B,
    PROF_2B_DIST,
    PROF_2B_EVAL,
    PROF_2B_FORCE_TALLY,

    PROF_3B_DIST,
    PROF_3B_EVAL,
    PROF_3B_FORCE_TALLY,

    PROF_4B_SEED,
    PROF_4B_DIST,
    PROF_4B_PREPARE,
    PROF_4B_EVAL,
    PROF_4B_EVAL_REUSE,
    PROF_4B_EVAL_CP_TAB,
    PROF_4B_EVAL_CP_DIRECT,
    PROF_4B_EVAL_COEFF,
    PROF_4B_EVAL_SVD3,
    PROF_4B_EVAL_SVD4,
    PROF_4B_EVAL_DIRECT,
    PROF_4B_FORCE_TALLY,

    PROF_FDOTR,
    PROF_COMPUTE_TOTAL,

    PROF_N
};

static const char *chimes_prof_name[PROF_N] =
{
    "ev_setup",

    "mb_build",

    "1B_energy",
    "2B_distance",
    "2B_eval",
    "2B_force_tally",

    "3B_distance",
    "3B_eval",
    "3B_force_tally",

    "4B_seed",
    "4B_l_distances",
    "4B_prepare_reuse",
    "4B_eval_total",
    "4B_eval_reuse",
    "4B_eval_cp_tab",
    "4B_eval_cp_direct",
    "4B_eval_coeff",
    "4B_eval_svd3",
    "4B_eval_svd4",
    "4B_eval_direct",
    "4B_force_tally",

    "virial_fdotr",
    "compute_total"
};

struct ChimesProfData
{
    double t[PROF_N];
    double n[PROF_N];

    ChimesProfData()
    {
        reset();
    }

    void reset()
    {
        for (int i = 0; i < PROF_N; i++)
        {
            t[i] = 0.0;
            n[i] = 0.0;
        }
    }

    void add(int id, double dt)
    {
        t[id] += dt;
        n[id] += 1.0;
    }
};

static ChimesProfData chimes_prof;

struct ChimesProfScope
{
    int id;
    double t0;

    ChimesProfScope(int id_in)
        : id(id_in), t0(MPI_Wtime())
    {
    }

    ~ChimesProfScope()
    {
        chimes_prof.add(id, MPI_Wtime() - t0);
    }
};

#define CHIMES_PROF_ADD(id, dt) \
    do { chimes_prof.add((id), (dt)); } while (0)

#define CHIMES_PROF_SCOPE(id) \
    ChimesProfScope chimes_prof_scope_##__LINE__((id))

#define CHIMES_PROF_BLOCK(id, block) \
    do { ChimesProfScope chimes_prof_scope_block_##__LINE__((id)); block } while (0)

static void chimes_profile_report(
    MPI_Comm world,
    int me,
    long long timestep
)
{
    int nprocs = 1;
    MPI_Comm_size(world, &nprocs);

    double local_t[PROF_N];
    double local_n[PROF_N];

    for (int i = 0; i < PROF_N; i++)
    {
        local_t[i] = chimes_prof.t[i];
        local_n[i] = chimes_prof.n[i];
    }

    double sum_t[PROF_N];
    double max_t[PROF_N];
    double sum_n[PROF_N];

    MPI_Reduce(local_t, sum_t, PROF_N, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(local_t, max_t, PROF_N, MPI_DOUBLE, MPI_MAX, 0, world);
    MPI_Reduce(local_n, sum_n, PROF_N, MPI_DOUBLE, MPI_SUM, 0, world);

    if (me == 0)
    {
        const double total = max_t[PROF_COMPUTE_TOTAL] > 0.0
                           ? max_t[PROF_COMPUTE_TOTAL]
                           : 1.0;

        std::cout << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "CHIMES_PROFILE timestep " << timestep
                  << " interval timing over last "
                  << CHIMES_PROFILE_EVERY << " steps" << std::endl;
        std::cout << "Times are seconds. max_rank_time is usually the bottleneck metric." << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;

        std::cout
            << std::setw(24) << "section"
            << std::setw(16) << "max_rank"
            << std::setw(16) << "avg_rank"
            << std::setw(14) << "%total"
            << std::setw(16) << "calls_sum"
            << std::setw(16) << "us_per_call"
            << std::endl;

        std::cout << "--------------------------------------------------------------------------------" << std::endl;

        for (int i = 0; i < PROF_N; i++)
        {
            if (sum_n[i] <= 0.0 && sum_t[i] <= 0.0)
                continue;

            const double avg_rank = sum_t[i] / (double)nprocs;
            const double pct = 100.0 * max_t[i] / total;
            const double us_per_call = (sum_n[i] > 0.0)
                                     ? 1.0e6 * sum_t[i] / sum_n[i]
                                     : 0.0;

            std::cout
                << std::setw(24) << chimes_prof_name[i]
                << std::setw(16) << std::scientific << std::setprecision(6) << max_t[i]
                << std::setw(16) << std::scientific << std::setprecision(6) << avg_rank
                << std::setw(14) << std::fixed << std::setprecision(2) << pct
                << std::setw(16) << std::fixed << std::setprecision(0) << sum_n[i]
                << std::setw(16) << std::scientific << std::setprecision(6) << us_per_call
                << std::endl;
        }

        std::cout << "================================================================================" << std::endl;
        std::cout << std::endl;
    }

    chimes_prof.reset();
}

#else

#define CHIMES_PROF_ADD(id, dt) \
    do { } while (0)

#define CHIMES_PROF_SCOPE(id) \
    do { } while (0)

#define CHIMES_PROF_BLOCK(id, block) \
    do { block } while (0)

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
#ifdef FINGERPRINT
    fingerprint = false;
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
	// if (maxcut_3b==0.0 && fingerprint){
	// 	double max_val = 20;
	// 	for (const auto& row : cutoff_2b) {
	// 		for (double val : row) {
	// 			if (val > max_val) {
	// 				max_val = val;
	// 			}
	// 		}
	// 	}
	// 	maxcut_3b=max_val;
	// }
	maxcut_4b = chimes_calculator.max_cutoff_4B();
	// if (maxcut_4b==0.0 && fingerprint){maxcut_4b=maxcut_3b;}
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
	
	// Set up neighbor lists... borrowing this from pair_airebo:
	// need a full neighbor list, including neighbors of ghosts

	// int irequest = neighbor->request(this,instance_me);
	// neighbor->requests[irequest]->half = 0;
	// neighbor->requests[irequest]->full = 1;
	// neighbor->requests[irequest]->ghost = 1;

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

void PairCHIMES::build_mb_neighlists()
{


	if ( (chimes_calculator.poly_orders[1] == 0) &&  (chimes_calculator.poly_orders[2] == 0) 
		#ifdef FINGERPRINT
			&& (!fingerprint)
		#endif
		)
		return;

	// List gets built based on atoms owned by calling proc. 
	
	neighborlist_3mers.clear();
	
	int i,j,k,l,inum,jnum,knum,lnum, ii, jj, kk, ll;		 // Local iterator vars
	int *ilist,*jlist,*klist,*llist, *numneigh,**firstneigh; // Local neighborlist vars
	tagint 	*tag   = atom -> tag;					         // Access to global atom indices
	int     itag, jtag, ktag, ltag;					         // holds tags	
	int     *type  = atom -> type;					         // Access to system atom types
	double 	**x    = atom -> x;					             // Access to system coordinates
	
	double maxcut_3b_padded = maxcut_3b + neighbor-> skin;
	
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
			jtag  = tag[j];		
			j    &= NEIGHMASK;

			if (chimes_type[type[j]-1] < 0)
				continue;

			if (j == i)
				continue;
			if (jtag < itag) 
				continue;				
				
			// Check ij distance

			dist_ij = get_dist(i,j);
			
			if ( (dist_ij >= maxcut_3b_padded))
				continue;

			klist = firstneigh[i];	// ChIMES assumes all atoms must be within cutoff of eachother for a valid interaction	
			knum  = numneigh[i];	
			
			for (kk = 0; kk < knum; kk++)	
			{
				k     = klist[kk];	
				ktag  = tag[k];		
				k    &= NEIGHMASK;

				if (chimes_type[type[k]-1] < 0)
					continue;

				if ( (k==i) || (k==j) )
					continue;
				if ( (ktag < itag) || (ktag < jtag) )
					continue;						
							
 	 			// Check ik distance			

				dist_ik = get_dist(i,k);

				if ( (dist_ik >= maxcut_3b_padded))
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
			}
		}
	}

// Add this at the end of build_mb_neighlists()
// Include <algorithm> at the top of your file if not already present

	return;

}

void PairCHIMES::compute(int eflag, int vflag)
{

#ifdef CHIMES_PROFILE
    const double chimes_compute_t0 = MPI_Wtime();
#endif
	// Vars for access to chimesFF compute_XB functions
	
	std::vector  <double>  stensor(6);	// pointers to system stress tensor
    
	// Temp vars to hold chimes output for passing to ev_tally function
	
    std::vector<double>  fscalar(6);
    std::vector<double>  tmp_dist(1);
    std::vector<double>  tmp_dr(6);
    int                  atmidxlst[6][2];
	
	// General LAMMPS compute vars
	
	int 	i,j,k,l,inum,jnum,lnum, ii, jj;	// Local iterator vars
	int 	*ilist,*jlist,*llist, *numneigh,**firstneigh;	// Local neighborlist vars
	int     idx;

	double 	**x    = atom -> x;		    // Access to system coordinates
	double 	**f    = atom -> f;		    // Access to system forces
	
	int 	*type  = atom -> type;		// Acces to system atom types (countng starts from 1, chimesFF class expects counting from 0!)
	tagint 	*tag   = atom -> tag;		// Access to global atom indices (sort of like "parent" indices)
	int     itag, jtag, ktag, ltag;		// holds tags
	int 	nlocal = atom -> nlocal;	// Number of real atoms owned by current process .. used used to assure force assignments aren't duplicated
	int 	newton_pair = force -> newton_pair;	// Should f_j be automatically set to -f_i (true) or manually calculated (false)
	double  energy;				        // pair energy 

	int me = comm->me;
	MPI_Comm_rank(world,&me);	
	
	// Set up vars controlling if energy/pressure (virial) contributions are computed

CHIMES_PROF_BLOCK(PROF_EV_SETUP, {
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
});

	// Compile if fingerprinting desired
#ifdef FINGERPRINT
	std::vector<std::vector<double>> tmp_dist_2b;
    std::vector<std::vector<double>> tmp_dist_3b;
    std::vector<std::vector<double>> tmp_dist_4b;
	bool 				 tmp_FP;
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
	
	// Build the ChIMES many-body neighbor lists.. only do so when LAMMPS neighborlist has been updated
	
if (neighbor->ago == 0)
{
    if (chimes_calculator.rank == 0)
        std::cout << "Updating chimesFF neighbor lists..." << std::endl;

    CHIMES_PROF_BLOCK(PROF_MB_BUILD, {
        build_mb_neighlists();
    });

    if (chimes_calculator.rank == 0)
    {
        std::cout << "  Rank " << me << " 3-body list size: "
                  << neighborlist_3mers.size() << std::endl;
        std::cout << "  Rank " << me << " 4-body list size: "
                  << neighborlist_4mers.size() << std::endl;
        std::cout << "  ...update complete" << std::endl;
    }
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
		CHIMES_PROF_BLOCK(PROF_1B, {
    chimes_calculator.compute_1B(chimes_type[type[i]-1], energy);
});
        
        atmidxlst[0][0] = i;
		
		if(evflag)
			ev_tally_mb(1, 0, atmidxlst, energy, stensor);

		// Now move on to two-body force, stress, and energy
		
		for (jj = 0; jj < jnum; jj++) // Loop over neighbors of i
		{
			j     = jlist[jj];			// Index of the jj atom
			jtag  = tag[j];				// Get j's global atom index (sort of like its "parent")
			j    &= NEIGHMASK;			// Strip possible extra bits of j

			if (chimes_type[type[j]-1] < 0)
				continue;
				
			
			if (jtag <= itag) // only allow calculation for j<i, since we've requested a full neighbor list
				continue;
				
			// Get distance using ghost atoms... don't need MIC since we're using ghost atoms
CHIMES_PROF_BLOCK(PROF_2B_DIST, {
    dist = get_dist(i, j, &dr[0]);
});
			
			typ_idxs_2b[0] = chimes_type[type[i]-1]; // Type (index) of the current atom... subtract 1 to account for chimesFF vs LAMMPS numbering convention
			typ_idxs_2b[1] = chimes_type[type[j]-1];

			// Using std::fill for maximum efficiency.
			std::fill(force_2b.begin(), force_2b.end(), 0.0) ;

			// Do the same for stress tensors
			std::fill(stensor.begin(), stensor.end(), 0.0) ;

			energy = 0.0;	
CHIMES_PROF_BLOCK(PROF_2B_EVAL, {
#ifdef TABULATION
    if (chimes_calculator.tabulate_2B)
    {
        chimes_calculator.compute_2B_tab(
            dist,
            dr,
            typ_idxs_2b,
            force_2b,
            stensor,
            energy,
            chimes_2btmp
        );
    }
    else
#endif
    {
#ifdef FINGERPRINT
        valid_order = (i < j);

        if (tmp_FP && valid_order)
        {
            double tmp_force_scalar;

            chimes_calculator.compute_2B(
                dist,
                dr,
                typ_idxs_2b,
                force_2b,
                stensor,
                energy,
                chimes_2btmp,
                tmp_force_scalar,
                tmp_dist_2b,
                tmp_FP && valid_order
            );
        }
        else
#endif
        {
            chimes_calculator.compute_2B(
                dist,
                dr,
                typ_idxs_2b,
                force_2b,
                stensor,
                energy,
                chimes_2btmp
            );
        }
    }
});
			CHIMES_PROF_BLOCK(PROF_2B_FORCE_TALLY, {
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

    tmp_dist[0] = dist;

    if (evflag)
        ev_tally_mb(2, 1, atmidxlst, energy, stensor);
});
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
    if (for_fitting)
        if(update->ntimestep % output->every_dump[0] == 0)
            badness_stream << update->ntimestep << " " <<  chimes_calculator.get_badness() << endl;

	// if (chimes_calculator.poly_orders[1] > 0 || tmp_FP)
	if (chimes_calculator.poly_orders[1] > 0 ||
			chimes_calculator.poly_orders[2] > 0)
		{
			std::vector<chimesFF::chimes4BTripletReuseTmp> reuse_cache;
			reuse_cache.reserve(8);

			for (ii = 0; ii < neighborlist_3mers.size(); ii++)
			{
				i = neighborlist_3mers[ii][0];
				j = neighborlist_3mers[ii][1];
				k = neighborlist_3mers[ii][2];

				itag = tag[i];
				jtag = tag[j];
				ktag = tag[k];

				if (chimes_type[type[i]-1] < 0) continue;
				if (chimes_type[type[j]-1] < 0) continue;
				if (chimes_type[type[k]-1] < 0) continue;

				// Shared triplet distances.
				CHIMES_PROF_BLOCK(PROF_3B_DIST, {
					dist_3b[0] = get_dist(i, j, &dr_3b[0*CHDIM]);
					dist_3b[1] = get_dist(i, k, &dr_3b[1*CHDIM]);
					dist_3b[2] = get_dist(j, k, &dr_3b[2*CHDIM]);
				});

				const double dist_ij = dist_3b[0];
				const double dist_ik = dist_3b[1];
				const double dist_jk = dist_3b[2];

				const int type_i = chimes_type[type[i]-1];
				const int type_j = chimes_type[type[j]-1];
				const int type_k = chimes_type[type[k]-1];

				/*
				* 3B calculation.
				*
				* Existing neighborlist_3mers should already satisfy maxcut_3b-padded
				* construction, but keep the explicit maxcut check here.
				*/
				if (chimes_calculator.poly_orders[1] > 0)
				{
					if (dist_ij < maxcut_3b &&
						dist_ik < maxcut_3b &&
						dist_jk < maxcut_3b)
					{
						typ_idxs_3b[0] = type_i;
						typ_idxs_3b[1] = type_j;
						typ_idxs_3b[2] = type_k;

						std::fill(force_3b.begin(), force_3b.end(), 0.0);
						std::fill(stensor.begin(), stensor.end(), 0.0);

						energy = 0.0;

		CHIMES_PROF_BLOCK(PROF_3B_EVAL, {
#ifdef TABULATION
    if (chimes_calculator.tabulate_3B)
    {
        chimes_calculator.compute_3B_tab(
            dist_3b,
            dr_3b,
            typ_idxs_3b,
            force_3b,
            stensor,
            energy,
            chimes_3btmp
        );
    }
    else
#endif
    {
        chimes_calculator.compute_3B(
            dist_3b,
            dr_3b,
            typ_idxs_3b,
            force_3b,
            stensor,
            energy,
            chimes_3btmp
        );
    }
});

												CHIMES_PROF_BLOCK(PROF_3B_FORCE_TALLY, {
							for (idx = 0; idx < CHDIM; idx++)
							{
								f[i][idx] += force_3b[0*CHDIM + idx];
								f[j][idx] += force_3b[1*CHDIM + idx];
								f[k][idx] += force_3b[2*CHDIM + idx];
							}

							if (vflag_atom)
							{
								atmidxlst[0][0] = i; atmidxlst[0][1] = j;
								atmidxlst[1][0] = i; atmidxlst[1][1] = k;
								atmidxlst[2][0] = j; atmidxlst[2][1] = k;
							}

							if (evflag)
								ev_tally_mb(3, 3, atmidxlst, energy, stensor);
						});
					} // end 3B cutoff check
				}     // end if poly_orders[1] > 0

				/*
				* 4B seed check.
				*
				* Assumption:
				*   maxcut_3b >= maxcut_4b
				*
				* Therefore neighborlist_3mers includes every possible 4B seed triplet.
				*/
				if (chimes_calculator.poly_orders[2] == 0)
					continue;

				bool chimes_4b_seed_ok = false;

				CHIMES_PROF_BLOCK(PROF_4B_SEED, {
					chimes_4b_seed_ok =
						(dist_ij < maxcut_4b &&
						dist_ik < maxcut_4b &&
						dist_jk < maxcut_4b);
				});

				if (!chimes_4b_seed_ok)
					continue;

				reuse_cache.clear();

				// Copy triplet side into 4B runtime pair slots:
				//
				//   0 = ij
				//   1 = ik
				//   3 = jk
				dist_4b[0] = dist_ij;
				dist_4b[1] = dist_ik;
				dist_4b[3] = dist_jk;

				for (idx = 0; idx < CHDIM; idx++)
				{
					dr_4b[0*CHDIM + idx] = dr_3b[0*CHDIM + idx];
					dr_4b[1*CHDIM + idx] = dr_3b[1*CHDIM + idx];
					dr_4b[3*CHDIM + idx] = dr_3b[2*CHDIM + idx];
				}

				typ_idxs_4b[0] = type_i;
				typ_idxs_4b[1] = type_j;
				typ_idxs_4b[2] = type_k;

				/*
				* Candidate fourth atoms.
				*
				* We scan i's full neighbor list and enforce the old uniqueness rule:
				*
				*   ltag > itag, jtag, ktag
				*
				* This preserves one evaluation per 4B cluster.
				*/
				llist = firstneigh[i];
				lnum  = numneigh[i];

				for (int ll = 0; ll < lnum; ll++)
				{
					l = llist[ll];
					l &= NEIGHMASK;

					if ((l == i) || (l == j) || (l == k))
						continue;

					ltag = tag[l];

					if ((ltag < itag) || (ltag < jtag) || (ltag < ktag))
						continue;

					if (chimes_type[type[l]-1] < 0)
						continue;
					bool chimes_4b_l_ok = false;

					CHIMES_PROF_BLOCK(PROF_4B_DIST, {
						dist_4b[2] = get_dist(i, l, &dr_4b[2*CHDIM]);

						if (dist_4b[2] < maxcut_4b)
						{
							dist_4b[4] = get_dist(j, l, &dr_4b[4*CHDIM]);

							if (dist_4b[4] < maxcut_4b)
							{
								dist_4b[5] = get_dist(k, l, &dr_4b[5*CHDIM]);

								if (dist_4b[5] < maxcut_4b)
									chimes_4b_l_ok = true;
							}
						}
					});

					if (!chimes_4b_l_ok)
						continue;

					const int type_l = chimes_type[type[l]-1];

					typ_idxs_4b[3] = type_l;

					/*
					* Reuse cache is keyed by type_l_context.
					*
					* For single-species systems this computes one left object per triplet.
					* For multi-species systems this computes one left object per distinct
					* fourth-atom type encountered for this triplet.
					*/
					chimesFF::chimes4BTripletReuseTmp *reuse_ptr = nullptr;

					for (int c = 0; c < (int)reuse_cache.size(); c++)
					{
						if (reuse_cache[c].type_l_context == type_l)
						{
							reuse_ptr = &reuse_cache[c];
							break;
						}
					}

					if (reuse_ptr == nullptr)
					{
						chimesFF::chimes4BTripletReuseTmp reuse_new;


						bool ok = false;

						CHIMES_PROF_BLOCK(PROF_4B_PREPARE, {
							ok = chimes_calculator.prepare_4B_triplet_reuse(
								dist_3b,
								typ_idxs_4b,
								chimes_4btmp,
								reuse_new
							);
						});

						if (ok)
						{
							reuse_cache.push_back(std::move(reuse_new));
							reuse_ptr = &reuse_cache.back();
						}
					}

					std::fill(force_4b.begin(), force_4b.end(), 0.0);
					std::fill(stensor.begin(), stensor.end(), 0.0);

					energy = 0.0;

					CHIMES_PROF_BLOCK(PROF_4B_EVAL, {
    if (reuse_ptr != nullptr && reuse_ptr->valid)
    {
        CHIMES_PROF_BLOCK(PROF_4B_EVAL_REUSE, {
            chimes_calculator.compute_4B_from_triplet_reuse(
                dist_4b,
                dr_4b,
                typ_idxs_4b,
                *reuse_ptr,
                force_4b,
                stensor,
                energy,
                chimes_4btmp
            );
        });
    }
    else
    {
#ifdef TABULATION
        if (chimes_calculator.tabulate_4B_cp_direct)
        {
            CHIMES_PROF_BLOCK(PROF_4B_EVAL_CP_DIRECT, {
                chimes_calculator.compute_4B_cp_direct(
                    dist_4b,
                    dr_4b,
                    typ_idxs_4b,
                    force_4b,
                    stensor,
                    energy,
                    chimes_4btmp
                );
            });
        }
        else if (chimes_calculator.tabulate_4B_cp)
        {
            CHIMES_PROF_BLOCK(PROF_4B_EVAL_CP_TAB, {
                chimes_calculator.compute_4B_cp_tab(
                    dist_4b,
                    dr_4b,
                    typ_idxs_4b,
                    force_4b,
                    stensor,
                    energy,
                    chimes_4btmp
                );
            });
        }
        else if (chimes_calculator.tabulate_4B_svd4x2)
        {
            CHIMES_PROF_BLOCK(PROF_4B_EVAL_SVD4, {
                chimes_calculator.compute_4B_svd4x2_tab(
                    dist_4b,
                    dr_4b,
                    typ_idxs_4b,
                    force_4b,
                    stensor,
                    energy,
                    chimes_4btmp
                );
            });
        }
        else if (chimes_calculator.tabulate_4B_svd3x3)
        {
            CHIMES_PROF_BLOCK(PROF_4B_EVAL_SVD3, {
                chimes_calculator.compute_4B_svd3x3_tab(
                    dist_4b,
                    dr_4b,
                    typ_idxs_4b,
                    force_4b,
                    stensor,
                    energy,
                    chimes_4btmp
                );
            });
        }
        else if (chimes_calculator.tabulate_4B_coeff)
        {
            CHIMES_PROF_BLOCK(PROF_4B_EVAL_COEFF, {
                chimes_calculator.compute_4B_tab_coeff(
                    dist_4b,
                    dr_4b,
                    typ_idxs_4b,
                    force_4b,
                    stensor,
                    energy,
                    chimes_4btmp
                );
            });
        }
        else
#endif
                {
            CHIMES_PROF_BLOCK(PROF_4B_EVAL_DIRECT, {
                chimes_calculator.compute_4B(
                    dist_4b,
                    dr_4b,
                    typ_idxs_4b,
                    force_4b,
                    stensor,
                    energy,
                    chimes_4btmp
                );
            });
        }
    } // end else no reuse
}); // end PROF_4B_EVAL

					CHIMES_PROF_BLOCK(PROF_4B_FORCE_TALLY, {
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
					});
				}
			}
		}
if (vflag_fdotr)
{
    CHIMES_PROF_BLOCK(PROF_FDOTR, {
        virial_fdotr_compute();
    });
}

#ifdef CHIMES_PROFILE
CHIMES_PROF_ADD(PROF_COMPUTE_TOTAL, MPI_Wtime() - chimes_compute_t0);

if (update->ntimestep % CHIMES_PROFILE_EVERY == 0)
{
    chimes_profile_report(
        world,
        comm->me,
        static_cast<long long>(update->ntimestep)
    );
}
#endif

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
