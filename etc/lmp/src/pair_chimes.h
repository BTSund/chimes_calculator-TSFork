/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS

PairStyle(chimesFF,PairCHIMES); // PairStyle(key, class)

#else

#ifndef LMP_PAIR_CHIMES_H
#define LMP_PAIR_CHIMES_H


#include "pair.h"

#include "chimesFF.h"
#include <vector>
#include <utility>


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

/*
   Compile-time options (define for BOTH chimesFF.cpp and pair_chimes.cpp):

     -DCHIMES_SUBGRAPH_REUSE   Enable 4-body subgraph (shared-triplet) reuse.
                               Without it every quadruplet is evaluated
                               independently with the standard evaluator chain.
     -DTABULATION              Tabulated 2B/3B/4B evaluators.
     -DCHIMES_PROFILE          Per-section timing report.
*/


namespace LAMMPS_NS
{
	class PairCHIMES : public Pair
	{
	 	public:

			// Variable definitions

			chimesFF chimes_calculator;   // chimesFF instance

			char * chimesFF_paramfile;	  // ChIMES parameter file

			std::vector<int> chimes_type; // For i = LMP atom type indx, chimes_type[i-1] gives the ChIMES parameter file type idx

			double maxcut_3b;
			double maxcut_4b;

			// ------------------------------------------------------------------
			// Many-body neighbor lists (original ChIMES construction; rebuilt only
			// when LAMMPS re-neighbors). neighborlist_Xmers[cluster idx][atom in cluster idx]
			// ------------------------------------------------------------------

			std::vector<std::vector<int> > neighborlist_3mers;
			std::vector<std::vector<int> > neighborlist_4mers;

			// [begin,end) ranges of neighborlist_4mers that share the same (i,j,k)
			// triplet. The original l-loop already emits them contiguously; this
			// just records the ranges so subgraph reuse can use them.
			std::vector<std::pair<int,int> > neighborlist_4mer_groups;

			// Vars for neighlist construction

			std::vector <int> tmp_3mer;
			std::vector <int> tmp_4mer;

#ifdef CHIMES_SUBGRAPH_REUSE
			// Per-triplet reuse objects, indexed by ChIMES type of the fourth atom.
			// A stamp marks which triplet each entry currently belongs to, so no
			// clearing, searching or re-allocation is needed between triplets.
			std::vector<chimesFF::chimes4BTripletReuseTmp> reuse_pool;
			std::vector<long long>                         reuse_stamp;
			std::vector<signed char>                       reuse_state;  // 0 reuse, 1 skip, 2 standard
			long long                                      reuse_epoch;
#endif

            // Prepare files necessary for ChIMES fitting

            bool     for_fitting;
            bool     fingerprint;
			int IO_freq;
            ofstream badness_stream;

			// 2-body vars for chimesFF access

			std::vector        <double>   dr;
		    std::vector        <double>   dr_3b;
		    std::vector        <double >  dr_4b;

			double                        dist;
			std::vector        <double>   dist_3b;
			std::vector        <double>   dist_4b;

			std::vector<double> force_2b;
		    std::vector<double> force_3b;
		    std::vector<double> force_4b;

			std::vector<int> typ_idxs_2b;
			std::vector<int> typ_idxs_3b;
			std::vector<int> typ_idxs_4b;

			// Constructor/Deconstructor

			PairCHIMES(class LAMMPS *);

			virtual ~PairCHIMES();

			// Functions that have been written

			void   settings(int narg, char **arg);
			void   init_style();
			void   coeff(int narg, char **arg);
			void   allocate();
			double init_one(int i, int j);
			void   compute(int eflag, int vflag);
			void   compute_body(int eflag, int vflag);   // compute() = profiling wrapper around this
			void   build_mb_neighlists();
		    inline double get_dist(int i, int j, double* dr);
		    inline double get_dist(int i, int j);
			void   eval_4b_standard(std::vector<double> & stensor, chimes4BTmp & tmp, double & energy);
			void   set_chimes_type();

			// Functions I haven't worked on

			void write_restart();
			void read_restart();
			void write_restart_settings();
			void read_restart_settings();
			void single();

		};
}



#endif
#endif