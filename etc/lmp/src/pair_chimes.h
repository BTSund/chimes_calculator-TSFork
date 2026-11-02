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

// Build options of the 3B/4B pair reuse (see pair_chimes.cpp).  Decided here, in the header
// every file that uses PairCHIMES includes, so the class layout is the same everywhere.
//   -DTABULATION             pair reuse on (cluster lists, seed triangles, exact 3B pair-matrix
//                            contraction, hashed ownership, 2B pair loop)
//   -DCHIMES_PAIR_CACHE      also keep the per-step pair cache: each pair function evaluated once
//                            per step and shared by all clusters (faster, more memory)
//   -DCHIMES_PAIR_REUSE      the former option name; same as -DCHIMES_PAIR_CACHE
//   -DCHIMES_NO_PAIR_REUSE   no pair reuse: clusters evaluated one at a time, as originally
#if defined(CHIMES_PAIR_REUSE) && !defined(CHIMES_PAIR_CACHE)
#define CHIMES_PAIR_CACHE
#endif

#if defined(TABULATION) && !defined(CHIMES_NO_PAIR_REUSE)
#ifndef CHIMES_PAIR_REUSE
#define CHIMES_PAIR_REUSE
#endif
#else
#undef CHIMES_PAIR_REUSE
#undef CHIMES_PAIR_CACHE
#endif

#include "pair.h"

#include "chimesFF.h"
#include <vector>


/*  Functions required by LAMMPS:


settings    (done)      reads the input script line with arguments defined here
coeff       (done)      set coefficients for one i,j pair type
compute     (done)      workhorse routine that computes pairwise interactions
init_one    (done)      perform initalization for one i,j type pair
init_style  (done)      initialization specific to this pair style

write_restart           write i,j pair coeffs to restart file
read_restart            read i,j pair coeffs from restart file
write_restart_settings  write global settings to restart file
read_restart_settings   read global settings from restart file
single                  force and energy fo a single pairwise interaction between two atoms
*/


namespace LAMMPS_NS
{
    class PairCHIMES : public Pair
    {
    public:

        // Variable definitions

        chimesFF chimes_calculator;   // chimesFF instance

        char * chimesFF_paramfile;    // ChIMES parameter file

        std::vector<int> chimes_type; // For i = LMP atom type indx, chimes_type[i-1] gives the ChIMES parameter file type idx

        double maxcut_3b;
        double maxcut_4b;

        int n_3mers;                   // number of neighborlist_Xmers entries
        int n_4mers;

        std::vector<std::vector<int> > neighborlist_3mers;  // custom neighbor list; neighborlist_Xmers[cluster idx][atom in cluster idx]
        std::vector<std::vector<int> > neighborlist_4mers;

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

        // Vars for neighlist construction

        std::vector <int> tmp_3mer;
        std::vector <int> tmp_4mer;
        std::vector <int> mb_short;   // per-atom 3B candidates while building the 3-mer list

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
        void   build_mb_neighlists();
        inline double get_dist(int i, int j, double* dr);
        inline double get_dist(int i, int j);
        void   set_chimes_type();

#ifdef CHIMES_PAIR_REUSE
        // ---------------------------------------------------------------------
        // Pair + triplet subgraph reuse (on with -DTABULATION), see pair_chimes.cpp.
        // Cluster lists are built when the neighbor list is rebuilt; every step
        // then forms each 4B seed triangle's product once and only the
        // fourth-atom side per quadruplet, and each exact 3B group's
        // pair-contracted coefficient matrix once.  With -DCHIMES_PAIR_CACHE each
        // pair's functions (CP rank terms or exact 3B polynomials) are also
        // evaluated once per step; without it, once per cluster that uses them.
        // The members are the same with and without the cache.
        // ---------------------------------------------------------------------

        bool reuse_greedy;   // pair_style keyword greedy_seeds: double-sweep seed choice
        int  reuse_block;    // pair_style keyword reuse_block N: rank terms per pass
        double reuse_skin;   // pair_style keyword mb_skin X: padding of the cluster lists (< 0: default)
        bool reuse_active;   // cluster lists are built and used for 3B/4B
        bool reuse_3b;       // 3B evaluated through the pair cache (CP tables or exact polynomial)
        bool reuse_4b;       // 4B evaluated through the pair cache (4B CP model); otherwise per cluster
        bool reuse_warned;
        bool reuse_mixed4;   // 4B types without a CP model stay per cluster, the others are cached
        std::vector<char> ru_cache4_type;   // per 4B type combination: 1 = handled by the pair cache
        int  ru_block;       // rank terms per pass actually used
        int  ru_rank_max;    // largest CP rank among the cached tables

        struct ReuseTrip     // one triplet, atoms in runtime order
        {
            int atom[3];
            int pid[3];      // runtime pairs 0 = ij, 1 = ik, 2 = jk
            int ent[3];      // cache entry of each canonical slot
            int c2r[3];      // canonical slot -> runtime pair
            int rank;
            int mslot;       // exact 3B: slot of the group's (first contracted) pair; -1 for CP
            bool inner;      // every pair stays inside its cutoff until the next list build
        };

        struct ReuseSeed     // seed triangle A0 A1 A2, shared by quadruplets A0 A1 A2 A3
        {
            int atom[3];
            int pid[3];      // runtime pairs 0 (A0A1), 1 (A0A2), 3 (A1A2)
            int ent[3];      // cache entries of those pairs
            int cls;         // exact 4B: class (3|3 coefficient matrix in runtime order); -1: CP
            int rank;
            int quad_begin;
            int quad_inner;  // quads [quad_begin, quad_inner) are inner (no per-step check)
            int quad_end;
            bool inner;      // the seed's three pairs are inner
        };

        struct ReuseQuad
        {
            int l;           // fourth atom A3
            int pid[3];      // runtime pairs 2, 4, 5
            int ent[3];
            bool inner;      // its three fourth-atom pairs are inner
        };

        struct ReuseX3Group  // exact 3B triplets that share the pair contracted first
        {
            int ent;         // cache entry of that pair
            int cls;         // contraction class (coefficient tensor with that slot first)
            int trip_begin;
            int trip_inner;  // triplets [trip_begin, trip_inner) are inner
            int trip_end;
        };

        std::vector<int> rp_a, rp_b;              // pairs used by the 3B/4B clusters (rp_a < rp_b)
        std::vector<double> rp_r, rp_dr, rp_fs;   // per step: distance, x[b]-x[a], summed force scalar

        std::vector<int> re_pair, re_table, re_ptype, re_rank;   // pair-cache entries (pair, CP table, cluster cutoff)
        std::vector<double> re_rc;                // cluster cutoff of the entry's pair
        std::vector<double> re_val;               // pair cache: per step, used entries packed: X at [slot*2*ru_block], dX/dr after it

        // Cluster lists are padded by ru_pad and rebuilt at neighbor-list builds and whenever an
        // atom has moved more than ru_pad/2 since the last build (own displacement check).  A
        // cluster whose pairs were all at least ru_pad inside their cutoffs at the build is
        // "inner": it stays inside until the next build and is never checked per step.
        double ru_pad;                            // padding of the current lists
        bool   ru_mid_ok;                         // lists may be rebuilt between neighbor-list builds
        int    ru_nbuild, ru_nmid;                // list builds, of which between neighbor-list builds
        std::vector<double> ru_xhold;             // positions (local + ghost) at the last build
        int    ru_nt_inner;                       // ru_trip[0, ru_nt_inner) are inner
        size_t ru_npair_prev;                     // largest pair count of a build so far (hash table size)
        std::vector<unsigned char> re_always;     // entry used by an inner cluster: always needed

        struct ShortNeigh                         // per-atom many-body neighbor list entry
        {
            int atom;
            int pid;                              // pair id, assigned on first use
            unsigned long long key;   // ownership key of the atom (owner_key)
            int ct;                   // ChIMES type of the atom
            double r;
        };
        std::vector<int> sl_start;                // local atom i: entries [sl_start[i], sl_start[i+1])
        std::vector<ShortNeigh> sl_list;

        std::vector<ReuseTrip> ru_trip;           // 3B CP triplets
        std::vector<ReuseSeed> ru_seed;
        std::vector<ReuseQuad> ru_quad;

        // Exact 3B: triplets (canonical slot = parameter slot) grouped by contracted pair
        std::vector<ReuseTrip> ru_trip3x;
        std::vector<ReuseX3Group> ru_x3group;
        int x3_rank;                              // N of the exact 3B: largest 3B power + 1 (0: no exact 3B)
        int x3_ntab_cp;                           // table ids >= this are exact pair-function specs
        std::vector<int> x3_spec_ptype;           // per spec: pair type, inner and outer cutoff, basis size
        std::vector<double> x3_spec_rin, x3_spec_rout;
        std::vector<int> x3_spec_n;

        // Exact 4B (parameter-file coefficients or Morse-space SVD), through the 4B seeds.
        // ru_seed[0, ru_nseed_cp) are CP seeds, the rest exact 4B seeds sorted by (entry of
        // pair A0A1, class): a run of them is a group whose pair A0A1 is contracted once.
        int x4_rank;                              // N of the exact 4B: largest 4B power + 1 (0: none)
        int ru_nseed_cp;
        int ru_nx4_group;                         // groups at the last build (for the message)
        std::vector<std::vector<double>> x4_cls_mat;   // per class and first seed pair: M[a][b c][d e f], N^6
        std::vector<std::vector<double>> x4_cls_matT;  // the same as M[a][d e f][b c] (seed gradients)
        std::vector<double> ru_x4_work;           // exact 4B scratch
        std::vector<std::vector<double>> x3_cls_tensor;   // per class: R[x][a][b], N^3 values

        // Per step: clusters inside the true cutoffs and the cache entries they use
        std::vector<int> vt_list;                 // valid triplets
        std::vector<int> vs_list, vs_qbeg;        // valid seeds, start of their quads in vq_list
        std::vector<int> vq_list;                 // valid quadruplets, grouped by seed
        std::vector<int> vg_list, vg_tbeg;        // exact 3B groups with valid triplets, start in vx_list
        std::vector<int> vx_list;                 // valid exact 3B triplets, grouped
        std::vector<int> re_need;                 // entries used this step
        std::vector<int> re_slot;                 // their position in the packed cache
        std::vector<unsigned char> re_ok;         // entry's pair is inside the cluster cutoff
        std::vector<unsigned char> re_used;       // entry is used by a selected cluster

        std::vector<double> ru_trip_acc, ru_seed_acc, ru_quad_acc;   // sums over rank passes (valid clusters)
        std::vector<double> ru_x3_acc;            // exact 3B: E and dE/dr per valid triplet
        std::vector<double> ru_scratch;           // seed products; without the cache also the pair functions of 3 entries
        std::vector<double> ru_x3_mat;            // exact 3B: M, dM and the work arrays of a batch of triplets

        void build_reuse_lists();
        bool reuse_lists_stale();
        void compute_mb_reuse();
        void compute_2b_fast(chimes2BTmp &tmp);   // 2B pair loop on chimesFF::eval_2B
#endif

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