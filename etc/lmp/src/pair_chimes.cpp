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

// ==== CHIMES_PROFILE BLOCK BEGIN ====
/* ----------------------------------------------------------------------
   Optional profiling

     -DCHIMES_PROFILE            coarse: one timer read per stage.
                                 Gives clean 2B / 3B / 4B totals, cluster
                                 counts and cost per cluster.
     -DCHIMES_PROFILE_FINE       fine (implies CHIMES_PROFILE): additionally
                                 splits each body order into its stages
                                 (distances, reuse prepare, eval variant,
                                 force+tally) with per-cluster timer reads.
                                 Cheap clusters get inflated; the tmr%
                                 column shows by how much.
     -DCHIMES_PROFILE_EVERY=N    report interval in timesteps (default 100).

   Exclusive "lap" timing: one stopwatch runs from the first line of
   compute() to its last line. Each lap charges the time since the previous
   lap to one section and restarts the watch, so sections never overlap,
   never leave gaps, and add up to 100% of compute(). Loop control, skipped
   neighbors and scratch zeroing are charged to the stage whose loop they
   belong to.

   Subgraph-reuse build: 3B and 4B run inside ONE loop over the triplet
   list. Each triplet is split into three laps even in coarse mode:
     shared triplet  loop control + the three triplet distances, which are
                     used by both 3B and 4B (own row, not split between them)
     3-body          3B eval + force/tally (only if inside the 3B cutoff)
     4-body          seed check, scan for the fourth atom l, l-distances,
                     reuse prepare, eval, force/tally
   That is up to three timer reads per triplet in coarse mode too; tmr%
   shows what they cost.

   The report is organized by body order. Each block shows:
     items/rank  clusters per rank: atoms, pairs, triplets in the list,
                 triplets evaluated for 3B, quadruplets evaluated for 4B.
     us/item     block time per cluster (all ranks summed / all clusters).
     nzE%        share of evaluated clusters that returned nonzero energy.
     ms/call     block time per compute() call (per rank).
     tmr%        estimated share of the block that is timer reads.
   A "4-body detail" line adds seed / candidate / reuse statistics.

   Section ids and row names are shared with the non-reuse pair_chimes.cpp,
   so the same block can be dropped into either file; rows and blocks a
   build never visits are not printed.
------------------------------------------------------------------------- */

#ifdef CHIMES_PROFILE_FINE
#ifndef CHIMES_PROFILE
#define CHIMES_PROFILE
#endif
#endif

#ifdef CHIMES_PROFILE

#ifndef CHIMES_PROFILE_EVERY
#define CHIMES_PROFILE_EVERY 100
#endif

enum ChimesProfGroup
{
    PG_1B = 0,
    PG_2B,
    PG_TRIP,       // shared triplet work (subgraph reuse: used by 3B and 4B)
    PG_3B,
    PG_4B,
    PG_NEIGH,      // many-body cluster list build
    PG_OVERHEAD,   // setup, io, virial, exit
    PG_N
};

static const char *chimes_prof_group_name[PG_N] =
{
    "1-body",
    "2-body",
    "shared 3B/4B triplet",
    "3-body",
    "4-body",
    "many-body list build",
    "overhead"
};

static const char *chimes_prof_group_item[PG_N] =
{
    "atoms",
    "pairs",
    "triplets",
    "triplets",
    "quads",
    "rebuilds",
    ""
};

enum ChimesProfID
{
    PROF_SETUP = 0,          // ev_setup, locals, scratch allocation
    PROF_MB_BUILD,           // many-body cluster list build (+ its log output)

    PROF_1B,

    PROF_2B_DIST,            // neighbor loop control + distance
    PROF_2B_EVAL,
    PROF_2B_FORCE_TALLY,     // also receives loop exit (and the whole loop in coarse mode)

    PROF_IO,                 // fingerprint / badness file output

    PROF_TRIP_DIST,          // reuse: triplet loop control + shared ij/ik/jk distances
    PROF_3B_DIST,            // non-reuse: 3B loop control + distances
    PROF_3B_EVAL,
    PROF_3B_FORCE_TALLY,

    PROF_4B_DIST,            // reuse: seed check + l scan + il/jl/kl distances; non-reuse: 6 distances
    PROF_4B_PREPARE,         // subgraph reuse preparation
    PROF_4B_EVAL_REUSE,
    PROF_4B_EVAL_CP_DIRECT,
    PROF_4B_EVAL_CP_TAB,
    PROF_4B_EVAL_SVD4,
    PROF_4B_EVAL_SVD3,
    PROF_4B_EVAL_COEFF,
    PROF_4B_EVAL_DIRECT,
    PROF_4B_FORCE_TALLY,

    PROF_FDOTR,
    PROF_OTHER,              // function exit: scratch destructors, etc.

    PROF_N
};

static const char *chimes_prof_name[PROF_N] =
{
    "setup",
    "build",

    "energy",

    "distance+loop",
    "eval",
    "force+tally",

    "io (fingerprint/badness)",

    "distances+loop",
    "distance+loop",
    "eval",
    "force+tally",

    "seed+l scan+l-distances",
    "prepare_reuse",
    "eval_reuse",
    "eval_cp_direct",
    "eval_cp_tab",
    "eval_svd4",
    "eval_svd3",
    "eval_coeff",
    "eval_direct",
    "force+tally",

    "virial_fdotr",
    "exit/destructors"
};

static const int chimes_prof_group[PROF_N] =
{
    PG_OVERHEAD, PG_NEIGH,
    PG_1B,
    PG_2B, PG_2B, PG_2B,
    PG_OVERHEAD,
    PG_TRIP,
    PG_3B, PG_3B, PG_3B,
    PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B, PG_4B,
    PG_OVERHEAD, PG_OVERHEAD
};

// 1 = evaluation kernel (the actual ChIMES energy/force math), 0 = bookkeeping
static const int chimes_prof_kernel[PROF_N] =
{
    0, 0,
    1,
    0, 1, 0,
    0,
    0,
    0, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0
};

// Extra event counters (always on in either mode; one add each)
enum ChimesProfExtra
{
    EX_4B_SEEDS = 0,         // triplets inside the 4B cutoff (4B seeds)
    EX_4B_CAND,              // fourth-atom candidates that passed tag/type checks
    EX_4B_PREP,              // prepare_4B_triplet_reuse calls
    EX_4B_REUSE,             // quads evaluated through the reuse path
    EX_N
};

static_assert(sizeof(chimes_prof_name)   / sizeof(chimes_prof_name[0])   == PROF_N, "chimes_prof_name size");
static_assert(sizeof(chimes_prof_group)  / sizeof(chimes_prof_group[0])  == PROF_N, "chimes_prof_group size");
static_assert(sizeof(chimes_prof_kernel) / sizeof(chimes_prof_kernel[0]) == PROF_N, "chimes_prof_kernel size");

struct ChimesProfData
{
    double t[PROF_N];       // seconds charged to each section
    double n[PROF_N];       // counted (per-item / per-event) laps
    double laps[PROF_N];    // all timer reads charged to the section (for overhead estimate)
    double items[PG_N];     // clusters handed to the evaluator, per body order
    double active[PG_N];    // ... of which returned a nonzero energy
    double extra[EX_N];

    double wall;            // independently measured compute() wall time
    double ncalls;          // compute() calls since the last report
    double last;            // stopwatch
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
            t[i]    = 0.0;
            n[i]    = 0.0;
            laps[i] = 0.0;
        }
        for (int g = 0; g < PG_N; g++)
        {
            items[g]  = 0.0;
            active[g] = 0.0;
        }
        for (int e = 0; e < EX_N; e++)
            extra[e] = 0.0;
        wall   = 0.0;
        ncalls = 0.0;
    }

    inline void begin()
    {
        start = last = MPI_Wtime();
    }

    inline void lap(int id, bool counted)
    {
        const double now = MPI_Wtime();
        t[id]    += now - last;
        laps[id] += 1.0;
        if (counted)
            n[id] += 1.0;
        last = now;
    }

    inline void end(int id)
    {
        lap(id, false);
        wall   += last - start;
        ncalls += 1.0;
    }
};

static ChimesProfData chimes_prof;

#define CHIMES_PROF_BEGIN()          chimes_prof.begin()
#define CHIMES_PROF_LAP(id)          chimes_prof.lap((id), true)    // one event of section id ended
#define CHIMES_PROF_BLOCK_END(id)    chimes_prof.lap((id), false)   // close a loop: charge remainder to id
#define CHIMES_PROF_END(id)          chimes_prof.end(id)
#define CHIMES_PROF_COUNT(g)         (chimes_prof.items[(g)] += 1.0)
#define CHIMES_PROF_ACTIVE(g, e)     do { if ((e) != 0.0) chimes_prof.active[(g)] += 1.0; } while (0)
#define CHIMES_PROF_EXTRA(x)         (chimes_prof.extra[(x)] += 1.0)

#ifdef CHIMES_PROFILE_FINE
#define CHIMES_PROF_FINE_LAP(id)     chimes_prof.lap((id), true)
// Eval-variant bookkeeping: remember which 4B evaluator ran, lap once after it
#define CHIMES_PROF_EVAL_DECL()      int chimes_prof_eval_id = PROF_4B_EVAL_DIRECT
#define CHIMES_PROF_EVAL_ID(id)      (chimes_prof_eval_id = (id))
#define CHIMES_PROF_EVAL_LAP()       chimes_prof.lap(chimes_prof_eval_id, true)
#else
#define CHIMES_PROF_FINE_LAP(id)     do { } while (0)
#define CHIMES_PROF_EVAL_DECL()      do { } while (0)
#define CHIMES_PROF_EVAL_ID(id)      do { } while (0)
#define CHIMES_PROF_EVAL_LAP()       do { } while (0)
#endif

static void chimes_profile_report(MPI_Comm world, int me, long long timestep)
{
    int nprocs = 1;
    MPI_Comm_size(world, &nprocs);
    const double P = (double) nprocs;

    // Cost of one timer read on this rank (used for the overhead estimate)
    double tick_cost;
    {
        const int NT = 20000;
        double t0 = MPI_Wtime(), dummy = 0.0;
        for (int it = 0; it < NT; it++)
            dummy += MPI_Wtime();
        tick_cost = (MPI_Wtime() - t0) / NT;
        if (dummy < 0.0) tick_cost = 0.0;   // keep the loop from being optimized away
    }

    // Pack everything into one buffer so a single SUM and a single MAX reduce do the job.
    // Group times are summed per rank BEFORE reducing, so the group max_rank is the
    // slowest rank for that whole body order, not a sum of per-section maxima.
    enum
    {
        O_T     = 0,
        O_N     = O_T    + PROF_N,
        O_LAPS  = O_N    + PROF_N,
        O_GT    = O_LAPS + PROF_N,
        O_ITEM  = O_GT   + PG_N,
        O_ACT   = O_ITEM + PG_N,
        O_EX    = O_ACT  + PG_N,
        O_WALL  = O_EX   + EX_N,
        O_CALLS = O_WALL + 1,
        NBUF    = O_CALLS + 1
    };

    std::vector<double> loc(NBUF, 0.0), sum(NBUF, 0.0), mx(NBUF, 0.0);

    for (int i = 0; i < PROF_N; i++)
    {
        loc[O_T    + i] = chimes_prof.t[i];
        loc[O_N    + i] = chimes_prof.n[i];
        loc[O_LAPS + i] = chimes_prof.laps[i];
        loc[O_GT   + chimes_prof_group[i]] += chimes_prof.t[i];
    }
    for (int g = 0; g < PG_N; g++)
    {
        loc[O_ITEM + g] = chimes_prof.items[g];
        loc[O_ACT  + g] = chimes_prof.active[g];
    }
    for (int e = 0; e < EX_N; e++)
        loc[O_EX + e] = chimes_prof.extra[e];
    loc[O_WALL]  = chimes_prof.wall;
    loc[O_CALLS] = chimes_prof.ncalls;

    MPI_Reduce(loc.data(), sum.data(), NBUF, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(loc.data(), mx.data(),  NBUF, MPI_DOUBLE, MPI_MAX, 0, world);

    if (me == 0)
    {
        double avg_sum = 0.0;
        for (int i = 0; i < PROF_N; i++)
            avg_sum += sum[O_T + i] / P;

        const double denom  = (avg_sum > 0.0) ? avg_sum : 1.0;
        const double ncalls = (sum[O_CALLS] > 0.0) ? sum[O_CALLS] / P : 1.0;

        double g_avg[PG_N], g_ker[PG_N], g_ovh[PG_N], g_laps[PG_N];
        int    g_rows[PG_N];
        for (int g = 0; g < PG_N; g++)
        {
            g_avg[g]  = sum[O_GT + g] / P;
            g_ker[g]  = 0.0;
            g_ovh[g]  = 0.0;
            g_laps[g] = 0.0;
            g_rows[g] = 0;
        }
        double ovh_total = 0.0;
        for (int i = 0; i < PROF_N; i++)
        {
            const int g = chimes_prof_group[i];
            const double ovh = (sum[O_LAPS + i] / P) * tick_cost;
            if (chimes_prof_kernel[i])
                g_ker[g] += sum[O_T + i] / P;
            g_ovh[g]  += ovh;
            g_laps[g] += sum[O_LAPS + i];
            ovh_total += ovh;
            if (sum[O_LAPS + i] > 0.0)
                g_rows[g]++;
        }

        const std::string rule(118, '-');
        const std::string dbl (118, '=');

#ifdef CHIMES_PROFILE_FINE
        const char *mode = "fine (per-cluster split)";
#else
        const char *mode = "coarse (per-block)";
#endif

        printf("\n%s\n", dbl.c_str());
        printf("CHIMES_PROFILE step %lld | PairCHIMES::compute() | %.0f call(s) since last report | %d rank(s) | %s\n", timestep, ncalls, nprocs, mode);
        printf("Exclusive timing: rows never overlap and sum to 100%% of compute(). %%comp uses rank-averaged time;" " imbal = max_rank/avg_rank.\n");
        printf("Indented rows split a block (items = timed calls); rows marked * are evaluation kernels, the rest bookkeeping.\n");
        printf("%s\n", rule.c_str());
        printf("%-34s%12s%12s%9s%7s%13s%11s%8s%10s%7s\n", "block", "avg_rank(s)", "max_rank(s)", "%comp", "imbal", "items/rank", "us/item", "nzE%", "ms/call", "tmr%");
        printf("%s\n", rule.c_str());

        const int order[PG_N] = { PG_1B, PG_2B, PG_TRIP, PG_3B, PG_4B, PG_NEIGH, PG_OVERHEAD };

        for (int oi = 0; oi < PG_N; oi++)
        {
            const int g = order[oi];

            if (g == PG_NEIGH)
                printf("%s\n", rule.c_str());

            const double avg   = g_avg[g];
            const double mxg   = mx[O_GT + g];
            const double items = sum[O_ITEM + g];
            const double act   = sum[O_ACT + g];

            // Blocks this build never visits are not printed
            if (g_laps[g] <= 0.0 && items <= 0.0)
                continue;

            char label[64];
            if (chimes_prof_group_item[g][0] != '\0')
                snprintf(label, sizeof(label), "%s [%s]", chimes_prof_group_name[g], chimes_prof_group_item[g]);
            else
                snprintf(label, sizeof(label), "%s", chimes_prof_group_name[g]);

            printf("%-34s%12.4e%12.4e%8.2f%%%7.2f", label, avg, mxg, 100.0 * avg / denom, (avg > 0.0) ? mxg / avg : 0.0);

            if (items > 0.0)
                printf("%13.5g%11.4g", items / P, 1.0e6 * sum[O_GT + g] / items);
            else
                printf("%13s%11s", "-", "-");

            if ((g == PG_2B || g == PG_3B || g == PG_4B) && items > 0.0)
                printf("%7.1f%%", 100.0 * act / items);
            else
                printf("%8s", "-");

            printf("%10.4f%6.1f%%\n", 1.0e3 * avg / ncalls, (avg > 0.0) ? 100.0 * g_ovh[g] / avg : 0.0);

            // Sub-rows only when the block was actually split (fine mode, or several stages)
            if (g_rows[g] < 2)
                continue;

            for (int i = 0; i < PROF_N; i++)
            {
                if (chimes_prof_group[i] != g || sum[O_LAPS + i] <= 0.0)
                    continue;

                const double a    = sum[O_T + i] / P;
                const double ovh  = (sum[O_LAPS + i] / P) * tick_cost;
                const double nsum = sum[O_N + i];

                char sublabel[64];
                snprintf(sublabel, sizeof(sublabel), "  %c %s", chimes_prof_kernel[i] ? '*' : ' ', chimes_prof_name[i]);

                printf("%-34s%12.4e%12.4e%8.2f%%%7.2f", sublabel, a, mx[O_T + i], 100.0 * a / denom, (a > 0.0) ? mx[O_T + i] / a : 0.0);

                if (nsum > 0.0)
                    printf("%13.5g%11.4g", nsum / P, 1.0e6 * sum[O_T + i] / nsum);
                else
                    printf("%13s%11s", "-", "-");

                printf("%8s%10.4f%6.1f%%\n", "", 1.0e3 * a / ncalls, (a > 0.0) ? 100.0 * ovh / a : 0.0);
            }
        }

        printf("%s\n", rule.c_str());
        printf("%-34s%12.4e%12s%8.2f%%\n", "SUM of rows", avg_sum, "", 100.0 * avg_sum / denom);
        printf("%-34s%12.4e%12.4e%8s%7.2f\n", "compute() wall", sum[O_WALL] / P, mx[O_WALL], "", (sum[O_WALL] > 0.0) ? mx[O_WALL] / (sum[O_WALL] / P) : 0.0);
        printf("%-34s%12.4e\n", "unaccounted", sum[O_WALL] / P - avg_sum);
        printf("%s\n", rule.c_str());

        // Headline: how the n-body cost splits
        const double b  = g_avg[PG_2B] + g_avg[PG_TRIP] + g_avg[PG_3B] + g_avg[PG_4B];
        const double bd = (b > 0.0) ? b : 1.0;
        printf("Body-order split of n-body time (%.4e s/rank, %.2f%% of compute):  2B %5.1f%% | 3B %5.1f%% | 4B %5.1f%%", b, 100.0 * b / denom, 100.0 * g_avg[PG_2B] / bd, 100.0 * g_avg[PG_3B] / bd, 100.0 * g_avg[PG_4B] / bd);
        if (g_laps[PG_TRIP] > 0.0)
            printf(" | shared triplet %5.1f%%", 100.0 * g_avg[PG_TRIP] / bd);
        printf("\n");
#ifdef CHIMES_PROFILE_FINE
        const double bk  = g_ker[PG_2B] + g_ker[PG_3B] + g_ker[PG_4B];
        const double bkd = (bk > 0.0) ? bk : 1.0;
        printf("  eval kernels only        (%.4e s/rank, %.2f%% of compute):  2B %5.1f%% | 3B %5.1f%% | 4B %5.1f%%\n", bk, 100.0 * bk / denom, 100.0 * g_ker[PG_2B] / bkd, 100.0 * g_ker[PG_3B] / bkd, 100.0 * g_ker[PG_4B] / bkd);
#else
        printf("  (build with -DCHIMES_PROFILE_FINE to split each body order into its stages)\n");
#endif

        const double seeds = sum[O_EX + EX_4B_SEEDS];
        if (seeds > 0.0)
        {
            const double quads = sum[O_ITEM + PG_4B];
            const double cand  = sum[O_EX + EX_4B_CAND];
            printf("4-body detail: %.5g seeds/rank | %.5g l-candidates/rank, %.1f%% become quads | %.2f quads/seed" " | %.3f prepares/seed | %.1f%% of quads via reuse\n", seeds / P, cand / P, (cand  > 0.0) ? 100.0 * quads / cand : 0.0, quads / seeds, sum[O_EX + EX_4B_PREP] / seeds, (quads > 0.0) ? 100.0 * sum[O_EX + EX_4B_REUSE] / quads : 0.0);
        }

        if (g_laps[PG_NEIGH] > 0.0)
            printf("Many-body list build: %.2f%% of compute over %.0f rebuild(s)/rank.\n", 100.0 * g_avg[PG_NEIGH] / denom, sum[O_ITEM + PG_NEIGH] / P);
        printf("Est. timer overhead: %.4e s/rank (%.2f%% of compute), %.1f ns per timer read; included in the rows above.\n", ovh_total, 100.0 * ovh_total / denom, 1.0e9 * tick_cost);
        printf("%s\n\n", dbl.c_str());
        fflush(stdout);
    }

    chimes_prof.reset();
}

#else

#define CHIMES_PROF_BEGIN()          do { } while (0)
#define CHIMES_PROF_LAP(id)          do { } while (0)
#define CHIMES_PROF_BLOCK_END(id)    do { } while (0)
#define CHIMES_PROF_END(id)          do { } while (0)
#define CHIMES_PROF_COUNT(g)         do { } while (0)
#define CHIMES_PROF_ACTIVE(g, e)     do { } while (0)
#define CHIMES_PROF_EXTRA(x)         do { } while (0)
#define CHIMES_PROF_FINE_LAP(id)     do { } while (0)
#define CHIMES_PROF_EVAL_DECL()      do { } while (0)
#define CHIMES_PROF_EVAL_ID(id)      do { } while (0)
#define CHIMES_PROF_EVAL_LAP()       do { } while (0)

#endif
// ==== CHIMES_PROFILE BLOCK END ====

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
    //  double max_val = 20;
    //  for (const auto& row : cutoff_2b) {
    //      for (double val : row) {
    //          if (val > max_val) {
    //              max_val = val;
    //          }
    //      }
    //  }
    //  maxcut_3b=max_val;
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
    double  **x    = atom -> x; // Access to system coordinates

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

    int i,j,k,l,inum,jnum,knum,lnum, ii, jj, kk, ll;         // Local iterator vars
    int *ilist,*jlist,*klist,*llist, *numneigh,**firstneigh; // Local neighborlist vars
    tagint  *tag   = atom -> tag;                            // Access to global atom indices
    int     itag, jtag, ktag, ltag;                          // holds tags
    int     *type  = atom -> type;                           // Access to system atom types
    double  **x    = atom -> x;                              // Access to system coordinates

    double maxcut_3b_padded = maxcut_3b + neighbor-> skin;

    double dist_ij, dist_ik, dist_il, dist_jk, dist_jl, dist_kl;

    ////////////////////////////////////////
    // Access to neighbor list vars
    ////////////////////////////////////////

    inum       = list -> inum;       // length of the list
    ilist      = list -> ilist;      // list of i atoms for which neighbor list exists
    numneigh   = list -> numneigh;   // length of each of the ilist neighbor lists
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

            klist = firstneigh[i];  // ChIMES assumes all atoms must be within cutoff of eachother for a valid interaction
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
    CHIMES_PROF_BEGIN();

    // Everything is inside this block so that the destruction of the scratch
    // objects (chimesXBTmp, reuse cache) is also charged to the profile.
    {
        // Vars for access to chimesFF compute_XB functions

        std::vector<double> stensor(6);     // stress tensor contribution
        int atmidxlst[6][2];

        // General LAMMPS compute vars

        int i, j, k, l, inum, jnum, lnum, ii, jj;
        int *ilist, *jlist, *llist, *numneigh, **firstneigh;
        int idx;

        double **f   = atom->f;            // Access to system forces
        int *type    = atom->type;         // LAMMPS types count from 1, chimesFF from 0
        tagint *tag  = atom->tag;          // Global atom indices
        tagint itag, jtag, ktag, ltag;
        double energy;

        int me = comm->me;
        MPI_Comm_rank(world, &me);

        // Set up vars controlling if energy/pressure (virial) contributions are computed

        if (eflag || vflag) {
            ev_setup(eflag, vflag);
        } else {
            evflag      = 0;
            vflag_fdotr = 0;
            vflag_atom  = 0;
        }

#ifdef FINGERPRINT
        std::vector<std::vector<double>> tmp_dist_2b;
        std::vector<std::vector<double>> tmp_dist_3b;
        std::vector<std::vector<double>> tmp_dist_4b;
        bool tmp_FP = false;
        bool valid_order;
        if (fingerprint && (update->ntimestep % IO_freq == 0))
            tmp_FP = true;
#endif

        // Access to (2-body) neighbor list vars

        inum       = list->inum;
        ilist      = list->ilist;
        numneigh   = list->numneigh;
        firstneigh = list->firstneigh;

        chimes2BTmp chimes_2btmp(chimes_calculator.poly_orders[0]);
        chimes3BTmp chimes_3btmp(chimes_calculator.poly_orders[1]);
        chimes4BTmp chimes_4btmp(chimes_calculator.poly_orders[2]);

        CHIMES_PROF_LAP(PROF_SETUP);

        // Build the ChIMES many-body (triplet) list only when the LAMMPS list was rebuilt

        if (neighbor->ago == 0) {
            if (chimes_calculator.rank == 0)
                std::cout << "Updating chimesFF neighbor lists..." << std::endl;

            build_mb_neighlists();

            if (chimes_calculator.rank == 0) {
                std::cout << "  Rank " << me << " 3-body list size: " << neighborlist_3mers.size() << std::endl;
                std::cout << "  Rank " << me << " 4-body list size: " << neighborlist_4mers.size() << std::endl;
                std::cout << "  ...update complete" << std::endl;
            }

            CHIMES_PROF_COUNT(PG_NEIGH);
            CHIMES_PROF_LAP(PROF_MB_BUILD);
        }

        chimes_calculator.reset_badness();

        ////////////////////////////////////////
        // Compute 1-body interactions
        // (own loop so its cost is not mixed into the 2-body timing)
        ////////////////////////////////////////

        std::fill(stensor.begin(), stensor.end(), 0.0);

        for (ii = 0; ii < inum; ii++)
        {
            i = ilist[ii];

            if (chimes_type[type[i]-1] < 0)
                continue;

            energy = 0.0;
            chimes_calculator.compute_1B(chimes_type[type[i]-1], energy);

            atmidxlst[0][0] = i;

            if (evflag)
                ev_tally_mb(1, 0, atmidxlst, energy, stensor);

            CHIMES_PROF_COUNT(PG_1B);
        }

        CHIMES_PROF_BLOCK_END(PROF_1B);

        ////////////////////////////////////////
        // Compute 2-body interactions
        ////////////////////////////////////////

        for (ii = 0; ii < inum; ii++)
        {
            i     = ilist[ii];
            itag  = tag[i];

            if (chimes_type[type[i]-1] < 0)
                continue;

            jlist = firstneigh[i];
            jnum  = numneigh[i];

            for (jj = 0; jj < jnum; jj++)
            {
                j     = jlist[jj];
                jtag  = tag[j];
                j    &= NEIGHMASK;

                if (chimes_type[type[j]-1] < 0)
                    continue;

                if (jtag <= itag) // full neighbor list: evaluate each pair once
                    continue;

                dist = get_dist(i, j, &dr[0]);

                typ_idxs_2b[0] = chimes_type[type[i]-1];
                typ_idxs_2b[1] = chimes_type[type[j]-1];

                CHIMES_PROF_FINE_LAP(PROF_2B_DIST);

                std::fill(force_2b.begin(), force_2b.end(), 0.0);
                std::fill(stensor.begin(), stensor.end(), 0.0);

                energy = 0.0;
#ifdef TABULATION
                if (chimes_calculator.tabulate_2B)
                    chimes_calculator.compute_2B_tab(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp);
                else
#endif
                {
#ifdef FINGERPRINT
                    valid_order = (i < j);
                    if (tmp_FP && valid_order) {
                        double tmp_force_scalar;
                        chimes_calculator.compute_2B(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp, tmp_force_scalar, tmp_dist_2b, tmp_FP && valid_order);
                    } else
#endif
                    {
                        chimes_calculator.compute_2B(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp);
                    }
                }

                CHIMES_PROF_FINE_LAP(PROF_2B_EVAL);

                for (idx = 0; idx < 3; idx++)
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

                CHIMES_PROF_COUNT(PG_2B);
                CHIMES_PROF_ACTIVE(PG_2B, energy);
                CHIMES_PROF_FINE_LAP(PROF_2B_FORCE_TALLY);
            } // end loop over j neighbors
        }     // end loop over i atoms

        CHIMES_PROF_BLOCK_END(PROF_2B_FORCE_TALLY);

#ifdef FINGERPRINT
        std::string ts = std::to_string(update->ntimestep);
        if (tmp_FP)
        {
            std::stringstream filename;
            filename << ts << "." << std::to_string(chimes_calculator.rank) << ".2b_clusters.txt";
            writeClusterDataComp(filename.str(), tmp_dist_2b);
        }
#endif

        // Document badness for configuration: current timestep, worst badness seen by rank
        if (for_fitting && (update->ntimestep % output->every_dump[0] == 0))
            badness_stream << update->ntimestep << " " << chimes_calculator.get_badness() << endl;

        CHIMES_PROF_LAP(PROF_IO);

        ////////////////////////////////////////
        // Compute 3- and 4-body interactions in one pass over the triplet list
        // (4B clusters are grown from each triplet seed; subgraph reuse)
        ////////////////////////////////////////

        if (chimes_calculator.poly_orders[1] > 0 || chimes_calculator.poly_orders[2] > 0)
        {
            std::vector<chimesFF::chimes4BTripletReuseTmp> reuse_cache;
            reuse_cache.reserve(8);

            const int ntrip = (int) neighborlist_3mers.size();

            for (ii = 0; ii < ntrip; ii++)
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

                // Shared triplet distances (used by both 3B and 4B)

                dist_3b[0] = get_dist(i, j, &dr_3b[0*CHDIM]);
                dist_3b[1] = get_dist(i, k, &dr_3b[1*CHDIM]);
                dist_3b[2] = get_dist(j, k, &dr_3b[2*CHDIM]);

                const double dist_ij = dist_3b[0];
                const double dist_ik = dist_3b[1];
                const double dist_jk = dist_3b[2];

                const int type_i = chimes_type[type[i]-1];
                const int type_j = chimes_type[type[j]-1];
                const int type_k = chimes_type[type[k]-1];

                CHIMES_PROF_COUNT(PG_TRIP);
                CHIMES_PROF_LAP(PROF_TRIP_DIST);

                /*
                 * 3B calculation.
                 *
                 * neighborlist_3mers is built with the skin-padded 3B cutoff,
                 * so keep the explicit (unpadded) maxcut check here.
                 */
                if (chimes_calculator.poly_orders[1] > 0 && dist_ij < maxcut_3b && dist_ik < maxcut_3b && dist_jk < maxcut_3b)
                {
                    typ_idxs_3b[0] = type_i;
                    typ_idxs_3b[1] = type_j;
                    typ_idxs_3b[2] = type_k;

                    std::fill(force_3b.begin(), force_3b.end(), 0.0);
                    std::fill(stensor.begin(), stensor.end(), 0.0);

                    energy = 0.0;

#ifdef TABULATION
                    if (chimes_calculator.tabulate_3B)
                        chimes_calculator.compute_3B_tab(dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);
                    else
#endif
                        chimes_calculator.compute_3B(dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);

                    CHIMES_PROF_FINE_LAP(PROF_3B_EVAL);

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

                    CHIMES_PROF_COUNT(PG_3B);
                    CHIMES_PROF_ACTIVE(PG_3B, energy);
                    CHIMES_PROF_LAP(PROF_3B_FORCE_TALLY);   // closes the 3B stage (both modes)
                }

                /*
                 * 4B: grow quadruplets from this triplet seed.
                 *
                 * Assumption: maxcut_3b >= maxcut_4b, so neighborlist_3mers holds
                 * every possible 4B seed triplet.
                 */
                if (chimes_calculator.poly_orders[2] > 0 && dist_ij < maxcut_4b && dist_ik < maxcut_4b && dist_jk < maxcut_4b)
                {
                    CHIMES_PROF_EXTRA(EX_4B_SEEDS);

                    reuse_cache.clear();

                    // Copy triplet side into 4B runtime pair slots: 0 = ij, 1 = ik, 3 = jk

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
                     * Candidate fourth atoms: scan i's full neighbor list and enforce
                     * ltag > itag, jtag, ktag so each 4B cluster is evaluated once.
                     */
                    llist = firstneigh[i];
                    lnum  = numneigh[i];

                    for (int ll = 0; ll < lnum; ll++)
                    {
                        l  = llist[ll];
                        l &= NEIGHMASK;

                        if ((l == i) || (l == j) || (l == k))
                            continue;

                        ltag = tag[l];

                        if ((ltag < itag) || (ltag < jtag) || (ltag < ktag))
                            continue;

                        if (chimes_type[type[l]-1] < 0)
                            continue;

                        CHIMES_PROF_EXTRA(EX_4B_CAND);

                        dist_4b[2] = get_dist(i, l, &dr_4b[2*CHDIM]);
                        if (dist_4b[2] >= maxcut_4b)
                            continue;

                        dist_4b[4] = get_dist(j, l, &dr_4b[4*CHDIM]);
                        if (dist_4b[4] >= maxcut_4b)
                            continue;

                        dist_4b[5] = get_dist(k, l, &dr_4b[5*CHDIM]);
                        if (dist_4b[5] >= maxcut_4b)
                            continue;

                        // Seed setup, rejected candidates and the accepted l's distances
                        CHIMES_PROF_FINE_LAP(PROF_4B_DIST);

                        const int type_l = chimes_type[type[l]-1];

                        typ_idxs_4b[3] = type_l;

                        /*
                         * Reuse cache is keyed by type_l_context: one left object per
                         * triplet for single-species systems, one per distinct
                         * fourth-atom type otherwise.
                         */
                        chimesFF::chimes4BTripletReuseTmp *reuse_ptr = nullptr;

                        for (int c = 0; c < (int) reuse_cache.size(); c++)
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

                            const bool ok = chimes_calculator.prepare_4B_triplet_reuse(dist_3b, typ_idxs_4b, chimes_4btmp, reuse_new);

                            if (ok)
                            {
                                reuse_cache.push_back(std::move(reuse_new));
                                reuse_ptr = &reuse_cache.back();
                            }

                            CHIMES_PROF_EXTRA(EX_4B_PREP);
                            CHIMES_PROF_FINE_LAP(PROF_4B_PREPARE);
                        }

                        std::fill(force_4b.begin(), force_4b.end(), 0.0);
                        std::fill(stensor.begin(), stensor.end(), 0.0);

                        energy = 0.0;

                        CHIMES_PROF_EVAL_DECL();

                        if (reuse_ptr != nullptr && reuse_ptr->valid)
                        {
                            CHIMES_PROF_EVAL_ID(PROF_4B_EVAL_REUSE);
                            CHIMES_PROF_EXTRA(EX_4B_REUSE);
                            chimes_calculator.compute_4B_from_triplet_reuse(dist_4b, dr_4b, typ_idxs_4b, *reuse_ptr, force_4b, stensor, energy, chimes_4btmp);
                        }
                        else
                        {
#ifdef TABULATION
                            if (chimes_calculator.tabulate_4B_cp_direct)
                            {
                                CHIMES_PROF_EVAL_ID(PROF_4B_EVAL_CP_DIRECT);
                                chimes_calculator.compute_4B_cp_direct(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);
                            }
                            else if (chimes_calculator.tabulate_4B_cp)
                            {
                                CHIMES_PROF_EVAL_ID(PROF_4B_EVAL_CP_TAB);
                                chimes_calculator.compute_4B_cp_tab(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);
                            }
                            else if (chimes_calculator.tabulate_4B_svd4x2)
                            {
                                CHIMES_PROF_EVAL_ID(PROF_4B_EVAL_SVD4);
                                chimes_calculator.compute_4B_svd4x2_tab(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);
                            }
                            else if (chimes_calculator.tabulate_4B_svd3x3)
                            {
                                CHIMES_PROF_EVAL_ID(PROF_4B_EVAL_SVD3);
                                chimes_calculator.compute_4B_svd3x3_tab(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);
                            }
                            else if (chimes_calculator.tabulate_4B_coeff)
                            {
                                CHIMES_PROF_EVAL_ID(PROF_4B_EVAL_COEFF);
                                chimes_calculator.compute_4B_tab_coeff(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);
                            }
                            else
#endif
                            {
                                CHIMES_PROF_EVAL_ID(PROF_4B_EVAL_DIRECT);
                                chimes_calculator.compute_4B(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);
                            }
                        } // end else no reuse

                        CHIMES_PROF_EVAL_LAP();

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

                        CHIMES_PROF_COUNT(PG_4B);
                        CHIMES_PROF_ACTIVE(PG_4B, energy);
                        CHIMES_PROF_FINE_LAP(PROF_4B_FORCE_TALLY);
                    } // end loop over candidate l

                    // Trailing l scan; in coarse mode this lap closes the whole 4B stage for this seed
                    CHIMES_PROF_BLOCK_END(PROF_4B_DIST);
                } // end 4B seed
            }     // end loop over triplets

            CHIMES_PROF_BLOCK_END(PROF_TRIP_DIST);   // triplet loop exit
        }

        if (vflag_fdotr)
            virial_fdotr_compute();

        CHIMES_PROF_LAP(PROF_FDOTR);
    }

#ifdef CHIMES_PROFILE
    CHIMES_PROF_END(PROF_OTHER);   // scratch destructors / function exit; closes the stopwatch

    if (update->ntimestep % CHIMES_PROFILE_EVERY == 0)
    {
        chimes_profile_report(world, comm->me, static_cast<long long>(update->ntimestep));
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