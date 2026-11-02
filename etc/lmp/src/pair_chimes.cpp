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
#ifdef CHIMES_PAIR_REUSE
#include <array>
#include <cstring>
#endif

using namespace LAMMPS_NS;

// Ownership key of an atom.  A pair is evaluated by its atom with the smaller key and a 3B/4B
// cluster by its atom with the smallest key (equal keys: periodic images of one atom).  With
// pair reuse the key is a bijective hash of the tag: with the tag itself, atoms numbered in
// spatial order (as most data files are) make the rank holding the lower tags own nearly all
// pairs and clusters that straddle its domain boundaries, which unbalances the MPI ranks.
// Without pair reuse the key is the tag, as before.
typedef unsigned long long chimes_key_t;

static inline chimes_key_t owner_key(tagint t)
{
#ifdef CHIMES_PAIR_REUSE
    chimes_key_t z = (chimes_key_t) t + 0x9E3779B97F4A7C15ULL;   // splitmix64 finalizer
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
#else
    return (chimes_key_t) t;
#endif
}

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
    PG_PAIR,       // shared pair work (pair reuse: CP functions of each pair, pair forces)
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
    "shared pair cache",
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
    "pair-slots",
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

    PROF_RU_BUILD,           // pair reuse: pair/cluster list build (sweep 1)
    PROF_RU_PAIRS,           // pair reuse: pair distances + CP functions of every pair
    PROF_RU_3B,              // pair reuse: 3B contraction
    PROF_RU_4B,              // pair reuse: 4B seed products + quadruplet contraction
    PROF_RU_FORCE,           // pair reuse: force scalars -> forces + tally

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

    "pair/cluster lists",
    "distances+CP functions",
    "eval_pair_cache",
    "eval_pair_cache",
    "forces+tally",

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
    PG_NEIGH, PG_PAIR, PG_3B, PG_4B, PG_PAIR,
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
    0, 1, 1, 1, 0,
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
#define CHIMES_PROF_COUNT_N(g, n)    (chimes_prof.items[(g)] += (double)(n))
#define CHIMES_PROF_ACTIVE_N(g, n)   (chimes_prof.active[(g)] += (double)(n))
#define CHIMES_PROF_EXTRA_N(x, n)    (chimes_prof.extra[(x)] += (double)(n))

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

        const int order[PG_N] = { PG_1B, PG_2B, PG_TRIP, PG_PAIR, PG_3B, PG_4B, PG_NEIGH, PG_OVERHEAD };

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
        const double b  = g_avg[PG_2B] + g_avg[PG_TRIP] + g_avg[PG_PAIR] + g_avg[PG_3B] + g_avg[PG_4B];
        const double bd = (b > 0.0) ? b : 1.0;
        printf("Body-order split of n-body time (%.4e s/rank, %.2f%% of compute):  2B %5.1f%% | 3B %5.1f%% | 4B %5.1f%%", b, 100.0 * b / denom, 100.0 * g_avg[PG_2B] / bd, 100.0 * g_avg[PG_3B] / bd, 100.0 * g_avg[PG_4B] / bd);
        if (g_laps[PG_TRIP] > 0.0)
            printf(" | shared triplet %5.1f%%", 100.0 * g_avg[PG_TRIP] / bd);
        if (g_laps[PG_PAIR] > 0.0)
            printf(" | shared pair cache %5.1f%%", 100.0 * g_avg[PG_PAIR] / bd);
        printf("\n");
#ifdef CHIMES_PROFILE_FINE
        const double bk  = g_ker[PG_2B] + g_ker[PG_PAIR] + g_ker[PG_3B] + g_ker[PG_4B];
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
#define CHIMES_PROF_COUNT_N(g, n)    do { } while (0)
#define CHIMES_PROF_ACTIVE_N(g, n)   do { } while (0)
#define CHIMES_PROF_EXTRA_N(x, n)    do { } while (0)
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

#ifdef CHIMES_PAIR_REUSE
    reuse_greedy = false;
    reuse_block  = 32;
    reuse_skin   = -1.0;
    ru_pad       = 0.0;
    ru_mid_ok    = false;
    ru_nbuild    = 0;
    ru_nmid      = 0;
    ru_nt_inner  = 0;
    ru_npair_prev = 0;
    reuse_active = false;
    reuse_3b     = false;
    reuse_4b     = false;
    reuse_warned = false;
    reuse_mixed4 = false;
    ru_block     = 1;
    ru_rank_max  = 0;
    x3_rank      = 0;
    x3_ntab_cp   = 0;
    x4_rank      = 0;
    ru_nseed_cp  = 0;
    ru_nx4_group = 0;
#endif

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
#ifdef CHIMES_PAIR_REUSE
    // Pair-reuse keywords; everything else is handled below as before
    //   greedy_seeds     choose seed triangles with the greedy double sweep
    //   reuse_block N    rank terms per pass over the cluster lists (default 32; with
    //                    -DCHIMES_PAIR_CACHE only, otherwise ignored)
    //   mb_skin X        padding of the 3B/4B cluster lists; with X < skin the lists are rebuilt
    //                    on their own displacement check.  Default: half the neighbor skin when
    //                    that is allowed (cutoff + X <= smallest 2B cutoff), else the skin
    std::vector<char *> kept;
    for (int a = 0; a < narg; a++)
    {
        if (strcmp(arg[a], "greedy_seeds") == 0)
        {
            reuse_greedy = true;
            continue;
        }
        if (strcmp(arg[a], "mb_skin") == 0)
        {
            if (a + 1 >= narg)
                error->all(FLERR, "Illegal pair_style chimesFF command: mb_skin needs a value");
            reuse_skin = utils::numeric(FLERR, arg[a+1], false, lmp);
            if (reuse_skin < 0.0)
                error->all(FLERR, "Illegal pair_style chimesFF command: mb_skin must be >= 0");
            a++;
            continue;
        }
        if (strcmp(arg[a], "reuse_block") == 0)
        {
            if (a + 1 >= narg)
                error->all(FLERR, "Illegal pair_style chimesFF command: reuse_block needs a value");
            reuse_block = utils::inumeric(FLERR, arg[a+1], false, lmp);
            if (reuse_block < 1)
                error->all(FLERR, "Illegal pair_style chimesFF command: reuse_block must be >= 1");
            a++;
            continue;
        }
        kept.push_back(arg[a]);
    }
    narg = (int) kept.size();
    arg  = kept.data();
#endif

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

    // Full list for the local atoms only: every cluster is built from its local atom with the
    // lowest tag, so the neighbor lists of ghost atoms were never used
    neighbor->add_request(this, NeighConst::REQ_FULL);
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

    int i,j,k,inum,jnum, ii, jj, kk;                         // Local iterator vars
    int *ilist,*jlist, *numneigh,**firstneigh;               // Local neighborlist vars
    tagint  *tag   = atom -> tag;                            // Access to global atom indices
    chimes_key_t ikey, jkey;                                 // ownership keys (owner_key)
    int     *type  = atom -> type;                           // Access to system atom types

    double maxcut_3b_padded = maxcut_3b + neighbor-> skin;

    ////////////////////////////////////////
    // Access to neighbor list vars
    ////////////////////////////////////////

    inum       = list -> inum;       // length of the list
    ilist      = list -> ilist;      // list of i atoms for which neighbor list exists
    numneigh   = list -> numneigh;   // length of each of the ilist neighbor lists
    firstneigh = list -> firstneigh; // point to the list of neighbors of i

    // j and k candidates of atom i: neighbors with a ChIMES type, key >= key of i and inside the
    // padded 3B cutoff, kept in neighbor-list order so the triplets come out in the same order
    // as from the full-list double loop.  Triplet storage from earlier builds is reused, so
    // there is no allocation per triplet.
    size_t n3 = 0;

    for (ii = 0; ii < inum; ii++) // Loop over real atoms (ai)
    {
        i     = ilist[ii];
        ikey  = owner_key(tag[i]);
        jlist = firstneigh[i];
        jnum  = numneigh[i];

        if (chimes_type[type[i]-1] < 0)
            continue;

        mb_short.clear();

        for (jj = 0; jj < jnum; jj++)
        {
            j = jlist[jj] & NEIGHMASK;

            if (j == i || chimes_type[type[j]-1] < 0 || owner_key(tag[j]) < ikey)
                continue;

            if (get_dist(i,j) >= maxcut_3b_padded)
                continue;

            mb_short.push_back(j);
        }

        const int nshort = (int) mb_short.size();

        for (jj = 0; jj < nshort; jj++)
        {
            j    = mb_short[jj];
            jkey = owner_key(tag[j]);

            for (kk = 0; kk < nshort; kk++)
            {
                k = mb_short[kk];

                if (k == j || owner_key(tag[k]) < jkey)
                    continue;

                if (get_dist(j,k) >= maxcut_3b_padded)
                    continue;

                if (n3 < neighborlist_3mers.size())
                {
                    neighborlist_3mers[n3][0] = i;
                    neighborlist_3mers[n3][1] = j;
                    neighborlist_3mers[n3][2] = k;
                }
                else
                {
                    tmp_3mer[0] = i;
                    tmp_3mer[1] = j;
                    tmp_3mer[2] = k;

                    neighborlist_3mers.push_back(tmp_3mer);
                }
                n3++;
            }
        }
    }

    neighborlist_3mers.resize(n3);

    return;

}

#ifdef CHIMES_PAIR_REUSE
/* ----------------------------------------------------------------------
   Pair + triplet subgraph reuse (on in -DTABULATION builds; off with
   -DCHIMES_NO_PAIR_REUSE; pair cache with -DCHIMES_PAIR_CACHE)

   Covers 3B CP tables, the exact 3B polynomial and 4B CP models; other
   models of a body order stay in the per-cluster loop of compute().

   A CP model writes each 3B/4B cluster energy as a sum over rank terms of
   products of one-dimensional pair functions,

       E_4B(A0 A1 A2 A3) = sum_q  L_q(A0 A1 A2) * X_A0A3,q X_A1A3,q X_A2A3,q
       L_q(A0 A1 A2)     = X_A0A1,q X_A0A2,q X_A1A2,q

   where every X depends on one pair distance only.  Reuse at three levels:

   pairs     With -DCHIMES_PAIR_CACHE, X_q(r) and dX_q/dr are computed once
             per step for each pair and each CP table (cluster slot) the pair
             is used with, and shared by every triplet and quadruplet that
             contains the pair.  Without it they are computed when a cluster
             is contracted: three per triplet, three per seed and three per
             quadruplet (the old per-cluster code: six per quadruplet).  That
             saves the cache's memory, X and dX/dr of 32 rank terms per pair
             and table, at the cost of more table lookups.
   triplets  Quadruplets are grouped by a seed triangle A0 A1 A2.  L_q and its
             three derivatives are formed once per seed; a quadruplet only
             adds its fourth-atom side.  The fourth-atom products are also
             summed over the seed's quadruplets, so the energy and the
             seed-side forces take one dot product per seed, not per quad.
   forces    Force scalars (dE/dr)/r are summed per pair, and each pair's
             force and virial are applied once.  When per-atom energy or
             virial is requested, clusters are tallied one at a time instead,
             so the per-atom partition is the same as without reuse.

   exact 3B  The pair functions G_n = fcut T_n (n = 0..order) are cached
             like the CP ones.  Triplets are grouped by one shared pair (slot
             s); per group M = sum_x G_s[x] T_s[x] (T_s = coefficient tensor
             with slot s first) is formed once, and each triplet adds the
             bilinear form G_sa M G_sb.  The shared pair is chosen by a greedy
             set cover (exact: only the summation order changes).

   Cluster lists (sweep 1)
     Ownership uses a hashed key of the tag (owner_key): a pair belongs to
     its atom with the smaller key, a cluster to its atom with the smallest
     key, which spreads the clusters on domain boundaries evenly over the
     MPI ranks.  Each local atom i gets a short list: neighbors with key >=
     key of i inside the padded many-body cutoff, sorted by key.  Triplets
     and quadruplets of i are enumerated from it alone, in one pass for 3B
     and 4B: each pair of entries is tested once into per-entry bitsets, and
     the clusters are the set bits of one row (3B) or of two rows ANDed
     (4B), 64 entries per word.
     The padding (mb_skin, default half the neighbor skin) is independent of
     the neighbor skin: when padding < skin the lists are rebuilt once any
     atom, local or ghost, has moved more than padding/2 since their build,
     which needs cutoff + padding <= smallest 2B cutoff.  Clusters whose
     pairs were all at least the padding inside their cutoffs at the build
     ("inner") stay valid until the next build and are not checked per step.
     The 3-body list (neighborlist_3mers) is only built for models the
     cache does not cover, and no ghost neighbor lists are requested.

   Seed choice
     default        quadruplet i<j<k<l (key order) uses seed i j k, the same
                    grouping as the triplet-seeded loop without pair reuse.
     greedy_seeds   greedy set cover: repeatedly take the triangle (plus the
                    type of the fourth atom) shared by the most unassigned
                    quadruplets, which roughly halves the number of seeds.  A
                    quadruplet is then evaluated with its atoms relabeled
                    (seed first).  The ChIMES coefficients are permutation
                    symmetric, so this changes results only at the level of
                    the CP fit / table interpolation error.

   Each step (sweep 2) first selects the clusters inside the true cutoffs
   and, with the cache, the pair-cache entries they use, then evaluates those
   entries once, packed (entries are numbered in the order the evaluation
   reads them), and contracts.  Without the cache it contracts the selected
   clusters directly, evaluating their entries into scratch.  Sums over rank
   terms run in RU_LANES (8) independent partial sums, one SIMD register
   each, and exact 3B triplets of a group in batches of 8, one triplet per
   SIMD lane.  With the cache, rank terms
   are processed in passes of reuse_block terms (default 32): the contraction
   streams the cache once per pass, so a small cache slice that stays in
   cache beats fewer, larger passes for high-rank models.  Without it there
   is one pass over all rank terms.
------------------------------------------------------------------------- */

namespace
{
// Open-addressing hash map from a non-negative 64-bit key to an int id
class ChimesIdMap
{
  public:
    void init(size_t expected)
    {
        size_t cap = 16;
        int bits = 4;

        while (cap < 2 * expected + 2)
        {
            cap <<= 1;
            bits++;
        }

        keys.assign(cap, -1);
        vals.assign(cap, 0);
        shift = 64 - bits;
        count = 0;
    }

    // Id stored for key; if the key is new, store next_id and return it
    int get(long long key, int next_id)
    {
        if (2 * (count + 1) > keys.size())
            grow();

        const size_t mask = keys.size() - 1;
        size_t h = slot(key);

        while (true)
        {
            if (keys[h] == key)
                return vals[h];

            if (keys[h] < 0)
            {
                keys[h] = key;
                vals[h] = next_id;
                count++;
                return next_id;
            }

            h = (h + 1) & mask;
        }
    }

  private:
    std::vector<long long> keys;
    std::vector<int> vals;
    int shift = 60;
    size_t count = 0;

    size_t slot(long long key) const
    {
        return (size_t) (((unsigned long long) key * 0x9E3779B97F4A7C15ULL) >> shift);
    }

    void grow()
    {
        std::vector<long long> old_keys;
        std::vector<int> old_vals;
        old_keys.swap(keys);
        old_vals.swap(vals);

        init(old_keys.size());

        for (size_t h = 0; h < old_keys.size(); h++)
            if (old_keys[h] >= 0)
                get(old_keys[h], old_vals[h]);
    }
};

// Independent partial sums per contraction (one SIMD register of doubles each with AVX-512;
// two with AVX2): rank terms of the CP contractions, triplets of an exact 3B group
#define RU_LANES 8

// Contraction kernels of the pair-cache path.  They are functions so that the __restrict
// parameters let the compiler vectorize them (restrict on block-scope pointers is not
// reliably honored).  Sums over rank terms q run in RU_LANES partial sums, then the
// remainder; out[] receives the totals.

// 3B CP triplet: E = sum_q x0 x1 x2 and dE/dr of slots 0, 1, 2
static inline void ru_k3_cp(int n, const double * __restrict x0, const double * __restrict x1, const double * __restrict x2, const double * __restrict d0, const double * __restrict d1, const double * __restrict d2, double * __restrict out)
{
    const int nmain = n - n % RU_LANES;
    double se[RU_LANES] = {}, s0[RU_LANES] = {}, s1[RU_LANES] = {}, s2[RU_LANES] = {};

    for (int q = 0; q < nmain; q += RU_LANES)
    {
        for (int l = 0; l < RU_LANES; l++)
        {
            const double x12 = x1[q + l] * x2[q + l];
            const double x02 = x0[q + l] * x2[q + l];
            se[l] += x0[q + l] * x12;
            s0[l] += d0[q + l] * x12;
            s1[l] += d1[q + l] * x02;
            s2[l] += x0[q + l] * x1[q + l] * d2[q + l];
        }
    }

    double e3 = 0.0, g0 = 0.0, g1 = 0.0, g2 = 0.0;

    for (int q = nmain; q < n; q++)
    {
        const double x12 = x1[q] * x2[q];
        const double x02 = x0[q] * x2[q];
        e3 += x0[q] * x12;
        g0 += d0[q] * x12;
        g1 += d1[q] * x02;
        g2 += x0[q] * x1[q] * d2[q];
    }

    for (int l = 0; l < RU_LANES; l++)
    {
        e3 += se[l];
        g0 += s0[l];
        g1 += s1[l];
        g2 += s2[l];
    }

    out[0] = e3;
    out[1] = g0;
    out[2] = g1;
    out[3] = g2;
}

// 4B seed products: L = x0 x1 x2 and its three slot derivatives; S cleared
static inline void ru_k4_seed(int n, const double * __restrict x0, const double * __restrict x1, const double * __restrict x2, const double * __restrict d0, const double * __restrict d1, const double * __restrict d2, double * __restrict Lq, double * __restrict L0, double * __restrict L1, double * __restrict L2, double * __restrict Sq)
{
    for (int q = 0; q < n; q++)
    {
        Lq[q] = x0[q] * x1[q] * x2[q];
        L0[q] = d0[q] * x1[q] * x2[q];
        L1[q] = x0[q] * d1[q] * x2[q];
        L2[q] = x0[q] * x1[q] * d2[q];
        Sq[q] = 0.0;
    }
}

// 4B quadruplet, energy summed per seed: S += y0 y1 y2; out = the three fourth-atom gradients
static inline void ru_k4_quad(int n, const double * __restrict Lq, const double * __restrict y0, const double * __restrict y1, const double * __restrict y2, const double * __restrict dy0, const double * __restrict dy1, const double * __restrict dy2, double * __restrict Sq, double * __restrict out)
{
    const int nmain = n - n % RU_LANES;
    double s0[RU_LANES] = {}, s1[RU_LANES] = {}, s2[RU_LANES] = {};

    for (int q = 0; q < nmain; q += RU_LANES)
    {
        for (int l = 0; l < RU_LANES; l++)
        {
            const double y12 = y1[q + l] * y2[q + l];
            const double Ly0 = Lq[q + l] * y0[q + l];

            Sq[q + l] += y0[q + l] * y12;
            s0[l]     += Lq[q + l] * dy0[q + l] * y12;
            s1[l]     += Ly0 * dy1[q + l] * y2[q + l];
            s2[l]     += Ly0 * y1[q + l] * dy2[q + l];
        }
    }

    double gr0 = 0.0, gr1 = 0.0, gr2 = 0.0;

    for (int q = nmain; q < n; q++)
    {
        const double y12 = y1[q] * y2[q];
        const double Ly0 = Lq[q] * y0[q];

        Sq[q] += y0[q] * y12;
        gr0   += Lq[q] * dy0[q] * y12;
        gr1   += Ly0 * dy1[q] * y2[q];
        gr2   += Ly0 * y1[q] * dy2[q];
    }

    for (int l = 0; l < RU_LANES; l++)
    {
        gr0 += s0[l];
        gr1 += s1[l];
        gr2 += s2[l];
    }

    out[0] = gr0;
    out[1] = gr1;
    out[2] = gr2;
}

// 4B quadruplet on its own (per-atom tallies): out = E, three fourth-atom gradients, three
// seed gradients
static inline void ru_k4_quad_full(int n, const double * __restrict Lq, const double * __restrict L0, const double * __restrict L1, const double * __restrict L2, const double * __restrict y0, const double * __restrict y1, const double * __restrict y2, const double * __restrict dy0, const double * __restrict dy1, const double * __restrict dy2, double * __restrict out)
{
    const int nmain = n - n % RU_LANES;
    double sv[7][RU_LANES] = {};

    for (int q = 0; q < nmain; q += RU_LANES)
    {
        for (int l = 0; l < RU_LANES; l++)
        {
            const double y12   = y1[q + l] * y2[q + l];
            const double Rprod = y0[q + l] * y12;
            const double L     = Lq[q + l];
            const double Ly0   = L * y0[q + l];

            sv[0][l] += L * Rprod;
            sv[1][l] += L * dy0[q + l] * y12;
            sv[2][l] += Ly0 * dy1[q + l] * y2[q + l];
            sv[3][l] += Ly0 * y1[q + l] * dy2[q + l];
            sv[4][l] += L0[q + l] * Rprod;
            sv[5][l] += L1[q + l] * Rprod;
            sv[6][l] += L2[q + l] * Rprod;
        }
    }

    double t[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (int q = nmain; q < n; q++)
    {
        const double y12   = y1[q] * y2[q];
        const double Rprod = y0[q] * y12;
        const double L     = Lq[q];
        const double Ly0   = L * y0[q];

        t[0] += L * Rprod;
        t[1] += L * dy0[q] * y12;
        t[2] += Ly0 * dy1[q] * y2[q];
        t[3] += Ly0 * y1[q] * dy2[q];
        t[4] += L0[q] * Rprod;
        t[5] += L1[q] * Rprod;
        t[6] += L2[q] * Rprod;
    }

    for (int c = 0; c < 7; c++)
    {
        for (int l = 0; l < RU_LANES; l++)
            t[c] += sv[c][l];
        out[c] = t[c];
    }
}

// 4B seed: E = L.S and the three seed gradients
static inline void ru_k4_seed_dot(int n, const double * __restrict Lq, const double * __restrict L0, const double * __restrict L1, const double * __restrict L2, const double * __restrict Sq, double * __restrict out)
{
    const int nmain = n - n % RU_LANES;
    double se[RU_LANES] = {}, s0[RU_LANES] = {}, s1[RU_LANES] = {}, s2[RU_LANES] = {};

    for (int q = 0; q < nmain; q += RU_LANES)
    {
        for (int l = 0; l < RU_LANES; l++)
        {
            se[l] += Lq[q + l] * Sq[q + l];
            s0[l] += L0[q + l] * Sq[q + l];
            s1[l] += L1[q + l] * Sq[q + l];
            s2[l] += L2[q + l] * Sq[q + l];
        }
    }

    double e = 0.0, g0 = 0.0, g1 = 0.0, g2 = 0.0;

    for (int q = nmain; q < n; q++)
    {
        e  += Lq[q] * Sq[q];
        g0 += L0[q] * Sq[q];
        g1 += L1[q] * Sq[q];
        g2 += L2[q] * Sq[q];
    }

    for (int l = 0; l < RU_LANES; l++)
    {
        e  += se[l];
        g0 += s0[l];
        g1 += s1[l];
        g2 += s2[l];
    }

    out[0] = e;
    out[1] = g0;
    out[2] = g1;
    out[3] = g2;
}

// Exact 4B kernels.  N: basis size per pair slot, N3 = N^3 (index s = (d*N + e)*N + f of
// the fourth-atom pairs A0A3, A1A3, A2A3), class matrix M[a][b][c][s] with a b c the seed
// pairs A0A1, A0A2, A1A2.

// Two dot products against the same vector, in RU_LANES partial sums
static inline void ru_dot2(int n, const double * __restrict x, const double * __restrict y, const double * __restrict z, double &xz, double &yz)
{
    const int nmain = n - n % RU_LANES;
    double s0[RU_LANES] = {}, s1[RU_LANES] = {};

    for (int q = 0; q < nmain; q += RU_LANES)
    {
        for (int l = 0; l < RU_LANES; l++)
        {
            s0[l] += x[q + l] * z[q + l];
            s1[l] += y[q + l] * z[q + l];
        }
    }

    double t0 = 0.0, t1 = 0.0;

    for (int q = nmain; q < n; q++)
    {
        t0 += x[q] * z[q];
        t1 += y[q] * z[q];
    }

    for (int l = 0; l < RU_LANES; l++)
    {
        t0 += s0[l];
        t1 += s1[l];
    }

    xz = t0;
    yz = t1;
}

// Group: the group's pair (index a) contracted with a class matrix stored as M[a][x],
// x over N^5:  Q = sum_a G[a] M[a]  and, when dQ is given, dQ = sum_a dG[a] M[a]
static inline void ru_k4x_group(int N, int N5, const double * __restrict M, const double * __restrict G, const double * __restrict dG, double * __restrict Q, double * __restrict dQ)
{
    for (int x = 0; x < N5; x++)
        Q[x] = 0.0;

    if (dQ)
        for (int x = 0; x < N5; x++)
            dQ[x] = 0.0;

    for (int a = 0; a < N; a++)
    {
        const double g = G[a];
        const double * __restrict Ma = M + (size_t) a * N5;

        for (int x = 0; x < N5; x++)
            Q[x] += g * Ma[x];

        if (dQ)
        {
            const double dg = dG[a];

            for (int x = 0; x < N5; x++)
                dQ[x] += dg * Ma[x];
        }
    }
}

// Sum over the two other seed pairs with coefficients k[b*N + c]:  out[s] = sum_bc k[bc] Q[bc][s]
// (Q in [bc][s] layout; every update is a vector of length N^3, so it vectorizes)
static inline void ru_k4x_bc_sum(int N2, int N3, const double * __restrict Q, const double * __restrict k, double * __restrict out)
{
    for (int x = 0; x < N3; x++)
        out[x] = 0.0;

    for (int bc = 0; bc < N2; bc++)
    {
        const double kb = k[bc];
        const double * __restrict Qbc = Q + (size_t) bc * N3;

        for (int x = 0; x < N3; x++)
            out[x] += kb * Qbc[x];
    }
}

// Quadruplet: E = sum_s w[s] a[d] b[e] c[f] and dE/dr of the pairs A0A3, A1A3, A2A3, by the
// chain t = sum_d w[d] a[d] (a matrix over e f), u = sum_e t[e] b[e], E = u.c.  t holds 2 N^2.
static inline void ru_k4x_quad(int N, const double * __restrict w, const double * __restrict a, const double * __restrict da, const double * __restrict b, const double * __restrict db, const double * __restrict c, const double * __restrict dc, double * __restrict t, double * __restrict out)
{
    const int N2 = N * N;
    double * __restrict dt = t + N2;

    for (int x = 0; x < N2; x++)
    {
        t[x]  = 0.0;
        dt[x] = 0.0;
    }

    for (int d = 0; d < N; d++)
    {
        const double ad  = a[d];
        const double dad = da[d];
        const double * __restrict wd = w + (size_t) d * N2;

        for (int x = 0; x < N2; x++)
        {
            t[x]  += ad  * wd[x];
            dt[x] += dad * wd[x];
        }
    }

    double E = 0.0, g0 = 0.0, g1 = 0.0, g2 = 0.0;

    for (int f = 0; f < N; f++)
    {
        double u = 0.0, u0 = 0.0, u1 = 0.0;

        for (int e = 0; e < N; e++)
        {
            u  += t[e*N + f]  * b[e];
            u0 += dt[e*N + f] * b[e];
            u1 += t[e*N + f]  * db[e];
        }

        E  += u  * c[f];
        g0 += u0 * c[f];
        g1 += u1 * c[f];
        g2 += u  * dc[f];
    }

    out[0] = E;
    out[1] = g0;
    out[2] = g1;
    out[3] = g2;
}

// Star product of the fourth-atom pairs: P[s] = a[d] b[e] c[f] (add: P += ...)
static inline void ru_k4x_star(int N, const double * __restrict a, const double * __restrict b, const double * __restrict c, double * __restrict P, bool add)
{
    for (int d = 0; d < N; d++)
    {
        for (int e = 0; e < N; e++)
        {
            const double ab = a[d] * b[e];
            double * __restrict Pde = P + (size_t) (d*N + e) * N;

            if (add)
                for (int f = 0; f < N; f++)
                    Pde[f] += ab * c[f];
            else
                for (int f = 0; f < N; f++)
                    Pde[f] = ab * c[f];
        }
    }
}

// Seed gradients from the summed star products S of the seed's quadruplets, with the
// group matrices in [s][bc] layout (Qt, dQt):  q[bc] = sum_s S[s] Qt[s][bc] (and dq), then
//     dE/dr_x = sum_bc gb gc dq[bc],  dE/dr_y = sum_bc dgb gc q[bc],  dE/dr_z = sum_bc gb dgc q[bc]
// The products over s are vector updates of length N^2 (no reductions over s).
static inline void ru_k4x_seed_grad(int N, int N3, const double * __restrict Qt, const double * __restrict dQt, const double * __restrict S, const double * __restrict gb, const double * __restrict dgb, const double * __restrict gc, const double * __restrict dgc, double * __restrict q, double * __restrict dq, double * __restrict out)
{
    const int N2 = N * N;

    for (int bc = 0; bc < N2; bc++)
    {
        q[bc]  = 0.0;
        dq[bc] = 0.0;
    }

    for (int x = 0; x < N3; x++)
    {
        const double sv = S[x];
        const double * __restrict Qx  = Qt  + (size_t) x * N2;
        const double * __restrict dQx = dQt + (size_t) x * N2;

        for (int bc = 0; bc < N2; bc++)
        {
            q[bc]  += sv * Qx[bc];
            dq[bc] += sv * dQx[bc];
        }
    }

    double gx = 0.0, gy = 0.0, gz = 0.0;

    for (int b = 0; b < N; b++)
    {
        for (int c = 0; c < N; c++)
        {
            gx += gb[b]  * gc[c]  * dq[b*N + c];
            gy += dgb[b] * gc[c]  * q[b*N + c];
            gz += gb[b]  * dgc[c] * q[b*N + c];
        }
    }

    out[0] = gx;
    out[1] = gy;
    out[2] = gz;
}

// Exact 3B, a batch of RU_LANES triplets of one group (lane l = triplet; unused lanes hold
// zeros).  At, dAt, Bt, dBt hold the pair functions of the two other slots lane-minor
// ([a*RU_LANES + l]).  With w = A M, wd = dA M, u = A dM:
//     E = w.B,  dE/dr_sa = wd.B,  dE/dr_sb = w.dB,  dE/dr_s = u.B
// Per triplet the sums run over a and b in the same order as one triplet at a time.
static inline void ru_k3x_batch(int N, const double * __restrict M, const double * __restrict dM, const double * __restrict At, const double * __restrict dAt, const double * __restrict Bt, const double * __restrict dBt, double * __restrict e3, double * __restrict gs, double * __restrict ga, double * __restrict gb)
{
    double se[RU_LANES] = {}, ss[RU_LANES] = {}, sa[RU_LANES] = {}, sb[RU_LANES] = {};

    for (int b = 0; b < N; b++)
    {
        double w[RU_LANES] = {}, wd[RU_LANES] = {}, u[RU_LANES] = {};

        for (int a = 0; a < N; a++)
        {
            const double mab  = M[a * N + b];
            const double dmab = dM[a * N + b];

            for (int l = 0; l < RU_LANES; l++)
            {
                w[l]  += At[a * RU_LANES + l] * mab;
                wd[l] += dAt[a * RU_LANES + l] * mab;
                u[l]  += At[a * RU_LANES + l] * dmab;
            }
        }

        for (int l = 0; l < RU_LANES; l++)
        {
            se[l] += w[l] * Bt[b * RU_LANES + l];
            sa[l] += wd[l] * Bt[b * RU_LANES + l];
            sb[l] += w[l] * dBt[b * RU_LANES + l];
            ss[l] += u[l] * Bt[b * RU_LANES + l];
        }
    }

    for (int l = 0; l < RU_LANES; l++)
    {
        e3[l] = se[l];
        gs[l] = ss[l];
        ga[l] = sa[l];
        gb[l] = sb[l];
    }
}

// Bitsets of list entries for the cluster sweep: 64 entries per word
typedef unsigned long long ru_bits_t;

static inline int ru_ctz(ru_bits_t b)   // position of the lowest set bit (b != 0)
{
    return __builtin_ctzll(b);
}

static inline ru_bits_t ru_mask_from(int word, int first)   // bits of word for entries >= first
{
    const int w0 = first >> 6;

    if (word > w0)
        return ~(ru_bits_t) 0;
    if (word < w0)
        return 0;
    return ~(ru_bits_t) 0 << (first & 63);
}

// v[renum[e]] = old v[e]
template <typename T>
static void ru_permute(std::vector<T> &v, const std::vector<int> &renum)
{
    const std::vector<T> old(v);

    for (size_t e = 0; e < old.size(); e++)
        v[renum[e]] = old[e];
}

// Greedy set cover.  Item i has k candidate keys cand[k*i + j] (keys 0..nkeys-1).  Repeatedly
// choose the key shared by the most unassigned items and assign those items to it; a bucket
// queue on the counts, updated eagerly, keeps this O(k*nitems + nkeys).  On return choice[i]
// is the candidate j that item i was assigned to.
static inline void ru_greedy_cover(int nitems, int k, const std::vector<int> &cand, int nkeys, std::vector<int> &choice)
{
    std::vector<int> kstart(nkeys + 1, 0), kfill(nkeys, 0), klist(cand.size());

    for (size_t e = 0; e < cand.size(); e++)
        kstart[cand[e] + 1]++;

    for (int kk = 0; kk < nkeys; kk++)
        kstart[kk + 1] += kstart[kk];

    for (size_t e = 0; e < cand.size(); e++)
    {
        const int kk = cand[e];
        klist[kstart[kk] + kfill[kk]++] = (int) e;   // e = k*item + candidate
    }

    std::vector<char> done(nitems, 0), chosen(nkeys, 0);
    std::vector<int> cnt(nkeys);
    int cmax = 0;

    for (int kk = 0; kk < nkeys; kk++)
    {
        cnt[kk] = kstart[kk + 1] - kstart[kk];
        cmax = MAX(cmax, cnt[kk]);
    }

    std::vector<std::vector<int>> bucket(cmax + 1);

    for (int kk = nkeys - 1; kk >= 0; kk--)   // lowest key on top of each bucket
        bucket[cnt[kk]].push_back(kk);

    choice.assign(nitems, 0);

    for (int c = cmax; c > 0; )
    {
        if (bucket[c].empty())
        {
            c--;
            continue;
        }

        const int kk = bucket[c].back();
        bucket[c].pop_back();

        if (chosen[kk] || cnt[kk] != c)   // stale entry
            continue;

        chosen[kk] = 1;

        for (int e = kstart[kk]; e < kstart[kk + 1]; e++)
        {
            const int it = klist[e] / k;

            if (done[it])
                continue;

            done[it]   = 1;
            choice[it] = klist[e] % k;

            // The item's other candidates lose one unassigned item
            for (int j = 0; j < k; j++)
            {
                const int k2 = cand[(size_t)k*it + j];

                if (k2 == kk || chosen[k2])
                    continue;

                if (--cnt[k2] > 0)
                    bucket[cnt[k2]].push_back(k2);
            }
        }
    }
}
}

void PairCHIMES::build_reuse_lists()
{
    reuse_active = false;
    reuse_3b     = false;
    reuse_4b     = false;
    reuse_mixed4 = false;
    ru_rank_max  = 0;
    x3_rank      = 0;
    x4_rank      = 0;
    ru_nseed_cp  = 0;
    ru_nx4_group = 0;

    ru_npair_prev = MAX(ru_npair_prev, rp_a.size());

    rp_a.clear();
    rp_b.clear();
    re_pair.clear();
    re_table.clear();
    re_ptype.clear();
    re_rank.clear();
    re_rc.clear();
    ru_trip.clear();
    ru_seed.clear();
    ru_quad.clear();
    ru_trip3x.clear();
    ru_x3group.clear();
    x3_spec_ptype.clear();
    x3_spec_rin.clear();
    x3_spec_rout.clear();
    x3_spec_n.clear();
    x3_cls_tensor.clear();
    x4_cls_mat.clear();
    x4_cls_matT.clear();

#ifdef TABULATION
#ifdef FINGERPRINT
    if (fingerprint)
        return;
#endif

    const bool do3 = chimes_calculator.poly_orders[1] > 0;
    const bool do4 = chimes_calculator.poly_orders[2] > 0;

    // 3B through pair reuse: CP tables, or the exact polynomial (3B grid tables are
    // evaluated per triplet).  4B through pair reuse: a CP model (tables or direct factors),
    // or the exact polynomial (parameter-file coefficients or a Morse-space SVD file); 4B
    // types with a grid table (r-space SVD, coefficient tables) run in the triplet loop of
    // compute().
    const bool cp3 = do3 && chimes_calculator.tabulate_3B_cp && !chimes_calculator.tabulate_3B;
    const bool ex3 = do3 && !chimes_calculator.tabulate_3B_cp && !chimes_calculator.tabulate_3B;
    bool use3 = cp3 || ex3;
    bool use4 = do4;
    bool ex4  = false;   // some 4B type goes through pair reuse as the exact polynomial

    if (!use3 && !use4)
        return;

    int *type   = atom->type;
    tagint *tag = atom->tag;
    int *numneigh    = list->numneigh;
    int **firstneigh = list->firstneigh;

    double **x = atom->x;
    const int nlocal = atom->nlocal;
    const long long nall = (long long) atom->nlocal + (long long) atom->nghost;
    const long long ntyp = (long long) chimes_calculator.natmtyps + 1;

    // Cluster info depends only on the atom types: computed once per type combination.
    // state: 0 = not yet computed, 1 = valid, -1 = the type has terms but no model for the cache.
    // A 3B type without a CP table in a CP model goes through the cache as the exact polynomial.
    const int nct = chimes_calculator.natmtyps;
    bool mixed3 = false;
    std::vector<chimesFF::CPClusterInfo> info3(use3 ? nct*nct*nct : 0), info4(use4 ? nct*nct*nct*nct : 0);
    std::vector<signed char> state3(info3.size(), 0), state4(info4.size(), 0);

    auto cluster_info = [&](int nbody, const int *typ) -> const chimesFF::CPClusterInfo *
    {
        int idx = 0;

        for (int m = 0; m < nbody; m++)
            idx = idx*nct + typ[m];

        std::vector<chimesFF::CPClusterInfo> &cache = (nbody == 3) ? info3 : info4;
        std::vector<signed char> &state = (nbody == 3) ? state3 : state4;

        if (state[idx] == 0)
        {
            bool ok;

            if (nbody == 4)
            {
                ok = chimes_calculator.cp4_cluster_info(typ, cache[idx]);

                if (!ok)   // no CP model: the exact polynomial, unless a 4B grid table is used
                {
                    ok = chimes_calculator.exact4_cluster_info(typ, cache[idx]);

                    if (ok && cache[idx].type_index >= 0 && cache[idx].rank > 0)
                        ex4 = true;
                }
            }
            else if (cp3)
            {
                ok = chimes_calculator.cp3_cluster_info(typ, cache[idx]);

                if (!ok)
                {
                    ok = chimes_calculator.exact3_cluster_info(typ, cache[idx]);

                    // A slot with outer <= inner cutoff: no cluster of this type can occur
                    // (util/tensor_tools/chimes_cpd.py skips such types), so it has no terms
                    bool dead = false;

                    for (int c = 0; c < 3; c++)
                        dead |= (cache[idx].rout[c] <= cache[idx].rin[c]);

                    if (dead)
                        cache[idx].rank = 0;
                    else
                        mixed3 = true;
                }
            }
            else
                ok = chimes_calculator.exact3_cluster_info(typ, cache[idx]);

            state[idx] = ok ? 1 : -1;
        }

        return (state[idx] > 0) ? &cache[idx] : nullptr;
    };

    // 3B: every cluster type must be covered (CP or exact), otherwise 3B stays per cluster.
    // 4B: types with a CP model or the exact polynomial go through pair reuse; types with a 4B
    // grid table stay in the per-cluster loop of compute() (reuse_mixed4, ru_cache4_type marks
    // the types done through pair reuse).
    //
    // Cluster cutoff per pair of ChIMES types (rc3p, rc4p): the largest runtime cutoff of that
    // pair over the cached cluster types with terms.  The lists use these, not the global 3B/4B
    // maxima: types without terms often keep the default (2B) cutoff, which would otherwise set
    // the list radius (Turbo water: every 4B type with terms uses 4 A, the global 4B maximum is
    // the 7 A default).
    static const int rt3[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };                                 // runtime pair -> atoms
    static const int rt4[6][2] = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 } };

    std::vector<double> rc3p(nct * nct, 0.0), rc4p(nct * nct, 0.0);
    bool any4 = false;

    reuse_mixed4 = false;
    ru_cache4_type.assign(use4 ? nct*nct*nct*nct : 0, 0);

    for (int a = 0; a < nct; a++)
    {
        for (int b = 0; b < nct; b++)
        {
            for (int c = 0; c < nct; c++)
            {
                const int typ[4] = { a, b, c, 0 };

                if (use3)
                {
                    const chimesFF::CPClusterInfo *inf = cluster_info(3, typ);

                    if (!inf)
                        use3 = false;
                    else if (inf->type_index >= 0 && inf->rank > 0)
                        for (int r = 0; r < 3; r++)
                        {
                            const int ta = typ[rt3[r][0]];
                            const int tb = typ[rt3[r][1]];
                            rc3p[ta*nct + tb] = rc3p[tb*nct + ta] = MAX(rc3p[ta*nct + tb], inf->rc_runtime[r]);
                        }
                }

                for (int d = 0; d < nct && use4; d++)
                {
                    const int typ4[4] = { a, b, c, d };
                    const chimesFF::CPClusterInfo *inf = cluster_info(4, typ4);

                    if (!inf)
                    {
                        reuse_mixed4 = true;
                        continue;
                    }

                    ru_cache4_type[((a*nct + b)*nct + c)*nct + d] = 1;

                    if (inf->type_index >= 0 && inf->rank > 0)
                    {
                        any4 = true;

                        for (int r = 0; r < 6; r++)
                        {
                            const int ta = typ4[rt4[r][0]];
                            const int tb = typ4[rt4[r][1]];
                            rc4p[ta*nct + tb] = rc4p[tb*nct + ta] = MAX(rc4p[ta*nct + tb], inf->rc_runtime[r]);
                        }
                    }
                }
            }
        }
    }

    if (use4 && !any4)   // no 4B type for pair reuse: all 4B per cluster
    {
        use4 = false;
        reuse_mixed4 = false;
    }

    if (chimes_calculator.rank == 0 && !reuse_warned)
    {
        if (do3 && !cp3 && !ex3)
            std::cout << "WARNING (pair_style chimesFF): 3-body pair reuse off: 3B grid tables (TABULATED) are evaluated per triplet" << std::endl;
        if (do4 && !use4)
            std::cout << "WARNING (pair_style chimesFF): 4-body pair reuse off: every 4-body cluster type uses a 4B grid table" << std::endl;
        if (mixed3)
            std::cout << "NOTE (pair_style chimesFF): 3-body types without a CP table go through pair reuse as the exact polynomial" << std::endl;
        if (reuse_mixed4)
            std::cout << "NOTE (pair_style chimesFF): 4-body types with a 4B grid table are evaluated per cluster, the others through pair reuse" << std::endl;
    }
    reuse_warned = true;

    if (!use3 && !use4)
        return;

    double rc3max = 0.0, rc4max = 0.0;

    for (int t = 0; t < nct * nct; t++)
    {
        rc3max = MAX(rc3max, rc3p[t]);
        rc4max = MAX(rc4max, rc4p[t]);
    }

    ////////////////////////////////////////
    // Padding of the cluster lists
    ////////////////////////////////////////

    // Rebuilding between neighbor-list builds needs every pair within the many-body cutoff
    // plus the padding to be in the (2B) neighbor list at any time, i.e. cutoff + padding
    // <= smallest 2B cutoff of the ChIMES types (the list itself holds 2B cutoff + skin).
    // A smaller padding means fewer listed clusters that are out of range, so less checking
    // per step and denser lists, for more frequent (and cheaper) builds; half the neighbor
    // skin was the faster choice in tests (Si, HN3).
    const double skin   = neighbor->skin;
    const double cut_mb = MAX(use3 ? rc3max : 0.0, use4 ? rc4max : 0.0);
    double cut2b_min = 1.0e30;

    for (int a = 1; a <= atom->ntypes; a++)
        for (int b = 1; b <= atom->ntypes; b++)
            if (chimes_type[a-1] >= 0 && chimes_type[b-1] >= 0 && cutsq[a][b] > 0.0)
                cut2b_min = MIN(cut2b_min, sqrt(cutsq[a][b]));

    if (reuse_skin >= 0.0)
        ru_pad = reuse_skin;
    else
        ru_pad = (cut_mb + 0.5 * skin <= cut2b_min + 1.0e-12) ? 0.5 * skin : skin;

    ru_mid_ok = (cut_mb + ru_pad <= cut2b_min + 1.0e-12);

    if (ru_pad < skin && !ru_mid_ok)
        error->all(FLERR, "pair_style chimesFF mb_skin: many-body cutoff + mb_skin must not exceed the smallest 2-body cutoff");

    if (ru_pad > skin && cut_mb + ru_pad > cut2b_min + skin + 1.0e-12)
        error->all(FLERR, "pair_style chimesFF mb_skin: many-body cutoff + mb_skin exceeds the neighbor list cutoff");

    const double cut3 = rc3max + ru_pad;   // list radii; per type pair: rc3p, rc4p + padding
    const double cut4 = rc4max + ru_pad;

    // Hash table sized for the pair count of the previous build (it grows if needed): a
    // table much larger than the number of keys costs more in cache misses than a rehash
    ChimesIdMap pair_map;
    pair_map.init(MAX((size_t) 4096, ru_npair_prev));

    std::vector<int> pe_head;   // first cache entry of each pair (-1: none)
    std::vector<int> pe_next;   // next cache entry of the same pair

    auto pair_id = [&](int a, int b) -> int
    {
        if (b < a)
            std::swap(a, b);

        const int id = pair_map.get((long long) a * nall + b, (int) rp_a.size());

        if (id == (int) rp_a.size())
        {
            rp_a.push_back(a);
            rp_b.push_back(b);
            pe_head.push_back(-1);
        }
        return id;
    };

    // Table ids below x3_ntab_cp are CP tables; exact 3B and 4B pair functions use
    // x3_ntab_cp + spec, one spec per (pair type, inner cutoff, outer cutoff, basis size)
    x3_ntab_cp = chimes_calculator.cp_num_tables();
    x3_rank    = (ex3 || mixed3) ? chimes_calculator.exact_n3 : 0;
    x4_rank    = (use4 && ex4) ? chimes_calculator.exact_n4 : 0;

    auto x3_spec = [&](int ptype, double rin, double rout, int n) -> int
    {
        for (int sp = 0; sp < (int) x3_spec_ptype.size(); sp++)
            if (x3_spec_ptype[sp] == ptype && x3_spec_rin[sp] == rin && x3_spec_rout[sp] == rout && x3_spec_n[sp] == n)
                return sp;

        x3_spec_ptype.push_back(ptype);
        x3_spec_rin.push_back(rin);
        x3_spec_rout.push_back(rout);
        x3_spec_n.push_back(n);

        return (int) x3_spec_ptype.size() - 1;
    };

    // A cache entry is a (pair, table, cluster cutoff of the pair).  The cutoff is part
    // of the key so that one flag per entry, r < rc, decides whether a cluster is inside
    // its cutoffs; it equals the table cutoff except for shared 3B tables whose slots
    // have different cutoffs.  A pair has only a few entries, kept in a short list.
    auto entry_id = [&](int pid, int table, int ptype, double rc) -> int
    {
        for (int e = pe_head[pid]; e >= 0; e = pe_next[e])
            if (re_table[e] == table && re_rc[e] == rc)
                return e;

        const int id = (int) re_pair.size();

        re_pair.push_back(pid);
        re_table.push_back(table);
        re_ptype.push_back(ptype);
        re_rank.push_back((table < x3_ntab_cp) ? chimes_calculator.cp_table_rank(table) : x3_spec_n[table - x3_ntab_cp]);
        re_rc.push_back(rc);
        pe_next.push_back(pe_head[pid]);
        pe_head[pid] = id;

        return id;
    };

    ////////////////////////////////////////
    // Per-atom many-body neighbor lists (sweep 1a)
    ////////////////////////////////////////

    // For each local atom i: neighbors with a ChIMES type, key >= key of i (owner_key) and
    // inside the largest padded many-body cutoff, sorted by key, with their distance.
    // Clusters are enumerated i < j < k (< l) by key from these lists alone (j, k, l all lie
    // within the cutoff of i), so the walk only moves forward.  Equal keys (periodic images
    // in small cells) keep both orders, as the 3-body list does.
    const double cut_list = MAX(use3 ? cut3 : 0.0, use4 ? cut4 : 0.0);

    sl_start.assign(nlocal + 1, 0);
    sl_list.clear();

    for (int i = 0; i < nlocal; i++)
    {
        sl_start[i] = (int) sl_list.size();

        if (chimes_type[type[i]-1] < 0)
            continue;

        const int *jlist = firstneigh[i];
        const int jnum   = numneigh[i];
        const chimes_key_t ikey = owner_key(tag[i]);

        for (int jj = 0; jj < jnum; jj++)
        {
            const int j = jlist[jj] & NEIGHMASK;

            if (j == i || owner_key(tag[j]) < ikey || chimes_type[type[j]-1] < 0)
                continue;

            const double dx = x[j][0] - x[i][0];
            const double dy = x[j][1] - x[i][1];
            const double dz = x[j][2] - x[i][2];
            const double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 >= cut_list * cut_list)
                continue;

            ShortNeigh sn;
            sn.atom = j;
            sn.pid  = -1;
            sn.key  = owner_key(tag[j]);
            sn.ct   = chimes_type[type[j]-1];
            sn.r    = sqrt(r2);
            sl_list.push_back(sn);
        }

        std::stable_sort(sl_list.begin() + sl_start[i], sl_list.end(), [](const ShortNeigh &u, const ShortNeigh &v)
        {
            return u.key < v.key;
        });
    }
    sl_start[nlocal] = (int) sl_list.size();

    auto slot_pid = [&](int i, int a) -> int   // pair id of (i, list entry a), cached
    {
        if (sl_list[a].pid < 0)
            sl_list[a].pid = pair_id(i, sl_list[a].atom);
        return sl_list[a].pid;
    };

    auto dist2 = [&](int a, int b) -> double
    {
        const double dx = x[b][0] - x[a][0];
        const double dy = x[b][1] - x[a][1];
        const double dz = x[b][2] - x[a][2];
        return dx*dx + dy*dy + dz*dz;
    };

    std::vector<ReuseTrip> x3_trips;   // exact 3B, before grouping
    std::vector<int> x3_tidx;          // their triplet type

    auto add_triplet = [&](int i, int j, int k, int pij, int pik, int pjk)
    {
        const int typ[3] = { chimes_type[type[i]-1], chimes_type[type[j]-1], chimes_type[type[k]-1] };
        const chimesFF::CPClusterInfo *pinfo = cluster_info(3, typ);

        if (!pinfo || pinfo->type_index < 0 || pinfo->rank <= 0)
            return;

        const chimesFF::CPClusterInfo &info = *pinfo;
        const bool cp = info.table[0] >= 0;   // CP tables, or the exact polynomial

        ReuseTrip T;
        T.atom[0] = i;
        T.atom[1] = j;
        T.atom[2] = k;
        T.pid[0] = pij;
        T.pid[1] = pik;
        T.pid[2] = pjk;
        T.rank = info.rank;
        T.mslot = -1;
        T.inner = false;

        for (int c = 0; c < 3; c++)
        {
            const int table = cp ? info.table[c] : x3_ntab_cp + x3_spec(info.pair_type[c], info.rin[c], info.rout[c], x3_rank);

            T.c2r[c] = info.c2r[c];
            T.ent[c] = entry_id(T.pid[info.c2r[c]], table, info.pair_type[c], info.rc_runtime[info.c2r[c]]);
        }

        ru_rank_max = MAX(ru_rank_max, T.rank);

        if (cp)
        {
            ru_trip.push_back(T);
        }
        else
        {
            x3_trips.push_back(T);
            x3_tidx.push_back(info.type_index);
        }
    };

    static const int left_rt[3]  = { 0, 1, 3 };   // runtime pairs A0A1, A0A2, A1A2
    static const int right_rt[3] = { 2, 4, 5 };   // runtime pairs A0A3, A1A3, A2A3

    // Exact 4B class: the coefficient tensor of a type ordered by the runtime pairs of a
    // quadruplet A0 A1 A2 A3 whose atoms have these ChIMES types, as the 3|3 matrix
    //     M[a][b][c][(d*N + e)*N + f],   a b c: pairs A0A1 A0A2 A1A2 (seed),
    //                                    d e f: pairs A0A3 A1A3 A2A3 (fourth atom)
    // One class per type combination (in runtime order); identical matrices are shared.
    // Each class is stored three times, with seed pair k = 0, 1, 2 first (the pair a seed
    // group contracts) and the other two in order: x4_cls_mat[3*class + k], and each of
    // these also with the last two index groups swapped (x4_cls_matT, M[a][s][b c]).
    std::vector<int> x4_cls_of(use4 && ex4 ? nct*nct*nct*nct : 0, -1);
    std::vector<double> x4_dense;

    auto x4_class = [&](const int *typ, const chimesFF::CPClusterInfo &info) -> int
    {
        const int key = ((typ[0]*nct + typ[1])*nct + typ[2])*nct + typ[3];

        if (x4_cls_of[key] >= 0)
            return x4_cls_of[key];

        chimes_calculator.exact4_dense_tensor(info.type_index, x4_dense);

        const int N  = x4_rank;
        const int N3 = N * N * N;
        int pr[6];   // runtime pair -> parameter slot

        for (int c = 0; c < 6; c++)
            pr[info.c2r[c]] = c;

        std::vector<double> M((size_t) N3 * N3);
        int n[6];

        for (int a = 0; a < N; a++)
            for (int b = 0; b < N; b++)
                for (int c = 0; c < N; c++)
                    for (int d = 0; d < N; d++)
                        for (int e = 0; e < N; e++)
                            for (int f = 0; f < N; f++)
                            {
                                n[pr[0]] = a;
                                n[pr[1]] = b;
                                n[pr[3]] = c;
                                n[pr[2]] = d;
                                n[pr[4]] = e;
                                n[pr[5]] = f;
                                M[(size_t) ((a*N + b)*N + c) * N3 + (d*N + e)*N + f] = x4_dense[((((n[0]*(size_t)N + n[1])*N + n[2])*N + n[3])*N + n[4])*N + n[5]];
                            }

        int id = -1;

        for (int k = 0; k < (int) x4_cls_mat.size() && id < 0; k += 3)
            if (x4_cls_mat[k] == M)
                id = k / 3;

        if (id < 0)
        {
            id = (int) x4_cls_mat.size() / 3;
            x4_cls_mat.push_back(M);

            for (int lead = 1; lead < 3; lead++)   // seed pair A0A2 or A1A2 first
            {
                std::vector<double> V(M.size());

                for (int a = 0; a < N; a++)
                    for (int b = 0; b < N; b++)
                        for (int c = 0; c < N; c++)
                        {
                            const int o = (lead == 1) ? (b*N + a)*N + c : (c*N + a)*N + b;
                            std::copy(&M[(size_t) ((a*N + b)*N + c) * N3], &M[(size_t) ((a*N + b)*N + c) * N3] + N3, &V[(size_t) o * N3]);
                        }

                x4_cls_mat.push_back(V);
            }

            for (int k = 0; k < 3; k++)
            {
                const std::vector<double> &V = x4_cls_mat[3*id + k];
                std::vector<double> Vt(V.size());
                const int N2 = N * N;

                for (int a = 0; a < N; a++)
                    for (int bc = 0; bc < N2; bc++)
                        for (int x = 0; x < N3; x++)
                            Vt[((size_t) a * N3 + x) * N2 + bc] = V[((size_t) a * N2 + bc) * N3 + x];

                x4_cls_matT.push_back(Vt);
            }
        }

        x4_cls_of[key] = id;
        return id;
    };

    // One seed: triangle A0 A1 A2 plus the ChIMES type tl of its fourth atoms ls[0..nl).
    // p01, p02, p12: pair ids of the triangle; pl0, pl1, pl2 (or null: looked up): pair ids of
    // (A0, ls[m]), (A1, ls[m]), (A2, ls[m])
    auto emit_seed = [&](int A0, int A1, int A2, int tl, int p01, int p02, int p12, const int *ls, const int *pl0, const int *pl1, const int *pl2, int nl)
    {
        const int typ[4] = { chimes_type[type[A0]-1], chimes_type[type[A1]-1], chimes_type[type[A2]-1], tl };
        const chimesFF::CPClusterInfo *pinfo = cluster_info(4, typ);   // null: per cluster (4B grid table)

        if (!pinfo || pinfo->type_index < 0 || pinfo->rank <= 0)
            return;

        const chimesFF::CPClusterInfo &info = *pinfo;
        const bool cp = info.table[0] >= 0;   // CP model, or the exact polynomial

        int r2c[6];
        for (int c = 0; c < 6; c++)
            r2c[info.c2r[c]] = c;

        int table[6];   // per parameter slot
        for (int c = 0; c < 6; c++)
            table[c] = cp ? info.table[c] : x3_ntab_cp + x3_spec(info.pair_type[c], info.rin[c], info.rout[c], x4_rank);

        ReuseSeed S;
        S.atom[0] = A0;
        S.atom[1] = A1;
        S.atom[2] = A2;
        S.pid[0] = p01;
        S.pid[1] = p02;
        S.pid[2] = p12;
        S.cls = cp ? -1 : 3 * x4_class(typ, info);   // pair A0A1 first until the groups are chosen
        S.rank = info.rank;
        S.inner = false;

        for (int k = 0; k < 3; k++)
        {
            const int c = r2c[left_rt[k]];
            S.ent[k] = entry_id(S.pid[k], table[c], info.pair_type[c], info.rc_runtime[left_rt[k]]);
        }

        S.quad_begin = (int) ru_quad.size();

        for (int m = 0; m < nl; m++)
        {
            const int A3 = ls[m];

            ReuseQuad Qd;
            Qd.l = A3;
            Qd.pid[0] = pl0 ? pl0[m] : pair_id(A0, A3);
            Qd.pid[1] = pl1 ? pl1[m] : pair_id(A1, A3);
            Qd.pid[2] = pl2 ? pl2[m] : pair_id(A2, A3);
            Qd.inner = false;

            for (int k = 0; k < 3; k++)
            {
                const int c = r2c[right_rt[k]];
                Qd.ent[k] = entry_id(Qd.pid[k], table[c], info.pair_type[c], info.rc_runtime[right_rt[k]]);
            }

            ru_quad.push_back(Qd);
        }

        S.quad_inner = S.quad_begin;
        S.quad_end = (int) ru_quad.size();
        ru_rank_max = MAX(ru_rank_max, S.rank);
        ru_seed.push_back(S);
    };

    ////////////////////////////////////////
    // Sweep 1b: 3B and 4B clusters from the per-atom lists
    ////////////////////////////////////////

    // For atom i the m entries of its list (in key order) form a small local table.  Each
    // pair of entries is tested once, and entry v is set in the 3B (4B) bitset row of entry u
    // when both lie inside the padded 3B (4B) cutoff of i and of each other, the cutoff
    // being that of the pair's ChIMES types (rc3p, rc4p).  The triplets
    // (i, j, k) are then the set bits of row j from k's key on, and the quadruplets
    // (i, j, k, l) the set bits of (row j AND row k) from l's key on, 64 entries per word:
    // no nested scan over entries that fail a cutoff.  Pair ids of entry pairs are looked up
    // on first use, so a pair shared by many clusters of i is hashed once.  With the default
    // seeds every triangle (i, j, k) is the seed of its fourth atoms (one seed per ChIMES
    // type of the fourth atom), so seeds are emitted as they are found.
    // Padded cutoff per pair of ChIMES types (negative: the pair is in no cached cluster type)
    std::vector<double> lc3(nct * nct), lc4(nct * nct), lc3sq(nct * nct), lc4sq(nct * nct);

    for (int t = 0; t < nct * nct; t++)
    {
        lc3[t]   = (use3 && rc3p[t] > 0.0) ? rc3p[t] + ru_pad : -1.0;
        lc4[t]   = (use4 && rc4p[t] > 0.0) ? rc4p[t] + ru_pad : -1.0;
        lc3sq[t] = (lc3[t] > 0.0) ? lc3[t] * lc3[t] : -1.0;
        lc4sq[t] = (lc4[t] > 0.0) ? lc4[t] * lc4[t] : -1.0;
    }

    std::vector<ru_bits_t> adj3, adj4;      // m rows of nw words: bitset of neighbor entries
    std::vector<int> npid;                  // m x m pair ids (-1: not looked up yet)
    std::vector<int> nft;                   // first entry with the same key
    std::vector<std::array<int,4>> quads;   // greedy seeds: all quadruplets first
    std::vector<int> lat, lp0, lp1, lp2, sat, sp0, sp1, sp2;

    for (int i = 0; i < nlocal; i++)
    {
        const int s0 = sl_start[i];
        const int m  = sl_start[i + 1] - s0;

        if (m < 2)
            continue;

        const ShortNeigh *nb = &sl_list[s0];
        const int nw = (m + 63) >> 6;
        const int ti = chimes_type[type[i]-1];

        if (use3)
            adj3.assign((size_t) m * nw, 0);
        if (use4)
            adj4.assign((size_t) m * nw, 0);
        npid.assign((size_t) m * m, -1);
        nft.resize(m);

        for (int u = 0; u < m; u++)
        {
            nft[u] = (u > 0 && nb[u - 1].key == nb[u].key) ? nft[u - 1] : u;

            const int tu = nb[u].ct;
            const bool u3 = nb[u].r < lc3[ti*nct + tu];
            const bool u4 = nb[u].r < lc4[ti*nct + tu];

            if (!u3 && !u4)
                continue;

            for (int v = u + 1; v < m; v++)
            {
                const int tv = nb[v].ct;
                const bool v3 = u3 && nb[v].r < lc3[ti*nct + tv];
                const bool v4 = u4 && nb[v].r < lc4[ti*nct + tv];

                if (!v3 && !v4)
                    continue;

                const double d2 = dist2(nb[u].atom, nb[v].atom);

                if (v3 && d2 < lc3sq[tu*nct + tv])
                {
                    adj3[(size_t) u * nw + (v >> 6)] |= (ru_bits_t) 1 << (v & 63);
                    adj3[(size_t) v * nw + (u >> 6)] |= (ru_bits_t) 1 << (u & 63);
                }

                if (v4 && d2 < lc4sq[tu*nct + tv])
                {
                    adj4[(size_t) u * nw + (v >> 6)] |= (ru_bits_t) 1 << (v & 63);
                    adj4[(size_t) v * nw + (u >> 6)] |= (ru_bits_t) 1 << (u & 63);
                }
            }
        }

        auto ipid = [&](int u) -> int   // pair id of (i, entry u)
        {
            return slot_pid(i, s0 + u);
        };

        auto upid = [&](int u, int v) -> int   // pair id of (entry u, entry v)
        {
            int &p = npid[(size_t) MIN(u, v) * m + MAX(u, v)];

            if (p < 0)
                p = pair_id(nb[u].atom, nb[v].atom);
            return p;
        };

        // 3B: j = entry u, k = entry v with key of k >= key of j (equal keys in both orders)
        for (int u = 0; u < m && use3; u++)
        {
            const ru_bits_t *ru = &adj3[(size_t) u * nw];

            for (int wd = nft[u] >> 6; wd < nw; wd++)
            {
                ru_bits_t b = ru[wd] & ru_mask_from(wd, nft[u]);

                while (b)
                {
                    const int v = (wd << 6) + ru_ctz(b);
                    b &= b - 1;
                    add_triplet(i, nb[u].atom, nb[v].atom, ipid(u), ipid(v), upid(u, v));
                }
            }
        }

        // 4B: k = entry v in row j, l = entry w in rows j and k with key of l >= key of k
        for (int u = 0; u < m && use4; u++)
        {
            const ru_bits_t *ru = &adj4[(size_t) u * nw];

            for (int vd = nft[u] >> 6; vd < nw; vd++)
            {
                ru_bits_t bv = ru[vd] & ru_mask_from(vd, nft[u]);

                while (bv)
                {
                    const int v = (vd << 6) + ru_ctz(bv);
                    bv &= bv - 1;

                    const ru_bits_t *rv = &adj4[(size_t) v * nw];

                    lat.clear();
                    lp0.clear();
                    lp1.clear();
                    lp2.clear();

                    for (int wd = nft[v] >> 6; wd < nw; wd++)
                    {
                        ru_bits_t bw = ru[wd] & rv[wd] & ru_mask_from(wd, nft[v]);

                        while (bw)
                        {
                            const int w = (wd << 6) + ru_ctz(bw);
                            bw &= bw - 1;

                            if (reuse_greedy)
                            {
                                quads.push_back({ { i, nb[u].atom, nb[v].atom, nb[w].atom } });
                            }
                            else
                            {
                                lat.push_back(nb[w].atom);
                                lp0.push_back(ipid(w));
                                lp1.push_back(upid(u, w));
                                lp2.push_back(upid(v, w));
                            }
                        }
                    }

                    if (lat.empty())
                        continue;

                    const int j   = nb[u].atom;
                    const int k   = nb[v].atom;
                    const int pij = ipid(u);
                    const int pik = ipid(v);
                    const int pjk = upid(u, v);

                    if (nct == 1)
                    {
                        emit_seed(i, j, k, 0, pij, pik, pjk, lat.data(), lp0.data(), lp1.data(), lp2.data(), (int) lat.size());
                        continue;
                    }

                    for (int t = 0; t < nct; t++)
                    {
                        sat.clear();
                        sp0.clear();
                        sp1.clear();
                        sp2.clear();

                        for (size_t n = 0; n < lat.size(); n++)
                        {
                            if (chimes_type[type[lat[n]]-1] == t)
                            {
                                sat.push_back(lat[n]);
                                sp0.push_back(lp0[n]);
                                sp1.push_back(lp1[n]);
                                sp2.push_back(lp2[n]);
                            }
                        }

                        if (!sat.empty())
                            emit_seed(i, j, k, t, pij, pik, pjk, sat.data(), sp0.data(), sp1.data(), sp2.data(), (int) sat.size());
                    }
                }
            }
        }
    }

    ////////////////////////////////////////
    // Exact 3B, sweep 1c: group the triplets by the pair contracted first
    ////////////////////////////////////////

    // E = sum_x G_s[x] sum_ab R_s[x][a][b] G_sa[a] G_sb[b], with R_s the coefficient tensor
    // with slot s moved first and sa < sb the other two slots.  For a group of triplets
    // sharing the pair (and class) in slot s, M = sum_x G_s[x] R_s[x] is formed once per
    // step and each triplet then costs a bilinear form.  Candidates per triplet: its three
    // slots; the greedy cover picks the (pair entry, class) shared by the most triplets.

    if (!x3_trips.empty())
    {
        const int N  = x3_rank;
        const int NN = N * N;
        const int X  = (int) x3_trips.size();

        std::vector<int> cls_of;   // (triplet type, slot) -> class
        std::vector<double> dense, R(NN * (size_t) N);

        auto class_id = [&](int tripidx, int sl) -> int
        {
            if ((int) cls_of.size() <= 3 * tripidx + sl)
                cls_of.resize(3 * (tripidx + 1), -1);

            if (cls_of[3 * tripidx + sl] >= 0)
                return cls_of[3 * tripidx + sl];

            chimes_calculator.exact3_dense_tensor(tripidx, dense);

            const int sa = (sl == 0) ? 1 : 0;
            const int sb = (sl == 2) ? 1 : 2;

            for (int x = 0; x < N; x++)
            {
                for (int a = 0; a < N; a++)
                {
                    for (int b = 0; b < N; b++)
                    {
                        int n[3];
                        n[sl] = x;
                        n[sa] = a;
                        n[sb] = b;
                        R[(size_t) x * NN + a * N + b] = dense[((size_t) n[0] * N + n[1]) * N + n[2]];
                    }
                }
            }

            // Slots with identical tensors (symmetric triplet types) share a class
            int id = -1;

            for (int c = 0; c < (int) x3_cls_tensor.size() && id < 0; c++)
                if (x3_cls_tensor[c] == R)
                    id = c;

            if (id < 0)
            {
                id = (int) x3_cls_tensor.size();
                x3_cls_tensor.push_back(R);
            }

            cls_of[3 * tripidx + sl] = id;
            return id;
        };

        std::vector<int> cand_ent(3 * (size_t) X), cand_cls(3 * (size_t) X);

        for (int t = 0; t < X; t++)
        {
            for (int sl = 0; sl < 3; sl++)
            {
                cand_ent[3*(size_t)t + sl] = x3_trips[t].ent[sl];
                cand_cls[3*(size_t)t + sl] = class_id(x3_tidx[t], sl);
            }
        }

        // Keys (entry, class): an entry has a key per class it is contracted with, kept in a
        // short chain per entry
        std::vector<int> cand(3 * (size_t) X), key_ent, key_cls, key_next;
        std::vector<int> key_head(re_pair.size(), -1);

        for (size_t e = 0; e < cand.size(); e++)
        {
            int id = key_head[cand_ent[e]];

            while (id >= 0 && key_cls[id] != cand_cls[e])
                id = key_next[id];

            if (id < 0)
            {
                id = (int) key_ent.size();
                key_ent.push_back(cand_ent[e]);
                key_cls.push_back(cand_cls[e]);
                key_next.push_back(key_head[cand_ent[e]]);
                key_head[cand_ent[e]] = id;
            }
            cand[e] = id;
        }

        std::vector<int> choice;
        ru_greedy_cover(X, 3, cand, (int) key_ent.size(), choice);

        // Groups in (pair entry, class) order, triplets in sweep order within a group
        const int K = (int) key_ent.size();
        std::vector<int> kstart(K + 1, 0), kfill(K, 0), klist(X), korder;

        for (int t = 0; t < X; t++)
            kstart[cand[3*(size_t)t + choice[t]] + 1]++;

        for (int kk = 0; kk < K; kk++)
            kstart[kk + 1] += kstart[kk];

        for (int t = 0; t < X; t++)
        {
            const int kk = cand[3*(size_t)t + choice[t]];
            klist[kstart[kk] + kfill[kk]++] = t;
        }

        for (int kk = 0; kk < K; kk++)
            if (kstart[kk + 1] > kstart[kk])
                korder.push_back(kk);

        std::sort(korder.begin(), korder.end(), [&](int x, int y)
        {
            return (key_ent[x] != key_ent[y]) ? (key_ent[x] < key_ent[y]) : (key_cls[x] < key_cls[y]);
        });

        for (int kk : korder)
        {
            ReuseX3Group G;
            G.ent = key_ent[kk];
            G.cls = key_cls[kk];
            G.trip_begin = (int) ru_trip3x.size();

            for (int e = kstart[kk]; e < kstart[kk + 1]; e++)
            {
                const int t = klist[e];

                ru_trip3x.push_back(x3_trips[t]);
                ru_trip3x.back().mslot = choice[t];
            }

            G.trip_end = (int) ru_trip3x.size();
            ru_x3group.push_back(G);
        }
    }

    ////////////////////////////////////////
    // 4B, greedy seeds: the seed triangle of each quadruplet by greedy set cover
    ////////////////////////////////////////

    if (use4 && reuse_greedy)
    {
        const int Q = (int) quads.size();

        // Seed groups: atoms A0 A1 A2 + chimes type of A3.  Each quad gets a group and the
        // position (0..3) of its fourth atom A3.

        ChimesIdMap head_map, key_map;    // (A0,A1) -> head id;  (head, A2, type) -> group id
        head_map.init(4096);
        key_map.init(4096);

        int nhead = 0;
        std::vector<std::array<int,4>> keys;   // A0, A1, A2, type of A3

        auto key_id = [&](int a, int b, int c, int tl) -> int
        {
            const int h = head_map.get((long long) a * nall + b, nhead);

            if (h == nhead)
                nhead++;

            const int id = key_map.get(((long long) h * nall + c) * ntyp + tl, (int) keys.size());

            if (id == (int) keys.size())
                keys.push_back({ { a, b, c, tl } });

            return id;
        };

        std::vector<int> qgroup(Q), qfourth(Q);
        std::vector<int> group_order;   // groups in the order their seeds are emitted

        // Greedy double sweep: candidate seeds = the four triangles of each quad (atoms
        // sorted by local index, plus the type of the remaining atom); greedy set cover

        std::vector<int> qkey(4 * (size_t) Q);

        for (int qi = 0; qi < Q; qi++)
        {
            const std::array<int,4> &a = quads[qi];

            for (int drop = 0; drop < 4; drop++)
            {
                int tri[3], n = 0;

                for (int m = 0; m < 4; m++)
                    if (m != drop)
                        tri[n++] = a[m];

                std::sort(tri, tri + 3);

                qkey[4*(size_t)qi + drop] = key_id(tri[0], tri[1], tri[2], chimes_type[type[a[drop]]-1]);
            }
        }

        const int K = (int) keys.size();

        std::vector<int> choice;
        ru_greedy_cover(Q, 4, qkey, K, choice);

        std::vector<char> chosen(K, 0);

        for (int qi = 0; qi < Q; qi++)
        {
            qfourth[qi] = choice[qi];
            qgroup[qi]  = qkey[4*(size_t)qi + choice[qi]];
            chosen[qgroup[qi]] = 1;
        }

        // Emit seeds sorted by their atoms (memory locality of the pair cache)
        for (int kk = 0; kk < K; kk++)
            if (chosen[kk])
                group_order.push_back(kk);

        std::sort(group_order.begin(), group_order.end(), [&](int x, int y)
        {
            return keys[x] < keys[y];
        });

        // Members of each group (CSR, quads in sweep order)
        const int G = (int) keys.size();
        std::vector<int> gstart(G + 1, 0), gfill(G, 0), glist(Q);

        for (int qi = 0; qi < Q; qi++)
            gstart[qgroup[qi] + 1]++;

        for (int g = 0; g < G; g++)
            gstart[g + 1] += gstart[g];

        for (int qi = 0; qi < Q; qi++)
            glist[gstart[qgroup[qi]] + gfill[qgroup[qi]]++] = qi;

        for (int g : group_order)
        {
            if (gstart[g + 1] == gstart[g])
                continue;

            lat.clear();

            for (int e = gstart[g]; e < gstart[g + 1]; e++)
                lat.push_back(quads[glist[e]][qfourth[glist[e]]]);

            const int A0 = keys[g][0];
            const int A1 = keys[g][1];
            const int A2 = keys[g][2];

            emit_seed(A0, A1, A2, keys[g][3], pair_id(A0, A1), pair_id(A0, A2), pair_id(A1, A2), lat.data(), nullptr, nullptr, nullptr, (int) lat.size());
        }
    }

    ////////////////////////////////////////
    // Exact 4B seeds: after the CP seeds, grouped by (entry of pair A0A1, class)
    ////////////////////////////////////////

    // A group's first pair is contracted with its class matrix once per step.  Each seed can
    // join a group through any of its three pairs; the greedy cover picks the (pair entry,
    // class, seed slot) shared by the most seeds, as for the exact 3B.
    {
        auto mid = std::stable_partition(ru_seed.begin(), ru_seed.end(), [](const ReuseSeed &S) { return S.cls < 0; });
        ru_nseed_cp = (int) (mid - ru_seed.begin());

        const int X = (int) ru_seed.size() - ru_nseed_cp;

        if (X > 0)
        {
            const long long nbase = (long long) x4_cls_mat.size() / 3 + 1;
            ChimesIdMap key_map;
            key_map.init(MAX((size_t) 4096, (size_t) X));

            std::vector<int> cand(3 * (size_t) X), choice;
            int nkeys = 0;

            for (int t = 0; t < X; t++)
            {
                const ReuseSeed &S = ru_seed[ru_nseed_cp + t];

                for (int k = 0; k < 3; k++)
                {
                    const int id = key_map.get(((long long) S.ent[k] * nbase + S.cls / 3) * 3 + k, nkeys);

                    if (id == nkeys)
                        nkeys++;

                    cand[3*(size_t)t + k] = id;
                }
            }

            ru_greedy_cover(X, 3, cand, nkeys, choice);

            for (int t = 0; t < X; t++)
                ru_seed[ru_nseed_cp + t].cls += choice[t];
        }

        std::stable_sort(mid, ru_seed.end(), [](const ReuseSeed &u, const ReuseSeed &v)
        {
            const int eu = u.ent[u.cls % 3];
            const int ev = v.ent[v.cls % 3];
            return (eu != ev) ? (eu < ev) : (u.cls < v.cls);
        });

        for (size_t s = ru_nseed_cp; s < ru_seed.size(); s++)
            if (s == (size_t) ru_nseed_cp || ru_seed[s].ent[ru_seed[s].cls % 3] != ru_seed[s-1].ent[ru_seed[s-1].cls % 3] || ru_seed[s].cls != ru_seed[s-1].cls)
                ru_nx4_group++;
    }

    ////////////////////////////////////////
    // Storage
    ////////////////////////////////////////

#ifdef CHIMES_PAIR_CACHE
    // Exact 3B and 4B pair functions are evaluated in the first pass and need all N terms
    ru_block = MAX(MAX(x3_rank, x4_rank), MAX(1, MIN(reuse_block, MAX(1, ru_rank_max))));
#else
    // Without the cache all rank terms are contracted in one pass (reuse_block is not used)
    ru_block = MAX(MAX(x3_rank, x4_rank), MAX(1, ru_rank_max));
#endif

    const size_t np = rp_a.size();
    const size_t ne = re_pair.size();

    rp_r.assign(np, 0.0);
    rp_dr.assign(3 * np, 0.0);
    rp_fs.assign(np, 0.0);
    re_ok.assign(ne, 0);
#ifdef CHIMES_PAIR_CACHE
    re_used.assign(ne, 0);
    re_need.resize(ne);
    re_slot.resize(ne);
#endif
    vt_list.resize(ru_trip.size());
    vs_list.resize(ru_seed.size());
    vs_qbeg.resize(ru_seed.size() + 1);
    vq_list.resize(ru_quad.size());
    vx_list.resize(ru_trip3x.size());
    vg_list.resize(ru_x3group.size());
    vg_tbeg.resize(ru_x3group.size() + 1);
#ifdef CHIMES_PAIR_CACHE
    ru_scratch.assign(5 * (size_t) ru_block, 0.0);   // seed products
#else
    ru_scratch.assign(11 * (size_t) ru_block, 0.0);  // seed products, pair functions of 3 entries
#endif
    ru_x3_mat.assign(2 * (size_t) x3_rank * x3_rank + 4 * (size_t) RU_LANES * x3_rank, 0.0);   // M, dM and the batch gathers

    {
        // Exact 4B: Q, dQ, Qt, dQt (N^5), w, S, dw x3, star product (N^3), t, dt, q, dq and
        // three seed-pair products (N^2), seed pair functions (4 N)
        const size_t N = x4_rank;
        ru_x4_work.assign(4*N*N*N*N*N + 6*N*N*N + 7*N*N + 4*N, 0.0);
    }

    ////////////////////////////////////////
    // Inner clusters
    ////////////////////////////////////////

    // An entry is inner when its pair is now at least ru_pad inside the entry's cutoff: no
    // pair distance changes by more than ru_pad before the next build, so clusters made of
    // inner entries are valid at every step.  They go first in their lists (order otherwise
    // kept) and are not checked per step; entries of always-valid clusters are always needed.
    std::vector<char> ein(ne);

    for (size_t e = 0; e < ne; e++)
        ein[e] = sqrt(dist2(rp_a[re_pair[e]], rp_b[re_pair[e]])) < re_rc[e] - ru_pad;

    re_always.assign(ne, 0);

    auto inner3 = [&](const int *ent) -> bool
    {
        return ein[ent[0]] && ein[ent[1]] && ein[ent[2]];
    };

    auto always3 = [&](const int *ent)
    {
        re_always[ent[0]] = re_always[ent[1]] = re_always[ent[2]] = 1;
    };

    auto is_inner_trip = [](const ReuseTrip &T) { return T.inner; };
    auto is_inner_quad = [](const ReuseQuad &Q) { return Q.inner; };

    for (ReuseTrip &T : ru_trip)
        T.inner = inner3(T.ent);

    ru_nt_inner = (int) (std::stable_partition(ru_trip.begin(), ru_trip.end(), is_inner_trip) - ru_trip.begin());

    for (int t = 0; t < ru_nt_inner; t++)
    {
        always3(ru_trip[t].ent);
        vt_list[t] = t;   // the valid-triplet list always starts with the inner ones
    }

    for (ReuseX3Group &G : ru_x3group)
    {
        for (int t = G.trip_begin; t < G.trip_end; t++)
            ru_trip3x[t].inner = inner3(ru_trip3x[t].ent);

        G.trip_inner = (int) (std::stable_partition(ru_trip3x.begin() + G.trip_begin, ru_trip3x.begin() + G.trip_end, is_inner_trip) - ru_trip3x.begin());

        for (int t = G.trip_begin; t < G.trip_inner; t++)
            always3(ru_trip3x[t].ent);
    }

    for (ReuseSeed &S : ru_seed)
    {
        S.inner = inner3(S.ent);

        for (int q = S.quad_begin; q < S.quad_end; q++)
            ru_quad[q].inner = inner3(ru_quad[q].ent);

        S.quad_inner = (int) (std::stable_partition(ru_quad.begin() + S.quad_begin, ru_quad.begin() + S.quad_end, is_inner_quad) - ru_quad.begin());

        if (S.inner && S.quad_inner > S.quad_begin)
        {
            always3(S.ent);

            for (int q = S.quad_begin; q < S.quad_inner; q++)
                always3(ru_quad[q].ent);
        }
    }

    ////////////////////////////////////////
    // Cache entries in the order the evaluation reads them
    ////////////////////////////////////////

    // Entries were numbered as the sweep met them; the evaluation walks the clusters in
    // another order (grouped exact 3B, inner clusters first).  Renumbering by first use in
    // the evaluation keeps the reads of the packed cache close to sequential.
    {
        std::vector<int> renum(ne, -1);
        int nnew = 0;

        auto touch = [&](int *ent, int n)
        {
            for (int c = 0; c < n; c++)
            {
                if (renum[ent[c]] < 0)
                    renum[ent[c]] = nnew++;
                ent[c] = renum[ent[c]];
            }
        };

        for (ReuseTrip &T : ru_trip)
            touch(T.ent, 3);

        for (ReuseX3Group &G : ru_x3group)
        {
            touch(&G.ent, 1);

            for (int t = G.trip_begin; t < G.trip_end; t++)
                touch(ru_trip3x[t].ent, 3);
        }

        for (ReuseSeed &S : ru_seed)
        {
            touch(S.ent, 3);

            for (int q = S.quad_begin; q < S.quad_end; q++)
                touch(ru_quad[q].ent, 3);
        }

        for (size_t e = 0; e < ne; e++)
            if (renum[e] < 0)
                renum[e] = nnew++;

        ru_permute(re_pair, renum);
        ru_permute(re_table, renum);
        ru_permute(re_ptype, renum);
        ru_permute(re_rank, renum);
        ru_permute(re_rc, renum);
        ru_permute(re_always, renum);
    }

    // Positions at this build, for the displacement check
    ru_xhold.resize(3 * (size_t) nall);

    for (long long a = 0; a < nall; a++)
    {
        ru_xhold[3*a]   = x[a][0];
        ru_xhold[3*a+1] = x[a][1];
        ru_xhold[3*a+2] = x[a][2];
    }

    ru_nbuild++;

    reuse_3b     = use3;
    reuse_4b     = use4;
    reuse_active = true;
#endif
}

bool PairCHIMES::reuse_lists_stale()
{
    // The cluster lists stay valid while no atom (local or ghost) has moved more than
    // ru_pad/2 since they were built: no pair distance can then have changed by more than ru_pad
    if (!reuse_active || !ru_mid_ok)
        return false;

    double **x = atom->x;
    const long long nall = (long long) atom->nlocal + (long long) atom->nghost;

    if ((long long) ru_xhold.size() != 3 * nall)
        return true;

    const double lim = 0.25 * ru_pad * ru_pad;

    for (long long a = 0; a < nall; a++)
    {
        const double dx = x[a][0] - ru_xhold[3*a];
        const double dy = x[a][1] - ru_xhold[3*a+1];
        const double dz = x[a][2] - ru_xhold[3*a+2];

        if (dx*dx + dy*dy + dz*dz > lim)
            return true;
    }

    return false;
}

/* ----------------------------------------------------------------------
   2B pair loop without per-pair overhead: E and dE/dr from chimesFF::eval_2B
   (2B tables or the Chebyshev sum), forces applied directly, and energy and
   global virial summed and tallied once.  Per-atom tallies keep ev_tally_mb
   per pair, as before.
------------------------------------------------------------------------- */

void PairCHIMES::compute_2b_fast(chimes2BTmp &tmp)
{
    double **x = atom->x;
    double **f = atom->f;
    int *type  = atom->type;
    tagint *tag = atom->tag;

    const int inum   = list->inum;
    int *ilist       = list->ilist;
    int *numneigh    = list->numneigh;
    int **firstneigh = list->firstneigh;

    // Per pair of ChIMES types: pair type and squared outer cutoff
    const int nct = chimes_calculator.natmtyps;
    std::vector<int> pidx(nct * nct);
    std::vector<double> rc2(nct * nct);

    for (int a = 0; a < nct; a++)
    {
        for (int b = 0; b < nct; b++)
        {
            const int p = chimes_calculator.pair_index(a, b);
            const double rc = chimes_calculator.pair_outer_cutoff(p);
            pidx[a*nct + b] = p;
            rc2[a*nct + b]  = rc * rc;
        }
    }

    const bool per_atom = (eflag_atom || vflag_atom);
    std::vector<double> stensor(6, 0.0);
    int atmidxlst[6][2];

    double esum = 0.0;
    double v[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };   // xx yy zz xy xz yz
    long long npair = 0, nactive = 0;

    for (int ii = 0; ii < inum; ii++)
    {
        const int i  = ilist[ii];
        const int ti = chimes_type[type[i]-1];

        if (ti < 0)
            continue;

        const chimes_key_t ikey = owner_key(tag[i]);
        const double xi = x[i][0];
        const double yi = x[i][1];
        const double zi = x[i][2];
        const int *jlist = firstneigh[i];
        const int jnum   = numneigh[i];

        double fx = 0.0, fy = 0.0, fz = 0.0;

        for (int jj = 0; jj < jnum; jj++)
        {
            const int j  = jlist[jj] & NEIGHMASK;
            const int tj = chimes_type[type[j]-1];

            if (tj < 0 || owner_key(tag[j]) <= ikey)   // full neighbor list: each pair once
                continue;

            const double dx = x[j][0] - xi;
            const double dy = x[j][1] - yi;
            const double dz = x[j][2] - zi;
            const double rsq = dx*dx + dy*dy + dz*dz;
            const int k = ti*nct + tj;

            if (rsq >= rc2[k])
                continue;

            const double r = sqrt(rsq);
            double E, D;
            chimes_calculator.eval_2B(pidx[k], r, E, D, tmp);

            const double fs = D / r;   // F_i = fs * (r_j - r_i), as in chimesFF::compute_2B

            fx += fs * dx;
            fy += fs * dy;
            fz += fs * dz;
            f[j][0] -= fs * dx;
            f[j][1] -= fs * dy;
            f[j][2] -= fs * dz;

            npair++;
            nactive += (E != 0.0);

            if (per_atom)
            {
                stensor[0] = -fs * dx * dx;
                stensor[1] = -fs * dx * dy;
                stensor[2] = -fs * dx * dz;
                stensor[3] = -fs * dy * dy;
                stensor[4] = -fs * dy * dz;
                stensor[5] = -fs * dz * dz;

                atmidxlst[0][0] = i;
                atmidxlst[0][1] = j;

                ev_tally_mb(2, 1, atmidxlst, E, stensor);
            }
            else
            {
                esum += E;
                v[0] -= fs * dx * dx;
                v[1] -= fs * dy * dy;
                v[2] -= fs * dz * dz;
                v[3] -= fs * dx * dy;
                v[4] -= fs * dx * dz;
                v[5] -= fs * dy * dz;
            }
        }

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
    }

    if (!per_atom)
    {
        if (eflag_global)
            eng_vdwl += esum;

        if (vflag_global)
            for (int c = 0; c < 6; c++)
                virial[c] += v[c];
    }

    CHIMES_PROF_COUNT_N(PG_2B, npair);
    CHIMES_PROF_ACTIVE_N(PG_2B, nactive);
}

void PairCHIMES::compute_mb_reuse()
{
#ifdef TABULATION
    double **x = atom->x;
    double **f = atom->f;
    int *type  = atom->type;

    const bool per_atom = (eflag_atom || vflag_atom);
    const int B = ru_block;
    const size_t W = 2 * (size_t) B;   // cache stride per entry: X[B] then dX/dr[B]

    const int np = (int) rp_a.size();
    const int nt = (int) ru_trip.size();
    const int ns = (int) ru_seed.size();
    const int ng = (int) ru_x3group.size();

    ////////////////////////////////////////
    // Pair distances (once per pair)
    ////////////////////////////////////////

    for (int p = 0; p < np; p++)
    {
        const int a = rp_a[p];
        const int b = rp_b[p];
        double *d = &rp_dr[3*p];

        d[0] = x[b][0] - x[a][0];
        d[1] = x[b][1] - x[a][1];
        d[2] = x[b][2] - x[a][2];

        rp_r[p]  = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        rp_fs[p] = 0.0;
    }

    const double *rr = rp_r.data();

    ////////////////////////////////////////
    // Sweep 2a: clusters inside the true cutoffs and the cache entries they use
    ////////////////////////////////////////

    // One flag per cache entry: its pair is inside the cluster cutoff.  A cluster is
    // inside its cutoffs when the flags of its three (3B) or six (4B) entries are set.
    const int ne = (int) re_pair.size();
    unsigned char * __restrict ok   = re_ok.data();
#ifdef CHIMES_PAIR_CACHE
    unsigned char * __restrict used = re_used.data();
#endif

    for (int e = 0; e < ne; e++)
        ok[e] = rr[re_pair[e]] < re_rc[e];

    // Branch-free compaction: write the index, advance the count only if valid.  Inner
    // clusters (always valid) come first in each list and are taken without a check.
    int nvt = ru_nt_inner;   // vt_list[0, ru_nt_inner) was filled at the build

    for (int t = ru_nt_inner; t < nt; t++)
    {
        const int *e = ru_trip[t].ent;

        vt_list[nvt] = t;
        nvt += ok[e[0]] & ok[e[1]] & ok[e[2]];
    }

    int nvs = 0;
    int nvq = 0;
    long long n_cand = 0;

    for (int s = 0; s < ns; s++)
    {
        const ReuseSeed &S = ru_seed[s];

        if (!S.inner && !(ok[S.ent[0]] & ok[S.ent[1]] & ok[S.ent[2]]))
            continue;

        const int begin = nvq;

        for (int qi = S.quad_begin; qi < S.quad_inner; qi++)
            vq_list[nvq++] = qi;

        for (int qi = S.quad_inner; qi < S.quad_end; qi++)
        {
            const int *e = ru_quad[qi].ent;

            vq_list[nvq] = qi;
            nvq += ok[e[0]] & ok[e[1]] & ok[e[2]];
        }

        n_cand += S.quad_end - S.quad_begin;

        if (nvq == begin)
            continue;

        vs_list[nvs] = s;
        vs_qbeg[nvs] = begin;
        nvs++;
    }

    vs_qbeg[nvs] = nvq;

    // Valid CP seeds come first (ru_seed[0, ru_nseed_cp) are the CP seeds), then exact 4B
    const int nvs_cp = (int) (std::lower_bound(vs_list.begin(), vs_list.begin() + nvs, ru_nseed_cp) - vs_list.begin());

    // Exact 3B: groups whose contracted pair is inside its cutoff, then their valid triplets
    int nvg = 0;
    int nvx = 0;

    for (int g = 0; g < ng; g++)
    {
        const ReuseX3Group &G = ru_x3group[g];

        if (!ok[G.ent])
            continue;

        const int begin = nvx;

        for (int t = G.trip_begin; t < G.trip_inner; t++)
            vx_list[nvx++] = t;

        for (int t = G.trip_inner; t < G.trip_end; t++)
        {
            const int *e = ru_trip3x[t].ent;

            vx_list[nvx] = t;
            nvx += ok[e[0]] & ok[e[1]] & ok[e[2]];
        }

        if (nvx == begin)
            continue;

        vg_list[nvg] = g;
        vg_tbeg[nvg] = begin;
        nvg++;
    }

    vg_tbeg[nvg] = nvx;

#ifdef CHIMES_PAIR_CACHE
    // Entries used by the selected clusters, in entry order.  The cache holds only these,
    // packed (slot = position in re_need), which keeps its memory footprint small.  Entries
    // of always-valid clusters are flagged in re_always at the build.
    const unsigned char * __restrict always = re_always.data();

    for (int v = ru_nt_inner; v < nvt; v++)
    {
        const int *e = ru_trip[vt_list[v]].ent;
        used[e[0]] = used[e[1]] = used[e[2]] = 1;
    }

    for (int v = 0; v < nvs; v++)
    {
        const ReuseSeed &S = ru_seed[vs_list[v]];
        const bool fixed = S.inner && S.quad_inner > S.quad_begin;   // seed and inner quads always used

        if (!fixed)
            used[S.ent[0]] = used[S.ent[1]] = used[S.ent[2]] = 1;

        for (int w = vs_qbeg[v]; w < vs_qbeg[v + 1]; w++)
        {
            const ReuseQuad &Qd = ru_quad[vq_list[w]];

            if (!(fixed && Qd.inner))
                used[Qd.ent[0]] = used[Qd.ent[1]] = used[Qd.ent[2]] = 1;
        }
    }

    for (int w = 0; w < nvx; w++)
    {
        const ReuseTrip &T = ru_trip3x[vx_list[w]];

        if (!T.inner)
            used[T.ent[0]] = used[T.ent[1]] = used[T.ent[2]] = 1;
    }

    int nne = 0;
    int * __restrict slot = re_slot.data();

    for (int e = 0; e < ne; e++)
    {
        slot[e] = nne;
        re_need[nne] = e;
        nne += used[e] | always[e];
        used[e] = 0;
    }

    if (re_val.size() < (size_t) nne * W)   // grow with headroom: resizing zero-fills
        re_val.resize((size_t) (1.25 * nne + 64) * W);
#endif

    ru_trip_acc.assign(4 * (size_t) nvt, 0.0);
    ru_x3_acc.resize(4 * (size_t) nvx);
    ru_seed_acc.assign(4 * (size_t) nvs, 0.0);
    ru_quad_acc.assign(7 * (size_t) nvq, 0.0);

    CHIMES_PROF_LAP(PROF_RU_PAIRS);

    ////////////////////////////////////////
    // 3B without CP tables: per-triplet evaluators, as without pair reuse
    ////////////////////////////////////////

    if (chimes_calculator.poly_orders[1] > 0 && !reuse_3b)
    {
        int atmidxlst[6][2];
        double energy;

        std::vector<double> stensor(6);
        chimes3BTmp chimes_3btmp(chimes_calculator.poly_orders[1]);

        const int ntrip = (int) neighborlist_3mers.size();

        for (int t = 0; t < ntrip; t++)
        {
            const int i = neighborlist_3mers[t][0];
            const int j = neighborlist_3mers[t][1];
            const int k = neighborlist_3mers[t][2];

            dist_3b[0] = get_dist(i, j, &dr_3b[0*CHDIM]);
            dist_3b[1] = get_dist(i, k, &dr_3b[1*CHDIM]);
            dist_3b[2] = get_dist(j, k, &dr_3b[2*CHDIM]);

            if (dist_3b[0] >= maxcut_3b || dist_3b[1] >= maxcut_3b || dist_3b[2] >= maxcut_3b)
                continue;

            typ_idxs_3b[0] = chimes_type[type[i]-1];
            typ_idxs_3b[1] = chimes_type[type[j]-1];
            typ_idxs_3b[2] = chimes_type[type[k]-1];

            std::fill(force_3b.begin(), force_3b.end(), 0.0);
            std::fill(stensor.begin(), stensor.end(), 0.0);

            energy = 0.0;

            if (chimes_calculator.tabulate_3B)
                chimes_calculator.compute_3B_tab(dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);
            else
                chimes_calculator.compute_3B(dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);

            for (int idx = 0; idx < CHDIM; idx++)
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
        }

        CHIMES_PROF_LAP(PROF_3B_EVAL);
    }

    ////////////////////////////////////////
    // Sweep 2b, in passes of B rank terms: pair cache -> 3B -> 4B seeds and quadruplets
    ////////////////////////////////////////

    double * __restrict Lq = ru_scratch.data();
    double * __restrict L0 = Lq + B;
    double * __restrict L1 = L0 + B;
    double * __restrict L2 = L1 + B;
    double * __restrict Sq = L2 + B;

#ifdef CHIMES_PAIR_CACHE
    double *val = re_val.data();
#else
    // Without the cache, a cluster's (seed's, exact 3B group's) pair functions are evaluated
    // when it is contracted, into one of three scratch slots of W values after the seed
    // products.  B is then the largest rank, so there is a single pass.
    double * __restrict pf = Sq + B;
    long long npf = 0;
#endif

    // Pair functions of entry e for the rank terms [q0, q0 + B): X, then dX/dr at X + B.  With
    // the cache they are read from it; without, evaluated into scratch slot s.
    auto pfun = [&](int e, int s, int q0) -> const double *
    {
#ifdef CHIMES_PAIR_CACHE
        (void) s;
        (void) q0;
        return val + slot[e]*W;
#else
        double *X = pf + s * W;
        const int n = MIN(re_rank[e] - q0, B);
        const int table = re_table[e];

        npf++;

        if (table < x3_ntab_cp)
            chimes_calculator.cp_table_eval(table, re_ptype[e], rr[re_pair[e]], q0, q0 + n, X, X + B);
        else
            chimes_calculator.exact_pair_eval(x3_spec_ptype[table - x3_ntab_cp], x3_spec_rin[table - x3_ntab_cp], x3_spec_rout[table - x3_ntab_cp], x3_spec_n[table - x3_ntab_cp], rr[re_pair[e]], X, X + B);

        return X;
#endif
    };

    const int nblk = (ru_rank_max + B - 1) / B;

    for (int blk = 0; blk < nblk; blk++)
    {
        const int q0 = blk * B;

#ifdef CHIMES_PAIR_CACHE
        // CP functions of the pairs used this step (each pair and table once)
        for (int m = 0; m < nne; m++)
        {
            const int e = re_need[m];
            const int n = MIN(re_rank[e] - q0, B);

            if (n <= 0)
                continue;

            const int table = re_table[e];

            if (table < x3_ntab_cp)
                chimes_calculator.cp_table_eval(table, re_ptype[e], rr[re_pair[e]], q0, q0 + n, val + m*W, val + m*W + B);
            else
                chimes_calculator.exact_pair_eval(x3_spec_ptype[table - x3_ntab_cp], x3_spec_rin[table - x3_ntab_cp], x3_spec_rout[table - x3_ntab_cp], x3_spec_n[table - x3_ntab_cp], rr[re_pair[e]], val + m*W, val + m*W + B);
        }

        CHIMES_PROF_LAP(PROF_RU_PAIRS);
#endif

        // 3B
        for (int v = 0; v < nvt; v++)
        {
            const ReuseTrip &T = ru_trip[vt_list[v]];
            const int n = MIN(T.rank - q0, B);

            if (n <= 0)
                continue;

            const double * __restrict x0 = pfun(T.ent[0], 0, q0);
            const double * __restrict x1 = pfun(T.ent[1], 1, q0);
            const double * __restrict x2 = pfun(T.ent[2], 2, q0);
            const double * __restrict d0 = x0 + B;
            const double * __restrict d1 = x1 + B;
            const double * __restrict d2 = x2 + B;

            double t[4];
            ru_k3_cp(n, x0, x1, x2, d0, d1, d2, t);

            double *acc = &ru_trip_acc[4*(size_t)v];
            acc[0] += t[0];
            acc[1] += t[1];
            acc[2] += t[2];
            acc[3] += t[3];
        }

        // 3B, exact polynomial (first pass only: all N pair-function terms are in the cache).
        // Per group M = sum_x G_s[x] R_s[x] and dM, then per triplet with A = G_sa, Bv = G_sb:
        //     w = A M,  wd = dA M,  u = A dM
        //     E = w.Bv,  dE/dr_sa = wd.Bv,  dE/dr_sb = w.dBv,  dE/dr_s = u.Bv
        if (blk == 0 && nvg > 0)
        {
            const int N  = x3_rank;
            const int NN = N * N;

            // Triplets of a group in batches of L = RU_LANES (ru_k3x_batch): each batch gathers
            // the pair functions of its triplets lane-minor, so each element of M is read once
            // per batch and every multiply-add runs across the L triplets.
            const int L = RU_LANES;

            double * __restrict M   = ru_x3_mat.data();
            double * __restrict dM  = M + NN;
            double * __restrict At  = dM + NN;
            double * __restrict dAt = At + N * L;
            double * __restrict Bt  = dAt + N * L;
            double * __restrict dBt = Bt + N * L;

            for (int v = 0; v < nvg; v++)
            {
                const ReuseX3Group &G = ru_x3group[vg_list[v]];
                const double * __restrict gm  = pfun(G.ent, 0, 0);
                const double * __restrict dgm = gm + B;
                const double * __restrict R   = x3_cls_tensor[G.cls].data();

                for (int ab = 0; ab < NN; ab++)
                {
                    M[ab]  = 0.0;
                    dM[ab] = 0.0;
                }

                for (int xx = 0; xx < N; xx++)
                {
                    const double g  = gm[xx];
                    const double dg = dgm[xx];
                    const double * __restrict Rx = R + (size_t) xx * NN;

                    for (int ab = 0; ab < NN; ab++)
                    {
                        M[ab]  += g * Rx[ab];
                        dM[ab] += dg * Rx[ab];
                    }
                }

                for (int w0 = vg_tbeg[v]; w0 < vg_tbeg[v + 1]; w0 += L)
                {
                    const int nl = MIN(L, vg_tbeg[v + 1] - w0);

                    for (int l = 0; l < L; l++)
                    {
                        if (l < nl)
                        {
                            const ReuseTrip &T = ru_trip3x[vx_list[w0 + l]];
                            const int sm = T.mslot;               // the group's pair
                            const int sa = (sm == 0) ? 1 : 0;     // the other two slots, ascending
                            const int sb = (sm == 2) ? 1 : 2;
                            const double * __restrict A   = pfun(T.ent[sa], 1, 0);
                            const double * __restrict dA  = A + B;
                            const double * __restrict Bv  = pfun(T.ent[sb], 2, 0);
                            const double * __restrict dBv = Bv + B;

                            for (int a = 0; a < N; a++)
                            {
                                At[a*L + l]  = A[a];
                                dAt[a*L + l] = dA[a];
                                Bt[a*L + l]  = Bv[a];
                                dBt[a*L + l] = dBv[a];
                            }
                        }
                        else
                        {
                            for (int a = 0; a < N; a++)
                                At[a*L + l] = dAt[a*L + l] = Bt[a*L + l] = dBt[a*L + l] = 0.0;
                        }
                    }

                    double e3[RU_LANES], gs[RU_LANES], ga[RU_LANES], gb[RU_LANES];
                    ru_k3x_batch(N, M, dM, At, dAt, Bt, dBt, e3, gs, ga, gb);

                    for (int l = 0; l < nl; l++)
                    {
                        const int sm = ru_trip3x[vx_list[w0 + l]].mslot;
                        const int sa = (sm == 0) ? 1 : 0;
                        const int sb = (sm == 2) ? 1 : 2;

                        double *acc = &ru_x3_acc[4*(size_t)(w0 + l)];
                        acc[0]      = e3[l];
                        acc[1 + sm] = gs[l];
                        acc[1 + sa] = ga[l];
                        acc[1 + sb] = gb[l];
                    }
                }
            }
        }

        CHIMES_PROF_LAP(PROF_RU_3B);

        // 4B, CP
        for (int v = 0; v < nvs_cp; v++)
        {
            const ReuseSeed &S = ru_seed[vs_list[v]];
            const int n = MIN(S.rank - q0, B);

            if (n <= 0)
                continue;

            {
                const double *x0 = pfun(S.ent[0], 0, q0);
                const double *x1 = pfun(S.ent[1], 1, q0);
                const double *x2 = pfun(S.ent[2], 2, q0);

                ru_k4_seed(n, x0, x1, x2, x0 + B, x1 + B, x2 + B, Lq, L0, L1, L2, Sq);
            }

            for (int w = vs_qbeg[v]; w < vs_qbeg[v + 1]; w++)
            {
                const ReuseQuad &Qd = ru_quad[vq_list[w]];

                const double *y0 = pfun(Qd.ent[0], 0, q0);
                const double *y1 = pfun(Qd.ent[1], 1, q0);
                const double *y2 = pfun(Qd.ent[2], 2, q0);

                double *acc = &ru_quad_acc[7*(size_t)w];
                double t[7];

                if (!per_atom)
                {
                    // Energy and seed-side gradients come from sum_l (Sq) below; here only the
                    // fourth-atom-side gradients
                    ru_k4_quad(n, Lq, y0, y1, y2, y0 + B, y1 + B, y2 + B, Sq, t);

                    acc[1] += t[0];
                    acc[2] += t[1];
                    acc[3] += t[2];
                }
                else
                {
                    // Per-quadruplet energy and all six gradients (per-atom tallies)
                    ru_k4_quad_full(n, Lq, L0, L1, L2, y0, y1, y2, y0 + B, y1 + B, y2 + B, t);

                    for (int c = 0; c < 7; c++)
                        acc[c] += t[c];
                }
            }

            if (!per_atom)
            {
                double t[4];
                ru_k4_seed_dot(n, Lq, L0, L1, L2, Sq, t);

                double *acc = &ru_seed_acc[4*(size_t)v];
                acc[0] += t[0];
                acc[1] += t[1];
                acc[2] += t[2];
                acc[3] += t[3];
            }
        }

        // 4B, exact polynomial (parameter-file coefficients or Morse-space SVD), first pass
        // only: all N pair-function terms are in the cache.  For a seed with pair functions
        // G01, G02, G12 and class matrix M[a][b][c][s]:
        //     w[s] = sum_abc M[a][b][c][s] G01[a] G02[b] G12[c],
        // contracted as Q = sum_a G_k[a] M_k[a] once per group (seeds sharing the pair in seed
        // slot k and the class; M_k: the class matrix with that pair first), then per seed
        // w = sum_bc G_y[b] G_z[c] Q[bc] (y < z: the other two pairs).  Each of the seed's
        // quadruplets then costs about 3 N^3: E = sum_s w[s] G03[d] G13[e] G23[f] and its three
        // fourth-atom pair gradients.  The seed-pair gradients follow from the star products
        // summed over the seed's quadruplets (S) and Q, dQ in [s][bc] layout (Qt, dQt, formed
        // with Q per group), so that every product is a vector update.
        if (blk == 0 && nvs > nvs_cp)
        {
            const int N  = x4_rank;
            const int N2 = N * N;
            const int N3 = N2 * N;
            const int N5 = N3 * N2;

            double *Q   = ru_x4_work.data();   // [bc][s]
            double *dQ  = Q + N5;               // [bc][s], per-atom tallies only
            double *Qt  = dQ + N5;              // [s][bc]
            double *dQt = Qt + N5;              // [s][bc]
            double *w   = dQt + N5;
            double *Sst = w + N3;
            double *dw0 = Sst + N3;
            double *dw1 = dw0 + N3;
            double *dw2 = dw1 + N3;
            double *P   = dw2 + N3;
            double *tq  = P + N3;               // 2 N^2
            double *qv  = tq + 2 * (size_t) N2;
            double *dqv = qv + N2;
            double *k0  = dqv + N2;             // gb gc, dgb gc, gb dgc
            double *k1  = k0 + N2;
            double *k2  = k1 + N2;
            double *gb  = k2 + N2;
            double *dgb = gb + N;
            double *gc  = dgb + N;
            double *dgc = gc + N;

            int g_ent = -1, g_cls = -1;

            for (int v = nvs_cp; v < nvs; v++)
            {
                const ReuseSeed &S = ru_seed[vs_list[v]];
                const int kx = S.cls % 3;              // seed slot contracted by the group
                const int ky = (kx == 0) ? 1 : 0;      // the other two, in order
                const int kz = (kx == 2) ? 1 : 2;

                if (S.ent[kx] != g_ent || S.cls != g_cls)   // next group
                {
                    g_ent = S.ent[kx];
                    g_cls = S.cls;

                    const double *Gx = pfun(S.ent[kx], 0, 0);

                    ru_k4x_group(N, N5, x4_cls_mat[S.cls].data(), Gx, Gx + B, Q, per_atom ? dQ : nullptr);

                    if (!per_atom)
                        ru_k4x_group(N, N5, x4_cls_matT[S.cls].data(), Gx, Gx + B, Qt, dQt);
                }

                // The other seed pairs' functions are kept here: the scratch slots of pfun are
                // reused by the quadruplets without the cache
                {
                    const double *Gy = pfun(S.ent[ky], 1, 0);
                    const double *Gz = pfun(S.ent[kz], 2, 0);

                    for (int n = 0; n < N; n++)
                    {
                        gb[n]  = Gy[n];
                        dgb[n] = Gy[n + B];
                        gc[n]  = Gz[n];
                        dgc[n] = Gz[n + B];
                    }

                    for (int b = 0; b < N; b++)
                    {
                        for (int c = 0; c < N; c++)
                        {
                            k0[b*N + c] = gb[b]  * gc[c];
                            k1[b*N + c] = dgb[b] * gc[c];
                            k2[b*N + c] = gb[b]  * dgc[c];
                        }
                    }
                }

                ru_k4x_bc_sum(N2, N3, Q, k0, w);   // w[s] = sum_bc gb gc Q[bc][s]

                if (per_atom)
                {
                    ru_k4x_bc_sum(N2, N3, dQ, k0, dw0);   // dw/dr of the seed pairs
                    ru_k4x_bc_sum(N2, N3, Q, k1, dw1);
                    ru_k4x_bc_sum(N2, N3, Q, k2, dw2);
                }
                else
                    for (int x = 0; x < N3; x++)
                        Sst[x] = 0.0;

                double esum = 0.0;

                for (int wq = vs_qbeg[v]; wq < vs_qbeg[v + 1]; wq++)
                {
                    const ReuseQuad &Qd = ru_quad[vq_list[wq]];

                    const double *y0 = pfun(Qd.ent[0], 0, 0);   // A0A3
                    const double *y1 = pfun(Qd.ent[1], 1, 0);   // A1A3
                    const double *y2 = pfun(Qd.ent[2], 2, 0);   // A2A3

                    double t[4];
                    ru_k4x_quad(N, w, y0, y0 + B, y1, y1 + B, y2, y2 + B, tq, t);

                    double *acc = &ru_quad_acc[7*(size_t)wq];
                    acc[1] = t[1];
                    acc[2] = t[2];
                    acc[3] = t[3];

                    if (!per_atom)
                    {
                        esum += t[0];
                        ru_k4x_star(N, y0, y1, y2, Sst, true);
                    }
                    else
                    {
                        acc[0] = t[0];
                        ru_k4x_star(N, y0, y1, y2, P, false);

                        double d0, d1, d2, d3;
                        ru_dot2(N3, dw0, dw1, P, d0, d1);
                        ru_dot2(N3, dw2, dw2, P, d2, d3);
                        acc[4 + kx] = d0;
                        acc[4 + ky] = d1;
                        acc[4 + kz] = d2;
                    }
                }

                if (!per_atom)
                {
                    double g[3];
                    ru_k4x_seed_grad(N, N3, Qt, dQt, Sst, gb, dgb, gc, dgc, qv, dqv, g);

                    double *acc = &ru_seed_acc[4*(size_t)v];
                    acc[0] = esum;
                    acc[1 + kx] = g[0];
                    acc[1 + ky] = g[1];
                    acc[1 + kz] = g[2];
                }
            }
        }

        CHIMES_PROF_LAP(PROF_RU_4B);
    }

#ifdef CHIMES_PAIR_CACHE
    CHIMES_PROF_COUNT_N(PG_PAIR, nne);
#else
    CHIMES_PROF_COUNT_N(PG_PAIR, npf);   // pair-function evaluations
    (void) npf;
#endif
    CHIMES_PROF_COUNT_N(PG_3B, nvt + nvx);
    CHIMES_PROF_COUNT_N(PG_4B, nvq);
    CHIMES_PROF_EXTRA_N(EX_4B_SEEDS, nvs);
    CHIMES_PROF_EXTRA_N(EX_4B_CAND, n_cand);
    CHIMES_PROF_EXTRA_N(EX_4B_PREP, nvs);
    CHIMES_PROF_EXTRA_N(EX_4B_REUSE, nvq);

    ////////////////////////////////////////
    // Forces, energy and virial
    ////////////////////////////////////////

    std::vector<double> stensor(6, 0.0);
    int atmidxlst[6][2];

    auto pair_force = [&](int p, double fs, std::vector<double> &st)
    {
        const double *d = &rp_dr[3*p];
        double *fa = f[rp_a[p]];
        double *fb = f[rp_b[p]];

        fa[0] += fs * d[0];
        fa[1] += fs * d[1];
        fa[2] += fs * d[2];
        fb[0] -= fs * d[0];
        fb[1] -= fs * d[1];
        fb[2] -= fs * d[2];

        st[0] -= fs * d[0] * d[0];
        st[1] -= fs * d[0] * d[1];
        st[2] -= fs * d[0] * d[2];
        st[3] -= fs * d[1] * d[1];
        st[4] -= fs * d[1] * d[2];
        st[5] -= fs * d[2] * d[2];
    };

    // CP and exact 3B triplets share one layout: acc[0] = E, acc[1 + c] = dE/dr of slot c
    auto trip_scalars = [&](const ReuseTrip &T, const double *acc, double &etot)
    {
        etot += acc[0];

        for (int c = 0; c < 3; c++)
        {
            const int p = T.pid[T.c2r[c]];
            rp_fs[p] += acc[1 + c] / rr[p];
        }
    };

    auto trip_tally = [&](const ReuseTrip &T, const double *acc)
    {
        std::fill(stensor.begin(), stensor.end(), 0.0);

        for (int c = 0; c < 3; c++)
        {
            const int p = T.pid[T.c2r[c]];
            pair_force(p, acc[1 + c] / rr[p], stensor);
        }

        atmidxlst[0][0] = T.atom[0]; atmidxlst[0][1] = T.atom[1];
        atmidxlst[1][0] = T.atom[0]; atmidxlst[1][1] = T.atom[2];
        atmidxlst[2][0] = T.atom[1]; atmidxlst[2][1] = T.atom[2];

        ev_tally_mb(3, 3, atmidxlst, acc[0], stensor);
    };

    if (!per_atom)
    {
        // Sum force scalars per pair, apply each pair once, one tally for the step

        double etot = 0.0;

        for (int v = 0; v < nvt; v++)
            trip_scalars(ru_trip[vt_list[v]], &ru_trip_acc[4*(size_t)v], etot);

        for (int v = 0; v < nvx; v++)
            trip_scalars(ru_trip3x[vx_list[v]], &ru_x3_acc[4*(size_t)v], etot);

        for (int v = 0; v < nvs; v++)
        {
            const ReuseSeed &S = ru_seed[vs_list[v]];
            const double *sacc = &ru_seed_acc[4*(size_t)v];

            etot += sacc[0];

            for (int k = 0; k < 3; k++)
                rp_fs[S.pid[k]] += sacc[1 + k] / rr[S.pid[k]];

            for (int w = vs_qbeg[v]; w < vs_qbeg[v + 1]; w++)
            {
                const ReuseQuad &Qd = ru_quad[vq_list[w]];
                const double *qacc = &ru_quad_acc[7*(size_t)w];

                for (int k = 0; k < 3; k++)
                    rp_fs[Qd.pid[k]] += qacc[1 + k] / rr[Qd.pid[k]];
            }
        }

        for (int p = 0; p < np; p++)
            if (rp_fs[p] != 0.0)
                pair_force(p, rp_fs[p], stensor);

        if (evflag)
        {
            atmidxlst[0][0] = 0;
            atmidxlst[0][1] = 0;
            ev_tally_mb(2, 1, atmidxlst, etot, stensor);
        }
    }
    else
    {
        // One cluster at a time, so per-atom energies and virials are partitioned as without reuse

        for (int v = 0; v < nvt; v++)
            trip_tally(ru_trip[vt_list[v]], &ru_trip_acc[4*(size_t)v]);

        for (int v = 0; v < nvx; v++)
            trip_tally(ru_trip3x[vx_list[v]], &ru_x3_acc[4*(size_t)v]);

        for (int v = 0; v < nvs; v++)
        {
            const ReuseSeed &S = ru_seed[vs_list[v]];

            for (int w = vs_qbeg[v]; w < vs_qbeg[v + 1]; w++)
            {
                const ReuseQuad &Qd = ru_quad[vq_list[w]];
                const double *qacc = &ru_quad_acc[7*(size_t)w];

                std::fill(stensor.begin(), stensor.end(), 0.0);

                for (int k = 0; k < 3; k++)
                {
                    pair_force(S.pid[k], qacc[4 + k] / rr[S.pid[k]], stensor);
                    pair_force(Qd.pid[k], qacc[1 + k] / rr[Qd.pid[k]], stensor);
                }

                atmidxlst[0][0] = S.atom[0]; atmidxlst[0][1] = S.atom[1];
                atmidxlst[1][0] = S.atom[0]; atmidxlst[1][1] = S.atom[2];
                atmidxlst[2][0] = S.atom[0]; atmidxlst[2][1] = Qd.l;
                atmidxlst[3][0] = S.atom[1]; atmidxlst[3][1] = S.atom[2];
                atmidxlst[4][0] = S.atom[1]; atmidxlst[4][1] = Qd.l;
                atmidxlst[5][0] = S.atom[2]; atmidxlst[5][1] = Qd.l;

                ev_tally_mb(4, 6, atmidxlst, qacc[0], stensor);
            }
        }
    }

    CHIMES_PROF_LAP(PROF_RU_FORCE);
#endif
}
#endif

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
        chimes_key_t ikey, jkey, kkey, lkey;   // ownership keys (owner_key)
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

        // Build the ChIMES many-body lists when the LAMMPS list was rebuilt.  With pair reuse
        // the cluster lists are also rebuilt when an atom has moved more than half their
        // padding since their last build (padding smaller than the neighbor skin).

        const bool nl_build = (neighbor->ago == 0);
        bool need_3mers = true;

#ifdef CHIMES_PAIR_REUSE
        if (nl_build || reuse_lists_stale())
        {
            if (!nl_build)
                ru_nmid++;

            build_reuse_lists();

            if (nl_build && chimes_calculator.rank == 0 && reuse_active) {
                std::cout << "  Rank " << me << " pair reuse: " << rp_a.size() << " pairs, " << re_pair.size() << " pair-slots";

                if (!ru_trip.empty())
                    std::cout << ", " << ru_trip.size() << " 3B CP triplets";

                if (!ru_trip3x.empty())
                    std::cout << ", " << ru_trip3x.size() << " exact 3B triplets in " << ru_x3group.size() << " groups";

                if (reuse_4b)
                {
                    std::cout << ", " << ru_seed.size() << " seeds, " << ru_quad.size() << " quads (" << (reuse_greedy ? "greedy" : "first-triangle") << " seeds)";

                    if ((int) ru_seed.size() > ru_nseed_cp)
                        std::cout << ", exact 4B: " << ru_seed.size() - ru_nseed_cp << " seeds in " << ru_nx4_group << " pair groups, " << x4_cls_mat.size() / 3 << " classes";

                    if (reuse_mixed4)
                        std::cout << ", 4B types with grid tables per cluster";
                }
                else if (chimes_calculator.poly_orders[2] > 0)
                    std::cout << ", 4B per cluster";

#ifdef CHIMES_PAIR_CACHE
                std::cout << ", pair cache, " << ru_block << " rank terms per pass";
#else
                std::cout << ", no pair cache";
#endif
                std::cout << ", padding " << ru_pad << ", " << ru_nbuild << " list builds (" << ru_nmid << " between neighbor-list builds)" << std::endl;
            }

            CHIMES_PROF_LAP(PROF_RU_BUILD);
        }

        // The triplet list is needed only for what the reuse path leaves to the triplet loop
        need_3mers = !reuse_active || (chimes_calculator.poly_orders[1] > 0 && !reuse_3b) || (chimes_calculator.poly_orders[2] > 0 && (!reuse_4b || reuse_mixed4));
#endif

        if (nl_build) {
            if (chimes_calculator.rank == 0)
                std::cout << "Updating chimesFF neighbor lists..." << std::endl;

            if (need_3mers)
                build_mb_neighlists();
            else
                neighborlist_3mers.clear();

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

        bool generic_2b = true;

#ifdef CHIMES_PAIR_REUSE
#ifdef FINGERPRINT
        if (!tmp_FP)
#endif
        {
            compute_2b_fast(chimes_2btmp);
            generic_2b = false;
        }
#endif

        for (ii = 0; ii < inum && generic_2b; ii++)
        {
            i     = ilist[ii];
            ikey  = owner_key(tag[i]);

            if (chimes_type[type[i]-1] < 0)
                continue;

            jlist = firstneigh[i];
            jnum  = numneigh[i];

            for (jj = 0; jj < jnum; jj++)
            {
                j     = jlist[jj];
                jkey  = owner_key(tag[j]);
                j    &= NEIGHMASK;

                if (chimes_type[type[j]-1] < 0)
                    continue;

                if (jkey <= ikey) // full neighbor list: evaluate each pair once
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

        bool old_3b = true;   // body orders still evaluated per cluster in the triplet loop below
        bool old_4b = true;

#ifdef CHIMES_PAIR_REUSE
        if (reuse_active)
        {
            compute_mb_reuse();

            old_3b = false;
            old_4b = !reuse_4b || reuse_mixed4;   // mixed: the 4B types without a CP model
        }
#endif
        if ((old_3b && chimes_calculator.poly_orders[1] > 0) || (old_4b && chimes_calculator.poly_orders[2] > 0))
        {
            std::vector<chimesFF::chimes4BTripletReuseTmp> reuse_cache;
            reuse_cache.reserve(8);

            const int ntrip = (int) neighborlist_3mers.size();

            for (ii = 0; ii < ntrip; ii++)
            {
                i = neighborlist_3mers[ii][0];
                j = neighborlist_3mers[ii][1];
                k = neighborlist_3mers[ii][2];

                ikey = owner_key(tag[i]);
                jkey = owner_key(tag[j]);
                kkey = owner_key(tag[k]);

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
                if (old_3b && chimes_calculator.poly_orders[1] > 0 && dist_ij < maxcut_3b && dist_ik < maxcut_3b && dist_jk < maxcut_3b)
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
                if (old_4b && chimes_calculator.poly_orders[2] > 0 && dist_ij < maxcut_4b && dist_ik < maxcut_4b && dist_jk < maxcut_4b)
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
                     * key of l >= keys of i, j, k so each 4B cluster is evaluated once.
                     */
                    llist = firstneigh[i];
                    lnum  = numneigh[i];

                    for (int ll = 0; ll < lnum; ll++)
                    {
                        l  = llist[ll];
                        l &= NEIGHMASK;

                        if ((l == i) || (l == j) || (l == k))
                            continue;

                        lkey = owner_key(tag[l]);

                        if ((lkey < ikey) || (lkey < jkey) || (lkey < kkey))
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

#ifdef CHIMES_PAIR_REUSE
                        // Mixed 4B model: types with a CP model are done through pair reuse
                        if (reuse_4b && ru_cache4_type[((type_i*chimes_calculator.natmtyps + type_j)*chimes_calculator.natmtyps + type_k)*chimes_calculator.natmtyps + type_l])
                            continue;
#endif

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