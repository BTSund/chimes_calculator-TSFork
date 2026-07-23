/* 
    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
    Contributing Author:  Rebecca K. Lindsey (2020) 
*/

#ifndef _chimesFF_h
#define _chimesFF_h


#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<map>

#define pi 3.14159265359

using namespace std;

// Notes:
//
// 1. A Morse-style coordinate transformation is hard-coded (see set_cheby_polys) 
// 2. Polynomials are hard-coded over the domain [-1,1]
// 3. A cubic style cutoff is assumed, and Tersoff is the only other style considered (see get_fcut)


#define CHDIM 3 // The number of spatial dimensions.
#define USE_DISTANCE_TENSOR 1 // Use tensor of distances in computing stresses.

// Temporary storage for ChIMES interaction.
class chimes2BTmp
{
public:
    inline chimes2BTmp(int poly_order) ;
    inline void resize(int poly_order) ;
    vector<double> Tn ;
    vector<double> Tnd ;
} ;


inline chimes2BTmp::chimes2BTmp(int poly_order) : Tn(poly_order+1), Tnd(poly_order+1) 
{
    ;
}

inline void chimes2BTmp::resize(int poly_order)
{
    
    if ( Tn.size() < poly_order + 1 ) 
        Tn.resize(poly_order+1) ;

    if ( Tnd.size() < poly_order + 1 ) 
        Tnd.resize(poly_order+1) ;
}

class chimes3BTmp
{
public:
    inline chimes3BTmp(int poly_order) ;
    inline void resize(int poly_order) ;

    vector<double>  Tn_ij,   Tn_ik,   Tn_jk;   // The Chebyshev polymonials
    vector<double>  Tnd_ij,  Tnd_ik,  Tnd_jk;  // The Chebyshev polymonial derivatives

} ;

inline chimes3BTmp::chimes3BTmp(int poly_order) : Tn_ij(poly_order+1), Tn_ik(poly_order+1), Tn_jk(poly_order+1),
                                                  Tnd_ij(poly_order+1), Tnd_ik(poly_order+1), Tnd_jk(poly_order+1)
{
    ;
}


class chimes4BTmp
{
public:
    inline chimes4BTmp(int poly_order);
    inline void resize(int poly_order);

    vector<double>  Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl;
    vector<double>  Tnd_ij,Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl;


#ifdef TABULATION
    // Scratch for SVD 4|2.
    std::vector<double> svd4_L;
    std::vector<double> svd4_L0;
    std::vector<double> svd4_L1;
    std::vector<double> svd4_L2;
    std::vector<double> svd4_L3;

    std::vector<double> svd4_R;
    std::vector<double> svd4_R0;
    std::vector<double> svd4_R1;

    inline void resize_svd4_rank(int R)
    {
        if ((int)svd4_L.size() < R)
        {
            svd4_L.resize(R);
            svd4_L0.resize(R);
            svd4_L1.resize(R);
            svd4_L2.resize(R);
            svd4_L3.resize(R);

            svd4_R.resize(R);
            svd4_R0.resize(R);
            svd4_R1.resize(R);
        }
    }
#endif
#ifdef TABULATION
    // Scratch buffers for SVD 3|3 4B tabulation.
    // These avoid heap allocations inside compute_4B_svd3x3_tab().
    vector<double> svd_X;
    vector<double> svd_X0;
    vector<double> svd_X1;
    vector<double> svd_X2;

    vector<double> svd_Y;
    vector<double> svd_Y0;
    vector<double> svd_Y1;
    vector<double> svd_Y2;

    inline void resize_svd_rank(int R)
    {
        if ((int)svd_X.size() < R)
        {
            svd_X.resize(R);
            svd_X0.resize(R);
            svd_X1.resize(R);
            svd_X2.resize(R);

            svd_Y.resize(R);
            svd_Y0.resize(R);
            svd_Y1.resize(R);
            svd_Y2.resize(R);
        }
    }
#endif
#ifdef TABULATION
    // Scratch buffers for CP 1D 4B tabulation.
    //
    // cp_val[m][q] = X_m[q]
    // cp_der[m][q] = dX_m[q]/dr_m
    std::vector<double> cp_val[6];
    std::vector<double> cp_der[6];

    inline void resize_cp_rank(int Q)
    {
        for (int m = 0; m < 6; m++)
        {
            if ((int)cp_val[m].size() < Q)
                cp_val[m].resize(Q);

            if ((int)cp_der[m].size() < Q)
                cp_der[m].resize(Q);
        }
    }
#endif
};
inline chimes4BTmp::chimes4BTmp(int poly_order) : Tn_ij(poly_order+1), Tn_ik(poly_order+1), Tn_il(poly_order+1),
                                                  Tn_jk(poly_order+1), Tn_jl(poly_order+1), Tn_kl(poly_order+1),
                                                  Tnd_ij(poly_order+1), Tnd_ik(poly_order+1), Tnd_il(poly_order+1),
                                                  Tnd_jk(poly_order+1), Tnd_jl(poly_order+1), Tnd_kl(poly_order+1)
{
    ;
}

inline void chimes3BTmp::resize(int poly_order)
{
    
    if ( Tn_ij.size() < poly_order + 1 ) 
        Tn_ij.resize(poly_order+1) ;

    if ( Tnd_ij.size() < poly_order + 1 ) 
        Tnd_ij.resize(poly_order+1) ;

    if ( Tn_ik.size() < poly_order + 1 ) 
        Tn_ik.resize(poly_order+1) ;

    if ( Tnd_ik.size() < poly_order + 1 ) 
        Tnd_ik.resize(poly_order+1) ;

    if ( Tn_jk.size() < poly_order + 1 ) 
        Tn_jk.resize(poly_order+1) ;

    if ( Tnd_jk.size() < poly_order + 1 ) 
        Tnd_jk.resize(poly_order+1) ;   
}

enum class fcutType
{
    CUBIC,
    TERSOFF,
} ;
    
class chimesFF
{
public:
    
    ////////////////////////
    // General parameters 
    ////////////////////////
    
    int              rank;           // Used to prevent multiple cout statements when accessed from MPI
    int              natmtyps;       // How many atom types are defined for this force field?

        
    vector<int>      poly_orders;    // [bodiedness-1]; i.e. 12 = 2-body only, 12th order; 12 5 = 2+3-body, 0 5 = 3-body only, 5th order
    vector<string>   atmtyps;        // Atom types 
    vector<double>   masses;         // Atom masses

    ////////////////////////
    // Functions
    ////////////////////////
    
    chimesFF();
    ~chimesFF();
        
    void init(int mpi_rank);
        
    void read_parameters(string paramfile); 
        
    void compute_1B(const int typ_idx, double & energy );
        
	// 2+B compute functions overloaded with force_scalar_in var for compatibility with LAMMPS

	void compute_2B(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmp &tmp);
	void compute_2B(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmp &tmp, double & force_scalar_in
                    #ifdef FINGERPRINT
                        , vector<vector<double>> & clusters_2b, bool fingerprint
                    #endif
                    ); 

	void compute_3B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force,vector<double> & stress, double & energy, chimes3BTmp &tmp);
	void compute_3B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force,vector<double> & stress, double & energy, chimes3BTmp &tmp, vector<double> & force_scalar_in
                    #ifdef FINGERPRINT
                        , vector<vector<double >> & clusters_3b, bool fingerprint
                    #endif
                    ); 

	void compute_4B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes4BTmp &tmp);
	void compute_4B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes4BTmp &tmp, vector<double> & force_scalar_in
                    #ifdef FINGERPRINT
                        , vector<vector<double>> & clusters_4b, bool fingerprint
                    #endif
                    );
        struct chimes4BTripletReuseTmp
    {
        bool valid = false;
        bool use_svd3x3 = false;

        int quadidx = -1;
        int type_l_context = -1;

        // Runtime pair slots:
        //   0 = ij
        //   1 = ik
        //   2 = il
        //   3 = jk
        //   4 = jl
        //   5 = kl
        //
        // Triplet-star SVD3x3:
        //   left  = ij,ik,jk = 0,1,3
        //   right = il,jl,kl = 2,4,5
        int left_runtime[3]  = {0, 1, 3};
        int right_runtime[3] = {2, 4, 5};

        int rank = 0;

        std::vector<double> X;
        std::vector<double> X0;
        std::vector<double> X1;
        std::vector<double> X2;

        // Sparse direct-polynomial partial contraction.
        std::vector<int> p_il;
        std::vector<int> p_jl;
        std::vector<int> p_kl;

        std::vector<double> A;
        std::vector<double> A_ij;
        std::vector<double> A_ik;
        std::vector<double> A_jk;

        void clear()
        {
            valid = false;
            use_svd3x3 = false;

            quadidx = -1;
            type_l_context = -1;
            rank = 0;

            left_runtime[0] = 0;
            left_runtime[1] = 1;
            left_runtime[2] = 3;

            right_runtime[0] = 2;
            right_runtime[1] = 4;
            right_runtime[2] = 5;

            X.clear();
            X0.clear();
            X1.clear();
            X2.clear();

            p_il.clear();
            p_jl.clear();
            p_kl.clear();

            A.clear();
            A_ij.clear();
            A_ik.clear();
            A_jk.clear();
        }
    };

    bool prepare_4B_triplet_reuse(
        const std::vector<double> & dx_trip_013,
        const std::vector<int> & typ_idxs,
        chimes4BTmp & tmp,
        chimes4BTripletReuseTmp & reuse
    );

    void compute_4B_from_triplet_reuse(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        const chimes4BTripletReuseTmp & reuse,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp
    );
    void get_cutoff_2B(vector<vector<double> >  & cutoff_2b);   // Populates the 2b cutoffs
    
    double max_cutoff_2B(bool silent = false);    // Returns the largest 2B cutoff
    double max_cutoff_3B(bool silent = false);    // Returns the largest 3B cutoff
    double max_cutoff_4B(bool silent = false);    // Returns the largest 4B cutoff
        
    void set_atomtypes(vector<string> & type_list);
    
    int get_atom_pair_index(int pair_id);
    void build_pair_int_trip_map() ;
    void build_pair_int_quad_map() ;
    
    // Functions to aid using ChIMES Calculator for fitting
    
    inline int  get_badness();
    inline void reset_badness();
    
    #ifdef TABULATION
    // -------------------------------------------------------------------------
    // Exact/on-the-fly CP 4-body evaluator
    // -------------------------------------------------------------------------

    bool tabulate_4B_cp_direct = false;

    // [quadidx] rank
    std::vector<int> tab_4b_cp_direct_rank;

    // [quadidx][canonical_slot] = parameter slot
    std::vector<std::vector<int>> tab_4b_cp_direct_canon_to_param;

    // [quadidx][canonical_slot] = max Chebyshev power/order for that slot
    std::vector<std::vector<int>> tab_4b_cp_direct_orders;

    // Flattened factor matrices:
    //
    // tab_4b_cp_direct_factor[quadidx][canonical_slot][p*rank + q]
    //
    // where:
    //   p = Chebyshev power index
    //   q = CP rank index
    std::vector<std::vector<std::vector<float>>> tab_4b_cp_direct_factor;

    void read_4B_cp_direct_file(std::string factor_file, int quadidx);

    void compute_4B_cp_direct(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp
    );

    void compute_4B_cp_direct(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp,
        std::vector<double> & force_scalar_in
    );
#endif
    // New for tabulation -- 2b
#ifdef TABULATION
    bool                    tabulate_2B;          
    vector<string>          tab_param_files;    // tab_param_files[pair type index]
    vector<vector<double> > tab_r;              // tab_r[pair type index][rij]
    vector<vector<double> > tab_e;              // tab_e[pair type index][energy]
    vector<vector<double> > tab_f;              // tab_f[pair type index][force]
    
    void   read_2B_tab(string tab_file, bool energy=true);
    void   compute_2B_tab(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmp &tmp);   
    void   compute_2B_tab(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmp &tmp, double & force_scalar_in);     
    double get_tab_2B(int pair_idx, double rij, bool for_energy);
    
    
    // New for tabulation -- 3b
    
    bool                    tabulate_3B;        
    vector<string>          tab_param_files_3B;     // tab_param_files[trip type index]
    vector<int> tab_index_3B;             // tab_rij_3B[pair type index][rij]
    vector<vector<double> > tab_rij_3B;             // tab_rij_3B[pair type index][rij]
    vector<vector<double> > tab_rik_3B;             // tab_rik_3B[pair type index][rik]
    vector<vector<double> > tab_rjk_3B;             // tab_rjk_3B[pair type index][rjk]
    vector<vector<double> > tab_e_3B;               // tab_e_3B[pair type index][energy]
    vector<vector<double> > tab_f_ij_3B;             // tab_f_ij_3B[pair type index][force ij]
    vector<vector<double> > tab_f_ik_3B;             // tab_f_ik_3B[pair type index][force ik]
    vector<vector<double> > tab_f_jk_3B;             // tab_f_jk_3B[pair type index][force jk]
    
    vector<double> interpolateTricubic(int tripidx, double rij, double rik, double rjk, const vector<double>& y, const vector<double>& y1, const vector<double>& y2, const vector<double>& y3);
    void   read_3B_tab(string tab_file, bool energy=true);
    void   compute_3B_tab(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes3BTmp &tmp); 
    void   compute_3B_tab(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes3BTmp &tmp, vector<double> & force_scalar_in);     
    double get_tab_3B(int tripidx, const std::string& pairtyp_ij, const std::string& pairtyp_ik, const std::string& pairtyp_jk,  double rij, double rik, double rjk, double (&force_scalar)[3]);
    #endif
    

#ifdef TABULATION
    // -------------------------------------------------------------------------
    // SVD 4|2 4-body tabulation
    // -------------------------------------------------------------------------

    bool tabulate_4B_svd4x2 = false;

    struct SVD4x2SideTable
    {
        int ndim = 0;       // 4 for left, 2 for right
        int rank = 0;
        int ngrid = 0;

        int stride[4] = {0, 0, 0, 0};

        double r0[4]    = {0.0, 0.0, 0.0, 0.0};
        double dr[4]    = {0.0, 0.0, 0.0, 0.0};
        double invdr[4] = {0.0, 0.0, 0.0, 0.0};

        // block[0] = values
        // block[1] = derivative wrt local dim 0
        // ...
        // left table has 5 blocks, right table has 3 blocks.
        std::vector<std::vector<float>> block;
    };

    std::vector<SVD4x2SideTable> tab_4b_svd4x2_left;
    std::vector<SVD4x2SideTable> tab_4b_svd4x2_right;

    std::vector<int> tab_4b_svd4x2_rank;
    std::vector<std::vector<int>> tab_4b_svd4x2_canon_to_param;
    std::vector<std::vector<std::string>> tab_4b_svd4x2_canonical_pair_types;

    void read_4B_svd4x2_meta(std::string meta_file, int quadidx);
    void read_4B_svd4x2_tab(std::string data_file, int quadidx, bool left_side);

    void interpolateSVD4x2SideLinear(
        const SVD4x2SideTable &tab,
        const double *rquery,
        double **out_blocks
    );

    void compute_4B_svd4x2_tab(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp
    );

    void compute_4B_svd4x2_tab(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp,
        std::vector<double> & force_scalar_in
    );
#endif

#ifdef TABULATION
    bool tabulate_4B_coeff = false;

    // Canonical ordering metadata
    std::vector<std::vector<std::string>> tab_4b_canonical_pair_types; // [quadidx][6]
    std::vector<std::vector<int>> tab_4b_canon_to_param;               // [quadidx][6]

    void build_runtime_canonical_maps_4B(
        int quadidx,
        const std::vector<int> & mapped_pair_idx,
        const std::vector<double> & dx,
        int *canon_to_runtime,
        int *runtime_to_canon
    );

    // Contracted grid coordinates:
    // [quadidx][contracted_dim_local][row]
    std::vector<std::vector<std::vector<double>>> tab_r_4B;

    // Flattened coefficient blocks:
    // [quadidx][block][row*ncoeff + coeff]
    std::vector<std::vector<std::vector<double>>> tab_coeffs_4B_blocks_flat;

    // Metadata
    std::vector<std::vector<int>> tab_4b_contracted_dims;
    std::vector<std::vector<int>> tab_4b_retained_dims;
    std::vector<std::vector<std::vector<int>>> tab_4b_coeff_powers;

    std::vector<int> tab_4b_ncoeff;
    std::vector<int> tab_4b_ngrid;
    std::vector<int> tab_4b_ncontracted;
    std::vector<int> tab_4b_nretained;

    // Precomputed interpolation metadata
    std::vector<std::vector<int>>    tab_4b_stride;
    std::vector<std::vector<double>> tab_4b_r0;
    std::vector<std::vector<double>> tab_4b_dr;
    std::vector<std::vector<double>> tab_4b_invdr;
#endif

#ifdef TABULATION
    void read_4B_coeff_meta(std::string meta_file, int quadidx);
    void read_4B_coeff_tab (std::string data_file, int quadidx);


    void interpolateCoeff4B_2D(
        int quadidx,
        const double *rquery,
        double *coeffE,
        double *coeffD0,
        double *coeffD1
    );

    void interpolateCoeff4B_3D(
        int quadidx,
        const double *rquery,
        double *coeffE,
        double *coeffD0,
        double *coeffD1,
        double *coeffD2
    );

    void interpolateCoeff4B_4D(
        int quadidx,
        const double *rquery,
        double *coeffE,
        double *coeffD0,
        double *coeffD1,
        double *coeffD2,
        double *coeffD3
    );

    void compute_4B_tab_coeff(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp
    );

    void compute_4B_tab_coeff(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp,
        std::vector<double> & force_scalar_in
    );
#endif

#ifdef TABULATION
    // -------------------------------------------------------------------------
    // SVD 3|3 4-body tabulation
    // -------------------------------------------------------------------------

    bool tabulate_4B_svd3x3 = false;

    struct SVD3x3SideTable
    {
        int rank = 0;
        int ngrid = 0;

        int stride0 = 0;
        int stride1 = 0;
        int stride2 = 1;

        double r0[3]    = {0.0, 0.0, 0.0};
        double dr[3]    = {0.0, 0.0, 0.0};
        double invdr[3] = {0.0, 0.0, 0.0};

        // Store table data as float to reduce memory bandwidth.
        // Interpolation and dot products still accumulate in double.
        std::vector<float> val;
        std::vector<float> d0;
        std::vector<float> d1;
        std::vector<float> d2;
    };

    // One left/right table per quad type
    std::vector<SVD3x3SideTable> tab_4b_svd3x3_left;
    std::vector<SVD3x3SideTable> tab_4b_svd3x3_right;

    // Metadata
    std::vector<int> tab_4b_svd3x3_rank;
    std::vector<std::vector<int>> tab_4b_svd3x3_canon_to_param;
    std::vector<std::vector<std::string>> tab_4b_svd3x3_canonical_pair_types;
    std::vector<std::vector<int>> tab_4b_svd3x3_left_dims;
    std::vector<std::vector<int>> tab_4b_svd3x3_right_dims;

    void read_4B_svd3x3_meta(std::string meta_file, int quadidx);
    void read_4B_svd3x3_tab(std::string data_file, int quadidx, bool left_side);

    void interpolateSVD3x3SideLinear(
        const SVD3x3SideTable &tab,
        const double *rquery,
        double *val,
        double *d0,
        double *d1,
        double *d2
    );

    void compute_4B_svd3x3_tab(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp
    );

    void compute_4B_svd3x3_tab(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp,
        std::vector<double> & force_scalar_in
    );
#endif
#ifdef TABULATION
    // -------------------------------------------------------------------------
    // CP 1D 4-body tabulation
    // -------------------------------------------------------------------------

    bool tabulate_4B_cp = false;

    struct CP1DTable
    {
        int rank = 0;
        int ngrid = 0;

        double r0 = 0.0;
        double dr = 0.0;
        double invdr = 0.0;

        // Row-major storage:
        //   val[row*rank + q]
        //   der[row*rank + q]
        std::vector<float> val;
        std::vector<float> der;
    };

    // One six-slot CP table per quad type:
    //   tab_4b_cp_slot[quadidx][canonical_slot]
    std::vector<std::vector<CP1DTable>> tab_4b_cp_slot;

    // Metadata
    std::vector<int> tab_4b_cp_rank;
    std::vector<std::vector<int>> tab_4b_cp_canon_to_param;
    std::vector<std::vector<std::string>> tab_4b_cp_canonical_pair_types;

    void read_4B_cp_meta(std::string meta_file, int quadidx);
    void read_4B_cp_tab(std::string data_file, int quadidx, int slot);

    void interpolateCP1DLinear(
        const CP1DTable &tab,
        double rquery,
        double *val,
        double *der
    );

    void compute_4B_cp_tab(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp
    );

    void compute_4B_cp_tab(
        const std::vector<double> & dx,
        const std::vector<double> & dr,
        const std::vector<int> & typ_idxs,
        std::vector<double> & force,
        std::vector<double> & stress,
        double & energy,
        chimes4BTmp & tmp,
        std::vector<double> & force_scalar_in
    );
#endif

private:
        
    string            xform_style;    //  Morse, direct, inverse, etc...
    fcutType          fcut_type;      // cutoff function style (tersoff/cubic)
    double            fcut_var;       // tersoff distance (if fcut_type)
    double            inner_smooth_distance ; // Used in smoothing the cutoff interaction.
    vector<double>    morse_var;      // [npairs]; morse_lambda
    vector<double>    penalty_params; // [2];  Second dimension: [0] = A_pen, [1] = d_pen
    vector<double>    energy_offsets; // [natmtyps]; Single atom ChIMES energies
    int               badness;        // Keeps track of whether any interactions for atoms owned by proc rank are below rcutin, in the penalty region, or in the r>rcutin+dp region. 0 = good, 1 = in penalty region, 2 = below rcutin 
        
    // Names (chemical symbols for constituent atoms) .. handled differently for 2-body versus >2-body interactions

    vector<string> pair_params_atm_chem_1;    //[npairs]; // first atom in pair
    vector<string> pair_params_atm_chem_2;    //[npairs]; // second atom in pair
        
    vector<vector<string> > trip_params_atm_chems;    //[ntrips][3]    // Gives chemical symbol  for each ATOM in the triplet (i.e. "Si")    
    vector<vector<string> > trip_params_pair_typs;    //[ntrips][3]    // Gives chemical symbols for each PAIR in the triplet (i.e. "SiO")    
        
    vector<vector<string> > quad_params_atm_chems;    //[quads][3]    // Gives chemical symbol  for each ATOM in the quadruplet (i.e. "Si")    
    vector<vector<string> > quad_params_pair_typs;    //[quads][3]    // Gives chemical symbols for each PAIR in the quadruplet (i.e. "SiO")            

    int n_pair_maps;    // Number of pair maps entries
    int n_trip_maps;    // Number of trip maps entries
    int n_quad_maps;    // Number of quad maps entries
    
    int pair_type_idx;
    int trip_type_idx;
    int quad_type_idx;    
        
    ////////////////////////
    // Definitions for pair, triplet, and quadruplet types
    ////////////////////////

    // 2-body maps

    vector    <string> atom_typ_pair_map; // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiO")
    vector       <int> atom_idx_pair_map; // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 5)
    vector       <int> atom_int_pair_map; // [nmaps] "fast" maps, based on atom type index
    vector    <string> atom_int_prpr_map; // [nmaps] "fast" maps, based on atom type index ... returns the "proper" pair type instead of an index
        
    // 3-body maps
        
    vector    <string>    atom_typ_trip_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiOSiOOO")
    vector       <int>    atom_idx_trip_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 3)
    vector       <int>    atom_int_trip_map;    // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of three atoms of arbitrary order
    vector<vector<int> >   pair_int_trip_map ;  // Gives the atom pair indices for an arbitrary triplet of atom types.  

    // 4-body maps
        
    vector    <string>    atom_typ_quad_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiOSiOOO")
    vector      <int>        atom_idx_quad_map; // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 3)
    vector      <int>        atom_int_quad_map; // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of four atoms of arbitrary order
    vector<vector<int> >    pair_int_quad_map ;  // Gives the atom pair indices for an arbitrary quad of atom types.
        
    ////////////////////////
    // Polynomial parameters 
    ////////////////////////
    
    // number of coefficients for the pair/triplet/quadruplet type

    vector          <int>   ncoeffs_2b;        // [npairs]

    vector<vector<int>    > chimes_2b_pows;    // [npairs][npowers] power for the coresponding parameter

    vector<vector<double> > chimes_2b_params;    // [npairs][npowers] 2-body polynomial coefficients 
    vector<vector<double> > chimes_2b_cutoff;    // [npairs][2] inner and outer cutoff for pair

    vector<int>                      ncoeffs_3b;          // [ntrips]
    vector<vector<vector<int> > >    chimes_3b_powers;    // [ntrips][nparams][constit. pair]
    vector<vector<double> >          chimes_3b_params;    // [ntrips][nparams]    
    vector<vector<vector<double> > > chimes_3b_cutoff;    // [ntrips][2][constit. pair] inner and outer cutoff for pair 1

        
    vector<int>                      ncoeffs_4b;          // [nquads]
    vector<vector<vector<int> > >    chimes_4b_powers;    // [nquads][nparams][constit. pair]
    vector<vector<double> >          chimes_4b_params;    // [nquads][nparams]    
    vector<vector<vector<double> > > chimes_4b_cutoff;    // [nquads][2][constit. pair] inner and outer cutoff for pair 1

    // Tools for compute functions
        
	inline void set_cheby_polys(vector<double> &Tn, vector<double> &Tnd, double dx, const double morse,
                                const double inner_cutoff, const double outer_cutoff, const int order) ;

	void set_polys_out_of_range(vector<double> &Tn, vector<double> &Tnd, double dx, double x,
								int poly_order, double inner_cutoff, double exprlen, double dx_dr) ;
    
    inline void get_fcut(const double dx, const double outer_cutoff, double & fcut, double & fcutderiv);
        
    inline void get_penalty(const double dx, const int & pair_idx, double & E_penalty, double & force_scalar);
        
    inline void build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs,
                                            const vector<string> & clu_params_atm_chems, vector<int >  & mapped_pair_idx);

    inline void build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs, const vector<string> & clu_params_atm_chems,
                                            int *mapped_pair_idx);  

    int get_proper_pair(string ty1, string ty2);
        
    double max_cutoff(int ntypes, vector<vector<vector<double> > > & cutoff_list);
        
    // Tools for reading the input file
        
    int split_line(string line, vector<string> & items);
        
    string get_next_line(istream& str);        
        
    // Fun stuff
        
    void print_pretty_stuff();

    inline double dr2_3B(const double *dr2, int i, int j, int k, int l) ;
    inline double dr2_4B(const double *dr2, int i, int j, int k, int l) ;
    inline void init_distance_tensor(double *dr2, const vector<double> & dr, int natoms)    ;
};



inline void chimesFF::get_fcut(const double dx, const double outer_cutoff, double & fcut, double & fcutderiv)
{

    double fcut0;
    double fcut0_deriv ;
    
    if(fcut_type == fcutType::CUBIC )
    {        
        fcut0 = (1.0 - dx/outer_cutoff);
        fcut        = pow(fcut0,3.0);
        fcutderiv  = pow(fcut0,2.0);
        fcutderiv *= -1.0 * 3.0 /outer_cutoff;

    }
    else if ( fcut_type == fcutType::TERSOFF )
    {
    
        double THRESH = outer_cutoff-fcut_var*outer_cutoff;    
    
    
        if      (dx < THRESH)        // Case 1: Our pair distance is less than the fcut kick-in distance
        {
            fcut       = 1.0;
            fcutderiv = 0.0;
        }
        else if (dx > outer_cutoff)        // Case 2: Our pair distance is greater than the cutoff
        {
            fcut       = 0.0;
            fcutderiv = 0.0;
        }                    
        else                // Case 3: We'll use our modified sin function
        {
            fcut0       = (dx-THRESH) / (outer_cutoff-THRESH) * pi + pi/2.0;
            fcut0_deriv = pi / (outer_cutoff - THRESH);
            
            fcut        = 0.5 + 0.5 * sin( fcut0 );
            fcutderiv  = 0.5 * cos( fcut0 ) * fcut0_deriv; 
        }
    }
}

inline void chimesFF::get_penalty(const double dx, const int & pair_idx, double & E_penalty, double & force_scalar)
{
    double r_penalty = 0.0;
    
    E_penalty    = 0.0;
    force_scalar = 1.0;

    if (dx - penalty_params[0] < chimes_2b_cutoff[pair_idx][0]) // Then we're within the penalty-enforced region of distance space
    {    
        r_penalty = chimes_2b_cutoff[pair_idx][0] + penalty_params[0] - dx;
        
        if(dx < chimes_2b_cutoff[pair_idx][0])
            badness = 2;
        else if (1 > badness) // Only update badness if candiate badness is worse than its current value
            badness = 1;
    }    
    if ( r_penalty > 0.0 ) 
    {        
        E_penalty    = r_penalty * r_penalty * r_penalty * penalty_params[1];

        force_scalar = -3.0 * r_penalty * r_penalty * penalty_params[1];

        //if (rank == 0) // Commenting out - we need all ranks to report if the penalty function has been sampled
        //{
            cout << "chimesFF: " << "Adding penalty in 2B Cheby calc, r < rmin+penalty_dist " << fixed 
                 << dx << " " 
                 << chimes_2b_cutoff[pair_idx][0] + penalty_params[0]  
                 << " pair type: " << pair_idx << endl;
            cout << "chimesFF: " << "\t...Penalty potential = "<< E_penalty << endl;
        //}
    }   
}

inline int chimesFF::get_badness()
{
    return badness;
}

inline void chimesFF::reset_badness()
{
    badness = 0;
}

inline void chimesFF::build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs,
                                                  const vector<string> & clu_params_pair_typs, vector<int>  & mapped_pair_idx)
// Interface to array-based version.
{
    build_atom_and_pair_mappers(natoms, npairs, typ_idxs, clu_params_pair_typs, mapped_pair_idx.data() ) ;
}

inline void chimesFF::build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs,
                                                  const vector<string> & clu_params_pair_typs, 
                                                  int *mapped_pair_idx)
{
    // Generate permutations for atoms... all we are doing is permuting the possible indices for typ_idxs


    // build a copy of the atom type vector for permuting

    vector<int> tmp_typ_idxs;
    int         nelements;
    
    nelements = typ_idxs.size();
    tmp_typ_idxs.resize(nelements);
    
    for(int i=0; i<nelements; i++)
        tmp_typ_idxs[i] = i;
        
    // Build a copy of the original pairs for comparison against permuted pairs
    
    
    vector<vector<int> > tmp_pairs;
    tmp_pairs.resize(npairs,vector<int>(2));
    vector<vector<int> > runtime_pairs;
    runtime_pairs.resize(npairs,vector<int>(2));
    
    int idx = 0;
    
    for(int i=0; i<natoms; i++) 
    {

        for (int j=i+1; j<natoms; j++)
        {
            tmp_pairs[idx][0] = i;
            tmp_pairs[idx][1] = j;
            
            idx++;
        }
    }
        
    vector<string> runtime_pair_typs(npairs);    
        
    do
    {
        // Check if the permutation leads to pair types that match the order specified by the force field type
    
        idx = 0;
        
        for(int i=0; i<natoms; i++) // Associate the current atom pairs with a "proper" 2-body force field name
        {
            for (int j=i+1; j<natoms; j++)
            {
                runtime_pair_typs[idx] = atom_int_prpr_map[ typ_idxs[tmp_typ_idxs[i]]*natmtyps + typ_idxs[tmp_typ_idxs[j]] ];
                        
                idx++;
            }
        }
        
        bool match = true;
        
        for(int i=0; i<npairs; i++)
        {
            if (clu_params_pair_typs[i] != runtime_pair_typs[i])
            {
                match = false;
                break;
            }
        }
        
        if (match) // Then we've found an appropriate atom ordering... now what?
        {
            idx = 0;
        
            for(int i=0; i<natoms; i++) // Associate the current atom pairs with a "proper" 2-body force field name
            {
                for (int j=i+1; j<natoms; j++)
                {
                    runtime_pairs[idx][0] = tmp_typ_idxs[i];
                    runtime_pairs[idx][1] = tmp_typ_idxs[j];
                    
                    idx++;
                }
            }
        
            break;
        }
        

    } while ( next_permutation(tmp_typ_idxs.begin(),tmp_typ_idxs.begin()+nelements)) ;
    
    // Once we've found a re-ordering of atoms that properly maps to the force field pair types, need to figure out how to convert that to a map between *pairs*
    
    idx = 0;
    
    for(int i=0; i<npairs; i++)
        for(int j=0; j<npairs; j++)
            if (    ((runtime_pairs[i][0] == tmp_pairs[j][0]) && (runtime_pairs[i][1] == tmp_pairs[j][1]))
                 || ((runtime_pairs[i][0] == tmp_pairs[j][1]) && (runtime_pairs[i][1] == tmp_pairs[j][0]))
                )
                mapped_pair_idx[j] = i;

}


inline void chimesFF::set_cheby_polys(vector<double> &Tn, vector<double> &Tnd, double dx, const double morse,
									  const double inner_cutoff, const double outer_cutoff, const int order) 
{
    // Currently assumes a Morse-style transformation has been requested
    
    // Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd).  Tnd is the derivative
    // with respect to the interatomic distance, not the transformed distance (x).
    
    // Do the Morse transformation


     double x_min = exp(-1*inner_cutoff/morse);
     double x_max = exp(-1*outer_cutoff/morse);
    
     double x_avg   = 0.5 * (x_max + x_min);
     double x_diff  = 0.5 * (x_max - x_min);
	
    x_diff *= -1.0; // Special for Morse style


     bool out_of_range ;
     double dx_orig = dx ;

	//  The case dx > outer_cutoff is not treated, because it is assumed that the outer smoothing
    //  function will be zero for dx > outer_cutoff.
    if ( dx < inner_cutoff )
    {
        out_of_range = true ;
        dx = inner_cutoff ;
    }
    else
        out_of_range = false ;
    
	 double exprlen = exp(-1*dx/morse);
	 double x  = (exprlen - x_avg)/x_diff;
	 double dx_dr = (-exprlen/morse)/x_diff;		

    if ( ! out_of_range )
    {
        // Generate Chebyshev polynomials by recursion. 
        // 
        // What we're doing here. Want to fit using Cheby polynomials of the 1st kinD[i]. "T_n(x)."
        // We need to calculate the derivative of these polynomials.
        // Derivatives are defined through use of Cheby polynomials of the 2nd kind "U_n(x)", as:
        //
        // d/dx[ T_n(x) = n * U_n-1(x)] 
        // 
        // So we need to first set up the 1st-kind polynomials ("Tn[]")
        // Then, to compute the derivatives ("Tnd[]"), first set equal to the 2nd-kind, then multiply by n to get the der's
     
        // First two 1st-kind Chebys:
        
        Tn[0] = 1.0;
        Tn[1] = x;
    
        // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

        Tnd[0] = 1.0;
        Tnd[1] = 2.0 * x;
    
        // Use recursion to set up the higher n-value Tn and Tnd's

        for ( int i = 2; i <= order; i++ ) 
        {
            Tn[i]  = 2.0 * x *  Tn[i-1] -  Tn[i-2];
            Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
        }
    
        // Now multiply by n to convert Tnd's to actual derivatives of Tn
    
        // The following dx_dr compuation assumes a Morse transformation
        // DERIV_CONST is no longer used. (old way: dx_dr = DERIV_CONST*cheby_var_deriv(x_diff, rlen, ff_2body.LAMBDA, ff_2body.CHEBY_TYPE, exprlen);)

        for ( int i = order; i >= 1; i-- ) 
            Tnd[i] = i * dx_dr * Tnd[i-1];

        Tnd[0] = 0.0;
    }
    else // out_of_range == true
    {
		cout << "Warning: An intermolecular distance less than the inner cutoff = " << inner_cutoff << " was found\n " ;
		cout << "         Distance = " << dx_orig << endl ;

		set_polys_out_of_range(Tn, Tnd, dx_orig, x, order, inner_cutoff, exprlen, dx_dr) ;
    }        

}

#endif
























