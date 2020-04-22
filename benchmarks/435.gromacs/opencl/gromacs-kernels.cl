#ifndef GROMACS_KERNELS
#define GROMACS_KERNELS

//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
typedef float   real;
typedef long    int64_t;
typedef int     int32_t;
#define nul     0.0
#define one     1.0
#define six     6.0
#define twelve  12.0

__kernel void
memcpyKernel(int32_t num_particles, __global real* src, __global real* dst) {
    int32_t gid = get_global_id(0);
    if(gid < 3*num_particles) {
        dst[gid] = src[gid];
    }
}

__kernel void 
outerLoopKernel( int32_t nri, real facel, int32_t ntype
               , __global int32_t* shift
               , __global real* shiftvec
               , __global int32_t* iinr
               , __global int32_t* types
               , __global real* pos
               , __global real* charge
               , __global real* ix1s
               , __global real* iy1s
               , __global real* iz1s
               , __global real* iqAs
               , __global int32_t* ntiAs
) {
    int32_t gid = get_global_id(0);
    if(gid < nri) {
        int32_t is3 = 3*shift[gid];
        real    shX = shiftvec[is3];
        real    shY = shiftvec[is3+1];
        real    shZ = shiftvec[is3+2];
        int32_t ii  = iinr[gid];
        int32_t ii3 = 3*ii;       
        ix1s[gid]   = shX + pos[ii3]; 
        iy1s[gid]   = shY + pos[ii3+1];
        iz1s[gid]   = shZ + pos[ii3+2];
        iqAs[gid]   = facel*charge[ii];
        ntiAs[gid]  = 2*ntype*types[ii];
    }
}


inline int32_t
binSearch(int32_t gid, int32_t N, volatile __global int32_t* A) {
    int32_t L = 0;
    int32_t R = N - 1;

    while (L <= R) {
        int32_t m = (L + R) / 2;
        if(A[m] <= gid) {
            if(m < N-1 && A[m+1] > gid) {
                return m;
            }
            L = m + 1;
        } else if (A[m] > gid) {
            if(m > 0 && A[m-1] <= gid) {
                return m-1;
            }
            R = m - 1;
        }
    }
}

#if 0
inline double atomic_fadd_f64_global(volatile __global double *p, double x) {
  union { int64_t i; double f; } old;
  union { int64_t i; double f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __global int64_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
}
#endif

inline float atomic_fadd_f32_global(volatile __global float *p, float x) {
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i); 
  return old.f;
}


inline void 
atomADD(volatile __global real *address, int32_t idx, real val) {
    atomic_fadd_f32_global(address+idx, val);
}

__kernel void 
innerLoopKernel ( int32_t len_flat
                , int32_t nri
                , __global int32_t* jindex 
                //, __global int32_t* out_inds, __global int32_t* inn_inds
                , __global int32_t* iinr
                , __global int32_t* jjnr
                , __global int32_t* types
                , __global real* pos
                , __global real* charge
                , __global real* nbfp
                , __global real* ix1s
                , __global real* iy1s
                , __global real* iz1s
                , __global real* iqAs
                , __global int32_t* ntiAs
                , volatile __global real* faction
               
) {
    int32_t gid = get_global_id(0);
    if(gid < len_flat) {
        int32_t oind = binSearch(gid, nri+1, jindex);
        int32_t k = gid;
        //int32_t iind = gid - jindex[oind];
        //int32_t oind = out_inds[gid];
        //int32_t iind = inn_inds[gid];
        //int32_t k    = jindex[oind] + iind;
        real    ix1  = ix1s[oind], 
                iy1  = iy1s[oind],
                iz1  = iz1s[oind],
                iqA  = iqAs[oind];
        int32_t ntiA = ntiAs[oind];

        int32_t jnr  = jjnr[k];
        int32_t j3   = 3 * jnr;
        real    jx1  = pos[j3],
                jy1  = pos[j3+1],
                jz1  = pos[j3+2];
        real    dx11 = ix1 - jx1,
                dy11 = iy1 - jy1,
                dz11 = iz1 - jz1;
        real   rsq11 = dx11*dx11+dy11*dy11+dz11*dz11;
        real  rinv11 = one / sqrt(rsq11);
        real rinvsq11= rinv11*rinv11;
        real rinvsix = rinvsq11*rinvsq11*rinvsq11;
        int32_t tjA  = ntiA + 2*types[jnr];
        real    vnb6 = rinvsix * nbfp[tjA];
        real    vnb12= rinvsix * rinvsix * nbfp[tjA+1];
        real    qq   = iqA * charge[jnr];
        real    vcoul= qq * rinv11;
        real    fs11 = (twelve*vnb12-six*vnb6+vcoul)*rinvsq11;
        real    tx11 = dx11*fs11;
        real    ty11 = dy11*fs11;
        real    tz11 = dz11*fs11;

        atomADD(faction, j3,   nul - tx11);
        atomADD(faction, j3+1, nul - ty11);
        atomADD(faction, j3+2, nul - tz11);

        int32_t i3 = 3*iinr[oind];
        atomADD(faction, i3,   tx11);
        atomADD(faction, i3+1, ty11);
        atomADD(faction, i3+2, tz11);     
    }
} 
#endif // GROMACS_KERNELS
