#ifndef GROMACS_KERNELS
#define GROMACS_KERNELS

class Add1 {
  public:
    typedef char InpElTp;
    typedef int32_t RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline InpElTp identInp()               { return (char)0;     }
    static __device__ __host__ inline RedElTp mapFun(const InpElTp& el){ return (RedElTp)el; }
    static __device__ __host__ inline RedElTp identity()               { return (RedElTp)0;  }
    static __device__ __host__ inline RedElTp apply(const RedElTp t1, const RedElTp t2) { return t1 + t2; }
    static __device__ __host__ inline bool   equals(const RedElTp t1, const RedElTp t2) { return (t1 == t2); }
    static __device__ __host__ inline RedElTp remVolatile(volatile RedElTp& t) { RedElTp res = t; return res; }
};

class Add2 {
  public:
    typedef char InpElTp;
    typedef int32_t RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline InpElTp identInp()               { return (char)0;     }
    static __device__ __host__ inline RedElTp mapFun(const InpElTp& el){ return (RedElTp)(1-el); }
    static __device__ __host__ inline RedElTp identity()               { return (RedElTp)0;  }
    static __device__ __host__ inline RedElTp apply(const RedElTp t1, const RedElTp t2) { return t1 + t2; }
    static __device__ __host__ inline bool   equals(const RedElTp t1, const RedElTp t2) { return (t1 == t2); }
    static __device__ __host__ inline RedElTp remVolatile(volatile RedElTp& t) { RedElTp res = t; return res; }
};

#if 1
/**************************************************/
/*** AtomicAdd demonstrated on real addition    ***/
/**************************************************/
__device__ inline static void
atomADD(volatile real* loc_hists, int idx, real v) {
    atomicAdd((real*)&loc_hists[idx], v);
}

#else
/**************************************************/
/*** CAS implementation of atomic add on floats ***/
/**************************************************/
__device__ inline static void 
atomADD(volatile real *address, uint32_t idx, real val) {
    int tmp0 = *address; 
    int i_val = __float_as_int(val + __int_as_float(tmp0)); 
    int tmp1;

    while( (tmp1 = atomicCAS((int *)(address+idx), tmp0, i_val)) != tmp0)
    { 
        tmp0 = tmp1; 
        i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}
#endif

#if 0
__device__ inline static void
atomCAS(volatile real* loc_hists, uint32_t idx, real v) {
    real old = loc_hists[idx];
    real assumed, upd;
    do {
        assumed = old;
        upd = assumed + v;
        old = atomicCAS( (real*)&loc_hists[idx], assumed, upd );
    } while(assumed != old);
}
#endif

/***********************/
/*** Various Kernels ***/
/***********************/

__global__ void mkFlagKernel(char* flag, int* inds, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        // first element is left zero
        if(gid!=0) flag[inds[gid]] = 1;
    }
}

__global__ void setFirstFlagElmKernel(char* flag) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid==0) flag[0] = 1;
}

__global__ void 
outerLoopKernel( int32_t nri, real facel, int32_t ntype
               , int32_t* shift, real* shiftvec, int32_t* iinr, int32_t* types
               , real* pos, real* charge
               , real* ix1s, real* iy1s, real* iz1s, real* iqAs, int32_t* ntiAs
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
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


__global__ void 
innerLoopKernel( int32_t len_flat, int32_t* jindex, int32_t* out_inds, int32_t* inn_inds
               , int32_t* iinr, int32_t* jjnr, int32_t* types, real* pos, real* charge
               , real* nbfp, real* ix1s, real* iy1s, real* iz1s, real* iqAs, int32_t* ntiAs
               , volatile real* faction
               
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < len_flat) {
        int32_t oind = out_inds[gid];
        int32_t iind = inn_inds[gid];
        int32_t k    = jindex[oind] + iind;
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
        real  rinv11 = one / sqrtf(rsq11);
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
