typedef struct OCLControl {
    cl_context          ctx;            // OpenCL context
    cl_device_id        device;      // OpenCL device list
    cl_program          prog;      // OpenCL program
    cl_command_queue    queue; // command queue of the targe GPU device
} OclControl;

typedef struct OCLKernels {
    cl_kernel outer_loop_ker;
    cl_kernel inner_loop_ker;
    cl_kernel memcpy_ker;
} OclKernels;

typedef struct gromacsBuffers {
    // constants
    int32_t nri;
    int32_t nrj;
    int32_t num_particles;
    int32_t ntype;
    int32_t shiftvec_len;
    int32_t nbfp_len;
    real    facel;

    int32_t *jindex;
    cl_mem jindex_d;
    int32_t *iinr, *jjnr; 
    cl_mem  iinr_d, jjnr_d;
    int32_t *shift, *types;
    cl_mem shift_d, types_d;

    real *shiftvec;
    cl_mem shiftvec_d;
    real *pos;
    cl_mem pos_d;
    real *faction0, *faction, *faction_dh;
    cl_mem faction0_d, faction_d;
    real *charge, *nbfp;
    cl_mem charge_d, nbfp_d; 

    // intermediate arrays
    cl_mem ix1s_d, iy1s_d, iz1s_d, iqAs_d, ntiAs_d; 
} Buffers;

//CpuArrays  arrs;
OclControl ctrl;
OclKernels kers;
Buffers buffs;

void initOclControl() {
    char    compile_opts[128];
    sprintf(compile_opts, "-D lgWARP=%d", lgWARP);
    
    opencl_init_command_queue_default(&ctrl.device, &ctrl.ctx, &ctrl.queue);
    ctrl.prog = opencl_build_program(ctrl.ctx, ctrl.device, "gromacs-kernels.cl", compile_opts);
}

void initOclBuffers() {
    cl_int error = CL_SUCCESS;
    size_t size;

    size = (buffs.nri+1)*sizeof(int32_t);
    buffs.jindex_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.jindex, &error);
    OPENCL_SUCCEED(error);

    size = buffs.nri*sizeof(int32_t);
    buffs.iinr_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.iinr, &error);
    OPENCL_SUCCEED(error);

    size = buffs.nrj*sizeof(int32_t);
    buffs.jjnr_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.jjnr, &error);
    OPENCL_SUCCEED(error);

    size = buffs.nri*sizeof(int32_t);
    buffs.shift_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.shift, &error);
    OPENCL_SUCCEED(error);

    size = buffs.num_particles*sizeof(int32_t);
    buffs.types_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.types, &error);
    OPENCL_SUCCEED(error);

    size = buffs.shiftvec_len*sizeof(real);
    buffs.shiftvec_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.shiftvec, &error);
    OPENCL_SUCCEED(error);

    size = 3*buffs.num_particles*sizeof(real);
    buffs.pos_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.pos, &error);
    OPENCL_SUCCEED(error);

    size = 3*buffs.num_particles*sizeof(real);
    buffs.charge_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.charge, &error);
    OPENCL_SUCCEED(error);

    size = 3*buffs.num_particles*sizeof(real);
    buffs.faction0_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, buffs.faction0, &error);
    OPENCL_SUCCEED(error);

    size = 3*buffs.num_particles*sizeof(real);
    buffs.faction_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, buffs.faction0, &error);
    OPENCL_SUCCEED(error);
    
    size = buffs.nbfp_len*sizeof(real);
    buffs.nbfp_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buffs.nbfp, &error);
    OPENCL_SUCCEED(error);

    size = buffs.nri*sizeof(int32_t);
    buffs.ntiAs_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error );
    OPENCL_SUCCEED(error);

    size = buffs.nri*sizeof(real);
    buffs.iqAs_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error );
    buffs.ix1s_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error );
    buffs.iy1s_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error );
    buffs.iz1s_d = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error );
    OPENCL_SUCCEED(error);
}

void initKernels() {
    cl_int ciErr;
    kers.memcpy_ker = clCreateKernel(ctrl.prog, "memcpyKernel", &ciErr);
    OPENCL_SUCCEED(ciErr);
    kers.outer_loop_ker = clCreateKernel(ctrl.prog, "outerLoopKernel", &ciErr);
    OPENCL_SUCCEED(ciErr);
    kers.inner_loop_ker = clCreateKernel(ctrl.prog, "innerLoopKernel", &ciErr);
    OPENCL_SUCCEED(ciErr);
}

void initKernelParams() {
    cl_int ciErr;
    unsigned int counter = 0;

    { // memcpy kernel
        ciErr |= clSetKernelArg(kers.memcpy_ker, counter++, sizeof(uint32_t), (void *)&buffs.num_particles);
        ciErr |= clSetKernelArg(kers.memcpy_ker, counter++, sizeof(cl_mem),   (void *)&buffs.faction0_d);
        ciErr |= clSetKernelArg(kers.memcpy_ker, counter++, sizeof(cl_mem),   (void *)&buffs.faction_d);
    }

    counter = 0;
    { // outer loop kernel
#if 0
outerLoopKernel( int32_t nri, real facel, int32_t ntype
               , int32_t* shift, real* shiftvec, int32_t* iinr, int32_t* types
               , real* pos, real* charge
               , real* ix1s, real* iy1s, real* iz1s, real* iqAs, int32_t* ntiAs
#endif
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(uint32_t), (void *)&buffs.nri);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(real),     (void *)&buffs.facel);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(uint32_t), (void *)&buffs.ntype);

        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.shift_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.shiftvec_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.iinr_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.types_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.pos_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.charge_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.ix1s_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.iy1s_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.iz1s_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.iqAs_d);
        ciErr |= clSetKernelArg(kers.outer_loop_ker, counter++, sizeof(cl_mem),   (void *)&buffs.ntiAs_d);  
        OPENCL_SUCCEED(ciErr);
    }

    counter = 0;
    { // inner loop

#if 0
innerLoopKernel( int32_t len_flat, int32_t nri, int32_t* jindex
                //, int32_t* out_inds, int32_t* inn_inds
               , int32_t* iinr, int32_t* jjnr, int32_t* types, real* pos, real* charge
               , real* nbfp, real* ix1s, real* iy1s, real* iz1s, real* iqAs, int32_t* ntiAs
               , volatile real* faction
#endif
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(int32_t), (void *)(&buffs.jindex[buffs.nri]));
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(int32_t), (void *)(&buffs.nri));
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.jindex_d);
        // out_inds, inn_inds we skip
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.iinr_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.jjnr_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.types_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.pos_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.charge_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.nbfp_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.ix1s_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.iy1s_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.iz1s_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.iqAs_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.ntiAs_d);
        ciErr |= clSetKernelArg(kers.inner_loop_ker, counter++, sizeof(cl_mem),  (void *)&buffs.faction_d);
        OPENCL_SUCCEED(ciErr);
    }
}

void gpuToCpuTransfer() {
    cl_int  ciErr;
    fprintf(stderr, "GPU-to-CPU Transfer %d ...\n", 3*buffs.num_particles);
    size_t size = 3*buffs.num_particles*sizeof(real);
    ciErr = clEnqueueReadBuffer (
                        ctrl.queue, buffs.faction_d, CL_TRUE,
                        0, size, buffs.faction_dh, 0, NULL, NULL
                );
    OPENCL_SUCCEED(ciErr);
}

void gpuToGpuTransfer() {
    cl_int  ciErr;
    size_t size = 3*buffs.num_particles*sizeof(real);
    ciErr = clEnqueueCopyBuffer( ctrl.queue, buffs.faction0_d, buffs.faction_d, 0, 0, size, 0, NULL, NULL);
    OPENCL_SUCCEED(ciErr);
}

void freeBuffers() {
    fprintf(stderr, "Releasing GPU buffers ...\n");
    clReleaseMemObject(buffs.jindex_d);
    clReleaseMemObject(buffs.iinr_d);
    clReleaseMemObject(buffs.jjnr_d);
    clReleaseMemObject(buffs.shift_d);
    clReleaseMemObject(buffs.types_d);

    clReleaseMemObject(buffs.shiftvec_d);
    clReleaseMemObject(buffs.pos_d);
    clReleaseMemObject(buffs.charge_d);
    clReleaseMemObject(buffs.nbfp_d);
    clReleaseMemObject(buffs.faction0_d);
    clReleaseMemObject(buffs.faction_d);
    clReleaseMemObject(buffs.ix1s_d);
    clReleaseMemObject(buffs.iy1s_d);
    clReleaseMemObject(buffs.iz1s_d);
    clReleaseMemObject(buffs.iqAs_d);
    clReleaseMemObject(buffs.ntiAs_d);

    free(buffs.jindex); free(buffs.iinr);  free(buffs.jjnr);
    free(buffs.shift);  free(buffs.types); free(buffs.shiftvec);
    free(buffs.pos); free(buffs.charge); free(buffs.nbfp);
    free(buffs.faction0); free(buffs.faction); free(buffs.faction_dh);
}

void freeOclControl() {
    fprintf(stderr, "Releasing GPU program ...\n");
    clReleaseProgram(ctrl.prog);

    fprintf(stderr, "Releasing Command Queue ...\n");
    clReleaseCommandQueue(ctrl.queue);
        
    fprintf(stderr, "Releasing GPU context ...\n");
    clReleaseContext(ctrl.ctx);

    fprintf(stderr, "Releasing Kernels...\n");
    clReleaseKernel(kers.memcpy_ker);
    clReleaseKernel(kers.outer_loop_ker);
    clReleaseKernel(kers.inner_loop_ker);
}
