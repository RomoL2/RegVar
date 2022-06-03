# cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, cdivision=True
### cython: boundscheck=True, wraparound=True, initializedcheck=True, overflowcheck=True, cdivision=False
#!python

__license__ = "MIT"
__version__ = "0.9.8"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

"""
Thermodynamic model code
"""

from types cimport *
from types import *

from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np
cimport cython
cimport openmp

from libc.math cimport exp, log, pow
from libc.stdlib cimport abort, malloc, free #, posix_memalign
from libc.string cimport memset #faster than np.zeros

cdef extern from "cmpxchg.h":
    # cdef extern int sync_bool_compare_and_swap_vp (void **ptr, void *oldval, void *newval) nogil
    cdef extern int cmpxchg_uint32 (UINT32_t *ptr, UINT32_t oldval, UINT32_t newval) nogil
    cdef extern int cmpxchg_uint64 (UINT64_t *ptr, UINT64_t oldval, UINT64_t newval) nogil
    # cdef extern int _cmpxchg_float32 (UINT32_t *ptr, UINT32_t oldval, UINT32_t newval) nogil

cdef inline int cmpxchg_float32 (FLOAT32_t *ptr, FLOAT32_t oldval, FLOAT32_t newval) nogil:
    # pointer cast shenanigans to trick __bool_sync_compare_and_exchange 
    # into swapping a float by treating the 4 bytes as uint32
    cdef UINT32_t ov = (<UINT32_t *>(<void*> &oldval))[0] 
    cdef UINT32_t nv = (<UINT32_t *>(<void*> &newval))[0]
    return cmpxchg_uint32(<UINT32_t*>ptr, ov, nv)


def SPA_partition_function(UINT32_t [:,:] index_matrix, UINT8_t [:,:] openen_matrix, FLOAT32_t [:] acc_lookup, FLOAT32_t [:] kmer_invkd, UINT64_t k, int n_max=0, int openen_ofs=0):
    #assert k <= 8 # must fit into UINT16 kmer-indices!

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]
    
    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z = np.empty(N, dtype=np.float32)
    
    # helper variables to tell cython the types
    cdef UINT8_t o=0
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index=0
    cdef FLOAT64_t w=0
    cdef FLOAT64_t Z1=0 # Single protein partition function

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='static'):
            Z1 = 0
            # iterate over all k-mers, always adding next base to index
            for i in range(0, l):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                o = openen_matrix[j, i + openen_ofs]
                w = kmer_invkd[index] * acc_lookup[o]
                Z1 = Z1 + w
            
            Z[j] = Z1

    return Z.base



def SPA_partition_function_raw(UINT32_t [:,:] index_matrix, FLOAT32_t [:,:] acc_matrix, FLOAT32_t [:] kmer_invkd, UINT64_t k, int n_max=0, int openen_ofs=0):
    assert k <= 16 # must fit into UINT32 kmer-indices!

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]
    
    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z = np.empty(N, dtype=np.float32)
    
    # helper variables to tell cython the types
    cdef FLOAT32_t a=0
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index=0
    cdef FLOAT32_t w=0
    cdef FLOAT64_t Z1=0 # Single protein partition function

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='static'):
            Z1 = 0
            # iterate over all k-mers
            for i in range(0, l):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                a = acc_matrix[j, i + openen_ofs]
                w = kmer_invkd[index] * a
                Z1 = Z1 + w
            
            Z[j] = Z1

    return Z.base

def seqm_pad_adapters(UINT8_t [:,:] seqm, UINT8_t [:] adap5, UINT8_t [:] adap3, UINT64_t k):
    cdef UINT64_t N = seqm.shape[0]
    cdef UINT64_t l = seqm.shape[1]
    cdef UINT64_t L = l + 2* (k-1)
    cdef UINT8_t [:,:] padded = np.empty((N,L), dtype=np.uint8)
    cdef UINT64_t l5 = len(adap5.base)
    cdef UINT64_t l3 = len(adap3.base)
    cdef UINT64_t i = 0, j=0

    with nogil:
        for j in prange(N):
            for i in range(k-1):
                padded[j,i] = adap5[l5 - k + i + 1]
            for i in range(l):
                padded[j,i+k-1] = seqm[j,i]
            for i in range(k-1):
                padded[j,i+k-1+l] = adap3[i]
    
    return padded.base


def clipped_sum_and_max(FLOAT32_t [:,:] Z, FLOAT32_t clip=100000.):
    cdef UINT64_t N = Z.base.shape[0]
    cdef UINT64_t l = Z.base.shape[1]
    cdef FLOAT32_t Z_max=0, Z_max_local=0
    cdef UINT64_t i=0, j=0, d=0, n=0
    # cdef UINT32_t index=0
    # cdef FLOAT32_t w=0
    # cdef FLOAT64_t Z1=0 # Single protein partition function
    cdef FLOAT32_t Z_sum=0
    cdef FLOAT32_t *ptr = &Z_max
    cdef FLOAT32_t [:] Z_read = np.zeros(N, dtype=np.float32)
    cdef int num_threads=8
    cdef int thread_num = -1
    cdef FLOAT32_t [:] Z_max_thread = np.zeros(num_threads, dtype=np.float32)

    with nogil, parallel():
        for j in prange(N, schedule='static'):
            # make these thread-local
            thread_num = openmp.omp_get_thread_num()

            Z_sum = 0
            # sum
            for i in range(l):
                Z_sum = Z_sum + Z[j,i]
            # clip
            Z_sum = min(Z_sum, clip)

            # keep sum
            Z_read[j] = Z_sum

            # keep max-value in thread-safe way
            Z_max_local = Z_max_thread[thread_num]
            if Z_sum > Z_max_local:
                Z_max_thread[thread_num] = Z_sum

    # find the max over all threads
    for thread_num in range(num_threads):
        if Z_max_thread[thread_num] > Z_max:
            Z_max = Z_max_thread[thread_num]

    return Z_read.base, Z_max


def pow_scale(FLOAT32_t [:,:] Z, FLOAT32_t a):
    cdef UINT64_t N = Z.base.shape[0]
    cdef UINT64_t l = Z.base.shape[1]
    cdef UINT64_t i=0, j=0, d=0, n=0
    # cdef FLOAT32_t [:,:] Z_scaled = np.empty(Z.base.shape, dtype=np.float32)
    cdef int num_threads=8
    cdef int thread_num = -1

    with nogil, parallel():
        for j in prange(N, schedule='static'):
            # make these thread-local
            # thread_num = openmp.omp_get_thread_num()

            for i in range(l):
                # Z[j, i] = pow(Z[j, i], a)
                Z[j, i] = exp(a * log(Z[j, i]))


def PSAM_partition_function(UINT8_t [:, :] seqm, FLOAT32_t [:, :] acc_matrix, FLOAT32_t [:, :] psam, int n_max=0, int openen_ofs=0, FLOAT32_t non_specific=0, FLOAT32_t alpha=1, int single_thread=False, int noacc=False):
    cdef UINT64_t N = seqm.base.shape[0]
    cdef UINT64_t L = seqm.base.shape[1]
    cdef UINT64_t k = psam.base.shape[0]
    cdef UINT64_t l = L - k + 1
    cdef UINT64_t L_acc = acc_matrix.base.shape[1]
    # print "part_func L-k+1", l, k
    # assert L_acc - openen_ofs >= l 
    # assert openen_ofs >= 0
    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:, :] Z = np.ones((N,l), dtype=np.float32)
    
    # helper variables to tell cython the types
    # cdef FLOAT32_t a=0
    cdef int i=0, j=0, d=0, n=0, ind=0
    cdef int acc_i=0
    # cdef UINT32_t index=0
    # cdef FLOAT32_t w=0
    cdef FLOAT32_t z=0 # Single protein partition function

    if n_max:
        N = min(N, n_max)

    if single_thread:
        for j in range(N):
            # iterate over all PSAM start positions
            for i in range(l):
                # specific binding: product of per-site affinities
                z = 1.
                for d in range(k):
                    n = seqm[j, i + d]
                    if n > 3:
                        z = 0. # skip N's
                    else:
                        z = z * psam[d, n]

                # add non-specific component (still reacts to accessbility)
                z = z + non_specific

                if not noacc:
                    acc_i = i + openen_ofs
                    if 0 <= acc_i < L_acc:
                        z = z * acc_matrix[j, i + openen_ofs]
                    else:
                        z = 0 # no valid accessibility footprint

                Z[j, i] = z

    elif alpha < 1:
        with nogil, parallel():
            for j in prange(N, schedule='static'):
                # iterate over all PSAM start positions
                for i in range(l):
                    # specific binding: product of per-site affinities
                    z = 1.
                    for d in range(k):
                        n = seqm[j, i + d]
                        z = z * psam[d, n]
                    
                    # add non-specific component (still reacts to accessbility)
                    z = z + non_specific

                    if not noacc:
                        acc_i = i + openen_ofs
                        if 0 <= acc_i < L_acc:
                            z = z * exp(acc_matrix[j, i + openen_ofs] * alpha) 
                        else:
                            z = 0 # no valid accessibility footprint

                    Z[j, i] = z

    else:
        with nogil, parallel():
            for j in prange(N, schedule='static'):
                # iterate over all PSAM start positions
                for i in range(l):
                    # specific binding: product of per-site affinities
                    z = 1.
                    for d in range(k):
                        n = seqm[j, i + d]
                        z = z * psam[d, n]
                    # add non-specific component (still reacts to accessbility)
                    z = z + non_specific
                    if not noacc:
                        acc_i = i + openen_ofs
                        if 0 <= acc_i < L_acc:
                            z = z * acc_matrix[j, i + openen_ofs]
                        else:
                            z = 0 # no valid accessibility footprint

                    Z[j, i] = z

    return Z.base


def PSAM_partition_function_gradient(state, params, FLOAT32_t [:,:] Z1m, FLOAT32_t [:] Z1rm):

    ### Relevant data from the state object
    cdef UINT8_t [:,:] seqm = state.mdl.seqm
    cdef UINT8_t *seqm_row
    cdef FLOAT32_t [:, :] Z1 = state.Z1 # this is the total partition function, sum of all motif contributions!
    cdef FLOAT32_t *Z1_row
    cdef FLOAT32_t *Z1m_row # this is used for the motif-specific terms *scaled by rel. affinity*!
    cdef FLOAT32_t [:] Z1_read = state.Z1_read
    cdef FLOAT32_t [:] rbp_free = state.rbp_free
    # do not even look at reads with Z1_read < this value
    cdef FLOAT32_t Z1_thresh = state.threshold * state.Z1_read_max # set to 0 to look at all reads
    cdef UINT32_t [:] F0 = state.mdl.F0 # 4^k kmer frequencies in input
    cdef FLOAT32_t [:] f0 = state.mdl.f0 # 4^k kmer rel. frequencies in input
    cdef FLOAT32_t [:,:] psi = state.psi
    cdef FLOAT32_t *psi_row
    cdef UINT32_t [:,:] im = state.mdl.im
    cdef UINT32_t *im_row
    cdef FLOAT32_t [:] W = state.W # normalization factors for each sample
    cdef FLOAT32_t A0 = params.A0
    cdef FLOAT32_t [:,:] w = state.w # n_samples x 4^k
    cdef FLOAT32_t [:,:] R = state.R # kmer enrichments
    cdef FLOAT64_t [:,:] E = state.R_errors # R - R0
    # cdef FLOAT32_t [:] E_weights = state.E_weights # EXPERIMENTAL!

    ### Important dimensions needed to allocate buffers
    cdef UINT64_t N = seqm.base.shape[0]
    cdef UINT64_t L = seqm.base.shape[1]
    cdef UINT64_t n_psam = len(params.psam_vec)
    cdef UINT64_t pad = 16 - (n_psam % 16) # 16 x FLOAT32 = 1 L1 Cache line
    cdef UINT64_t n_psam_padded = n_psam + pad
    cdef UINT64_t n_samples = params.n_samples
    cdef UINT64_t n_params = n_psam + n_samples # affinity + beta values
    cdef UINT64_t k = (n_psam - 1) / 4
    cdef UINT64_t l = L - k + 1 # no. of positions for the PSAM
    cdef UINT64_t lam = im.shape[1] # no. of kmers in each read
    cdef UINT64_t Nk = state.mdl.nA # 4^k
    cdef FLOAT32_t Nk_inv = 1./Nk
    cdef int n_threads = 8

    ### Local variables and flags
    cdef int tid = 0 # thread number
    cdef int lin_occ = state.mdl.linear_occ # pretend we're far from saturation
    cdef UINT32_t index=0
    cdef UINT64_t i=0, j=0, d=0, n=0, x=0, y=0, r=0
    cdef FLOAT32_t Z1r=0, Z1r_inv = 0, dZ=0, p=0, pp2=0, to_w=0, dbeta=-1, norm=-1
    cdef FLOAT32_t dR_dA, pre1, pre2, Eji, rA

    ### Static vectors needed during computation
    # mul is faster than div
    cdef FLOAT32_t [:] psam_inv = 1./params.psam_vec
    # self consistent free protein, made dimensionless to fit Z
    cdef FLOAT32_t [:] rbp_free_inv = 1./ (params.A0 * state.rbp_free) 

    ### (Thread-)local buffers
    cdef UINT32_t [:] skipped = np.zeros(n_threads, dtype=np.uint32)
    
    # change in partition function (per read, thread-local). Gets zeroed a lot.
    # cdef FLOAT32_t [:,:] dZr_dA = np.zeros((n_threads, n_psam_padded), dtype=np.float32)

    # print "setup1", n_samples, n_psam, Z1_thresh
    # get dZr_dA cache aligned
    # cdef FLOAT32_t *ptr = <FLOAT32_t*> malloc(4*n_threads*n_psam_padded+64)
    # if ptr == NULL:
    #     raise ValueError('could not allocate cache-line aligned buffer')
    # cdef unsigned int base = <unsigned int> ptr
    # if base % 64 > 0: # not cache aligned?
    #     base = base + 64 - (base % 64) # use the padding
    # # print base, base % 64, <int> ptr
    # cdef FLOAT32_t [:,:] dZr_dA = <FLOAT32_t [:n_threads, :n_psam_padded]> <FLOAT32_t*>base
    cdef FLOAT32_t [:, :] dZr_dA = np.zeros((n_threads, n_psam_padded), dtype=np.float32)
    cdef FLOAT32_t *dZr_dA_row

    cdef UINT64_t zero_bytes = (n_psam-1)*4 # 4 = sizeof(FLOAT32_t)

    # change in weight assigned to each kmer in pulldown (thread-local)
    cdef FLOAT32_t [:,:,:,:] dw = np.zeros((n_threads, n_samples, n_psam, Nk), dtype=np.float32)
    cdef FLOAT32_t *dw_row

    # change in normalization factor
    cdef FLOAT32_t [:,:,:] dW = np.zeros((n_threads, n_samples, n_psam_padded), dtype=np.float32)

    # where to store the final gradient
    gradient = params.copy()
    gradient.data[:] = 0
    cdef FLOAT32_t [:] grad = gradient.data

    # store kmer-specific dR_dA
    cdef FLOAT32_t [:,:,:] gradi = np.zeros( (n_samples, Nk, n_psam), dtype=np.float32)

    # The loop over all reads is parallelized using thread-local dZ_dA and dw.
    # dw size is 4^k * n_samples * n_threads * n_psam * sizeof(FLOAT32_t)
    # for 3 samples, a 11nt PSAM, and k=6 dw requires only ~2MB per thread
    # aggregation from threads is only over 4**k cycles, not number of reads.
    from time import time
    t0 = time()
    # print "setup2"
    # main loop over all reads. compute dw_dA. in threads
    for r in prange(N, schedule='dynamic', nogil=True, num_threads=3):
        tid = cython.parallel.threadid() #openmp.omp_get_thread_num()

    # # single threaded version for testing
    # tid = 0
    # n_threads = 1
    # for r in range(N):
        Z1r = Z1_read[r]
        if Z1r < Z1_thresh:
            # skip early and save time
            skipped[tid] += 1
            continue
        # print "0.5", tid, psam_inv, dZr_dA, dZr_dA.base #, "%x" % base, n_psam_padded, psam_inv[0]
        # print dZr_dA[tid, 0]
        # since A0 is not inside Zr
        dZr_dA_row = &dZr_dA[tid, 0]
        dZr_dA_row[0] = Z1rm[r] #* psam_inv[0]

        # initialize other elements to 0
        memset(&dZr_dA_row[1], 0, zero_bytes)
        seqm_row = &seqm[r, 0]
        Z1_row = &Z1[r, 0]
        Z1m_row = &Z1m[r, 0]
        for x in range(l):
            for d in range(k):
                n = seqm_row[x+d]
                y = (d << 2) + n + 1
                dZr_dA_row[y] += Z1m_row[x]

        # compute dPsi/dA. up to the (psi - psi^2) factor 
        Z1r_inv = 1./Z1r
        for y in range(0, n_psam):
            dZr_dA_row[y] = psam_inv[y] * dZr_dA_row[y] * Z1r_inv

        # push dpsi_dA. to individual kmer weights dw_dA.
        im_row = &im[r, 0]
        for j in range(n_samples):
            if lin_occ:
                pp2 = 1. #rbp_free[j]
            else:
                p = psi[j, r]
                # (psi - psi^2) is the only sample/concentration dependent term
                pp2 = (p - (p * p))

            for y in range(n_psam):
                to_w = dZr_dA_row[y] * pp2 

                # propagate to individual kmers
                dw_row = &dw[tid, j, y, 0]
                for x in range(lam):
                    dw_row[im_row[x]] += to_w

                dW[tid, j, y] += lam * to_w
    t1 = time()
    # print "t1"
    ## accumulate data from threads into the 0-th entry
    for tid in range(1, n_threads):
        skipped[0] += skipped[tid]
        for j in range(n_samples):
            for y in range(n_psam):
                dW[0, j, y] += dW[tid, j, y]
                for i in range(Nk):
                    dw[0, j, y, i] += dw[tid, j, y, i]

    t2 = time()
    # print "t2"
    ## compute gradient matrix element sum
    for j in range(n_samples):
        for i in range(Nk):
            Eji = E[j,i]
            pre1 = R[j,i] / w[j,i]
            pre2 = f0[i] * R[j,i]

            for y in range(n_psam):
                dR_dA = pre1 * (dw[0, j, y, i] - pre2 * dW[0, j, y])
                gradi[j, i, y] += dR_dA
                grad[y] += Eji * dR_dA

            dbeta = Eji * pre1 * F0[i] * (1 - R[j, i])
            grad[n_psam + j] += dbeta

    ## normalize gradient elements sum -> mean 
    # factor 2 is from dE_dA. = - _2_ * (R - R0) sum dR_dA.
    norm = 2. / (Nk * n_samples)
    for y in range(n_params):
        grad[y] *= norm

    t3 = time()
    print "dw/dA. over reads {0:.2f}ms, thread-acc {1:.2f}ms, grad-matrix {2:.2f} ms, total {3:.2f}ms".format(
        1000. * (t1-t0), 1000. * (t2-t1), 1000. * (t3-t2), 1000. * (t3-t0))
    # print "done"
    state.skipped = skipped.base[0]
    state.gradi = gradi
    # free(ptr)
    # state.n_eval = n_eval.base
    return gradient


def PSAM_kmer_gradient(UINT8_t [:,:] seqm, FLOAT32_t [:,:] Z, FLOAT32_t [:] Zj, FLOAT32_t [:] psi, UINT32_t [:,:] index_matrix, FLOAT32_t [:] psam, UINT64_t k_mer, int n_max=0):
    cdef UINT64_t N = seqm.base.shape[0]
    cdef UINT64_t L = seqm.base.shape[1]
    cdef UINT64_t n_psam = len(psam.base)
    cdef UINT64_t k = (n_psam - 1) / 4
    cdef UINT64_t l = L - k + 1
    # print "L-k+1", l, "Z.shape", Z.shape, 'k', k
    cdef UINT64_t l_im = index_matrix.shape[1]
    cdef UINT64_t zero_bytes = n_psam*4

    # dpi 
    cdef FLOAT32_t [:,:] dpi = np.zeros((4**k_mer, n_psam), dtype=np.float32)
    cdef FLOAT32_t [:] pi = np.zeros(4**k_mer, dtype=np.float32)
    cdef int thread_num = 0
    cdef UINT64_t i=0, j=0, d=0, n=0, x=0
    cdef UINT32_t index=0
    cdef FLOAT32_t p=0, dpsi=0
    cdef FLOAT32_t [:] dpsi_dM = np.zeros(n_psam, dtype=np.float32)
    # cdef FLOAT64_t Z1=0 # Single protein partition function
    
    # mutliplication is faster than division. So divide outside of loop.
    cdef FLOAT32_t [:] psam_inv = 1./psam.base

    if n_max:
        N = min(N, n_max)

    # TODO: make safe for parallelization by thread-local dpsi_dM and CAS for dpi access
    # with nogil, parallel():
    #     for j in prange(N, schedule='static'):
    #         thread_num = openmp.omp_get_thread_num()
    
    for j in range(N):
        p = psi[j]
        # chain rule: how changes in per-sequence partition function
        # carry over to changes in binding probability
        dpsi = (p - (p * p)) / Zj[j]

        # zero out dZj_dM bc we accumulate this for each sequence separately
        memset(&dpsi_dM[0], 0, zero_bytes)
        
        # compute dZj_dM gradient matrix
        # dZj/dA0 first
        dpsi_dM[0] = Zj[j] * psam_inv[0] * dpsi

        # now dZj/dM with M being the matrix elements of the psam (here flattened)
        for i in range(l):
            for d in range(k):
                n = seqm[j, i+d]
                x = (d << 2) + n + 1
                dpsi_dM[x] += Z[j,i] * psam_inv[x] * dpsi

        # propagate effect to pulldown kmer frequencies
        for i in range(l_im):
            index = index_matrix[j,i]
            # build weighted kmer-frequencies on the fly
            pi[index] += p
            
            # dpi/dA0
            dpi[index, 0] += dpsi_dM[0] 
            
            # dpi/dM
            for d in range(k):
                for n in range(4):
                    # if dpsi_dM[(d << 2) + n + 1] == np.NaN:
                    #     print "encountered NaN in gradient computation", j, dpsi_dM, psam_inv, dpsi, Z[j]
                    x = (d << 2) + n + 1
                    dpi[index, x] += dpsi_dM[x]

    return pi.base, dpi.base
                    

def params_from_pwm(FLOAT32_t [:,:] pwm, FLOAT32_t A0=1., FLOAT32_t aff0=1e-5):
    cdef UINT64_t k = pwm.base.shape[0]
    cdef UINT64_t Na = 4**k

    cdef int thread_num = 0
    cdef int n_threads = 8

    # store parameters here
    cdef FLOAT32_t [:] params = np.zeros(Na, dtype = np.float32) + aff0
    # indices of kmers with A_i > aff0
    cdef UINT32_t [:,:] indices = np.empty((n_threads, Na), dtype = np.uint32)

    cdef UINT32_t [:] n_indices = np.zeros(n_threads, dtype = np.uint32)
    cdef int i,j,n,ind,l
    cdef FLOAT32_t A=0

    with nogil, parallel(num_threads=8):
        for i in prange(Na, schedule='static'):

    # ugcacgu = seq_to_index('ugcacgu')

    # for i in range(Na):
            A = A0
            ind = i
            l = k-1
            for j in range(k):
                n = ind & 3
                A = A * pwm[l,n]
                ind = ind >> 2
                # if i == ugcacgu:
                #     print "ugcacgu", j, "ACGU"[n], pwm[l,n], A
                l = l - 1

            A = max(A, aff0)
            params[i] = A

            thread_num = openmp.omp_get_thread_num()
            if params[i] > aff0:
                indices[thread_num, n_indices[thread_num]] = i
                n_indices[thread_num] += 1
            
    cat = []
    for i in range(n_threads):
        cat.append(indices.base[i,:n_indices[i]])

    return params.base, np.concatenate(cat)


            
#@cython.boundscheck(True)
#@cython.wraparound(True)
#@cython.initializedcheck(True)
#@cython.overflowcheck(True)
def PSAM_mean_field_eval(state):
    cdef UINT64_t k = state.params.k
    cdef UINT64_t n_samples = state.params.n_samples
    cdef UINT64_t Nk = 4**k
    cdef UINT64_t Nr = len(state.I)
    cdef UINT32_t [:] I = state.I # indices of kmers with A > aff0
    cdef FLOAT32_t [:] rbp_conc = state.rbp_conc
    cdef FLOAT32_t [:] params = state.params.data
    cdef FLOAT32_t aff0 = state.mdl.aff0
    cdef FLOAT32_t A0 = state.params.A0
    cdef FLOAT32_t [:] betas = params[state.params.betas_start:]
    cdef FLOAT32_t [:] f0 = state.mdl.f0
    cdef FLOAT32_t [:,:] R0 = state.mdl.R0
    cdef FLOAT32_t [:,:] M = state.mdl.xm.M

    cdef int thread_num = 0
    cdef int n_threads = 8

    cdef FLOAT32_t [:] Z1 = state.A #np.empty(Nk, dtype=np.float32)
    cdef FLOAT32_t [:,:] occ = np.empty((n_samples,Nk), dtype=np.float32)
    cdef FLOAT32_t [:,:] pi = np.empty((n_samples,Nk), dtype=np.float32)
    cdef FLOAT32_t [:,:] sum_pi = np.zeros((n_threads, n_samples), dtype=np.float32)
    cdef FLOAT32_t [:] _sum_pi = np.zeros(n_samples, dtype=np.float32)
    cdef FLOAT32_t [:,:] R = np.empty((n_samples,Nk), dtype=np.float32)
    cdef FLOAT32_t [:] error = np.zeros(n_threads, dtype=np.float32)

    cdef UINT64_t i=0,j=0,l=0,ind=0,n=0,nt=0,x=0,rowbase=0
    cdef FLOAT32_t A = 0,Z=0, z=0,o=0, Mil = 0, err=0

    with nogil:
        # with parallel():
        #     for i in prange(Nk):
        #         ind = i
        #         rowbase = (k-1) << 2
        #         A = A0
        #         for j in range(k):
        #             nt = ind & 3
        #             x = rowbase + nt + 1
        #             A = A * params[x]
        #             ind = ind >> 2
        #             rowbase = rowbase >> 2

        #         Z1[i] = max(A, aff0)
        
        for n in range(n_samples):
            with parallel():
                for i in prange(Nk):
                    z = rbp_conc[n]*Z1[i]
                    occ[n,i] = z / (z + 1)

            with parallel():
                for i in prange(Nk):
                    thread_num = openmp.omp_get_thread_num()
                    Mil = 0
                    # for l in range(Nk): # speed up by keeping explicitly the relevant indices and weights!
                    #     Mil = Mil + M[i,l] * occ[n,l]
                    for j in range(Nr):
                        l = I[j] # only iterate over kmers with A > aff0
                        Mil = Mil + M[i,l] * occ[n,l]

                    pi[n,i] = f0[i] * (Mil + betas[n])
                    sum_pi[thread_num, n] += pi[n,i]
                
            for i in range(n_threads):
                _sum_pi[n] += sum_pi[i, n]

            with parallel():
                for i in prange(Nk):
                    thread_num = openmp.omp_get_thread_num()
                    R[n,i] = pi[n,i]/_sum_pi[n] / f0[i]
                    err = R[n,i] - R0[n,i]
                    error[thread_num] += err * err

    state.A = Z1.base
    state.occ = occ.base
    state.pi = pi.base
    state.sum_pi = _sum_pi.base
    state.R = R.base
    state.error = error.base.sum() / (Nk * n_samples)


#@cython.boundscheck(True)
#@cython.wraparound(True)
#@cython.initializedcheck(True)
#@cython.overflowcheck(True)
def PSAM_mean_field_gradient(state):
    cdef UINT64_t n_samples = state.params.n_samples
    cdef UINT64_t k = state.params.k
    cdef UINT64_t Nk = 4**k
    
    cdef FLOAT32_t [:] psam_inv = 1./state.params.psam_vec # all PSAM matrix elements and A0 at index 0
    # shape = (n_samples, Nk)
    cdef FLOAT32_t [:,:] occ = state.occ 
    cdef FLOAT32_t [:,:] pi = state.pi
    cdef FLOAT32_t [:,:] R = state.R
    cdef FLOAT32_t [:,:] R0 = state.mdl.R0
    
    cdef UINT64_t Nr = len(state.I)
    cdef UINT32_t [:] I = state.I # indices of kmers with A > aff0

    # shape = (n_samples)
    cdef FLOAT32_t [:] sum_pi_inv = 1./pi.base.sum(axis=1)
    # shape = (Nk, Nk)
    cdef FLOAT32_t [:,:] M = state.mdl.xm.M
    # shape = (Nk) [abundance weighted row mean]
    cdef FLOAT32_t [:] wrm = state.mdl.xm.wrm
    
    cdef int thread_num = 0
    cdef int n_threads = 8
    cdef UINT64_t i=0, j=0, d=0, n=0, x=0, l=0, nt=0

    cdef FLOAT32_t pre, o, w
    cdef FLOAT32_t [:,:] grad = np.zeros((n_threads, state.params.n), dtype=np.float32)
    
    
    # mutliplication is faster than division. So divide outside of loop.
    cdef FLOAT32_t [:] params_inv = 1./state.params.data

    with nogil, parallel():
        for n in range(n_samples):
            for i in prange(Nk, schedule='dynamic'):
                thread_num = openmp.omp_get_thread_num()
                # make variabls thread-private
                o = 0
                w = 0
                pre = 0
                pre = 2 * (R[n,i] - R0[n,i]) * sum_pi_inv[n]
                # for l in range(Nk):
                for j in range(Nr):
                    l = I[j] # only iterate over kmers with A > aff0

                    o = occ[n,l]
                    w = pre * (M[i,l] - R[n,i] * wrm[l]) * (o - o*o)
                    grad[thread_num, 0] += w * psam_inv[0] # dE/dA0 (always contributes)
                    for d in range(k): 
                        # deconstruct kmer into matrix element coordinates
                        nt = l >> (2 * (k - d -1)) & 3
                        x = (d << 2) + nt + 1
                        grad[thread_num, x] += w * psam_inv[x] #dE/dAm,n (only for the elements that contribute)
                    
                grad[thread_num, 1+k*4+n] += (R[n,i] - R0[n,i]) * (1 - R[n,i]) #accumulate dE/dbeta terms

            grad[thread_num, 1+k*4+n] *= 2 / sum_pi_inv[n] # dE/dbeta pre-factor

    return grad.base.sum(axis=0)
        

    


#@cython.boundscheck(True)
#@cython.wraparound(True)
#@cython.initializedcheck(True)
#@cython.overflowcheck(True)
def PSAM_inv_mean_field_gradient(state):
    cdef UINT64_t n_samples = state.params.n_samples
    cdef UINT64_t k = state.params.k
    cdef UINT64_t Nk = 4**k
    
    cdef FLOAT32_t [:] psam_inv = 1./state.params.psam_vec # all PSAM matrix elements and A0 at index 0
    # shape = (n_samples, Nk)
    cdef FLOAT32_t [:,:] occ = state.occ 
    cdef FLOAT32_t [:,:] pi = state.pi
    cdef FLOAT32_t [:,:] R = state.R
    cdef FLOAT32_t [:,:] R0 = state.mdl.R0
    
    cdef UINT64_t Nr = len(state.I)
    cdef UINT32_t [:] I = state.I # indices of kmers with A > aff0

    # shape = (n_samples)
    cdef FLOAT32_t [:] sum_pi_inv = 1./pi.base.sum(axis=1)
    # shape = (Nk, Nk)
    cdef FLOAT32_t [:,:] M = state.mdl.xm.M
    # shape = (Nk) [abundance weighted row mean]
    cdef FLOAT32_t [:] wrm = state.mdl.xm.wrm
    
    cdef int thread_num = 0
    cdef int n_threads = 8
    cdef UINT64_t i=0, j=0, d=0, n=0, x=0, l=0, nt=0

    cdef FLOAT32_t pre, o, w
    cdef FLOAT32_t [:,:] grad = np.zeros((n_threads, state.params.n), dtype=np.float32)
    
    
    # mutliplication is faster than division. So divide outside of loop.
    cdef FLOAT32_t [:] params_inv = 1./state.params.data

    # for n in range(n_samples):
    #     for i in range(Nk):


    with nogil, parallel():
        for n in range(n_samples):
            for i in prange(Nk, schedule='dynamic'):
                thread_num = openmp.omp_get_thread_num()
                # make variabls thread-private
                o = 0
                w = 0
                pre = 0
                pre = 2 * (R[n,i] - R0[n,i]) * sum_pi_inv[n]
                # for l in range(Nk):
                for j in range(Nr):
                    l = I[j] # only iterate over kmers with A > aff0

                    o = occ[n,l]
                    w = pre * (M[i,l] - R[n,i] * wrm[l]) * (o - o*o)
                    grad[thread_num, 0] += w * psam_inv[0] # dE/dA0 (always contributes)
                    for d in range(k): 
                        # deconstruct kmer into matrix element coordinates
                        nt = l >> (2 * (k - d -1)) & 3
                        x = (d << 2) + nt + 1
                        grad[thread_num, x] += w * psam_inv[x] #dE/dAm,n (only for the elements that contribute)
                    
                grad[thread_num, 1+k*4+n] += (R[n,i] - R0[n,i]) * (1 - R[n,i]) #accumulate dE/dbeta terms

            grad[thread_num, 1+k*4+n] *= 2 / sum_pi_inv[n] # dE/dbeta pre-factor

    return grad.base.sum(axis=0)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.initializedcheck(False)
# @cython.cdivision(True)
# @cython.overflowcheck(False)
# def SPA_gradient_raw(UINT32_t [:,:] index_matrix, FLOAT32_t [:,:] acc_matrix, FLOAT32_t [:] kmer_invkd, UINT64_t k, int n_max=0, int openen_ofs=0):
#     assert k <= 16 # must fit into UINT32 kmer-indices!

#     cdef UINT64_t N = index_matrix.base.shape[0]
#     cdef UINT64_t l = index_matrix.base.shape[1]
#     cdef UINT64_t n_k = 4**k
#     # result will be stored here (Z = 'Zustandssumme' sum of states)
#     cdef FLOAT32_t [:] Z = np.empty(N, dtype=np.float32)
    
#     # Jacobi matrix, how affinity of each kmer will change the weighted of each kmer in the pulldown
#     cdef FLOAT32_t [:,:] J = np.zeros((n_k,n_k), dtype=np.float32)

#     # accumulated kmer Boltzmann weights from all reads that contain them
#     cdef FLOAT32_t [:] W = np.zeros(n_k, dtype=np.float32)

#     # helper variables to tell cython the types
#     cdef FLOAT32_t a=0
#     cdef UINT64_t i=0, j=0
#     cdef UINT32_t index=0
#     cdef FLOAT32_t w=0
#     cdef FLOAT64_t Z1=0 # Single protein partition function

#     if n_max:
#         N = min(N, n_max)

#     with nogil, parallel():
#         for j in prange(N, schedule='static'):
#             Z1 = 0
#             # iterate over all k-mers in this read
#             for i in range(0, l):
#                 # assigned variables are thread-local
#                 index = index_matrix[j, i]
#                 a = acc_matrix[j, i + openen_ofs]
#                 w = kmer_invkd[index] * a
#                 Z1 = Z1 + w

#             # re-iterate to update Jacobi matrix and weighted kmer table
#             for i in range(0, l):
#                 index = index_matrix[j, i]
#                 a = acc_matrix[j, i + openen_ofs]
#                 w = kmer_invkd[index] * a
#                 Z1 = Z1 + w
            
#             Z[j] = Z1

#     return Z.base



def SPA_bipartite_partition_function_raw(
    UINT32_t [:,:] index_matrix, 
    FLOAT32_t [:,:] acc_matrix, 
    FLOAT32_t [:] aff_A, 
    FLOAT32_t [:] aff_B, 
    FLOAT32_t [:] dist_cost, 
    UINT64_t k, 
    int n_max=0, 
    int openen_ofs=0
    ):

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t L = index_matrix.base.shape[1]
    cdef UINT64_t d_max = len(dist_cost.base)

    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z = np.empty(N, dtype=np.float32)
    
    # helper variables to tell cython the types
    cdef FLOAT32_t a=0
    cdef int i=0, j=0, m=0, d=0
    cdef UINT32_t index=0
    cdef FLOAT32_t w_A=0, w_B=0
    cdef FLOAT32_t [:] Z_A = np.zeros(L, dtype=np.float32) # Single protein partition function terms
    cdef FLOAT32_t [:] Z_B = np.zeros(L, dtype=np.float32) # Single protein partition function terms

    cdef FLOAT32_t Z1 = 0

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='static'):
            Z1 = 0 # make thread-local
            # iterate over all k-mers and fill in single motif partition functions
            for i in range(L):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                a = acc_matrix[j, i + openen_ofs]
                w_A = aff_A[index] * a
                w_B = aff_B[index] * a
                Z_A[i] = w_A
                # Z_B[i] = w_B
                
                Z1 = Z1 + w_A + w_B # add single motif contributions

                # scan "backwards" to add bi-partite contributions
                # w_B fixed, w_A is read from already populated part of Z_A
                for m in range(i):
                    d = i - m
                    Z1 = Z1 + w_B * dist_cost[d] * Z_A[m]
            
            Z[j] = Z1

    return Z.base


def xcorr_Z(FLOAT32_t [:,:] Z_A, FLOAT32_t [:,:] Z_B, UINT64_t k1, UINT64_t k2):
    cdef UINT64_t N = len(Z_A)
    assert N == len(Z_B)
    # assert k1 >= k2

    cdef UINT64_t L1 = len(Z_A[0])
    cdef UINT64_t L2 = len(Z_B[0])
    
    cdef UINT64_t L_max = max(L1, L2)

    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z_corr = np.zeros(2*L_max, dtype=np.float32)
    cdef FLOAT32_t [:] n_corr = np.zeros(2*L_max, dtype=np.float32)

    # helper variables to tell cython the types
    cdef FLOAT32_t a=0
    cdef int i=0, j=0, n=0, ofs = k1 - k2, d =0, shift = 0
    cdef UINT64_t L1_max, L2_max, L2_start

    # if k1 <= k2:
    #     ofs = k2 - k1
    #     L1_max = L1 - k2
    #     L2_start = k1 - ofs # first k2 mer that does not overlap first k1 mer
    #     L2_max = L2
    
    # else:
    #     ofs = k1 - k2
    #     L1_max = L2 + ofs - k1
    #     L2_start = 0
    #     L2_max = L2

    cdef UINT32_t index=0
    cdef FLOAT32_t w=0
    cdef FLOAT64_t Z1=0 # Single protein partition function

    # with nogil, parallel():
        # for j in prange(N, schedule='static'):
    # with nogil:
    for n in range(N):
        # iterate over all k-mers
        for i in range(L1):
            for j in range(L2):
                d = j-i+ofs # separation between the two mers
                # print k1, k2, ofs, "i,j", i,j, "d",d
                Z_corr[d+L_max] += Z_A[n,i] * Z_B[n,j]
                n_corr[d+L_max] += 1
            

    # print L_max
    # return (Z_corr.base / n_corr.base)[L_max+1:]
    return Z_corr.base

def p_bound(FLOAT32_t [:] Z1, FLOAT32_t [:] rbp_conc_vector, int n_threads = 8):
    cdef UINT64_t N = Z1.base.shape[0]
    cdef int n_conc = len(rbp_conc_vector)
    # store weighted k-mer counts here (for each thread)
    cdef FLOAT32_t [:,:] p_bound = np.empty((n_conc, N), dtype = np.float32)

    # helper variables to tell cython the types
    cdef FLOAT32_t conc=0, Z=0, w=0
    cdef UINT64_t i=-1,j=-1

    with nogil, parallel(num_threads=8):
        for i in range(n_conc):
            conc = rbp_conc_vector[i]
            for j in prange(N, schedule='static'):
                Z = Z1[j] * conc
                w = Z / (Z + 1.) 
                p_bound[i,j] = w
            
    return p_bound.base

