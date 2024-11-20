#cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, cdivision=True
###cython: boundscheck=True, wraparound=True, initializedcheck=True, overflowcheck=True, cdivision=False
#!python

__license__ = "MIT"
__version__ = "0.9.8"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

"""
A fast Cython implementation of pseudo-random numbers and implementation for simulating RBNS reads.

Uses xorshift128plus by Vigna, Sebastiano 
    https://arxiv.org/abs/1404.0390

"""

from types cimport *
from types import *

from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np
cimport cython
cimport openmp

from libc.math cimport exp, log
from libc.stdlib cimport abort, malloc, free

cdef UINT64_t rand_state[2]
cdef UINT64_t RAND_MAX = 2**64 - 1
cdef FLOAT32_t FRAND_MAX = RAND_MAX

@cython.overflowcheck(False)
cdef inline UINT64_t randint():
    """
    Cython version of xorshift128plus by Vigna, Sebastiano 
    https://arxiv.org/abs/1404.0390
    """
    cdef UINT64_t x = rand_state[0]
    cdef UINT64_t y = rand_state[1]
    rand_state[0] = y
    x ^= x << 23
    rand_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26)
    return rand_state[1] + y

cdef inline FLOAT32_t rand():
    """
    Returns uniform random FLOAT32 between 0 and 1
    """
    cdef FLOAT32_t x = randint()
    return x / FRAND_MAX

@cython.overflowcheck(False)
def rand_seed(UINT64_t seed, burn=1000):
    cdef UINT64_t rnd
    
    rand_state[0] = seed
    rand_state[1] = (~ seed) << 3
    
    for i in range(burn):
        rnd = randint()

def fast_randint(int N, UINT64_t max=RAND_MAX):
    cdef UINT64_t [:] rnd = np.empty(N, dtype=np.uint64)
    cdef int i
    
    for i in range(N):
        rnd[i] = randint() % max

    return rnd.base

def fast_rand(int N):
    cdef np.ndarray[FLOAT32_t] rnd = np.empty(N, dtype=np.float32)
    cdef int i
    
    for i in range(N):
        rnd[i] = rand()

    return rnd

def random():
    return rand()
    
cdef inline UINT8_t rand_choice_uint8(UINT64_t [:] cum, int ofs, int n):
    cdef UINT64_t rnd = randint()
    cdef UINT8_t x = 0

    for x in range(n-1):
        if rnd < cum[ofs+x]:
            return x

    return n-1

def generate_random_sequence_matrix(UINT32_t l, UINT32_t N):
    cdef np.ndarray[UINT8_t, ndim=2] seqm_ = np.empty((N, l), dtype=np.uint8)
    cdef UINT8_t [:, :] seqm = seqm_ # MemoryView
    
    cdef UINT64_t i, j
    
    for j in range(N):
        for i in range(l):
            seqm[j,i] = randint() & 3 # use lower 2 bits
            
    return seqm_

def generate_random_sequence_matrix_dinuc(UINT32_t l, UINT32_t N, np.ndarray[FLOAT32_t] nt_freqs, np.ndarray[FLOAT32_t, ndim=2] di_freqs):
    cdef np.ndarray[UINT8_t, ndim=2] seqm_ = np.empty((N, l), dtype=np.uint8)
    cdef UINT8_t [:, :] seqm = seqm_ # MemoryView
    
    cdef UINT64_t i, j, nuc
    cdef UINT64_t [:] cum_nt
    cdef UINT64_t [:] cum_di
    
    cum_nt = np.array(nt_freqs.cumsum() * FRAND_MAX, dtype=np.uint64)
    cum_di = np.array(di_freqs.cumsum(axis=1) * FRAND_MAX, dtype=np.uint64).flatten()

    #print "nt", cum_nt
    #print "di", cum_di
    
    for j in range(N):
        nuc = rand_choice_uint8(cum_nt, 0, 4)
        seqm[j,0] = nuc
        for i in range(1,l):
            nuc = rand_choice_uint8(cum_di, nuc << 2, 4)
            seqm[j,i] = nuc
            
    return seqm_



def simulate_rbns_reads(
        UINT32_t L, 
        UINT32_t N,
        UINT64_t k,
        np.ndarray[FLOAT32_t] nt_freqs, 
        np.ndarray[FLOAT32_t, ndim=2] di_freqs,
        np.ndarray[FLOAT32_t] scaled_kmer_energies_, # already in units of RT
        FLOAT32_t P, # protein concentration in nM
        FLOAT32_t p_ns, # prob. for non-specific binding
    ):

    cdef np.ndarray[UINT8_t, ndim=2] seqm_ = np.empty((N, L), dtype=np.uint8)

    cdef UINT8_t [:, :] seqm = seqm_ # MemoryView
    cdef UINT64_t [:] cum_nt
    cdef UINT64_t [:] cum_di
    cdef FLOAT32_t [:] kmer_energies =  scaled_kmer_energies_
    
    cum_nt = np.array(nt_freqs.cumsum() * FRAND_MAX, dtype=np.uint64)
    cum_di = np.array(di_freqs.cumsum(axis=1) * FRAND_MAX, dtype=np.uint64).flatten()

    cdef FLOAT32_t mu = np.log(P*1e-9)
    cdef FLOAT32_t [:] boltzmann_weights = np.exp(-scaled_kmer_energies_ + mu)

    
    cdef UINT8_t [:] seq_bits
    cdef FLOAT64_t Z, w, p_bound, p_obs
    cdef UINT64_t i, j, nuc, index, n_bound=0, n_simulated=0, l=L-k+1
    cdef UINT64_t MAX_INDEX = 4**k - 1, ofs
    cdef UINT8_t s
    
    #P *= 1e-9 # convert from nano Molars to Molars

    j = 0
    while j < N:
        # generate a random read with dinuc frequencies
        nuc = rand_choice_uint8(cum_nt, 0, 4)
        seqm[j,0] = nuc
        for i in range(1,L):
            nuc = rand_choice_uint8(cum_di, nuc << 2, 4)
            #nuc = rand_choice_uint8(cum_nt, 0, 4)
            seqm[j,i] = nuc
        n_simulated += 1
        
        # partition function for binding of a single protein
        Z = 0
        #ofs = j+l
        # compute index of first k-1 mer by bit-shifts
        index = 0
        for i in range(k-1):
            index += seqm[j, i] << 2 * (k - i - 2)

        # iterate over all k-mers in the read
        for i in range(k-1, L):
            # get next "letter"
            s = seqm[j, i]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            
            # Boltzmann weight for binding here
            Z += boltzmann_weights[index]
            #print "weight", exp(-kmer_energies[index]), "index", index, "energy", kmer_energies[index]

        #Z *= P # times protein concentration
        
        #print p_bound
        # do we observe this read?
        p_bound = Z / (Z + 1.)
        p_obs = 1 - (1-p_bound)*(1-p_ns)

        #if (seqm_[j,:] == 3).all(): #polyU
            #print "UUUUUU Z={0} p_bound={1} p_obs={2}".format(Z, p_bound, p_obs) 

        #print "Z={0} p_obs={1}".format(Z, p_obs)
        if rand() < p_obs:
            # was pulled down
            j += 1
    
    return seqm_, float(N)/n_simulated


# default fastrand initialization
import time
rand_seed(int(1000*time.time()) + 11)
