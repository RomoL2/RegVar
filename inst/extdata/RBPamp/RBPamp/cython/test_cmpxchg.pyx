#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, cdivision=True
###cython: boundscheck=True, wraparound=True, initializedcheck=True, overflowcheck=True, cdivision=False

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

from libc.math cimport exp, log
from libc.stdlib cimport abort, malloc, free

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


def test_cmpxchg():
    cdef int N = 1000
    cdef FLOAT32_t [:] Z = np.zeros(N, dtype=np.float32)
    cdef FLOAT32_t old, x
    cdef int i,j, o

    with nogil, parallel():
        for j in prange(10000000, schedule='guided'):
            i = j % N
            x = 0
            for o in range(i):
                x = x + o

            # ofs = base + i*8
            old = Z[i]
            while not cmpxchg_float32(&Z[i], old, old+x):
            # while not cmpxchg_uint(<UINT64_t*>ofs, old, old+1):
                old = Z[i]
                
    
    print Z.base.sum()
    return Z.base


def test_naive():
    cdef int N = 1000
    cdef UINT64_t [:] Z = np.zeros(N, dtype=np.uint64)
    cdef int i,j, old, x

    with nogil, parallel():
        for j in prange(10000000, schedule='guided'):
            i = j % N
        
            x = 0
            for old in range(i):
                x = x + old

            Z[i] += x
    
    print Z.base.sum()
    return Z.base

def test_sequential():
    cdef int N = 1000
    cdef FLOAT32_t [:] Z = np.zeros(N, dtype=np.float32)
    cdef FLOAT32_t old, x
    cdef int i,j, o

    # with nogil:
    for j in range(10000000):
        i = j % N
        old = Z[i]
        
        x = 0
        for o in range(i):
            x = x + o

        Z[i] += x
        # while not cmpxchg_uint64(&Z[i], old, old+x):
        # # while not cmpxchg_uint(<UINT64_t*>ofs, old, old+1):
        #     # print old, Z[i], i
        #     old = Z[i]
        # Z[i] += 1
    
    print Z.base.sum()
    return Z.base
