#cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, cdivision=True
###cython: boundscheck=True, wraparound=True, initializedcheck=True, overflowcheck=True, cdivision=False
#!python

__license__ = "MIT"
__version__ = "0.9.8"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

"""
A fast Cython implementation of the "Streaming K-mer Assignment" 
algorithm initially described in Lambert et al. 2014 (PMID: 24837674)
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

# maps ASCII values of A,C,G,T (U) to correct bits
cdef UINT8_t letter_to_bits[256]
for i in range(256):
    letter_to_bits[i] = 255

letter_to_bits[ord('a')] = 0 
letter_to_bits[ord('c')] = 1 
letter_to_bits[ord('g')] = 2
letter_to_bits[ord('t')] = 3
letter_to_bits[ord('u')] = 3

letter_to_bits[ord('A')] = 0 
letter_to_bits[ord('C')] = 1 
letter_to_bits[ord('G')] = 2
letter_to_bits[ord('T')] = 3
letter_to_bits[ord('U')] = 3


cdef UINT8_t bits_to_letters[4]
bits_to_letters[:] = [ord('A'), ord('C'), ord('G'), ord('U')]
    
def yield_kmers(k):
    import itertools
    """
    An iterater to all kmers of length k in alphabetical order
    """
    bases = 'ACGU'
    for kmer in itertools.product(bases, repeat=k):
        yield ''.join(kmer)
                                

def weighted_kmer_shifts(UINT64_t index, UINT64_t k, UINT64_t L, UINT64_t x, FLOAT32_t [:] kfreqs):
    """
    Used to compute the overlap matrix.
    """
    
    cdef UINT64_t i = 0, j = 0, s = 0
    cdef UINT64_t N = 4**x
    cdef UINT64_t l = L - k + 1
    
    cdef UINT64_t OV_MAX_INDEX, MAX_INDEX = (4**k) -1 
    
    cdef FLOAT32_t w

    cdef np.ndarray[UINT64_t] sindices_  = np.zeros(2*N, dtype=np.uint64)
    cdef np.ndarray[FLOAT32_t] sweights_ = np.zeros(2*N, dtype=np.float32)
    
    cdef UINT64_t [:] sindices = sindices_
    cdef FLOAT32_t [:] sweights = sweights_
    
    
    i = 0
    w = 1. * (l - 2*x) / l 
    OV_MAX_INDEX = 4**(k-x) - 1
    
    # shifted to the right, overlaps on the left with index
    ov = ( index << (2 * x) ) & MAX_INDEX
    for j in range(4**x): # all possible extensions
        s = j | ov
        sindices[i] = s
        sweights[i] = w * kfreqs[j]
        i += 1

    # shifted to the left, overlaps on the right with index
    ov = ( index >> (2 * x) ) & OV_MAX_INDEX
    for j in range(4**x): # all possible extensions
        s = (j << (2*(k-x)) ) | ov
        sindices[i] = s
        sweights[i] = w * kfreqs[j]
        i += 1

    return sweights_, sindices_
        
    
def write_seqm(np.ndarray[UINT8_t, ndim=2] seqm_, f):
    cdef UINT64_t N = seqm_.shape[0], l = seqm_.shape[1], i, j
    
    cdef UINT8_t [:, :] seqm = seqm_
    cdef np.ndarray[UINT8_t] seq_ = np.zeros(l+1, dtype=np.uint8)
    cdef UINT8_t [:] seq = seq_
    seq[l] = '\n'
    
    for j in range(N):
        # convert seq entries back to string
        for i in range(l):
            seq[i] = bits_to_letters[ seqm[j, i] ]

        f.write(seq_)
    
    
def seq_to_bits(str py_str):
    cdef UINT32_t x, L
    cdef UINT8_t n
    cdef bytes py_byte_str = py_str.encode('ascii')
    cdef unsigned char *seq = py_byte_str
    
    L = len(seq)
    cdef np.ndarray[UINT8_t] _res = np.zeros(L, dtype=np.uint8)
    cdef UINT8_t [::1] res = _res
    
    for x in range(L):
        n = seq[x]
        res[x] = letter_to_bits[n]

    return _res


cdef inline UINT64_t kbits_to_index(UINT8_t[:] kbits, UINT32_t k) nogil:
    cdef UINT64_t i, index = 0
    
    for i in range(k):
        index += kbits[i] << 2 * (k - i - 1)
    
    return index


def seq_to_index(seq):
    k = len(seq)
    bits = seq_to_bits(seq)
    return kbits_to_index(bits, k)

def index_to_seq(index, k):
    nucs = ['a','c','g','u']
    seq = []
    for i in range(k):
        j = index >> ((k-i-1) * 2)
        seq.append(nucs[j & 3])

    return "".join(seq)

def read_raw_seqs_chunked(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0, int chunklines=1000000):
    cdef unsigned char* l
    cdef UINT64_t i, N=0, n=0, n0=0, L=0
    cdef UINT8_t x=0
    cdef UINT32_t chunkbytes = 0
    cdef list chunks = list()
    cdef str line
    cdef bytes py_byte_str
    cdef bytes _pre = <bytes>pre
    cdef bytes _post = <bytes>post
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf # fast memoryview into current buffer

    for line in src:
        if n_skip and N <= n_skip:
            continue

        #if 'N' in line:
            #continue

        line = line.rstrip() # remove trailing new-line characters

        #if pre or post:
            #line = _pre + line + _post
        
        if not L:
            L = len(line)
            chunkbytes = chunklines*L
            # print "discovered L=",L
        
        if N % chunklines == 0:
            if n: 
                # print "new chunk at", N, n
                chunks.append(_buf)
            _buf = np.empty(L*chunklines, dtype=np.uint8)
            buf = _buf # initialize the view
            n = 0

        # l = line # extract raw string content
        py_byte_str = line.encode('ascii')
        l = py_byte_str
        n0 = n
        for i in range(0,L):
            x = letter_to_bits[l[i]]
            if x > 3:
                # non-ACGT character!
                n = n0
                break
            
            buf[n] = x
            n += 1

        if n > n0:
            # we have actually read a sequence!
            N += 1
            
        if N >= n_max + n_skip and n_max:
            break

    # print "N",N, len(chunks), "chunks", n, "left in _buf"
    chunks.append(_buf[:n])
    cat = np.concatenate(chunks)
    # print cat.shape

    return cat.reshape((N,L))


def seq_set_kmer_count(np.ndarray[UINT8_t, ndim=2] seq_matrix, UINT64_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1

    # store k-mer counts here
    _counts = np.zeros(4**k, dtype = np.uint32)
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] counts = _counts

    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j
    
    with nogil:
        for j in range(N):
            seq_bits = _seq_matrix[j*L:(j+1)*L]
            # compute index of first k-mer by bit-shifts
            index = kbits_to_index(seq_bits, k) 
            # count first k-mer
            counts[index] += 1
            # iterate over remaining k-mers
            for i in range(0, L-k):
                # get next "letter"
                s = seq_bits[i+k]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                # count
                counts[index] += 1
            
    return _counts


def seq_to_kmer_count(seq, UINT64_t k):
    # store k-mer counts here
    cdef UINT32_t [:] counts = np.zeros(4**k, dtype = np.uint32)
    cdef UINT64_t L = len(seq)
    cdef unsigned char* cseq = seq
    cdef UINT8_t n, b
    cdef UINT32_t index = 0
    cdef UINT32_t max_index = 4**k - 1
    cdef int last_invalid=-1, x
    
    for x in range(L):
        n = cseq[x]
        b = letter_to_bits[n]
        index = ((index << 2) | b) & max_index
        if b > 3:
            last_invalid = x

        if x >= (k-1):
            if x - last_invalid >= k:
                counts[index] += 1

    return counts.base


def seq_set_kmer_count_matrix(UINT8_t [:,:] seq_matrix, UINT64_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1
    cdef int N = len(seq_matrix.base)
    cdef int L = len(seq_matrix.base[0])

    # store k-mer counts here
    cdef UINT32_t [:,:] counts = np.zeros((N, 4**k), dtype = np.uint32)

    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j
    
    with nogil:
        for j in range(N):
            index = 0
            for i in range(L):
                # get next "letter"
                s = seq_matrix[j, i]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                if i >= k-1:
                    # count
                    counts[j, index] += 1
            
    return counts.base



def kmer_crosstalk_matrix(UINT32_t [:,:] im1, UINT32_t [:,:] im2, UINT64_t k1, UINT64_t k2):

    assert k1 <= k2
    cdef int N = im1.base.shape[0]
    assert im2.base.shape[0] == N
    cdef int L1 = im1.base.shape[1]
    cdef int L2 = im2.base.shape[1]

    # store k-mer counts here
    cdef FLOAT32_t [:,:] overlaps = np.zeros((4**k1, 4**k2), dtype = np.float32)

    # helper variables to tell cython the types
    cdef UINT64_t index1, index2, i, j, l, ofs
    
    # because we start inside the 5' adapter,
    # k2-mer indices start earlier in the sequence if k2 > k1.
    ofs = k2 - k1
    
    with nogil:
        for j in range(N):
            for i in range(L1):
                index1 = im1[j,i]
                for l in range(max(0, i - k1 + ofs), min(L2, i + k1 + ofs)):
                    index2 = im2[j,l]
                    overlaps[index1, index2] += 1
                    
    return overlaps.base




#def seq_set_kmer_flag(UINT8_t [:,:] seq_matrix, UINT64_t k, UINT64_t kmer_index):
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1

    #cdef UINT64_t N = seq_matrix.base.shape[0]
    #cdef UINT64_t L = seq_matrix.base.shape[1]
    #cdef UINT64_t l = L - k + 1

    ## store k-mer counts here
    #cdef UINT8_t [::1] flags = np.zeros(N, dtype = np.uint8)

    ## helper variables to tell cython the types
    #cdef UINT8_t s
    #cdef UINT64_t index, i, j
    
    #with nogil:
        #for j in range(N):
            #index = 0
            #for i in range(k-1):
                #index += seq_matrix[j, i] << 2 * (k - i - 2)
            
            #for i in range(0, l):
                ## get next "letter"
                #s = seq_matrix[j, i+k-1]
                
                ## compute next index from previous by shift + next letter
                #index = ((index << 2) | s ) & MAX_INDEX
                
                ## count
                #if index == kmer_index:
                    #flags[index] += 1
            
    #return flags.base



def seq_matrix_to_index_matrix(UINT8_t [:,:] seq_matrix, UINT64_t k, UINT8_t [:] adap5, UINT8_t [:] adap3):
    """
    Converts a matrix of nucleotide values (0..3 instead of ACGT) into
    a matrix of kmer index values. (AAA -> 0, AAC -> 1, ..., TTT -> 63).
    """
    #assert k <= 8 # for k > 8 indices do not fit into UINT16 anymore!

    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef UINT64_t l = L - k + 1
    cdef UINT64_t Lt = L + k - 1

    # store k-mer indices here
    cdef UINT32_t [:,:] indices = np.zeros( (N, Lt), dtype=np.uint32)

    # helper variables to tell cython the types
    cdef UINT64_t i, j
    cdef UINT32_t index, s, a5_index

    # largest kmer index 
    cdef UINT16_t MAX_INDEX = 4**k - 1

    a5_index = 0
    for i in range(k-1):
        s = adap5[i]
        a5_index = ((a5_index << 2) | s ) & MAX_INDEX
        
    with nogil, parallel(num_threads=8):
        for j in prange(N):
            index = 0 # make thread-local
            index = a5_index # first kmer overlaps (k-1) with 5'adapter
            for i in range(0, L):
                # get next "letter"
                s = seq_matrix[j, i]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                indices[j, i] = index
            
            for i in range(k-1): # last kmers read into 3'adapter
                s = adap3[i]
                index = ((index << 2) | s ) & MAX_INDEX
                indices[j, L+i] = index
                
    return indices.base


def index_matrix_rows_with_kmer(UINT32_t [:,:] index_matrix, UINT64_t k, UINT32_t kmer_index, int n_threads=8):
    """
    searches for sequences that contain the desired kmer at least once.
    returns a vector with row-indices into the index-matrix with hits.
    Used to speed up updating of the thermodynamic model by restricting
    updates to the sequences that actually change their contribution.
    """
    #assert k <= 8 # for k > 8 indices do not fit into UINT16 anymore!

    cdef UINT64_t N = len(index_matrix)
    cdef UINT64_t l = len(index_matrix[0])

    # store row indices here.
    cdef UINT32_t [:,:] row_indices = np.zeros((n_threads, N), dtype=np.uint32)
    cdef UINT32_t [:] n_hits = np.zeros(n_threads, dtype=np.uint32)
    
    # helper variables to tell cython the types
    cdef UINT64_t i, j, t
    cdef UINT32_t index, s

    with nogil, parallel(num_threads=8):
        for j in prange(N):
            t = openmp.omp_get_thread_num()
            for i in range(0, l):
                if index_matrix[j,i] == kmer_index:
                    row_indices[t, n_hits[t]] = j
                    n_hits[t] += 1
                    break

    # merge results
    res = [row_indices.base[t,:n_hits[t]] for t in range(n_threads)]
    return np.concatenate(res)


def kmer_counts_acc_weighted(UINT32_t [:,:] index_matrix, FLOAT32_t [:,:] acc_matrix, UINT64_t k, int n_max=0, int openen_ofs=0, int num_threads=8):
    assert k <= 16 # must fit into UINT32 kmer-indices!

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]
    
    # result will be stored here
    cdef FLOAT32_t [:] weights = np.zeros(4**k, dtype=np.float32)

    # helper variables to tell cython the types
    cdef int thread_num
    cdef FLOAT32_t a=0
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index=0

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='static'):
            thread_num = openmp.omp_get_thread_num()
            # iterate over all k-mers
            for i in range(0, l):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                a = acc_matrix[j, i + openen_ofs]
                weights[index] += a
            
    return weights.base


def collect_kmer_acc(UINT32_t [:,:] index_matrix, FLOAT32_t [:,:] acc_matrix, UINT64_t kmer_i, int ofs):
    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]
    
    # result will be stored here
    cdef FLOAT32_t [:] accs = np.empty(N, dtype=np.float32)
    cdef UINT64_t n_max = N - 1
    cdef UINT64_t n_acc = 0

    # helper variables to tell cython the types
    cdef int thread_num
    cdef FLOAT32_t a=0
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index=0

    # with nogil:
    for j in range(N):
        for i in range(l):
            if index_matrix[j, i] == kmer_i:
                accs[n_acc] = acc_matrix[j, i + ofs]
                if n_acc < n_max:
                    # WARNING! Upon overflow we are overwriting already seen data
                    n_acc += 1

    return accs.base[:n_acc]


    # with nogil, parallel():
    #     for j in prange(N, schedule='static'):
    #         thread_num = openmp.omp_get_thread_num()
    #         # iterate over all k-mers
    #         for i in range(0, l):
    #             # assigned variables are thread-local
    #             index = index_matrix[j, i]
    #             a = acc_matrix[j, i + openen_ofs]
    #             weights[index] += a
            
    # return weights.base





def index_matrix_kmer_counts(UINT32_t [:,:] index_matrix, UINT64_t k, int n_threads = 8):
    assert k <= 16 # must fit into UINT32 kmer-indices!
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]

    # store weighted k-mer counts here (for each thread)
    cdef UINT32_t [:,:] counts = np.zeros((n_threads, 4**k), dtype = np.uint32)

    # helper variables to tell cython the types
    cdef int thread_num
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index

    with nogil, parallel(num_threads=8):
        for j in prange(N, schedule='static'):
            thread_num = openmp.omp_get_thread_num()

            for i in range(0, l):
                index = index_matrix[j, i]
                counts[thread_num, index] += 1
            
    return counts.base.sum(axis=0)


def extrapolate_kmer_freqs(UINT64_t k, FLOAT32_t [:] init, FLOAT32_t [:,:] p_transition, UINT64_t level=2):
    
    cdef UINT64_t i, j, c, n
    cdef UINT64_t Nk = 4**k
    cdef UINT64_t l = level-1
    cdef UINT64_t shift_init = (k - l) * 2
    cdef UINT64_t MAX_TRANS = 4**l - 1
    
    cdef FLOAT32_t [:] freqs = np.empty(Nk, dtype=np.float32)
    
    cdef FLOAT32_t f, q
    cdef int special = seq_to_index('aucc')
    
    for i in range(Nk):
        if i == special:
            print index_to_seq(i,k)
        j = shift_init
        c = i >> j # current l-1 mer
        f = init[c]
        if i == special:
            print "  init", index_to_seq(c,l), f
            
        for m in range(k-l):
            j = j - 2
            n = i >> j # next nucleotide
            
            q = p_transition[c & MAX_TRANS, n & 3]
            f *= q # += q
            if i == special:
                print "  transition", index_to_seq(c & MAX_TRANS, l), "to", "ACGU"[n & 3], q, f
            c = n
        
        freqs[i] = f #exp(f)
    
    return freqs.base
    
    





def weighted_kmer_counts(UINT32_t [:,:] index_matrix, FLOAT32_t [:] weights, UINT64_t k, int n_threads = 8):
    assert k <= 16 # must fit into UINT32 kmer-indices!
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]

    # store weighted k-mer counts here (for each thread)
    cdef FLOAT32_t [:,:] counts = np.zeros((n_threads, 4**k), dtype = np.float32)

    # helper variables to tell cython the types
    cdef int thread_num
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index
    cdef FLOAT32_t w=0

    with nogil, parallel(num_threads=8):
        for j in prange(N, schedule='static'):
            thread_num = openmp.omp_get_thread_num()

            w = weights[j]
            for i in range(0, l):
                index = index_matrix[j, i]
                counts[thread_num, index] += w
        
    return counts.base.sum(axis=0)





def kmer_openen_counts(UINT8_t [:,:] seq_matrix, UINT8_t [:,:] openen_matrix, UINT64_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef UINT64_t l = L - k + 1

    # store joint frequencies here
    cdef UINT32_t [:,:] counts = np.zeros((4**k, 256), dtype = np.uint32)
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j
    
    with nogil:
        for j in range(N):
            
            # prepare index from first k-1 positions
            index = 0
            for i in range(k-1):
                index += seq_matrix[j, i] << 2 * (k - i - 2)

            # iterate over all k-mers, always adding next base to index
            for i in range(0, l):
                # get next "letter"
                s = seq_matrix[j, i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                counts[index, openen_matrix[j,i]] += 1
            
    return counts.base



def kmer_acc_counts(reads, UINT64_t k):
    cdef UINT32_t [:,:] im = reads.get_index_matrix(k) #seq_matrix, UINT8_t [:,:] openen_matrix, ):
    cdef UINT64_t N = im.base.shape[0]
    cdef UINT64_t L = im.base.shape[1]
    binned, openen = reads.get_acc_matrix(k, disc_mode='linear')
    cdef UINT8_t [:,:] acc = binned
    # store joint frequencies here
    cdef UINT32_t [:,:] counts = np.zeros((4**k, 256), dtype = np.uint32)
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j
    
    with nogil:
        for j in range(N):
            # iterate over all k-mers i in read j
            for i in range(L):
                index = im[j,i]
                counts[index, acc[j,i]] += 1
            
    return counts.base, openen





def kmer_mean_openen_profiles(UINT8_t [:,:] seq_matrix, UINT8_t [:,:] openen_matrix, FLOAT32_t [:] openen_lookup, int k_seq, int k_openen, int ofs):
    """
    Compute the mean open-energy levels relative to the 
    position of the kmer, for all kmers.
    
    returns a (4^k_seq, 2*(L-k_openen)+1) shaped array with mean 
    open-energies. If a kmer occurs multiple times in a read
    the open-energies will be counted multiple times 
    (into different position).
    """
    
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX_SEQ = 4**k_seq - 1
    cdef UINT64_t MAX_INDEX_OE = 4**k_openen - 1
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef int l_seq = L - k_seq + 1
    cdef int l_openen = L - k_openen + 1

    # store observations here to compute means upon exit
    cdef UINT32_t [:,:,:] counts = np.zeros((4**k_seq, 2*l_openen+1, 256), dtype = np.uint32)
    #cdef FLOAT32_t [:,:] sums = np.zeros((4**k_seq, 2*l_openen+1), dtype = np.float32)
    #cdef FLOAT32_t [:] openens = np.zeros(l_openen, dtype=np.float32)
    cdef UINT8_t [:] openens = np.zeros(l_openen, dtype=np.uint8)
    cdef UINT64_t [:] indices = np.zeros(l_seq, dtype=np.uint64)
    
    # helper variables to tell cython the tqypes
    cdef UINT8_t s
    cdef int index_seq, index_oe, index, i, j, pos, x, m
    
    #with gil:
    for j in range(N):
        # iterate over all k-mers, always adding next base to index
        index = 0
        for i in range(L):
            s = seq_matrix[j, i]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s )
            
            if i >= k_seq-1:
                index_seq = index & MAX_INDEX_SEQ
                indices[i-k_seq+1] = index_seq
                
            if i >= k_openen-1:
                index_oe = index & MAX_INDEX_OE
                #openens[i-k_openen+1] = openen_lookup[openen_matrix[j,i-k_openen+1]]
                openens[i-k_openen+1] = openen_matrix[j,i-k_openen+1+ofs]

        for i in range(0,l_seq):
            index = indices[i]
            #print i, index, range(-i, l-i)
            for m in range(0, l_openen):
                x = m - i
                pos = l_openen + x
                #print x, pos
                #if index == 0b1001001110:
                    #print "i={0}, x={1}, pos={2}, openens[x+i] = {3}, sums[index, pos] = {4}, counts[index,pos]={5}".format(i, x, pos, openens[x+i], sums[index, pos], counts[index, pos])

                #sums[index, pos] += openens[m]
                counts[index, pos, openens[m]] += 1
            
    return counts.base


def kmer_openen_profile(UINT32_t [:,:] index_matrix, UINT8_t [:,:] openen_matrix, int k_seq, UINT32_t kmer_index, int k_openen, int ofs):
    """
    Compute the mean open-energy levels relative to the 
    position of the kmer, for all kmers.
    
    returns a (4^k_seq, 2*(L-k_openen)+1) shaped array with mean 
    open-energies. If a kmer occurs multiple times in a read
    the open-energies will be counted multiple times 
    (into different position).
    """
    
    # print k_seq, k_openen
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX_SEQ = 4**k_seq - 1
    cdef UINT64_t MAX_INDEX_OE = 4**k_openen - 1
    cdef UINT64_t N = len(index_matrix)
    cdef UINT64_t l_seq = len(index_matrix[0])
    cdef int l_openen = l_seq + k_seq - k_openen

    # store observations here to compute means upon exit
    cdef UINT32_t [:,:] counts = np.zeros((2*l_openen+1, 256), dtype = np.uint32)
    #cdef FLOAT32_t [:,:] sums = np.zeros((4**k_seq, 2*l_openen+1), dtype = np.float32)
    #cdef FLOAT32_t [:] openens = np.zeros(l_openen, dtype=np.float32)
    cdef UINT8_t [:] openens = np.zeros(l_openen, dtype=np.uint8)
    cdef UINT64_t [:] indices = np.zeros(l_seq, dtype=np.uint64)
    
    # helper variables to tell cython the tqypes
    cdef UINT8_t s
    cdef int index_seq, index_oe, index, i, j, pos, x, m
    
    #with gil:
    for j in range(N):
        # iterate over all k-mers, always adding next base to index
        #index = 0
        for i in range(l_seq):
            index = index_matrix[j, i]
            if index != kmer_index:
                continue
            
            for m in range(l_openen):
                pos = l_openen + m - i
                counts[pos, openen_matrix[j, m + ofs]] += 1

    return counts.base




def kmer_openen_profiles(UINT32_t [:,:] index_matrix, UINT8_t [:,:] openen_matrix, int k_seq, int k_openen, int ofs):
    """
    Compute the mean open-energy levels relative to the 
    position of the kmer, for all kmers.
    
    returns a (4^k_seq, 2*(L-k_openen)+1) shaped array with mean 
    open-energies. If a kmer occurs multiple times in a read
    the open-energies will be counted multiple times 
    (into different position).
    """
    
    # print k_seq, k_openen
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX_SEQ = 4**k_seq - 1
    cdef UINT64_t MAX_INDEX_OE = 4**k_openen - 1
    cdef UINT64_t N = len(index_matrix)
    cdef UINT64_t l_seq = len(index_matrix[0])
    cdef int l_openen = l_seq + k_seq - k_openen

    # store observations here to compute means upon exit
    cdef UINT32_t [:,:,:] counts = np.zeros((4**k_seq, 2*l_openen+1, 256), dtype = np.uint32)
    #cdef FLOAT32_t [:,:] sums = np.zeros((4**k_seq, 2*l_openen+1), dtype = np.float32)
    #cdef FLOAT32_t [:] openens = np.zeros(l_openen, dtype=np.float32)
    cdef UINT8_t [:] openens = np.zeros(l_openen, dtype=np.uint8)
    cdef UINT64_t [:] indices = np.zeros(l_seq, dtype=np.uint64)
    
    # helper variables to tell cython the tqypes
    cdef UINT8_t s
    cdef int index_seq, index_oe, index, i, j, pos, x, m
    
    with nogil:
        for j in range(N):
            # iterate over all k-mers, always adding next base to index
            #index = 0
            for i in range(l_seq):
                index = index_matrix[j, i]
                for m in range(l_openen):
                    pos = l_openen + m - i
                    counts[index, pos, openen_matrix[j, m + ofs]] += 1
           
    return counts.base



def joint_kmer_profiles(UINT8_t [:,:] seq_matrix, int k_core, int k_flank, int pseudo=1):
    """
    Compute the co-occurrence frequency of k_flank mers relative to the 
    position of k_core mers, for all k_core mers.
    
    returns a (4^k_core, 2*(L-k_flank)+1) shaped array with frequencies.
    If a kmer occurs multiple times in a read it will be counted multiple 
    times (into different position).
    """
    
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX_CORE = 4**k_core - 1
    cdef UINT64_t MAX_INDEX_FLANK = 4**k_flank - 1
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef int l_core = L - k_core + 1
    cdef int l_flank = L - k_flank + 1
    cdef int l_min = min(l_core, l_flank)
    cdef int l_max = max(l_core, l_flank)

    # store observations here to compute means upon exit
    cdef FLOAT32_t [:,:,::1] freqs = np.ones((l_core + l_flank, 4**k_core, 4**k_flank), dtype = np.float32) * pseudo
    cdef UINT64_t [::1] indices_flank = np.zeros(l_flank, dtype=np.uint64)
    cdef UINT64_t [::1] indices_core = np.zeros(l_core, dtype=np.uint64)
    
    # helper variables to tell cython the tqypes
    cdef UINT8_t s
    cdef int index_core, index_flank, index, i, j, pos, x, m
    
    #with gil:
    for j in range(N):
        # iterate over all k-mers, always adding next base to index
        index = 0
        for i in range(L):
            s = seq_matrix[j, i]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s )
            
            if i >= k_core-1:
                index_core = index & MAX_INDEX_CORE
                indices_core[i-k_core+1] = index_core
                
            if i >= k_flank-1:
                index_flank = index & MAX_INDEX_FLANK
                indices_flank[i-k_flank+1] = index_flank

        for i in range(0,l_core):
            index_core = indices_core[i]
            #print i, index, range(-i, l-i)
            for m in range(0, l_flank):
                x = m - i
                pos = l_core + x
                #print x, pos
                #if index == 0b1001001110:
                    #print "i={0}, x={1}, pos={2}, openens[x+i] = {3}, sums[index, pos] = {4}, counts[index,pos]={5}".format(i, x, pos, openens[x+i], sums[index, pos], counts[index, pos])

                index_flank = indices_flank[m]
                #print i, m, x, index_core, pos, index_flank
                freqs[pos, index_core, index_flank] += 1
                #counts[index_core, pos] += 1
            
    return freqs.base #/ counts.base[:,:,np.newaxis]

def seq_set_SKA(np.ndarray[UINT8_t, ndim=2] seq_matrix, np.ndarray[FLOAT32_t] _weights, np.ndarray[FLOAT32_t] _background, UINT32_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])

    # store k-mer indices here
    _mer_indices = np.zeros(L-k+1, dtype=np.uint32)
    
    # store current k-mer weights here
    _mer_weights = np.zeros(L-k+1, dtype=np.float32)
    
    
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] mer_indices = _mer_indices
    cdef FLOAT32_t [::1] mer_weights = _mer_weights
    cdef FLOAT32_t [::1] weights = _weights
    cdef FLOAT32_t [::1] background = _background

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef int index, i, j, ofs
    cdef FLOAT32_t w=0, total_w=0, current_weights_sum = 0, weights_sum = MAX_INDEX+1, Z=0
    
    current_weights_sum = _weights.sum()
    Z = weights_sum / current_weights_sum
    
    #with nogil, parallel(num_threads=8):
        #for j in prange(N):
    with nogil:
        for j in range(N):
            ofs = j*L

            # compute index of first k-1 mer by bit-shifts
            index = 0
            for i in range(k-1):
                index += _seq_matrix[ofs+i] << 2 * (k - i - 2)
            
            total_w = 0
            
            # iterate over k-mers
            for i in range(0, L-k+1):
                # get next "letter"
                s = _seq_matrix[ofs+i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                mer_indices[i] = index
                w = weights[index] / background[index] * Z
                mer_weights[i] = w
                total_w += w

            # update weights
            for i in range(0, L-k+1):
                weights[mer_indices[i]] += mer_weights[i]/total_w

            current_weights_sum += 1
            Z = weights_sum / current_weights_sum

    # normalize such that all weights sum up to 4**k
    _weights *= Z
    return _weights


def store_pure_reads(
        out_file,
        UINT64_t k,
        np.ndarray[UINT8_t, ndim=2] seq_matrix, 
        np.ndarray[INT32_t, ndim=1] _flags, # one entry for each read. 
        np.ndarray[UINT32_t, ndim=1] _indices, # one entry for each read.
        UINT32_t n_sample,
    ):
    
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])
    cdef UINT32_t l = L-k+1

    cdef np.ndarray[UINT32_t] _n = np.zeros(4**k, dtype=np.uint32)
    cdef np.ndarray[UINT8_t] _seq = np.zeros(L, dtype=np.uint8)
    cdef np.ndarray[UINT8_t] _kmer = np.zeros(k, dtype=np.uint8)
    
    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq = _seq
    cdef UINT8_t [::1] kmer = _kmer
    cdef INT32_t [::1] flags = _flags
    cdef UINT32_t [::1] indices = _indices
    cdef UINT32_t [::1] n = _n
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t hit_id = 0
    cdef INT32_t flag = 0
    cdef UINT64_t ofs, index, i, j, n_hits=0, hit_index=0
    
    for j in range(N):
        
        flag = flags[j]
        if flag < 1:
            # not pure
            continue

        hit_index = indices[j]
        if n[hit_index] >= n_sample:
            # already enough of these
            continue

        n[hit_index] += 1
        ofs = j*L

        # convert seq entries back to string
        for i in range(L):
            seq[i] = bits_to_letters[ _seq_matrix[ofs+i] ]

        # convert hit index back to kmer
        for i in range(k):
            s = hit_index >> ((k - i-1) * 2)
            kmer[i] = bits_to_letters[s & 3]
        
        out_file.write(">{0} | r={1} | n={2}\n{3}\n".format(_kmer.tobytes(), flag, n[hit_index], _seq.tobytes()) )


def count_reads_with_kmers(UINT32_t [:,:] index_matrix, UINT64_t k):
    cdef UINT32_t N = len(index_matrix)
    cdef UINT32_t L = len(index_matrix[0])

    # count reads containing a given kmer, for all kmers
    cdef UINT32_t [:] hit_counts = np.zeros(4**k ,dtype=np.uint32)
    # keep distinct kmer indices from each read here
    cdef UINT32_t [:] dindices = np.zeros(L ,dtype=np.uint32)
    
    # helper variables to tell cython the types
    cdef UINT64_t index, i, j, m, n_distinct=0, append=1
    
    with nogil:
        for j in range(N):
            n_distinct = 0
            # iterate over all k-mers in the read
            for i in range(L):
                index = index_matrix[j,i]

                # make sure we do not have this index already
                append = 1
                for m in range(n_distinct):
                    if index == dindices[m]:
                        append = 0
                
                # it's a new index
                if append:
                    dindices[n_distinct] = index
                    n_distinct += 1
                    # record the read for each of the contained kmers *once*
                    hit_counts[index] += 1
            

    return hit_counts.base




def count_reads_with_kmap_hit(UINT32_t [:,:] index_matrix, UINT8_t [:] kmap):
    cdef UINT32_t N = len(index_matrix)
    cdef UINT32_t L = len(index_matrix[0])

    # helper variables to tell cython the types
    cdef UINT64_t index, i, j, hit=0, n_reads=0
    
    with nogil:
        for j in range(N):
            hit = 0
            # iterate over all k-mers in the read
            for i in range(L):
                index = index_matrix[j,i]
                hit += kmap[index]

            n_reads += (hit > 0)

    return n_reads




def joint_freq_at_distance(UINT32_t [:,:] index_matrix, UINT64_t k):
    cdef UINT32_t N = len(index_matrix)
    cdef UINT32_t L = len(index_matrix[0])
    cdef UINT32_t Nk = 4**k

    cdef UINT32_t [:,:,:] joint = np.zeros((Nk, Nk, L-k) ,dtype=np.uint32)

    # helper variables to tell cython the types
    cdef UINT64_t index_A, index_B, i, j, d, hit=0, n_reads=0
    
    
    # with nogil:
    for j in range(N):
        hit = 0
        # iterate over all k-mers in the read
        for i in range(L-k):
            index_A = index_matrix[j,i]
            for d in range(k, L-i):
                
                index_B = index_matrix[j,i+d]
                if d < k:
                    print j,i,k,d
                assert d-k >= 0
                assert d-k < L-k
                assert index_A < Nk
                assert index_B < Nk
                joint[index_A, index_B, d-k] += 1

    return joint.base




def kmer_count_pos_per_read(UINT8_t [:,:] seq_matrix, UINT64_t kmer_index, UINT8_t k):
    """
    Examine each read for occurrences of the indicated kmer. Record the number of
    occurrences, as well as the the index of the last occurrence, for each read.
    """
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(seq_matrix.base)
    cdef UINT32_t L = len(seq_matrix.base[0])
    cdef UINT32_t l = L-k+1

    # count occurrences of the kmer, for each read
    cdef UINT8_t [:] counts = np.zeros(N, dtype=np.uint8)
    # keep position of first hit
    cdef UINT8_t [:] last_pos = np.zeros(N, dtype=np.uint8)
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j, m
    
    with nogil:
        for j in range(N):
            # compute index of first k-1 mer by bit-shiftkmer_profile(self.seqm, k)s
            index = 0
            for i in range(k-1):
                index += seq_matrix[j,i] << 2 * (k - i - 2)

            # iterate over remaining k-mers
            for i in range(0, l):
                # get next "letter"
                s = seq_matrix[j,i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                
                if index == kmer_index:
                    counts[j] += 1
                    last_pos[j] = i

    return counts.base, last_pos.base



def digitize_32fp_8bit(np.ndarray[FLOAT32_t, ndim=2] data, FLOAT32_t [:] bins):
    cdef UINT64_t n = len(bins.base)
    assert n <= 257
    cdef UINT64_t N = len(data)
    cdef UINT64_t L = len(data[0])
    cdef FLOAT32_t [:] flat = data.flatten()
    cdef UINT8_t [:] res = np.zeros(N*L, dtype=np.uint8)

    cdef UINT64_t i,j,m, pivot
    cdef FLOAT32_t x
    #res = np.zeros(data.size, dtype=int)
    with nogil, parallel(num_threads=8):
        for m in prange(N*L):
            x = flat[m]
            i = 0
            j = n-1
            #print "value", x
            #c = 0
            while j - i > 1:
                pivot = max(1, (j - i)/2 ) + i
                #print pivot, bins[pivot]
                #print "indices",i,j, pivot
                #print "values",'?', bins[i], bins[j], bins[pivot]
                
                if x >= bins[pivot]:
                    i = pivot
                else:
                    j = pivot
                #c += 1
                
            if x >= bins[j]:
                res[m] = j
            else:
                res[m] = i

            #print x,"->", res[n], "in {0} steps".format(c)
            #steps.append(c)
            
    #print np.array(steps).mean(), "average steps"
    return np.reshape(res.base, (N,L)) + 1


def aggregate_binned_profiles(UINT8_t [:,:] bin_matrix, UINT8_t [:] pos, UINT8_t upstream, UINT8_t downstream):
    """
    Examine each read for occurrences of the indicated kmer. Record the number of
    occurrences, as well as the the index of the last occurrence, for each read.
    """
    
    cdef UINT64_t N = len(bin_matrix.base)
    cdef UINT64_t L = len(bin_matrix.base[0])

    cdef UINT64_t l = downstream + upstream + 1
    cdef UINT64_t rightmost = L - downstream - 1
    cdef UINT64_t leftmost = upstream

    ## count bin occupancies for downstream and upstream positions relative to the hit pos
    cdef UINT32_t [:,:] profile = np.zeros((l,2**8), dtype=np.uint32)
    
    ## keep position of first hit
    #cdef UINT8_t [:] last_pos = np.zeros(N, dtype=np.uint8)
    
    ## helper variables to tell cython the types
    cdef UINT8_t o
    cdef UINT64_t i, j, m
    
    with nogil:
        for j in range(N):
            i = pos[j]
            if i < leftmost or i > rightmost:
                # does not fit
                continue

            for m in range(l):
                # fetch the bin-value of the position rel to hit
                o = bin_matrix[j, i - leftmost + m] 
                # and record
                profile[m,o] += 1
                
    return profile.base



def kmer_profiles(np.ndarray[UINT8_t, ndim=2] _seq_matrix, UINT64_t k):
    """
    count the occurrences of each kmer at each position from 0-L-k+1 
    across all reads.
    input: 
      _seq_matrix NxL UINT8_t array of all reads
      k : kmer size
    
    returns:
      4^k x (L-k+1) array of kmer counts along read positions (profiles)
    
    """
    #print "kmer_profiles"
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    cdef UINT32_t N = len(_seq_matrix)
    cdef UINT32_t L = len(_seq_matrix[0])
    cdef UINT32_t l = L-k+1 # maximum spacing

    # count reads containing a given kmer, for all kmers
    cdef np.ndarray[UINT32_t, ndim=2] _profiles = np.zeros( (4**k, l) ,dtype=np.uint32)

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    
    cdef UINT8_t [::1] seqm = _seq_matrix.flatten()
    cdef UINT32_t [:,:] profiles = _profiles
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef int ofs, index, i, j
    
    #with gil:
    for j in range(N):
        ofs = j*L
        
        # compute index of first k-1 mer by bit-shifts
        index = 0
        for i in range(k-1):
            index += seqm[ofs+i] << 2 * (k - i - 2)

        # iterate over k-mers
        for i in range(0, l):
            # get next "letter"
            s = seqm[ofs+i+k-1]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            # count occurrence
            profiles[index][i] += 1
            #print ">", index, index*l + 1, profiles[index*l + i]

    #print "outof",_profiles.sum()
    return _profiles


def kmer_cooccurrence_distance_tensor(np.ndarray[UINT8_t, ndim=2] _seq_matrix, np.ndarray[UINT64_t, ndim=1] _kmer_lookup, UINT64_t k, UINT64_t n_kmers):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(_seq_matrix)
    cdef UINT32_t L = len(_seq_matrix[0])
    cdef UINT32_t l = L-k+1 # maximum spacing

    # count reads containing a given kmer, for all kmers
    cdef np.ndarray[UINT32_t, ndim=3] _tensor = np.zeros( (n_kmers, n_kmers, l) ,dtype=np.uint32)

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    
    cdef UINT8_t [::1] seq_matrix = _seq_matrix.flatten()
    cdef UINT32_t [:,:,:] tensor = _tensor
    cdef UINT64_t [::1] kmer_lookup = _kmer_lookup.flatten()
    
    # running variables
    cdef UINT64_t [::1] encountered_vec = np.zeros(l, dtype=np.uint64)
    cdef UINT64_t [::1] spacing_vec = np.zeros(l, dtype=np.uint64)
    cdef UINT64_t n_tracing = 0
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef int ofs, index, i, j, m, kmer_i=0, kmer_j=0, spacing=0, kmer_hit=0, n2=n_kmers*n_kmers
    
    #with gil:
    for j in range(N):
        ofs = j*L
        
        # compute index of first k-1 mer by bit-shifts
        index = 0
        for i in range(k-1):
            index += seq_matrix[ofs+i] << 2 * (k - i - 2)

        kmer_i = 0
        kmer_j = 0
        
        n_tracing = 0
        # iterate over remaining k-mers
        for i in range(0, l):
            # get next "letter"
            s = seq_matrix[ofs+i+k-1]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            
            kmer_hit = kmer_lookup[index]
            
            if kmer_hit:
                for m in range(n_tracing):
                    # record occurrence relative to previously encountered kmers
                    kmer_i = encountered_vec[m]
                    kmer_j = kmer_hit
                    spacing = spacing_vec[m]
                    tensor[kmer_i-1, kmer_j-1, spacing] += 1
                    
                # record as new hit
                encountered_vec[n_tracing] = kmer_hit
                spacing_vec[n_tracing] = 0
                n_tracing += 1
            
            # and update all spacings
            for m in range(n_tracing):
                spacing_vec[m] += 1
                
                           
    return _tensor
      
    

def kmer_flank_profiles(np.ndarray[UINT8_t, ndim=2] seq_matrix, str kmer, int k_flank=3):
    cdef UINT64_t k = len(kmer)
    
    # number of bits to shift right to convert k-mer index to k_flank-mer index
    cdef UINT64_t k_diff_bits = (k-k_flank)*2 

    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1
    # the index we are looking for
    cdef UINT64_t k_index = seq_to_index(kmer)
    
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef UINT64_t l = L-k+1
    
    # aggregate flanking kmer counts - at each relative position - here
    _profile = np.zeros( ((4**k_flank) * 2*l), dtype=np.uint32)

    # store k-mer hits here
    _mask = np.zeros(N * l, dtype = np.uint8)

    # remember the hit positions
    _hit_pos = np.zeros(l, dtype = np.uint64)
    
    # buffer
    _indices = np.zeros(l, dtype = np.uint64)
    
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT8_t [::1] mask = _mask
    cdef UINT32_t [::1] profile = _profile
    cdef UINT64_t [::1] indices = _indices
    cdef UINT64_t [::1] hit_pos = _hit_pos

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, findex, i, j, r, o, q
    cdef UINT64_t hits = 0

    with nogil:#, parallel(num_threads=8):
        for j in range(N):
            seq_bits = _seq_matrix[j*L:(j+1)*L]
            # compute index of first k-1-mer by bit-shifts
            index = kbits_to_index(seq_bits, k-1) 
            # iterate over remaining k-mers
            hits = 0
            #print list(_seq_matrix[j*L:(j+1)*L])
            for i in range(0, l):
                # get next "letter"
                s = seq_bits[i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                indices[i] = index
                #print i, index, index_to_seq(index, k), _indices[i]
                if index == k_index:
                    mask[j*l+i] = 1
                    hit_pos[hits] = i
                    hits += 1

            #print "pos 0 index", _indices[0]
            for q in range(0, hits):
                o = hit_pos[q]
                #print "hit at",o
                for i in range(0,l):
                    # compute shorter kmer index from longer ones
                    index = indices[i]
                    findex = index >> k_diff_bits
                    #print index, index_to_seq(index, k), index_to_seq(findex, k_flank), k_flank
                    r = findex*2*l + i-o+l-1
                    profile[r] += 1

    return _profile.reshape( (4**k_flank, 2*l) ), _mask.reshape( (N, l) )

def acc_scale_Z1(FLOAT32_t [:, :] Z1, FLOAT32_t [:, :] lacc, FLOAT32_t a, int ofs=0):
    cdef UINT64_t N = Z1.base.shape[0]
    cdef UINT64_t L = Z1.base.shape[1]
    cdef FLOAT32_t [:, :] Z1_scaled = np.empty(Z1.base.shape, dtype=np.float32)
    cdef int j, i

    # for j in prange(N, schedule='static', nogil=True, num_threads=8):
    for j in range(N):
        for i in range(L):
            Z1_scaled[j, i] = Z1[j, i] * exp(lacc[j, i + ofs] * a)

    return Z1_scaled.base

def acc_footprints(FLOAT32_t [:, :] Z1, FLOAT32_t [:,:] acc, int w, int k, int ofs=0, int pad=5, row_w=None, int n_threads = 1):
    cdef UINT64_t N = Z1.base.shape[0]
    cdef UINT64_t L = Z1.base.shape[1]
    cdef int tid=-1
    cdef int l = w + 2 * pad
    cdef FLOAT32_t [:,:] rw
    cdef int n_cols=0, col=0
    if row_w is None:
        n_cols = 1
        rw = np.ones((N, 1), dtype=np.float32)
    else:
        n_cols = row_w.shape[1]
        rw = row_w

    # print "n_cols=", n_cols
    cdef FLOAT32_t [:,:,:] footprint = np.zeros((n_threads, n_cols, (l % 64 + 1) * 64), dtype=np.float32)
    cdef FLOAT32_t fp = 0.
    cdef FLOAT32_t [:,:] Z = np.zeros((n_threads, n_cols), dtype=np.float32)

    cdef int j,x
    cdef int d,x0,x1,x2,x3
    cdef FLOAT32_t f0,f1,f2,f3
    # cdef FLOAT32_t weight=1.
    cdef int ofsx = 0
    cdef FLOAT32_t *Z_row
    cdef FLOAT32_t *Z1_row
    cdef FLOAT32_t *acc_row
    cdef FLOAT32_t *rw_row
    cdef FLOAT32_t Z1x = 0.
    # for j in prange(N, schedule='static')
    if n_threads==1:
        # print "single threaded"
        with nogil:
            tid = 0
            for j in range(N):
                Z_row = &Z[tid, 0]
                for x in range(pad, L-pad):
                    ofsx = ofs + x
                    Z1_row = &Z1[j, x]
                    Z1x = Z1_row[0]
                    rw_row = &rw[j, 0]
                    for col in range(n_cols):
                        Z_row[col] += Z1x * rw_row[col]

                    acc_row = &acc[j, ofsx]
                    for d in range(-pad, w + pad):
                        f0 = Z1x * acc_row[d]
                        for col in range(n_cols):
                            fp = footprint[tid, col, d + pad]
                            footprint[tid, col, d + pad] = fp + f0 * rw_row[col]

    else:    
        raise ValueError("n_threads > 1 no longer supported!")
    #     for j in prange(N, schedule='dynamic', nogil=True, num_threads=3):
    #         tid = openmp.omp_get_thread_num()
    #     # n_threads = 1
    #     # for j in range(N):
    #     #     tid = 0
    #     # for x in range(pad, L-pad):
    #     #     for d in range(-pad, w+pad):
    #     #         footprint[d+pad] += Z1[j, x] * acc[j, ofs + x + d]
    #         Z_row = &Z[tid, 0]
    #         for x in range(pad, L-pad):
    #             ofsx = ofs + x
    #             Z1_row = &Z1[j, x]
    #             Z1x = Z1_row[0]
    #             for col in range(n_cols):
    #                 Z_row[col] += Z1x * rw[col, j]

    #             acc_row = &acc[j, ofsx]
    #             for d in range(-pad, w + pad):
    #                 f0 = Z1x * acc_row[d]
    #                 for col in range(n_cols):
    #                     fp = footprint[tid, col, d + pad]
    #                     footprint[tid, col, d + pad] = fp + f0 * rw[col, j]

    # collect data from all threads
    for tid in range(1, n_threads):
        for col in range(n_cols):
            for d in range(w + 2 * pad):
                footprint[0, col, d] += footprint[tid, col, d]
            Z[0, col] += Z[tid, col]

    return footprint.base[0,:,:l] / Z.base[0, :, np.newaxis]
