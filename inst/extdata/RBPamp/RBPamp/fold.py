#!/usr/bin/env python
# coding=future_fstrings
from __future__ import print_function

__license__ = "MIT"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import sys
import os
import logging
import copy
import time
import numpy as np
import pickle as pickle
from subprocess import PIPE, Popen
from multiprocessing import Process, Event, JoinableQueue as Queue
import multiprocessing
from collections import defaultdict
from RBPamp.caching import CachedBase, cached, pickled
import RBPamp.cyska as cyska
logger = logging.getLogger("fold")

# This global variable is used by the keyboard interrupt 
# handler to decide if we need to flush stuff to disk
folding_in_progress = False
        
class RBNSOpenen(CachedBase):
    """
    Analogous to RBNSReads, which holds the raw sequences, instances of this class hold 
    open-energies for all kmer start-positions inside the raw sequences. 
    Since one file with reads produces multiple openen files (one for each k) and
    these furthermore may exist in raw or discretized form, an OpenenStorage instance
    should be used to encapsulate transparent access to the underlying 
    files.
    """
    def __init__(self, fname, rbns_reads, k, oem=[], disc=None, dummy=False, acc_scale=1., **kwargs):

        CachedBase.__init__(self, **kwargs)

        self.fname = fname
        self.rbns_reads = rbns_reads
        
        self.k = k
        self.discretized = ("discretized" in self.fname)
        self.T = rbns_reads.temp
        self.RT = rbns_reads.RT
        self.acc_scale = acc_scale
        self.logger = logging.getLogger('fold.RBNSOpenen')
        self.missing_data = False
        
        # to be initialized upon first access to oem
        self.include_adapters = None
        self.ofs = 0
        self.dummy = dummy
        
        if self.discretized:
            # recovering discretization scheme from file-name
            self.disc = OpenenDiscretization.from_filename(fname)
            self.dtype = self.disc.dtype
            self.acc_lookup = np.exp(- self.disc.x/self.RT)
        else:
            # we have the raw floating point values
            self.disc = None
            self.dtype = np.float32
            self.acc_lookup = []

        if len(oem):
            self.cache_preload("oem", oem)
            N, L = oem.shape
            self.cache_preload("N", N)
            self.cache_preload("L", L)
            self.is_subsample = True
        else:
            self.is_subsample = False

        self.logger.debug("initialized for data from '{self.fname}' @{self.T} C".format(self=self))
    
    @classmethod
    def from_array(cls, reads, k, oem, dtype=np.uint8, mode='gamma', **kwargs):
        
        L = oem.shape[1]
        fname = "discretized_{mode}_L{L}_k{k}_{dtype.__name__}".format(**locals())
        openen = cls(fname, reads, k, **kwargs)
        openen._do_not_unpickle = True
        openen._do_not_pickle = True
        N, L = oem.shape
        
        openen.cache_preload("oem", oem)
        openen.cache_preload("N", N)
        openen.cache_preload("L", L)
        
        return openen
        
    @property
    def cache_key(self):
        return "RBNSOpenen({self.rbns_reads.cache_key}) k={self.k} disc={self.disc} nmax={self.rbns_reads.n_max}".format(self=self)

    @property
    @cached
    @pickled
    def N(self):
        N, L = self.oem.shape
        return N

    @property
    @cached
    @pickled
    def L(self):
        N, L = self.oem.shape
        return L
   
    def count_records(self, with_adapter=True):
        """return the number of complete records based on file-size"""
        if not os.path.exists(self.fname):
            return 0

        l = self.rbns_reads.L - self.k + 1
        if with_adapter:
            l += self.rbns_reads.l5 + self.rbns_reads.l3

        itemsize = np.dtype(self.dtype).itemsize
        N_items = os.path.getsize(self.fname) / itemsize
        
        return int(np.floor(N_items / l))

    def check_data(self, with_adapter=True):
        if self.count_records(with_adapter=with_adapter) == self.rbns_reads.N:
            return True

    @property
    # @cached
    def oem(self):
        """
        load and keep all open-energies in memory (optionally discretized)
        """
        self.logger.debug("loading open energies from {self.fname}".format(self=self) )
        #oem = np.fromfile(self.fname, dtype=self.dtype)
        N = self.rbns_reads.N
        l = self.rbns_reads.L - self.k + 1
        l_adap = l + self.rbns_reads.l5 + self.rbns_reads.l3

        oem = None
        if not os.path.exists(self.fname):
            self.logger.warning("file '{}' not found. Assuming accessibility = 1".format(self.fname))
            self.l_row = l_adap
            self.include_adapters = True
            self.ofs = self.rbns_reads.l5
            self.missing_data = True
            return np.zeros( (N, self.l_row), dtype=self.dtype)

        # we need to load from disk
        itemsize = np.dtype(self.dtype).itemsize
        N_items = os.path.getsize(self.fname) / itemsize
        L = N_items / float(N)

        self.logger.debug("open-energy row l={0}".format(L))
        if L == l:
            self.logger.info("data excludes adapters L={0}".format(L))
            self.l_row = l
            self.include_adapters = False
            self.ofs = 0
        
        elif L == l_adap:
            self.logger.info("data covers adapters L={0}".format(L))
            self.l_row = l_adap
            self.include_adapters = True
            self.ofs = self.rbns_reads.l5

        elif L > l_adap:
            n_file = N_items / l_adap
            if self.rbns_reads.n_max:
                pass
                # it's okay that we truncate!
            else:
                self.logger.warning("file contains {n_file} rows (assuming it includes adapters) but only {self.rbns_reads.N} reads are loaded. Truncating!".format(**locals()) )

            self.ofs = self.rbns_reads.l5
            self.include_adapters = True
            self.l_row = l_adap
        else:
            delta = L - ( l + self.rbns_reads.l5 + self.rbns_reads.l3 )
            raise ValueError("size of open energy matrix {L} does not match the reads {self.rbns_reads.L} even when accounting for 5' {self.rbns_reads.l5} and 3' {self.rbns_reads.l3} adapters. Delta = {delta}!".format(**locals()) )
        
        # read the actual data. Only as much as needed!
        import time
        import mmap
        from contextlib import closing
        N_bytes = self.l_row * N * itemsize
        # N_bytes = self.l_adap * N * itemsize
        N_needed = self.l_row * N

        t0 = time.time()
        oem = np.memmap(self.fname, dtype=self.dtype, mode='r')[:N_needed]
        # with open(self.fname, 'rb') as f:
        #     with closing(mmap.mmap(f.fileno(), length=N_bytes, access=mmap.ACCESS_READ)) as m:
        #         oem = np.frombuffer(m, dtype=self.dtype)

        dt = 1000. * (time.time() - t0)
        self.logger.debug("loading {N} rows of open-energy from {self.fname} took {dt:.2f} ms.".format(**locals()))
        oem = oem.reshape( (N, self.l_row) )
        return oem
    
    @property
    # @cached
    def acc(self):
        if self.dummy:
            return np.ones(self.oem.shape, dtype=np.float32)

        if not self.discretized:
            return np.exp(-self.oem*self.acc_scale/self.RT)
        else:
            return self.acc_lookup[self.oem]
        
    def discretize(self, disc=None, dname=None):
        
        if not disc:
            disc = OpenenDiscretization.from_filename(dname)

        self.logger.debug("discretizing {0} using {1}".format(self.fname, disc.to_filename(N=self.rbns_reads.N)) )
        t0 = time.time()
        
        d_oem = disc.discretize(self.oem)
        if not dname:
            path, fname = os.path.split(self.fname)
            dname = os.path.join(path, "{0}.{1}".format(disc.to_filename(N=self.rbns_reads.N), fname) )

        doe = RBNSOpenen(
            dname,
            self.rbns_reads,
            self.k,
            oem = d_oem,
        )
        doe.include_adapters = self.include_adapters
        doe.ofs = self.ofs
        
        dt = time.time() - t0
        self.logger.debug("discretization took {0:.1f} seconds".format(dt) )
        return doe
    
    def integrate(self, integrand, bin_weights):
        assert self.discretized
        return np.trapz(integrand * bin_weights, self.disc.x)

    @cached
    @pickled
    def kmer_openen_counts(self):
        return cyska.kmer_openen_counts(self.rbns_reads.seqm, self.oem, self.k)
        
    @cached
    #@pickled
    def get_kmer_openen_profile(self, kmer):
        kmer_index = cyska.seq_to_index(kmer)
        k_seq = len(kmer)
        return cyska.kmer_openen_profile(self.rbns_reads.get_index_matrix(k_seq), self.oem, k_seq, kmer_index, self.k, self.ofs)
        
    #@cached
    #@pickled
    def kmer_mean_openen_profiles(self, k_seq=None):
        self.logger.debug("computing kmer_mean_openen_profiles..." )
        
        if k_seq == None:
            k_seq = self.k

        seqm = self.rbns_reads.seqm
        oem = self.oem

        t0 = time.time()
        res = cyska.kmer_mean_openen_profiles(seqm, oem, np.array(self.disc.x), k_seq, self.k, self.ofs)
        dt = time.time() - t0
        self.logger.debug("kmer_mean_openen_profiles took {0:.1f} seconds".format(dt) )

        return res

    def store(self):
        self.logger.info("storing open-energies as '{0}'".format(self.fname) )
        self.oem.tofile(self.fname)




class ViennaOpenen(object):
    """
    Wrapper around an RNAplfold_RBPamp (modified RNAplfold) subprocess.
    """
    def __init__(self, k_min=3, k_max=8, temp=22., adap5="gggaguucuacaguccgacgauc", adap3="uggaauucucgggugucaagg", vienna_bin="RNAplfold_RBPamp", l_insert=20, skip_adap=False, **kwargs):
        
        L = len(adap5) + l_insert + len(adap3)
        self.logger = logging.getLogger('fold.ViennaOpenen')
        
        # create the folding sub-process
        cmd=[vienna_bin, "-O", "-u {0}".format(k_max), "-W {0}".format(L), "-L {0}".format(L), "-T {0}".format(temp)]
        self.cmd = " ".join(cmd)
        try:
            self.p = Popen(
                cmd, 
                stdin=PIPE, 
                stdout=PIPE, 
                bufsize=1, 
                close_fds=True
            )
        except FileNotFoundError:
            self.logger.error(f"Can not start the RNAplfold binary '{vienna_bin}'")
            sys.exit(1)
        
        # prepare constant variables needed in batch-processing
        self.k_indices = np.arange(k_min, k_max+1)
        
        # which part of the sequence we are actually interested in
        self.first = 0
        self.last = L
        
        if skip_adap:
            self.first = len(adap5)
            self.last = L - len(adap3)

        self.l_insert = self.last - self.first
        self.krange = np.arange(k_min, k_max+1)
        self.k_min = k_min
        self.k_max = k_max
        
        self.adap5 = adap5
        self.adap3 = adap3
        
        self.n_total = 0
        self.logger.info("initialized for adap5='{self.adap5}' adap3='{self.adap3}' l_insert = {self.l_insert} k_min={self.k_min} k_max={self.k_max} first={self.first} last={self.last}".format(self=self) )

    def process_sequences(self, seq_src):
        """
        Generator that folds the sequences from seq_src and yields (krange, data)
        tuples for each sequence.
        krange is range(self.k_min, self.k_max+1) as specified in __init__ and
        data[k][i] is the open-energy for window of size k starting at position i (zero-based).
        
        Can be called multiple times. The subprocess will not be closed unless close() is called.
        """
        n = 0
        for seq in seq_src:
            S = self.adap5 + seq.rstrip() + self.adap3
            l = len(S)
            #print "folding", S, len(S)
            self.p.stdin.write("{0}\n".format(S).encode('ascii'))
            self.p.stdin.flush()
            
            data = [ np.zeros(self.l_insert - k + 1, dtype=np.float32) for k in self.krange ]
            for i in range(l+2):
                j = i - 2
                line = self.p.stdout.readline()
                
                if j < self.first:
                    continue
                
                if j >= self.last:
                    continue
                
                cols = line.split(b'\t')
                # print(cols)
                for k in self.krange:
                    if j >= (k-1):
                        data[k - self.k_min][j-k-self.first+1] = float(cols[k])

            yield self.krange, data
            n += 1

        #self.logger.debug('folded {0} sequences'.format(n))
        self.n_total += n
        

    def close(self):
        """
        Close subprocess file-descriptors and wait for clean exit.
        """
        self.p.stdin.close()
        self.p.stdin.close()
        ex = self.p.wait()
        self.logger.debug('close(): {0} exited with code {1} after folding {2} sequences'.format(self.cmd, ex, self.n_total) )
    
class DummySink(object):
    def __init__(self, fname):
        self.fname = fname
        self.bytes_written = 0
        self.bytes_found = 0

    def write(self, *argcs, **kwargs):
        pass

    def close(self):
        pass

class FileSink(object):
    def __init__(self, fname, bytes_keep=0):
        self.fname = fname
        self.f = open(fname, 'ab+')
        logger = logging.getLogger("fold.FileSink")
        logger.warning(f'opened file {fname} in mode ab+')

        if bytes_keep:
            # print "keeping {} bytes".format(bytes_keep)
            self.f.seek(bytes_keep)
            self.f.truncate()

        self.bytes_written = 0
        self.bytes_found = bytes_keep
        
    def write(self, *argcs, **kwargs):
        self.f.write(*argcs, **kwargs)
        self.bytes_written += len(argcs[0])

    def close(self):
        self.f.flush()
        self.f.close()
        # print "expected file-size", self.fname, self.bytes_written + self.bytes_found, os.path.getsize(self.fname)
        

class OpenenStorage(CachedBase):
    def __init__(self, reads, path='./', discretize=False, raw_dtype=np.float32, disc_dtype=np.uint8, disc_mode='gamma', dummy=False, T=22., **kwargs):
        
        CachedBase.__init__(self)
        
        self.path = path
        self.reads = reads
        self.raw_dtype = raw_dtype
        self.disc_dtype = disc_dtype
        self.disc_mode = disc_mode
        self.T = T
        self.k_sinks = {}
        self.k_disc = {}
        self.logger = logging.getLogger('fold.OpenenStorage({self.reads.name})'.format(self=self))
        self.n_sets = 0
        self.discretize = discretize
        self.dummy = dummy
        self.kwargs = kwargs
        if dummy:
            self.write = self.dummy_write

    @property
    def cache_key(self):
        return "OpenenStorage({})".format(self.reads.cache_key)

    @cached
    def get_raw(self, k):
        return RBNSOpenen(self._make_filename(k), self.reads, k, dummy=self.dummy, **self.kwargs)
        
    @cached
    def get_discretized(self, k, disc_mode = ''):
        if not disc_mode:
            disc_mode = self.disc_mode
        
        disc = OpenenDiscretization(k, self.reads.L, self.disc_dtype, mode=disc_mode)
        self.k_disc[k] = disc
        fname_disc = self._make_filename(k, disc=disc)
        # print fname_disc
        if os.path.exists(fname_disc):
            return RBNSOpenen(fname_disc, self.reads, k, dummy=self.dummy)
        else:
            raw = self.get_raw(k)
            self.logger.info("discretizing '{0}' to satisfy get_discretized({1}) request".format(raw.fname, k) )
            discretized = raw.discretize(disc=disc, dname=fname_disc)
            discretized.store()

            return discretized
            
    def _make_filename(self, k, disc=None):
        if disc:
            #self.k_disc[k] = OpenenDiscretization(k, self.reads.L, self.dtype)
            fmt = disc.to_filename(N=self.reads.N)
        else:
            fmt = "raw_L{0}_k{1}_{2}".format(self.reads.L, k, self.raw_dtype.__name__)

        base, ext = os.path.splitext(os.path.basename(self.reads.fname))
        fname = os.path.join(self.path, "{0}.{1}.bin".format(base, fmt) )
        
        return fname

    def _record_raw_bytes(self, k, with_adapter=True):
        itemsize = np.dtype(self.raw_dtype).itemsize
        l = self.reads.L - k + 1
        if with_adapter:
            l += self.reads.l5 + self.reads.l3

        return l * itemsize

    def has_data(self, k, with_adapter=True):
        fname = self._make_filename(k)
        openen = self.get_raw(k, _do_not_cache=True)
        return openen.check_data(with_adapter=with_adapter)

    def has_data_range(self, kmin, kmax):
        yes = True
        for k in range(kmin, kmax+1):
            if not self.has_data(k):
                yes = False
                break

        return yes

    def count_complete_records_range(self, kmin, kmax, with_adapter=True):
        n = self.reads.N
        for k in range(kmin, kmax+1):
            fname = self._make_filename(k)
            openen = self.get_raw(k, _do_not_cache=True)
            nk = openen.count_records(with_adapter=with_adapter)
            # print fname, nk
            n = min(n, nk)
        
        return n


    def get_or_create(self, k, records_present=0):
        ### TODO: Add resume suppport by giving expected total number of records nd number of
        ### records to be skipped (both need to be determined prior to folding)
        if not k in self.k_sinks:
            fname = self._make_filename(k)
            self.logger.debug(f"get_or_create({k}) -> {fname}")
            if self.has_data(k):
                self.logger.info("data for '{}' already in place. Will leave '{}' untouched.".format(k, fname))
                self.k_sinks[k] = DummySink(fname)

            else:
                bytes_keep = self._record_raw_bytes(k) * records_present
                sink = FileSink(fname, bytes_keep = bytes_keep)
                self.k_sinks[k] = sink
                self.logger.info("prepared '{0}'".format(fname))
    
        return self.k_sinks[k]
    
    def prepare_sinks(self, kmin, kmax, n_complete=0):
        self.logger.info("preparing sinks to resume after {} records".format(n_complete))
        for k in range(kmin, kmax+1):
            sink = self.get_or_create(k, n_complete)

    def write(self, k, vec):
        sink = self.get_or_create(k)
        if self.discretize:
            vec = self.k_disc[k].discretize(vec)

        sink.write(vec.tobytes())

    def dummy_write(self, k, vec):
        print("writing", k, vec)

    def write_set(self, krange, data):
        for k, vec in zip(krange, data):
            self.write(k, vec)
        self.n_sets += 1
        
    def close(self):
        for sink in list(self.k_sinks.values()):
            sink.close()
            
        self.logger.info("closed all files after writing {0} data sets".format(self.n_sets) )


    # def fix_skipped_reads(self, krange):
    #     keep = []
    #     for i, line in enumerate(file(self.reads.fname,'r')):
    #         if 'N' in line.upper():
    #             keep.append(False)
    #         else:
    #             keep.append(True)
        
    #     keep = np.array(keep, dtype=bool)

    #     n_skip = (keep == False).sum()
    #     print "rows to drop", n_skip
        
    #     n_reads = self.reads.N
        
    #     def fixit(openen):
    #         openen._do_not_unpickle = True
    #         print "checking", openen.fname
    #         if openen.N - n_skip == n_reads:
    #             print "removing extra rows!"
    #             oem = openen.oem[keep]
    #             assert len(oem) == n_reads
                
    #             new = RBNSOpenen(openen.fname, self.reads, k, oem = oem, disc=openen.disc)
    #             new.debug_caching = True
    #             new._do_not_unpickle = True
    #             assert len(new.oem) == n_reads
    #             assert new.N == n_reads
    #             new.store()
                
    #             check = RBNSOpenen(openen.fname, self.reads, k, oem = oem, disc=openen.disc, _do_not_unpickle=True)
    #             assert len(check.oem) == n_reads
    #             assert check.N == n_reads
                
    #         else:
    #             print "File is already correct!"
        
    #     for k in krange:
    #         try:
    #             fixit(self.get_raw(k))
    #         except IOError:
    #             pass
            
    #         try:
    #             fixit(self.get_discretized(k))
    #         except IOError:
    #             pass
        
    #         self.cache_flush()

class OpenenDiscretization(object):
    """
    To save space and time, open-energy values are discretized. This class offers the tools
    to determine optimal bins, discretize floating point raw values, and convert 
    (approximately) back.
    """
    
    # parameters of the gamma distribution that best approximate the k-mer open energy
    # distribution observed for a sample of real RBNS input reads of length L. 
    # key is (L, k)
    opt_gamma_params = {
        (20,3) : (0.81302029638841211, -4.8516107653608754e-11, 1.6349961361376417),
        (20,4) : (0.76963400095617862, 5.0499197572751555e-12, 2.4086755693934219),
        (20,5) : (0.94430616311716076, -5.4020522985407265e-08, 3.1066652289455501),
        (20,6) : (1.3817737299304014, -0.064612855898649235, 2.4524518251440384),
        (20,7) : (2.2979596639625823, -0.41897928143131891, 1.8285816311839422),
        (20,8) : (3.3450059235858616, -0.88683930998285865, 1.5279577374464515),

        (40,1) : (0.56359570197492048, -5.8776308767500606e-30, 0.86206668592641167),
        (40,3) : (1.0628281681967706, -6.8214607159662128e-05, 1.2402078509734689),
        (40,4) : (1.2674929471836962, -0.0020650480683494271, 1.2929616570525289),
        (40,5) : (1.4888004568584394, -0.0081636662031228657, 1.3237981203329996),
        (40,6) : (1.7285627206360989, -0.021580429627936448, 1.3376752736147013),
        (40,7) : (2.0032560511697439, -0.046740618049563296, 1.3323232531823281),
        (40,8) : (2.2847020288586801, -0.086073860842223043, 1.3242768850690814),
    }

    opt_E_max = {
        (20,1) : 6.,
        (20,2) : 6.5,
        (20,3) : 7.,
        (20,4) : 8.,
        (20,5) : 10.,
        (20,6) : 10.,
        (20,7) : 10.,
        (20,8) : 11.,
        (20,9) : 12.,
        (20,10) : 13.,
        (20,11) : 14.,
        (40,1) : 3.,
        (40,2) : 4.,
        (40,3) : 6.,
        (40,4) : 8.,
        (40,5) : 9.,
        (40,6) : 10.,
        (40,7) : 11.,
        (40,8) : 12.,
    }
    def __init__(self, k, L, dtype=np.uint8, mode='gamma', N=0):
        self.n = 2**(dtype().nbytes*8) # highest number of bins encodable by dtype
        self.k = k
        self.L = L
        self.N = N
        self.dtype = dtype
        self.mode = mode
        
        if mode == 'gamma':
            import scipy.stats
            step = 1./self.n
            q = np.arange(0,1.+step,step) # n+1 "percentiles"
            params = OpenenDiscretization.opt_gamma_params[(L, k)]
            
            # compute optimal bin boundaries
            self.bins = scipy.stats.gamma.ppf(q, *params)
            
            # compute openen values that optimally represent each bin
            q_x = q[:-1] + 0.5*step
            self.x = np.array(scipy.stats.gamma.ppf(q_x, *params), dtype=np.float32)
        
        elif mode == 'linear':
            E_max = OpenenDiscretization.opt_E_max[(L, k)]
            step = E_max / self.n
            self.bins = np.arange(0, E_max + step, step, dtype=np.float32)
            self.bins[0] -= step
            self.x = 0.5*(self.bins[1:] + self.bins[:-1])
        else:
            raise ValueError("unknown discretization mode '{0}'".format(mode))
                                  
        self.dx = self.bins[1:] - self.bins[:-1]

    def __str__(self):
        return "OpenenDiscretization(L={self.L} k={self.k} dtype={self.dtype} mode={self.mode})".format(self=self)

    @staticmethod
    def optimal_gamma_params_from_raw(openen, n_max=0):
        L = openen.rbns_reads.L
        k = openen.k
        import scipy.stats
        if n_max:
            data = openen.oem[:n_max]
        else:
            data = openen.oem
        opt = scipy.stats.gamma.fit(data)
        
        return (L,k),opt
        
    @staticmethod
    def from_filename(fname):
        import re
        M = re.search(r'discretized_(?P<mode>\w+)_L(?P<L>\d+)_k(?P<k>\d+)_N(?P<N>\d+)_(?P<dtype>\w+)', fname)
        d = M.groupdict()
        L = int(d['L'])
        k = int(d['k'])
        N = int(d['N'])
        mode = d['mode']
        dtype_name = d['dtype']
        dtype = getattr(np, dtype_name)
        
        return OpenenDiscretization(k, L, dtype, mode=mode, N=N)
        
    def to_filename(self,N=0):
        if not N:
            N = self.N
        return "discretized_{self.mode}_L{self.L}_k{self.k}_N{N}_{self.dtype.__name__}".format(self = self, N=N)
        
    def discretize(self, data):
        #print "data", data.shape
        #print "examples", data[:10,:]
        #print "minmax", data.min(), data.max(), np.median(data),  (1 - np.isfinite(data)).sum()
        #print "issues"
        #mask = (1 - np.isfinite(data))
        #print mask.sum()
        #x = mask.nonzero()
        #for i in x:
            #print i, data[i]
        
        #print "bins", self.bins
        #print "dtype", self.dtype
        
        #import RBPamp.digitize as cd
        #return cd.digitize(data, self.bins, dtype=self.dtype) - 1
        #return np.array(np.digitize(data, self.bins) - 1, dtype=self.dtype)
        
        res = cyska.digitize_32fp_8bit(data, self.bins) - 1
        return res
        
        

    def get_hist_xy(self, counts, normed=False):
        """
        Takes a vector with bin-counts.
        Returns x and y coordinates that represent the underlying density 
        (ready for plotting). If normed==True, the trapz integral is 1. 
        If normed==False, the integral is counts.sum().
        """
        y = counts/self.dx
        y /= np.trapz(y, self.x)

        if not normed:
            y *= counts.sum()

        return self.x, y
        

interrupt_folding = Event()

def interrupt():
    logger = logging.getLogger('fold.interrupt')
    interrupt_folding.set()
    if folding_in_progress:
        logger.warning("parallel folding run interrupted")

# Here come a couple of functions that allow parallel folding using the multiprocessing 
# module and RNAplfold
def queue_iter(queue, stop_item = None, interrupt_event=interrupt_folding):
    """
    Small generator/wrapper around multiprocessing.Queue allowing simple
    for-loop semantics: 
    
        for item in queue_iter(queue):
            ...

    """
    while True:
        if interrupt_event.is_set():
            break
        
        item = queue.get()
        if item == stop_item:
            # signals end->exit
            break
        else:
            yield item


def seq_dispatcher(src, queue, chunk_size=100, max_depth=50, throttle_sleep=1., n_max=0, interrupt_event=interrupt_folding, **kwargs):
    """
    Reads sequences from src and groups them in chunks of up to chunk_size.
    Each chunk is enumerated and the tuple (n_chunk, chunk) is pushed to the queue 
    for processing. Avoids overly inflating the queue by sleeping if max_depth chunks are
    already queued.
    """

    logger = logging.getLogger('fold.seq_dispatcher')
    chunk = []
    n_chunk = 0
    n_seqs = 0
    n_skipped = 0
    for read in src:
        read = read.upper()
        if 'N' in read:
            n_skipped += 1
            continue

        chunk.append( read )
        n_seqs += 1
        if len(chunk) >= chunk_size:
            # avoid overloading the queue
            while queue.qsize() > max_depth:
                #logger.debug('qsize > {0} -> sleeping for {1} second'.format(max_depth, throttle_sleep) )
                time.sleep(throttle_sleep)

            queue.put( (n_chunk, chunk) )
            n_chunk += 1
            chunk = []

        if n_max and n_seqs >= n_max:
            break

        if interrupt_event.is_set():
            logger.info('interrupted after {0} sequences dispatched ({2} skipped bc of non-ACGT letters) in {1} chunks. Removing unprocessed chunks from the queue.'.format(n_seqs, n_chunk, n_skipped) )
            while not queue.empty():
                queue.get(False)

            return
        
    if chunk:
        queue.put( (n_chunk, chunk) )
        n_chunk += 1

    logger.info('{0} sequences dispatched ({2} skipped bc of non-ACGT letters) in {1} chunks. Closing down.'.format(n_seqs, n_chunk, n_skipped) )

def fold_worker(seq_queue, data_queue, interrupt_event=interrupt_folding, **vienna_kwargs):
    """
    Use a ViennaOpenen RNAplfold wrapper instance to compute open-energies for
    chunks of sequences from seq_queue. Results are also grouped into chunks and
    pushed (with the original chunk number) onto data_queue. This allows to order 
    the chunks later and write the results in the same order as the original 
    sequences.
    """
    vienna = ViennaOpenen(**vienna_kwargs)
    for n_block, block in queue_iter(seq_queue, interrupt_event=interrupt_event):
        # received a chunk of sequences. Fold them en-bloc
        results = list(vienna.process_sequences(block))
        
        # and return results
        data_queue.put( (n_block, results) )
        
    # cleaning up
    vienna.close()


def result_collector(storage, res_queue, n_complete=0, n_left=-1, k_min=1, k_max=12, interrupt_event = interrupt_folding, log_address="", log_format="", **kwargs):
    """
    Pops (n_chunk, results) from res_queue and inserts them into a heap
    (sorted on n_chunk). Keeping track of how many chunks were already passed on
    to storage, it uses the heap to make sure chunks are stored in the correct 
    order and no chunk gets skipped.
    """
    import heapq
    heap = []
    n_chunk_needed = 0
    t0 = time.time()
    t1 = t0
    n_rec = 0

    from RBPamp.zmq_logging import LoggerFactory
    zmq_logging = LoggerFactory(address=log_address, format_str=log_format)
    logger = zmq_logging.getLogger('fold.result_collector')

    if n_complete:
        storage.prepare_sinks(k_min, k_max, n_complete=n_complete)

    for n_chunk, results in queue_iter(res_queue, interrupt_event=interrupt_event):
        heapq.heappush(heap, (n_chunk, results) )
        
        # as long as the root of the heap is the next needed chunk
        # pass results on to storage
        while(heap and (heap[0][0] == n_chunk_needed)):
            n_chunk, results = heapq.heappop(heap) # retrieves heap[0]
            for krange, data in results:
                storage.write_set(krange, data)
                n_rec += 1
        
            n_chunk_needed += 1

        # debug output on average throughput
        t2 = time.time()
        if t2-t1 > 30:
            dT = t2 - t0
            rate = n_rec/dT
            
            n_remain = n_left - n_rec
            eta = n_remain / rate / 60 / 60
            logger.debug("processed {0} records in {1:.0f} seconds (average {2:.3f} records/second). ETA={3:.2f} hours".format(n_rec, dT, rate, eta) )
            t1 = t2
    
    # by the time None pops from the queue, all chunks 
    # should have been processed!
    if not interrupt_event.is_set():
        assert len(heap) == 0

    # close all open files and make sure stuff is on disk
    storage.close()
    dT = time.time() - t0
    logger.debug("finished processing {0} records in {1:.0f} seconds (average {2:.3f} records/second)".format(n_rec, dT, n_rec/dT) )
    

def parallel_fold(reads, n_complete=0, n_parallel=8, skip_records=0, k_min=1, k_max=12, **kwargs):
    """
    Top-level function for parallel folding. Constructs all the subprocesses
    and ensures proper shutdown. kwargs are passed to ViennaRNA instances, 
    as well as the dispatcher.
    """
    # This global variable is used by the keyboard interrupt 
    # handler to decide if we need to flush stuff to disk
    global folding_in_progress
    folding_in_progress = True
    
    import multiprocessing
    seq_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    
    # fire up the dispatcher: 
    #
    #  seqs from src-> 
    #  enumerated chunks->
    #  seq_queue
    #
    kwargs['interrupt_event'] = interrupt_folding
    kwargs['temp'] = reads.temp
    kwargs['adap5'] = reads.adap5
    kwargs['adap3'] = reads.adap3
    kwargs['l_insert'] = reads.L
    kwargs['k_min'] = k_min
    kwargs['k_max'] = k_max
    
    n_left = reads.N - n_complete
    kwargs['n_left'] = n_left
    kwargs['n_max'] = n_left
    kwargs['n_complete'] = n_complete

    dispatcher = multiprocessing.Process(
        target = seq_dispatcher, 
        name='seq_dispatcher', 
        args=(reads.iter_reads(n_skip=n_complete), seq_queue),
        kwargs=kwargs 
    )
    dispatcher.daemon = True
    dispatcher.start()
    
    # fire up multiple workers: 
    #
    #  seq_queue-> enumerated seq. chunks-> \
    #       ViennaOpenen(RNAplfold_RBPamp)-> \
    #  enumerated result chunks-> res_queue
    #
    workers = []
    for n in range(n_parallel):
        worker = multiprocessing.Process(
            target = fold_worker, 
            name='fold_worker_{0}'.format(n), 
            args=(seq_queue, res_queue), 
            kwargs=kwargs
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    # fire up result collector:
    #
    #  res_queue-> enumerated result chunks->
    #           re-order->
    #  write data using storage
    #
    collector = multiprocessing.Process(
        target = result_collector,
        name = 'result_collector',
        args = (reads.acc_storage, res_queue),
        kwargs = kwargs,
    )
    collector.daemon = True
    collector.start()
    
    # wait until all sequences have been thrown onto seq_queue
    dispatcher.join()
    # signal all fold-workers to finish
    for n in range(n_parallel):
        seq_queue.put(None) # each worker consumes exactly one None

    for worker in workers:
        # make sure all results are on res_queue by waiting for 
        # workers to exit.
        worker.join()
    
    # signal the collector to stop
    res_queue.put(None)
    # and wait until everything has reached storage. 
    # collector calls storage.close() bc it is in its own subprocess.
    collector.join()
   
    folding_in_progress = False
   
def test_discretization(N=10000):
    import matplotlib.pyplot as pp
    x = np.random.gamma(3, size=N)
    print(pp.hist(x, bins=100, normed=True))
    #print hcounts, hbins
    
    disc = OpenenDiscretization(3, 20, np.uint8, 'gamma')
    counts = np.bincount(disc.discretize(x), minlength=256)
    print(disc.x.shape, disc.dx.shape, counts)
    y = counts/disc.dx
    y *= counts.sum() / np.trapz(y, disc.x)

    print(np.trapz(y, disc.x), len(x))
    pp.plot(disc.x, y)
    
    pp.show()
    
def test_vienna():
    V = ViennaOpenen(k_min=5, k_max=5, l_insert = 40)
    seqs = [
        #'TATACACGCCAGGATGAGCATAGAATCCGCTATCTTTTTT',
        #'AACCGGCTAGTGTATCTAGAGTGGACCAATATTCTTTTGT',
        'AATTATACCCAACACTTTTTTCCGCATCAAAGATATATAG',
    ]
    
    correct_data = [
        np.array([  1.37758803e+00,   4.65885401e-02,   4.44491990e-02,
         4.44544517e-02,   4.37189484e+00,   5.28840590e+00,
         5.44863987e+00,   6.44182396e+00,   6.40045404e+00,
         6.11836720e+00,   6.08953190e+00,   6.57561302e+00,
         6.58033609e+00,   7.80855417e+00,   5.67592812e+00,
         6.03012180e+00,   6.29798985e+00,   6.51980209e+00,
         6.32596684e+00,   5.20592594e+00,   5.21245909e+00,
         5.19603205e+00,   3.84870291e+00,   6.88292726e-04,
         7.41391385e-04,   3.55043197e+00,   4.00692511e+00,
         4.00891781e+00,   4.00772381e+00,   4.01971483e+00,
         4.02041817e+00,   5.34093809e+00,   5.63969707e+00,
         5.71804190e+00,   5.54887295e+00,   4.88211823e+00], dtype=np.float32
        ),
        np.array([ 1.21388996,  1.29570901,  3.08945203,  1.72082102,  1.58345401,                                                                                                                                          
        2.14634705,  2.86216402,  2.59196711,  2.52647901,  2.56132007,                                                                                                                                           
        0.1191803 ,  2.34980106,  2.32055211,  2.19572902,  2.50372696,                                                                                                                                           
        2.59294009,  2.074687  ,  3.11443305,  4.6174159 ,  2.78021598,                                                                                                                                           
        2.777704  ,  3.70185995,  3.6581161 ,  1.99287498,  1.13532197,
        1.26033103,  0.69037437,  0.70658243,  0.75592059,  0.71988487,
        0.69261771,  0.07472514,  0.1399269 ,  2.3060441 ,  3.10272098,
        2.76067591], dtype=np.float32),
    ]

    U5a = seqs[0].index('TTTTT')
    print(U5a)
    U5b = U5a + 1
    for kr, data in V.process_sequences(seqs):
        
        print(data[0][U5a], data[0][U5b])
        
    #for (krange, data), correct in zip(V.process_sequences(seqs), correct_data):
        #print data
        #assert (data == correct).all()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    #test_discretization()
    # test_vienna()
    # sys.exit(0)

    #src = file('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads') #.readlines()[:30051]
    #storage = OpenenStorage(path='tmp')
    #parallel_fold(src, storage, n_parallel=10)
    #sys.exit(1)

    #test_memory_consumption()

    from RBPamp.reads import RBNSReads
    reads = RBNSReads('/scratch2/RBNS/RBFOX2/RBFOX2_input.reads', n_max=10000)
    storage = reads.acc_storage.get_raw(1)
    print(storage.oem)
    sys.exit(0)    
    
    dopenen = openen.discretize()
    for i in range(10):
        print(dopenen.rbns_reads.seqm[i])
        print(dopenen.oem[i])
    
    
    sys.exit(0)
    
    k8 = np.fromfile('openen.8.raw-float32.bin', dtype=np.float32)
    k7 = np.fromfile('openen.7.raw-float32.bin', dtype=np.float32)
    k6 = np.fromfile('openen.6.raw-float32.bin', dtype=np.float32)
    k5 = np.fromfile('openen.5.raw-float32.bin', dtype=np.float32)
    k4 = np.fromfile('openen.4.raw-float32.bin', dtype=np.float32)
    k3 = np.fromfile('openen.3.raw-float32.bin', dtype=np.float32)
    
    import matplotlib.pyplot as pp
    
    L=40
    k=7
    disc = OpenenDiscretization(7,40,np.uint8)
    print(disc.bins)
    mid = 0.5*(disc.bins[1:] + disc.bins[:-1])
    pp.loglog(mid, disc.x)
    pp.show()
    sys.exit(1)
    
    

    #dist = scipy.stats.beta # gamma
    dist = scipy.stats.gamma
    data = k8
    
    #params = dist.fit(data)
    #print params
    bins = gamma_bins_k(8,dtype=np.uint8)
    print("low bins",bins[:10])
    print("low data",sorted(data)[:10])
    print("low data->bins", np.digitize(sorted(data)[:10], bins) - 1)
    
    dig = np.digitize(data, bins) -1
    
    print(dig.min(), dig.max())
    print(bins, np.bincount(dig))
    
    mid = 0.5*(bins[1:] + bins[:-1]) # mid-points
    print(len(bins))
    x = np.arange(0, data.max(), .01)
    #pp.hist(data, bins=bins, normed=True)
    #pp.plot(x, dist.pdf(x, *params))
    pp.loglog(data, mid[dig],'ob')
    
    RMSD = np.sqrt(np.mean((mid[dig] - data)**2))
    print("RMSD",RMSD)
    pp.show()
    sys.exit(1)
    
    print(make_bins(5))
    src = open('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads')
    store = OpenenStorage()
    vienna = ViennaOpenen()

    import time
    t0 = time.time()
    for n, krange, data in vienna.process_sequences(src):
    #for n, krange, data in vienna_openen(src, L=64):
        store.store_set(krange, data)
        if n and not n % 1000:
            t1 = time.time()
            print("{0:.2f} seqs/second".format(1000./(t1-t0)))
            t0 = t1
    
    #oa = OpenenHistCollection(name=sys.argv[1])
    
    #tm = ThreadManager(n_threads=4)
    #tm.process_reads(sys.stdin, oa)
    
    #print oa['TGCATGT']
             
