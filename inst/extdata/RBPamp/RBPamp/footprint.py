# -*- coding: future_fstrings -*-
from __future__ import print_function
import numpy as np
# from RBPamp.npwrap import npmonitored
import gc
import os
import sys
import shelve
import logging
from RBPamp.cmdline import ensure_path
from RBPamp.caching import pickled, cached, monitored, CachedBase, get_cache_sizes
import RBPamp.cyska as cyska
from RBPamp.sc import SelfConsistency

from scipy.optimize import minimize
from time import time
from copy import deepcopy
gc.enable()
# gc.set_debug(gc.DEBUG_LEAK)

def dump_garbage():
    # force collection
    gc.collect()
    print("GARBAGE OBJECTS:")
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80: s = s[:80]
        print(type(x),"\n  ", s)

def dump_caches():
    for size, cache in get_cache_sizes():
        print(cache, size/1024.)


class RowOptimization(object):
    def __init__(self, cal, acc_k):
        self.logger = logging.getLogger("opt.footprint.row_opt")
        self.cal = cal
        self.acc_k = acc_k
        self.params = deepcopy(self.cal.params)
        self.params.acc_k = acc_k
        
        self.lacc0, self.punp1 = self.cal.get_lacc_punp_cached(acc_k)
        self.openen = self.cal.get_input_openen_cached(acc_k)
        self.openen_punp = self.cal.get_input_openen_cached(1)
        self.Z1 = self.cal.Z1
        self.zw = self.Z1.shape[1]

    def predict_profiles(self, acc_shift, a, A0):
        from time import time
        t0 = time()
        
        # select shifted accessibilities
        self.params.acc_shift = acc_shift
        ofs = self.openen.ofs - self.params.k + 1 + acc_shift
        if ofs < 0:
            self.logger.warning("ofs underflow")

        # compute partition function with scaled acc.
        # # scale accessibilities
        # acc1 = np.exp(self.lacc0 * np.float32(a))
        # Z1_acc = self.Z1 * acc1[:, ofs:ofs + self.zw]
        assert len(self.Z1) == len(self.lacc0) # make sure the subsetting was done on both!
        Z1_acc = cyska.acc_scale_Z1(self.Z1, self.lacc0, a, ofs)
        # aggregate to read-level
        t1 = time()
        Z1_read, Z1_read_max = cyska.clipped_sum_and_max(Z1_acc, clip=1E6)
        t2 = time()

        # predict binding probability
        sc = SelfConsistency(Z1_read, self.cal.input_reads.rna_conc, bins=1000)
        rbp_free = sc.free_rbp_vector(self.cal.rbp_conc, Z_scale=A0)
        t3 = time()

        psi = cyska.p_bound(Z1_read, np.array(rbp_free*A0,dtype=np.float32))
        t4 = time()

        # compute base-p_unpaired profiles
        assert len(self.Z1) == len(self.punp1)

        punp_expect = cyska.acc_footprints(
            self.Z1,
            self.punp1,
            self.params.k, 
            1, 
            self.openen_punp.ofs - self.params.k + 1, 
            pad=self.cal.pad, 
            row_w = np.ascontiguousarray(psi.T),
            n_threads=1
        )
        t_prof = time() - t4

        times = 1000 * np.array([t1-t0, t2-t1, t3-t2, t4-t3, t_prof])
        N = len(self.Z1)
        self.logger.debug(f"N={N} t_partfunc={times[0]:.2f} t_Zread={times[1]:.2f} t_sc={times[2]:.2f} t_psi={times[3]:.2f} t_prof={times[4]:.2f}")
        return np.array(punp_expect)

    def optimize_a(self, acc_shift, A0=None):
        if A0 is None:
            A0 = self.params.A0
        
        self.logger.debug(f"optimize_a({acc_shift}, A0={A0})")
        def to_opt_a(a):
            punp_expect = self.predict_profiles(acc_shift, a, A0)
            err = np.sum((punp_expect - self.cal.punp_profiles[1:])**2)
            return err

        from scipy.optimize import minimize
        res = minimize(
            to_opt_a,
            (0, ),  # start with no secondary structure data A0
            bounds= [ (0, 1.), ], 
            options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
        )
        res.a = res.x
        res.A0 = A0
        punp_expect = self.predict_profiles(acc_shift, res.a, res.A0)
        return punp_expect, res

    def optimize_A0(self, acc_shift, a=1):
        self.logger.debug(f"optimize_A0({acc_shift}, a={a})")
        def to_opt_A0(A0):
            punp_expect = self.predict_profiles(acc_shift, a, A0)
            err = np.sum((punp_expect - self.cal.punp_profiles[1:])**2)
            return err

        res = minimize(
            to_opt_A0,
            (1., ),
            bounds= [ (1e-4, 1e3), ], 
            options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
        )
        res.a = a
        res.A0 = res.x
        punp_expect = self.predict_profiles(acc_shift, res.a, res.A0)
        return punp_expect, res

    def optimize(self, acc_shift):
        self.logger.debug(f"optimize({acc_shift})")
        def to_opt(args):
            a, A0 = args
            punp_expect = self.predict_profiles(acc_shift, a, A0)
            err = np.sum((punp_expect - self.cal.punp_profiles[1:])**2)
            # print punp_expect, err
            return err

        res = minimize(
            to_opt,
            (0, 1., ),
            bounds= [ (0, 1), (1e-4, 1e3), ], 
            options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
        )
        res.a = res.x[0]
        res.A0 = res.x[1]
        punp_expect = self.predict_profiles(acc_shift, res.a, res.A0)
        # print "opt res", res
        return punp_expect, res



class FootprintCalibration(CachedBase):
    def __init__(self, rbns, params, pad=5, thresh=1e-3, subsample=False, redo=False):
         
        CachedBase.__init__(self)

        # print ">>> before initialization"
        # dump_caches()
        self.params = params.copy()
        self.consensus = self.params.as_PSAM().consensus
        self.consensus_ul = self.params.as_PSAM().consensus_ul
        self.logger = logging.getLogger(f'opt.FootprintCalibration({self.consensus_ul})')
        self.log_mem_usage("before init")
        self.path = ensure_path(os.path.join(rbns.out_path, 'footprint/'))
        self.params.acc_k = 0
        self.params.acc_scale = 0
        self.params.non_specific = 0
        self.rbns = rbns
        self.input_reads = rbns.reads[0]
        self.subsample = subsample
        self.pd_names = [reads.name for reads in rbns.reads[1:]]
        self.rbp_conc = rbns.rbp_conc
        self.pad = pad
        self.thresh = thresh
        self.result_log = logging.getLogger('results.footprint')

        self.shelve = shelve.open(
            os.path.join(self.path, f"history_{self.consensus_ul}"), 
            protocol=-1, 
            flag='n' if redo else 'c'
        )
        self.shelve["rbp_conc"] = self.rbp_conc

        self.results = {}
        self._openen_cache = {}
        self._lacc_cache = {}
        self._punp_input = None

        fp = os.path.join(self.path, f'footprints_{self.consensus_ul}.tsv')
        # if os.path.exists(fp):
        #     self.load_footprints(fp)
        # no need to load these, as we now keep pickled results from optimize()

        self.fp_file = open(fp, 'w')
        self.fp_file.write('acc_k\tacc_shift\tacc_scale\tA0\terror\n')
        self._partfunc_done = False
        self.log_mem_usage("after init")

    def log_mem_usage(self, when='SIZE', n_max=10):
        for size, cache in get_cache_sizes()[:n_max]:
            s = size/1024.**2
            if s < 1:
                continue
            self.logger.debug(f"MEM {when}: {cache} {s:.3f} MB")

    # @npmonitored
    def prepare_partition_functions(self):
        if self._partfunc_done:
            return

        # Z1 = np.array([reads.PSAM_partition_function(self.params) for reads in rbns.reads])
        self.logger.info("evaluating partition function")
        from RBPamp.params import ModelSetParams
        self.Z1_full = np.array([reads.PSAM_partition_function(ModelSetParams([self.params, ]), subsample=self.subsample) for reads in self.rbns.reads])
        self.Z1_in_noacc = self.Z1_full[0]
        
        Z1_read = self.Z1_in_noacc.sum(axis=1)
        self.logger.debug("Z1_read percentiles 5,25,50,75,95,99 {}".format(np.percentile(Z1_read, [5, 25, 50, 75, 95, 99])))
        thresh = self.thresh * Z1_read.max()
        self.I = Z1_read > thresh
        N = self.I.sum()
        kept_ratio = N/float(len(Z1_read))
        if kept_ratio > .5:
            self.logger.warning(f"motif-based thresholding would keep {kept_ratio:.2f} of reads. Choosing median instead")
            self.I = Z1_read > np.median(Z1_read)
            N = self.I.sum()
        else:
            self.logger.info(f"subsetting to {N} reads with Z1 > {thresh}")
        self.Z1 = self.Z1_in_noacc[self.I,:]

        for reads in self.rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

        self.punp_profiles, self.naive_profiles = self.compute_initial_profiles()
        # print "punp_profiles", self.punp_profiles
        # print "naive_profiles", self.naive_profiles
        self.store_shelve("params_initial", self.params)
        self.store_shelve("punp_profiles", self.punp_profiles)
        self.store_shelve("naive_profiles", self.naive_profiles)

        # self.logger.debug("plotting naive punp profiles")
        # self.plot_profiles(self.naive_profiles, 0, 0, None)
        self.err0 = np.sum((self.naive_profiles - self.punp_profiles[1:])**2)
        self.logger.debug(f"naive error: {self.err0}")
        self.store_shelve("err0", self.err0)
        self.store_shelve("indices_threshold", self.I)

        un_opt = (self.err0, 0, 0, 0, self.params.A0)
        self.results[(0, 0)] = un_opt
        self.store_footprint(un_opt)

        self.logger.debug("done, freeing some memory")
        for reads in self.rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

        self._partfunc_done = True
        self.log_mem_usage("after prepare_partition_functions")

    def store_shelve(self, key, value):
        self.shelve["{0}_{1}".format(self.consensus_ul, key)] = value
        self.shelve.sync()        

    def load_shelve(self, key, default=None):
        k = "{0}_{1}".format(self.consensus_ul, key)
        if k in self.shelve:
            return self.shelve[k]
        return default

    def load_profile(self, k, s):
        key = "opt_profile_{k}_{s}".format(k=k, s=s)
        # print "loading", key
        stored = self.load_shelve(key)
        # if stored is None:
        #     l = 'na'
        # else:
        #     l = len(stored)
        # print "load_profile",k,s,"->" , l
        return stored

    def store_profile(self, k, s, value):
        key = "opt_profile_{k}_{s}".format(k=k, s=s)
        return self.store_shelve(key, value)

    @property
    def cache_key(self):
        return f"{self.params}.{self.rbp_conc}.{self.input_reads.cache_key}.{self.subsample}.{self.thresh}"

    # @npmonitored
    def optimize_row(self, acc_k, shift_range, from_scratch=False):
        results = [self.load_profile(acc_k, s) for s in shift_range]

        if from_scratch:
            missing = list(shift_range)
        else:
            missing = [s for s, res in zip(shift_range, results) if res is None]      
        self.logger.info(f"scanning missing footprints for acc_k={acc_k} shift_range={missing}")

        new_results = []
        if missing:
            self.prepare_partition_functions()

            def _optimize(s):
                try:
                    punp_expect, res = row.optimize(s)
                    return (punp_expect, res)
                except KeyboardInterrupt:
                    pass

            from multiprocessing.pool import ThreadPool
            row = RowOptimization(self, acc_k)
            pool = ThreadPool()
            # new_results = pool.map(_optimize, missing, chunksize=1)
            # using map_async instead of pool.map makes it possible to catch KeyboardInterrupt exceptions
            p = pool.map_async(_optimize, missing)
            try:
                new_results = p.get(0xFFFF)
            except KeyboardInterrupt:
                pool.terminate()
                pool.close()
                pool.join()
                raise
            else:
                pool.terminate()
                pool.close()
                pool.join()
        
            # new_results = map(_optimize, missing)  # single-thread version for debugging

        for s, (punp_expect, res) in zip(missing, new_results):
            results[shift_range.index(s)] = (punp_expect, res)

        return results

    def optimize_a1(self, acc_k, s, from_scratch=False):
        key = "opt_profile_a_one_{acc_k}_{s}".format(**locals())
        res = self.load_shelve(key)
        if res is None:
            self.prepare_partition_functions()

            row = RowOptimization(self, acc_k)
            punp_a_one, res_a_one = row.optimize_A0(s, a=1)
            self.store_shelve(key, (punp_a_one, res_a_one))

        return self.load_shelve(key)

    # def get_err0(self):
    #     if hasattr(self, "err0"):
    #         return self.err0
        
    #     if not "err0" in self.shelve:
    #         self.prepare_partition_functions()

    #     return self.shelve["err0"]

    def calibrate(self, k_core_range=[3, None], plot=True, pad=5, from_scratch=False, heuristic=0):
        kmin, kmax = k_core_range
        if kmax is None:
            kmax = self.params.k + 2

        self.logger.debug(f"scanning acc_k = {kmin} .. {kmax}")

        for acc_k in range(kmax, kmin - 1, -1):
            d = self.params.k - acc_k + 1
            shift_range = range(-pad, max(d + pad, 2)) # maybe better but don't read over adapter into next read!!!
            for s, (punp_predict, res) in zip(shift_range, self.optimize_row(acc_k, shift_range, from_scratch)):
                err = res.fun
                err0 = self.load_shelve('err0')
                rel_err = err / err0
                a = res.a
                A0 = res.A0

                if not res.success:
                    self.logger.warning(f"unable to optimize footprint k={acc_k}, s={s}.")
                    a = 0.
                    A0 = self.params.A0
                    err = self.load_shelve('err0')
                    rel_err = 1.
                
                opt = (err, acc_k, s, a, A0)
                self.results[(acc_k, s)] = opt
                self.store_profile(acc_k, s, (punp_predict, res))
                self.store_footprint(opt)
                self.result_log.info(f"{self.consensus} k={acc_k} s={s} a_opt={a} A0_opt={A0} err={err} rel_err={rel_err}")

            self.log_mem_usage(f"after optimizing acc_k={acc_k}")


        results = sorted(self.results.values())
        err, acc_k, s, a, A0 = results[0]
        rel_err = err/err0
        self.result_log.critical(f"OPTIMUM {self.consensus} k={acc_k} s={s} a_opt={a} A0_opt={A0} err={err} rel_err={rel_err}")
        self.params.acc_k = acc_k
        self.params.acc_shift = s
        self.params.acc_scale = a
        self.params.rel_err = rel_err
        # self.params.A0 = A0
        self.params.rbp_name = self.input_reads.rbp_name
        self.store_shelve("params_calibrated", self.params)

        # for the optimum, also compute profile for a=1
        punp_a_one, res_a_one = self.optimize_a1(acc_k, s)
        self.params.save(os.path.join(self.path, f'calibrated_{self.consensus_ul}.tsv'))

        return self.params

    def store_footprint(self, opt):
        err, k, s, a, A0 = opt
        out = [k, s, a, A0, err]
        self.fp_file.write("\t".join([str(o) for o in out]) + "\n")
        self.fp_file.flush()
        self.store_shelve("{0}_{1}".format(k, s), opt)

    # @monitored
    # @pickled
    def compute_initial_profiles(self):

        punp_profiles = np.array([
            reads.weighted_accessibility_profile(z, self.params.k, pad=self.pad, subsample=self.subsample)[0]
            for reads, z in zip(self.rbns.reads, self.Z1_full)])

        for reads in self.rbns.reads:
            reads.cache_flush(deep=True)

        # print "right here", punp_profiles
        # 1/0
        # predict profiles w/o accessibility footprint

        # these are computed without thresholding, so on the full set of reads!
        Z1_read, Z1_read_max = cyska.clipped_sum_and_max(self.Z1_in_noacc, clip=1E6) # aggregate to read-level
        sc = SelfConsistency(Z1_read, self.input_reads.rna_conc, bins=1000)
        rbp_free = sc.free_rbp_vector(self.rbp_conc, Z_scale=self.params.A0)
        psi = cyska.p_bound(Z1_read, rbp_free*self.params.A0)

        punp_openen = self.get_input_openen_cached(1)
        punp_acc = punp_openen.acc
        # if self.subsample:
        #     punp_acc = self.input_reads.sub_sampler.draw(data=punp_acc)

        assert len(self.Z1_in_noacc) == len(punp_acc)
        naive_profiles = cyska.acc_footprints(
            self.Z1_in_noacc, 
            punp_acc, 
            self.params.k, 
            1, 
            punp_openen.ofs - self.params.k + 1, 
            pad=self.pad, 
            row_w = np.ascontiguousarray(psi.T),
        )

        return punp_profiles, naive_profiles

    def compute_kmer_acc_profiles(self):
        self.logger.debug("compute_kmer_acc_profiles")
        motif = self.consensus_ul
        psam = self.params.as_PSAM()
        highest_affinity = psam.highest_scoring_kmers()
        self.shelve['{}_high_affinity_kmers'.format(motif)] = highest_affinity
        
        for score, kmer in highest_affinity:
            self.logger.debug(f"compute_kmer_acc_profiles({kmer})")
            key = '{}_acc_data'.format(kmer)
            if not key in self.shelve:
                all_data = [reads.get_kmer_raw_unfolding_energy(kmer) for reads in self.rbns.reads]
                self.shelve[key] = all_data

    def get_input_openen_cached(self, k):
        if not k in self._openen_cache:
            self.logger.debug(f"get_input_openen_cached({k}) not found")
            self._openen_cache[k] = self.input_reads.acc_storage.get_raw(k, _do_not_cache=True)
            # self.input_reads.acc_storage.cache_flush() # free up memory
        
        for x in list(self._openen_cache.keys()):
            # drop everything that's not p-unpaired or current k
            if x > 1 and x != k and k > 1:
                self.logger.debug(f"get_input_openen_cached({k}) dropping {x}")
                self._openen_cache[x].cache_flush()
                del self._openen_cache[x]

        return self._openen_cache[k]

    def get_punp_cached(self):
        if self._punp_input is None:
            openen_punp = self.get_input_openen_cached(1)
            I = self.load_shelve('indices_threshold')
            punp = openen_punp.acc[I, :]
            self._punp_input = punp

        return self._punp_input

    def get_lacc_punp_cached(self, k):
        if not k in self._lacc_cache:
            I = self.load_shelve('indices_threshold')
            self.logger.debug(f"get_lacc_punp_cached({k}) not found")
            openen = self.get_input_openen_cached(k)

            acc0 = openen.acc[I, :]
            lacc0 = np.log(acc0)

            # if self.subsample:
            #     acc0 = self.input_reads.sub_sampler.draw(data=acc0)
            #     punp = self.input_reads.sub_sampler.draw(data=punp)

            # acc0 = acc0[I, :]
            # punp = punp[I, :]
            punp = self.get_punp_cached()
            self._lacc_cache = { k : (lacc0, punp) }  # always keep only one item!
        
        return self._lacc_cache[k]

    def close(self):
        for reads in self.rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

        self.shelve.close()
        import RBPamp.caching
        RBPamp.caching._dump_cache_sizes()
        import gc
        gc.collect()