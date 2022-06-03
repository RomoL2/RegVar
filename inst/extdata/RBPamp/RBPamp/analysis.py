# coding=future_fstrings
from __future__ import print_function
__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import os
import numpy as np
import time
import logging
import RBPamp.cyska
import RBPamp.cyska as cyska
from RBPamp.cmdline import ensure_path
from RBPamp.caching import cached, pickled, CachedBase


class RBNSSample(CachedBase):
    def __init__(self, reads):
        self.reads = reads
        self.nt_counts_profile = np.array(cyska.kmer_profiles(self.reads.seqm, 1), dtype=np.float32)
        self.nt_counts = self.nt_counts_profile.sum(axis=1)
        self.nt_freqs = self.nt_counts / self.nt_counts.sum()
        self.nt_freqs_profile = self.nt_counts_profile / self.nt_counts_profile.sum(axis=0)[np.newaxis,:]
    
        self.nt_entropy = -(np.log2(self.nt_freqs_profile) * self.nt_freqs_profile).sum(axis=0)

        im = self.reads.get_index_matrix(1)
        counts = np.array(cyska.joint_freq_at_distance(im,1), dtype=np.float32)
        freq = counts / counts.sum(axis=(0,1))[np.newaxis, np.newaxis,:]
        indep = np.outer(self.nt_freqs, self.nt_freqs)

        self.MI = (freq * np.log2(freq / indep[:,:,np.newaxis])).sum(axis=(0,1))

        print(self.nt_freqs)

    def nt_entropy_profile(self):
        freq = self.nt_freqs_profile
        ent = -(np.log2(freq) * freq).sum(axis=0)
        import matplotlib.pyplot as pp
        pp.subplot(211)
        pp.plot(ent, linestyle='steps-mid', label=self.reads.name)
        pp.ylim(1,2)
        pp.ylabel("nt. entropy [bits]")
        pp.xlabel("read position [nt]")
        pp.legend(loc='lower center')
        pp.subplot(212)
        pp.pcolor(freq, cmap='viridis')
        pp.yticks(np.arange(4)+.5, list('ACGU'))
        pp.colorbar(label="rel. frequency", orientation='horizontal', fraction=.05)

    def MI_profile(self):
        im = self.reads.get_index_matrix(1)
        print(im.min(), im.max())
        counts = np.array(cyska.joint_freq_at_distance(im,1), dtype=np.float32)
        freq = counts / counts.sum(axis=(0,1))[np.newaxis, np.newaxis,:]
        indep = np.outer(self.nt_freqs, self.nt_freqs)

        import matplotlib.pyplot as pp
        MI = (freq * np.log2(freq / indep[:,:,np.newaxis])).sum(axis=(0,1))
        pp.figure()
        pp.plot(MI)
        pp.show()
        print(MI)


from RBPamp.partfunc import PartFuncModel

class RBNSComparison(CachedBase):
    def __init__(self, in_reads, pd_reads, ska_runner = None):
        
        CachedBase.__init__(self)
        
        self.logger = logging.getLogger('rbns.RBNSComparison')
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        self.ska_runner = ska_runner
        
        self.name = 'RBNS:{pd_reads.rbp_conc}nM:{in_reads.rbp_conc}nM'.format(**locals())
        
    @property
    def cache_key(self):
        return "{self.name}.{self.pd_reads.cache_key}.{self.in_reads.cache_key}".format(self=self)
        
    def _subsampled(self, func):
        """
        Adds error estimates using subsamples of the underlying RBNSReads instance
        for the pulldown sample.
        """
        
        res = func(self.pd_reads, self.in_reads)

        sampled = np.array([
            func(sample, self.in_reads)
            for sample in self.pd_reads.subsamples
        ])
        errors = sampled.std(axis=0)
        assert res.shape == errors.shape
        return res, errors
        
    @cached
    @pickled
    def R_values(self, k):
        self.logger.debug("computing R-values")
        def compute_R(sample, control):
            return sample.kmer_frequencies(k) / control.kmer_frequencies(k)
        
        return self._subsampled(compute_R)

    @cached
    @pickled
    def W_values(self, k):
        self.logger.debug("computing W-values")
        def compute_W(sample, control):
            fs = (sample.kmer_counts_acc_weighted(k) + 1.)/ sample.N
            fc = (control.kmer_counts_acc_weighted(k) + 1.)/ control.N
            return fs / fc
        # subsampling currently not working with OpenenStorage!
        res = compute_W(self.pd_reads, self.in_reads)
        err = np.zeros(res.shape, dtype=res.dtype)
        return res, err


    @cached
    def O_values(self, k):
        self.logger.debug("computing approximate occupancies by fitting R-values to linear overlap model")
        R, R_err = self.R_values(k)
        from RBPamp.crosstalk_matrix import CrosstalkMatrix
        cm = CrosstalkMatrix(k, self.in_reads)
        
        occ = cm.fit_occupancies(R)
        occ_min = cm.fit_occupancies(R - R_err)
        occ_max = cm.fit_occupancies(R + R_err)
        
        err_m = np.fabs(occ - occ_min)
        err_M = np.fabs(occ - occ_max)
        occ_err = np.where(err_m > err_M, err_m, err_M )
        return occ, occ_err


    @cached
    #@pickled
    def DI_values(self, k):
        self.logger.debug("computing differential information (DI) values")
        def compute_DI(sample, control):
            
            c_pd = sample.kmer_counts(k)
            c_in = control.kmer_counts(k)
            
            n_pd = float(c_pd.sum())
            n_in = float(c_in.sum())
            
            p_pd = n_pd / (n_pd + n_in)
            p_in = 1. - p_pd
            
            #print "p_pd=",p_pd
            px_pd = c_pd / n_pd
            px_in = c_in / n_in
            px = (c_pd + c_in)/(n_pd + n_in)
            
            #print "norms=", px_pd.sum(), px_in.sum(), px.sum()
            
            DI = p_pd * np.log2(px_pd / (px * p_pd) ) + p_in * np.log2(px_in / (px * p_in) )
            #print k, "DI", DI[-5:], DI.sum()
            #print "UUUUU", px[-1], px_pd[-1], px_in[-1]
            
            return DI 
        
        return self._subsampled(compute_DI)


    @cached
    @pickled
    def SKA_weights(self, k):
        self.logger.debug("computing SKA-weights")
        def compute_SKA(sample, control):
            return self.ska_runner.stream_counts(k, sample, control)
        
        return self._subsampled(compute_SKA)

    @cached
    @pickled
    def F_ratios(self, k):
        self.logger.debug("computing F-ratios")
        def compute_F_ratio(sample, control):
            return sample.fraction_of_reads_with_kmers(k) / control.fraction_of_reads_with_kmers(k)
        
        return self._subsampled(compute_F_ratio)

    @cached
    @pickled
    def recall_ratios(self, k, kmer_order):
        self.logger.debug("computing recall ratios")
        def compute_recall_ratio(sample, control):
            return sample.recall(k, kmer_order, reorder=False) / control.recall(k, kmer_order, reorder=False)
        
        return self._subsampled(compute_recall_ratio)

    @cached
    @pickled
    def pure_F_ratios(self, k, candidates, out_path = "", n_sample = 100000):
        self.logger.debug("computing pure F-ratios")
        def compute_pure_F_ratio(sample, control):
            if not sample.is_subsample and out_path:
                out_file_pd = os.path.join(out_path, "pure_{k}mer_reads_{sample.name}.fa".format(k = k, sample=sample ))
                out_file_in = os.path.join(out_path, "pure_{k}mer_reads_{control.name}.fa".format(k = k, control=control ))
            else:
                out_file_pd = None
                out_file_in = None
                
            f_pd = sample.fraction_of_reads_with_pure_kmers(k, candidates, out_file=out_file_pd, n_sample = n_sample)
            f_in = control.fraction_of_reads_with_pure_kmers(k, candidates, out_file=out_file_in, n_sample = n_sample)
                
            return f_pd / f_in

        return self._subsampled(compute_pure_F_ratio)
            
    def __str__(self):
        return self.name


class RBNSAnalysis(CachedBase):
    def __init__(self, rbp_name ='RBP', out_path='ska_results', ska_runner=None, known_kd="", n_pure_samples = 100000):
        
        CachedBase.__init__(self)
        
        self.reads = []
        self.n_samples = 0
        self.acc_storages = []
        self.rbp_name = rbp_name
        self.out_path = out_path
        self.ska_runner = ska_runner
        #self.write_fasta = write_fasta
        self.known_kd = known_kd
        self.n_pure_samples = n_pure_samples
        self.logger = logging.getLogger('rbns.Analysis({self.rbp_name}) -> "{self.out_path}"'.format(self=self))
        
        self.rbp_conc = []
        self.comparisons = []

        self.k_range = []
        self.runs = {}
        self.AUCs = {}
        #self.pair_screens = collections.defaultdict(dict)
   
    @property
    def cache_key(self):
        return ".".join([r.cache_key for r in self.reads])

    @property
    def sample_labels(self):
        return ["input"] + ["{} nM".format(conc) for conc in self.rbp_conc]

    def flush(self, all=False):
        for comp in self.comparisons:
            comp.cache_flush()
        for reads in self.reads:
            if all:
                reads.cache_flush(deep=True)
            else:
                reads.cache_flush('get_index_matrix')

    def add_reads(self, rbns_reads):
        self.logger.info("adding {0}".format(rbns_reads.name) )
        self.reads.append(rbns_reads)
        # sample = RBNSSample(rbns_reads)
        # sample.nt_entropy_profile()
        # sample.MI_profile()

        # secondary structure open-energies/accessibility storage
        from RBPamp.fold import OpenenStorage
        self.acc_storages.append(rbns_reads.acc_storage)
        
        if len(self.reads) > 1:
            self.comparisons.append(RBNSComparison(self.reads[0], rbns_reads, self.ska_runner) )
            self.rbp_conc.append(rbns_reads.rbp_conc)

        self.n_samples = len(self.reads) - 1
    
    @property
    def input_reads(self):
        # return reads with lowest concentration (should be 0)
        i = np.array(self.rbp_conc).argsort()[0]
        return self.reads[i]

    def _make_matrices(self, comp_attr, *argc, **kwargs):
        self.logger.debug("gathering data matrices for {0}".format(comp_attr) )

        M = np.array([getattr(comp, comp_attr)(*argc, **kwargs) for comp in self.comparisons])
        values = M[:,0,:]
        errors = M[:,1,:]
        
        return values, errors
        
    def R_value_matrix(self, k):
        return self._make_matrices("R_values", k)

    def W_value_matrix(self, k):
        return self._make_matrices("W_values", k)

    def DI_value_matrix(self, k):
        DI, DI_err = self._make_matrices("DI_values", k)
        print(DI.shape, end=' ') 
        for conc,di in zip(self.rbp_conc, DI):
            print(di.argmax(), di.argmin())
            self.logger.info("total mutual information between {0}mer-frequencies and pd/in variable @{2:.1f}nM is {1:.3e} bits".format(k, di.sum(), conc))

        return self._make_matrices("DI_values", k)

    def O_value_matrix(self, k):
        return self._make_matrices("O_values", k)

    def SKA_weight_matrix(self, k):
        return self._make_matrices("SKA_weights", k)

    def F_ratio_matrix(self, k):
        return self._make_matrices("F_ratios", k)
            
    def recall_ratio_matrix(self, k):
        kmer_order = self.get_optimal_kmer_ranking(k)
        return self._make_matrices("recall_ratios", k, kmer_order)

    def pure_F_ratio_matrix(self, k):
        self.logger.debug("computing pure F-ratio matrix for k={0}".format(k) )

        # TODO: replace by select_significant_kmers
        order = self.get_optimal_kmer_ranking(k)
        R, R_err = self.F_ratio_matrix(k)
        Rm = np.median(R - R_err, axis=0)[order]
        
        i_cut = (Rm > 2).argmin()
        candidates = np.zeros(4**k, dtype=np.uint32)
        candidates[order[:i_cut]] = np.arange(i_cut) + 1

        out_path = None
        return self._make_matrices("pure_F_ratios", k, candidates, out_path=out_path, n_sample = self.n_pure_samples)

    def explained_matrix(self, k):
        X = []
        X_err = []
        for reads in self.reads:
            x = reads.fraction_of_reads_with_kmers(k)
            X.append(x)

            xx = np.array([sub.fraction_of_reads_with_kmers(k) for sub in reads.subsamples])
            print(xx[:,xx.argsort(axis=1)[-10:]])
            err = np.sqrt(((xx - x[np.newaxis,:])**2).mean(axis=0) / reads.n_subsamples)
            
            X_err.append(err)

        return np.array(X), np.array(X_err)


        
    @cached
    def get_optimal_kmer_ranking(self, k):
        # TODO: factor in consistently elevated scores with increasing protein concentration?
        from scipy.stats.mstats import gmean
        all_R = self.R_value_matrix(k)[0]

        #return gmean(all_ska_weights, axis=0).argsort()[::-1]
        return np.median(all_R, axis=0).argsort()[::-1]
    
    def select_diagnostic_kmers(self, k=7, n=10):
        R = self.R_value_matrix(k)[0]
        # print "R-shape", R.shape
        Rm = R.max(axis=1)
        
        from .cyska import index_to_seq
        Rm_i = sorted(set(R.argmax(axis=1)))

        n_choices = len(R)
        self.logger.debug("highest enriched kmers {}".format([index_to_seq(i, k) for i in Rm_i]) )
        med_R = []
        enr_R = []
        for i in Rm_i:
            print("checking kmer", index_to_seq(i, k))
            for reads, r, sample_r in zip(self.reads[1:], R[:, i], R):
                lo_quant = np.percentile(sample_r, 25)
                print(reads.name, r, lo_quant, 1./lo_quant)

            med_R.append(np.min(R[:, i]))
            enr_R.append((R[:, i] > 1).sum()/float(n_choices))
        
        enr_R = np.array(enr_R)
        med_R = np.array(med_R)
        self.logger.debug("min enrichment observed for these kmers {}".format(med_R))
        self.logger.debug("samples that showed any enrichment for these kmers {}".format(enr_R))

        kmer_score = med_R * enr_R
        # print kmer_score.shape, kmer_score
        I = kmer_score.argsort()[::-1]
        best = []
        for i in I:
            kmer_i = Rm_i[i]
            best.append( (index_to_seq(kmer_i, k), kmer_i, R[:, kmer_i]) )

        return best

    def keep_best_samples(self, ranks=[1, 2, 3], k=7, min_R=1.1):
        n_samples = len(self.reads[1:])
        n_wanted = len(ranks)
        if not n_wanted:
            # everything selected
            return
        
        R = self.R_value_matrix(k)[0]
        top_kmer_per_sample = R.argmax(axis=1)
        R_max = np.array([r[i] for r, i in zip(R, top_kmer_per_sample)])
        sample_max_R = R_max.argmax()
        kmer_i = top_kmer_per_sample[sample_max_R]
        import RBPamp.cyska
        kmer = RBPamp.cyska.index_to_seq(kmer_i, 7)
        sample_score = np.array([r[kmer_i] for r in R])

        QC_pass = sample_score >= min_R
        QC_fail = ~QC_pass
        n_fail = QC_fail.sum()
        if n_fail > 0:
            self.logger.warning("the following concentrations did not pass QC and are not considered further: {}".format(np.array(self.rbp_conc)[QC_fail]))

        # sample_score = []
        # for reads, sample_r in zip(self.reads[1:], R):
        #     lo_q = np.percentile(sample_r, 25)
        #     sample_score.append(1. / lo_q)
        
        # sample_score = np.array(sample_score)

        final_score = (sample_score * QC_pass)
        sample_i = final_score.argsort()[::-1] # failed experiments get 0 sample score
        self.logger.debug(f"ordering samples by enrichment of {kmer}: {sample_score} -> {sample_i}")

        original_ranks = ranks
        ranks = np.array(ranks, dtype=int) - 1
        ranks = ranks[ranks < (n_samples - n_fail)]
        indices = np.arange(n_samples)
        chosen = sorted(indices[sample_i[ranks]])

        chosen_scores = final_score[chosen]
        chosen_ranks = np.empty_like(chosen)
        chosen_ranks[chosen_scores.argsort()[::-1]] = np.arange(len(chosen))

        self.logger.debug(f"chosen sample indices: {chosen} scores: {chosen_scores}, ranks: {chosen_ranks}")

        for j in set(list(indices)) - set(list(chosen)):
            self.reads[j].cache_flush(deep=True)

        reads = [self.reads[0],] + list(np.array(self.reads[1:])[chosen])
        self.reads = []

        self.logger.info("keeping samples with RBP concentrations {}".format([r.rbp_conc for r in reads]))
        rbns = RBNSAnalysis(rbp_name = self.rbp_name, out_path=self.out_path, ska_runner=self.ska_runner, known_kd=self.known_kd, n_pure_samples = self.n_pure_samples)
        for r in reads:
            rbns.add_reads(r)
            
        if len(chosen) < n_wanted:
            self.logger.warning("less samples available than ranks requested. Analysis will use only {} samples".format(len(chosen)))
            if len(chosen) < 1:
                raise ValueError(f"no sample left! You asked for ranks={original_ranks} with n_samples={n_samples} n_fail={n_fail}")

        return chosen_ranks, rbns
        
    def select_significant_kmers(self,k, z_cut=2, n_min=1, n_max=None):
        ska, ska_err = self.SKA_weight_matrix(k)
        
        # be conservative rg. error of SKA weight, but keep it non-negative
        s = np.where(ska > ska_err, ska - ska_err, 0)
        # z-score across kmers, mean across protein concentations
        z = np.mean( (s - ska.mean(axis=1)[:, np.newaxis]) / ska.std(axis=1)[:, np.newaxis], axis=0)
        
        order = z.argsort()[::-1]
        #print z[order][:20]
        rank_cut = max((z[order] < z_cut).argmax(), n_min)
        if n_max:
            rank_cut = min(n_max, rank_cut)
        
        best_sample_i = ska[:,order[0]].argmax()
        kmers = [cyska.index_to_seq(i, k) for i in order[:rank_cut]]
        return kmers, order[:rank_cut], best_sample_i+1
        
    def compute_results(self, k, options, results=["R_value", "affinities", "pure_F_ratio", "recall_ratio", "SKA_weight", "F_ratio"]):
        if len(self.reads) < 2:
            self.logger.warning("need at least two samples to compute '{0}'".format(results))
            return

        order = self.get_optimal_kmer_ranking(k)
        all_kmers = np.array(list(cyska.yield_kmers(k)))
        
        for name in results:
            if not name:
                continue
            fname = "{self.rbp_name}.{name}.{k}mer.tsv".format(**locals())
            path = ensure_path(os.path.join(self.out_path, "metrics", fname))

            if name == 'cooccurrence_tensor':
                rbns.cooccurrence_tensor_analysis(k)
                continue

            else:
                values, errors = getattr(self, "{name}_matrix".format(name=name) )(k)
                self.write_kmer_matrix(path, all_kmers, values.T, errors.T, order)
                # if report and name == "R_value":
                #     from RBPamp.rbns_reports import EnrichmentBarPlot
                #     for comp in self.comparisons:
                #         path = os.path.join(self.out_path, "{0}nM".format(comp.pd_reads.rbp_conc))
                #         if not os.path.xists(path):
                #             os.makedirs(path)
                #         plot = EnrichmentBarPlot(comp)
                #         plot.make_plot(k, dest=path)
                yield values, errors
                    

    def cooccurrence_tensor_analysis(self, k):
        kmers, indices, best_sample_i = self.select_significant_kmers(k)
        print(kmers)
        reads = self.reads[best_sample_i]
        inrds = self.reads[0] # input control
        
        expect = np.array(reads.expected_kmer_cooccurrence_distance_tensor(kmers), dtype=np.float32)
        obsrvd = np.array(reads.kmer_cooccurrence_distance_tensor(kmers), dtype=np.float32)
        inpool = np.array(inrds.kmer_cooccurrence_distance_tensor(kmers), dtype=np.float32)
                          
        lratio =np.log2((obsrvd+1) / (expect+1) )
        sratio =np.log2((obsrvd+1) / (inpool+1) )
        
        #print "expect kmer co-occurrence", expect[0,1,:].sum()
        #print "obsrvd kmer co-occurrence", obsrvd[0,1,:].sum()
        
        #import matplotlib.pyplot as pp
        #pp.plot( lratio[2,2,:] ) 
        #pp.plot( sratio[2,2,:] ) 
        #pp.show()


    def write_kmer_matrix(self, out_path, kmers, values, errors, order=[], err_str='error', header=None):
        self.logger.info("writing data matrix '{out_path}'".format(out_path=out_path) )

        if header == None:
            header = ['# kmer'] + ['{0}nM\nerr'.format(c) for c in self.rbp_conc]

        if not len(order):
            order = np.arange(len(kmers))

        def round_to_2(x):
            if x:
                return round(x, max(-int(np.floor(np.log10(abs(x)))), 2) ) 
            else:
                return x
        
        def round_to_err(x, x_err):
            if x_err:
                n_dig = int(np.ceil(-np.log10(abs(x_err))))+1
                x_err = round(x_err,n_dig)

                n_dig = int(np.ceil(-np.log10(abs(x_err))))+1
                x = round(x,n_dig)
            
            return [str(x), str(x_err)]

        with open(os.path.join(out_path), 'w') as of:

            of.write("\t".join(header) + '\n')
            for mer, values_row, error_row in zip(kmers[order], values[order], errors[order]):
                
                cols = [str(mer), ]
                for x, err in zip(values_row, error_row):
                    cols.extend(round_to_err(x, err))

                of.write("\t".join(cols) + '\n')
            of.close()
 

def read_kmer_matrix(path):
    import re
    kmers = []
    data = []
    for line in open(path):
        if line.startswith('#'):
            head = re.split('\s+', line.rstrip())
            rbp_conc = [float(h.replace('nM','')) for h in head[2::2]]
        else:
            parts = line.split('\t')
            kmers.append(parts[0])
            data.append(np.array(parts[1:], dtype=np.float32))
    
    kmers = np.array(kmers)
    I = kmers.argsort()
    data = np.array(data)[I,:].T
    values = data[::2,:]
    errors = data[1::2,:]

    return rbp_conc, values, errors


