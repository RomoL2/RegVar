# coding=future_fstrings
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp
import logging
import os
import RBPamp.cyska as cyska
from .cyska import yield_kmers
import RBPamp
from RBPamp.caching import CachedBase, cached, pickled

class Alignment(object):
    def __init__(self, seqs=[], weights=[]):
        self.matrix = []
        # for s, w in izip_longest(seqs, weights, fillvalue=1.):
        #     ofs, score = self.align(s)

        # self.ext_cost = np.ones(125)
        # self.ext_cost[:9] = 0.01
        # self.ext_cost[9:18] = [.01, .02, .03, .04, .05, .06, .07, .08, .1, ]
        # print self.ext_cost
        self.seqs = []
        self.ofs = []
        self.weights = []

    def align(self, seq, normalize=False, multiply=False, contain=False, end_weight=False, end_ignore=False, min_overlap=1, core_k=None, core_start=None, debug=False):
        # TODO: handle core_k and core_start 
        bits = cyska.seq_to_bits(seq)
        l = len(seq)
        n = len(self.matrix)
        if not len(self.matrix):
            return 0, 1 # offset, alignment score
        
        scores = []
        if contain:
            assert n > l
            d = n - l
            ofs_range = list(range(-d, d+1))
        else:
            ofs_range = list(range(-l + min_overlap, n + 1 - min_overlap))
        # print seq
        if end_weight:
            func = np.mean
        else:
            func = np.min

        for ofs in ofs_range:
            m_start = max(0, ofs)
            m_end = min(n,ofs+l)
            
            if multiply:
                start_avg = 1.
                if m_start and not end_ignore:
                    start_avg = func(self.matrix[:m_start], axis=1).prod()
                
                end_avg = 1.
                if m_end < n and not end_ignore:
                    end_avg = func(self.matrix[m_end:], axis=1).prod()
            else:
                start_avg = 0
                if m_start and not end_ignore:
                    start_avg = func(self.matrix[:m_start], axis=1).sum()

                end_avg = 0.  # Arg, there was a bug here! Redo?
                if m_end < n and not end_ignore:
                    end_avg = func(self.matrix[m_end:], axis=1).sum()


            n_cols = m_end - m_start

            s_start = max(-ofs, 0)
            s_end = s_start + n_cols
            col_scores = []
            # if end_weight:
            #     score = start_avg*end_avg if multiply else start_avg + end_avg
            # else:
            #     score = 1 if multiply else 0
            score = start_avg*end_avg if multiply else start_avg + end_avg
            # score = 1 if multiply else 0

            for i in range(n_cols):
                if bits[i+s_start] > 3:
                    continue # skip gaps
                
                S = self.matrix[i+m_start, bits[i+s_start]]

                col_scores.append(S)
                score = score * S if multiply else score + S

            # if normalize:
            #     # score /= self.max_kmer_score_at(m_start, k=n_cols)
            #     # score *= float(n_cols) / len(seq)
            #     score /= self.max_score

            scores.append(score)
            if debug:
                print(ofs, s_start,":",s_end, seq[s_start:s_end], m_start,":", m_end, col_scores, "->", score)

        x = np.array(scores).argmax()
        S = scores[x]
        if normalize:
            S /= self.max_score
            # S /= self.max_kmer_score(k=len(seq))

        return ofs_range[x], S


    def blend(self, seq, ofs, weight, normalize=False):
        self.seqs.append(seq)
        self.weights.append(weight)
        if ofs < 0:
            self.ofs = [o - ofs for o in self.ofs]
            matrix = np.zeros((len(self.matrix)-ofs,4))
            matrix[-ofs:] = self.matrix[:]
            self.matrix = matrix
            ofs = 0
        
        d = ofs + len(seq) - len(self.matrix)
        if d > 0:
            matrix = np.zeros((len(self.matrix)+d,4))
            if len(self.matrix):
                matrix[:len(self.matrix)] = self.matrix[:]
            self.matrix = matrix
        
        self.ofs.append(ofs)

        bits = cyska.seq_to_bits(seq)
        l = len(seq)
        m = self.matrix.max()
        if m == 0:
            m = np.inf

        for i in range(l):
            if bits[i] > 3:
                continue # skip gaps
            self.matrix[i+ofs, bits[i]] += weight #min(max(weight, self.matrix[i+ofs, bits[i]]), m)
        
        if normalize:
            self.matrix /= self.max_score(k=len(seq))

    def add(self, seq, weight=1.):
        ofs, score = self.align(seq)
        self.blend(seq, ofs, weight)

        return score

    @property
    def score(self):
        return self.matrix.max(axis=0).mean()
    
    def max_kmer_score(self, k=8):
        if len(self.matrix):
            ma = self.matrix.max(axis=1)
            slices = np.array([ma[i:i+k].sum() for i in range(len(self.matrix)-k+1)])
            return slices.max()
        else:
            return 1.

    def max_kmer_score_at(self, i, k=8):
        if len(self.matrix):
            return self.matrix[i:i+k].max(axis=1).sum()
        else:
            return 1.

    @property
    def max_score(self):
        if len(self.matrix):
            return self.matrix.max(axis=1).sum()
        else:
            return 1.

    @property
    def max_weight(self):
        return np.array(self.weights).max()

    @property
    def wlen(self):
        colw = self.matrix.max(axis=1) / self.matrix.max()
        return colw.sum()

    def __str__(self):
        buf = []
        for s, o, w in zip(self.seqs, self.ofs, self.weights):
            spacer = " "*o
            buf.append("{w:3.3e}  {spacer}{s}".format(**locals()))

        ms = self.max_score(k=len(self.matrix))
        perc = 100. * self.score / ms
        buf.append("average max. column score {0:.2f} of {1:.2f} ({2:.2f}%)".format(self.score, ms, perc))
        return "\n".join(buf)

    def save_logo(self, fname):
        from RBPamp.pwm import weblogo_save
        weblogo_save(self.matrix, fname)

    def to_PSAM(self, keep_weight=1., n_max=0, pseudo=0, col_scale=True, A0=None):
        matrix = np.array(self.matrix)
        def find_best():
            if keep_weight == 1 and n_max == 0:
                return (1, 0, len(matrix))

            frac = matrix.sum(axis=1) 
            F = matrix.sum()
            n = len(matrix)

            best = {n : (1 ,0 ,n)}
            for i in range(n):
                for j in range(i, n+1):
                    
                    f = frac[i:j].sum()/F
                    l = j-i
                    if l in best:
                        if f < best[l][0]:
                            continue

                    best[l] = (f, i, j)

            bylength = sorted(best.keys())
            for l in bylength:
                if n_max and l > n_max:
                    # we exhausted all motifs of allowed length
                    break
                
                f,i,j = best[l]
                if f >= keep_weight:
                    # found the shortest motif that satisfies keep_weight
                    return f, i, j 

            # no allowed length satisfies keep_weight cutoff.
            # select the shortest motif that is as good as the longest allowed motif
            f_cut = best[n_max][0]
            for l in bylength:
                f,i,j = best[l]
                if f >= f_cut:
                    # found the shortest motif that satisfies keep_weight
                    return f, i, j 


        f, i, j = find_best()
        m = matrix[i:j] 
        if col_scale:
            # add pseudo-scores to columns 
            # with fewer observations/lower score
            M = m.max(axis=1)
            # print "maxima along positions", M
            # 0 for col with highest score, 
            # approaching 1 for lowest
            inc = 1 - M / M.max() 
            m += inc[:,np.newaxis]

        psam = m / m.max(axis=1)[:, np.newaxis]
        if pseudo:
            psam += pseudo
            psam /= psam.max(axis=1)[:, np.newaxis]

        # A0 = m.max(axis=1).sum()
        from RBPamp.pwm import PSAM
        if A0 is None:
            A0 = self.max_weight

        P = PSAM(psam, A0=A0)
        P._n_seqs = len(self.seqs)
        P._max_weight = self.max_weight
        return P
        
    @property
    def consensus(self):
        psam = self.to_PSAM(pseudo=0)
        return psam.consensus_ul



from RBPamp.pwm import PSAM, project_column
class PSAMSetBuilder(object):
    def __init__(self, kmer_set, k=8, keep_weight=.99, n_max=11, m_max=5, thresh = .75, z_cut=4, n_min=10, q_ns=5., A0=.01, pseudo=1e-3, **kwargs):
        self.logger = logging.getLogger("opt.seed.PSAMSetBuilder")

        param_keys = ['k', 'keep_weight', 'n_max', 'm_max', 'thresh', 'z_cut', 'n_min', 'q_ns', 'pseudo', 'A0'] + list(kwargs.keys())
        d = dict(locals())
        d.update(kwargs)
        param_str = ", ".join(["{}={}".format(key, d[key]) for key in sorted(param_keys)])
        self.logger.info("parameters: '{}'".format(param_str))

        self.k = k
        # self.debug = debug
        self.keep_weight = keep_weight
        self.n_max = n_max
        self.n_min = n_min
        self.m_max = m_max
        self.thresh = thresh
        self.pseudo = pseudo
        self.A0 = A0
        self.kmer_set = kmer_set
        self.assignments = {}
        self.n_enriched = len(kmer_set)
        self.z_cut = z_cut
        self.r0 = np.array([r for kmer, r in self.kmer_set]).max()
        print("PSB.r0", self.r0)
        self.logger.debug("building PSAMs from {0} significantly enriched {1}-mers".format(self.n_enriched, self.k))

    def new(self, kmer, r):
        aln = Alignment()
        aln.blend(kmer, 0, r, normalize=False)
        return aln

    def make_psam(self, aln, pseudo=0, **kwargs):
        return aln.to_PSAM(
            pseudo=pseudo, 
            keep_weight=self.keep_weight, 
            A0=aln.max_weight/self.r0 * self.A0,
            **kwargs
        )

    def get_motifs(self):
        psams = [self.make_psam(a) for a in self.alns]
        return ",".join([p.consensus_ul for p in psams])

    def build(self):
        while self.kmer_set:
            self.logger.debug("{} kmers left".format(len(self.kmer_set)))
            # align all remaining enriched kmers to all motifs
            scores = []
            ofs = []
            # print "re-aligning"
            for kmer, r in self.kmer_set:
                o, s = np.array([aln.align(kmer, normalize=True) for aln in self.alns]).T
                ofs.append(o)
                scores.append(s)

            scores = np.array(scores)
            ofs = np.array(ofs)

            # find best aligning kmer and add to best matching motif
            mer_scores = scores.max(axis=1)
            best_i = mer_scores.argmax()
            j = scores[best_i].argmax()
            self.logger.debug("best matching kmer is {}:{} with score {:.3f}".format(self.kmer_set[best_i], self.alns[j].consensus, mer_scores[best_i]) )
            if (scores[best_i, j] < self.thresh) and (len(self.alns) < self.m_max):
                kmer, r = self.kmer_set[0]
                current_motifs = self.get_motifs()
                self.logger.debug("{} R_est={:.1f} does not match existing motifs ({}) scores={}. Seeding new motif".format(kmer, r, current_motifs, scores[0]))
                # print "starting NEW MOTIF", kmer, r, scores[0]
                self.kmer_set.pop(0)
                self.alns.append(self.new(kmer, r))
            else:
                kmer, r = self.kmer_set.pop(best_i)
                s = scores[best_i, j]
                o = ofs[best_i, j]
                self.alns[j].blend(kmer, int(o), r, normalize=False)
   
        return self.alns

    def drop_low_support(self, alns):
        alns = list(alns)
        keep = list(alns)
        if len(alns) > 1:
            alns = sorted(alns, key=lambda a : len(a.seqs), reverse=True)
            keep = [alns[0], ]
            drop = []
            orphan_set = []
            for aln in alns[1:]:
                if len(aln.seqs) >= self.n_min:
                    keep.append(aln)
                else:
                    drop.append(aln)
                    ks = list(zip(aln.seqs, aln.weights))
                    orphan_set.extend(ks)

            orphan_set = sorted(orphan_set, key = lambda x : x[1], reverse=True)
            self.logger.info("need to drop {} motifs with {} kmers".format(len(drop), len(orphan_set)))
            self.logger.info("re-distributing kmers of weakest motifs {}".format(orphan_set))
            while orphan_set:
                # align all remaining enriched kmers to all motifs
                scores = []
                ofs = []
                # print "re-aligning"
                for kmer, r in orphan_set:
                    o, s = np.array([aln.align(kmer, normalize=True) for aln in keep]).T
                    ofs.append(o)
                    scores.append(s)

                scores = np.array(scores)
                ofs = np.array(ofs)

                mer_scores = scores.max(axis=1)
                best_i = mer_scores.argmax()
                self.logger.debug("best matching kmer is {} with score {:.3f}".format(orphan_set[best_i], mer_scores[best_i]) )

                kmer, r = orphan_set.pop(best_i)
                j = scores[best_i].argmax()
                s = scores[best_i, j]
                o = ofs[best_i, j]
                self.assignments[kmer] = j
                keep[j].blend(kmer, int(o), r, normalize=False)
        
        keep = sorted(keep, key=lambda a: a.max_weight, reverse=True)
        return keep

    def make_PSAM_set(self):
        kmer, r = self.kmer_set.pop(0)
        self.logger.debug("starting first motif with {0} R_est={1:.1f}".format(kmer, r))
        self.alns = [self.new(kmer, r), ]

        alns = self.build()
        keep = self.drop_low_support(alns)
        self.founders = [aln.seqs[0] for aln in keep]
        for i, aln in enumerate(keep):
            for kmer in aln.seqs:
                self.assignments[kmer] = i

        psams = [self.make_psam(aln, n_max=self.n_max, pseudo=self.pseudo) for aln in keep]
        w = np.array([p.n for p in psams])
        wm = w.max()
        motifs = ",".join([p.consensus_ul for p in psams])
        self.logger.info("done assembling {0} motifs from {1} kmers (at least {4} per motif) with z > {2}: {3}".format(len(psams), self.n_enriched, self.z_cut, motifs, self.n_min))

        # second pass -> pad motifs to equal size
        [p.pad_to_size(wm) for p in psams]

        return psams


class PSAMSeeding(object):
    def __init__(self, rbns):
        self.rbns = rbns
        self.logger = logging.getLogger("seed.PSAMSeeding")
        import shelve
        from RBPamp.cmdline import ensure_path
        self.shelf = shelve.open(os.path.join(ensure_path(os.path.join(self.rbns.out_path,'seed/')), 'history'), 'c')

    def primer_analysis(self, k=7):
        from RBPamp.seed import Alignment
        import RBPamp.cyska as cyska

        dG = np.fromfile(
            os.path.join(
                os.path.dirname(__file__), '../adapters/7mer_adap3.dG'
            ),
            sep='\n'
        )
        print(dG)
        low_dG = np.percentile(dG, 50)
        mask = (dG <= low_dG)
        print(low_dG, len(mask))
        g = dG[mask]

        alns = []
        R, R_err = self.rbns.R_value_matrix(k)
        from scipy.stats import spearmanr, pearsonr
        for j, r in enumerate(R):
            print("sample", j)
            r_dG = np.log2(r[mask])
            print(pearsonr(g, r_dG))
            print(spearmanr(g, r_dG))

            import RBPamp.report
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(g, r_dG, '.')
            plt.xlabel('dG')
            plt.ylabel('log2 R')
            plt.savefig('r_dG_{}.pdf'.format(j))
            plt.close()

    def motifs_from_R(self, k=8, z_cut=4, n_min=10, q_ns=5., **kwargs): # UNDO HERE!!!
        from RBPamp.seed import Alignment
        import RBPamp.cyska as cyska
        kwargs['z_cut'] = z_cut
        kwargs['n_min'] = n_min
        kwargs['q_ns'] = q_ns

        R, R_err = self.rbns.R_value_matrix(k)
        # from scipy.stats.mstats import gmean
        # R = gmean(R, axis=0) # geometric mean
        Rm = np.mean(R, axis=0)
        Rns = np.percentile(Rm, q_ns)
        # print "non-specific quantile", Rns
        R_err = R_err.mean(axis=0)
        R = Rm + Rns * ( (Rm - 1)/ (1 - Rns)) # corrected R-value, see suppl. methods

        I = R.argsort()[::-1]
        R_sort = R[I]
        z = (R_sort - R_sort.mean())/R_sort.std()
        # print R_sort[:10]
        # print "z-scores", z.min(), z.max(), z.mean()
        i_cut = (z < z_cut).argmax()
        # print i_cut
        R_cut = max(1, R[I[i_cut]])
        # print i_cut, "R_values above z-cut", R_cut

        self.shelf['R0'] = Rm
        self.shelf['Rns'] = Rns
        self.shelf['R'] = R
        self.shelf['I'] = I
        self.shelf['z'] = z
        self.shelf['i_cut'] = i_cut
        self.shelf['i_ns'] = (R_sort > Rns).argmax() - 1

        r0 = R[I[0]]
        print("maxR", r0)
        n = 0
        kmer_set = []

        for i in I:
            kmer = cyska.index_to_seq(i, k)
            n += 1
            r = R[i] 
            rerr = R_err[i]
            # self.logger.debug( "{i}, {kmer}, {r}, +/- {rerr}, {R_cut}".format(**locals()))
            if r - rerr <= R_cut and len(kmer_set) > n_min:
                break
            
            kmer_set.append( (kmer, r) )
        
        self.shelf['kmer_set'] = kmer_set

        PSB = PSAMSetBuilder(kmer_set, **kwargs)
        psams = PSB.make_PSAM_set()
        self.shelf['psams'] = psams
        self.shelf['width'] = psams[0].n
        self.shelf['n_psam'] = len(psams)
        self.shelf['assignments'] = PSB.assignments
        self.shelf['founders'] = PSB.founders

        return psams

    def seeded_multi_params(self, n_samples, max_motifs=4, k_seed=7, thresh=.7, **kwargs):
        from RBPamp.params import ModelSetParams, ModelParametrization
        params = []

        for i, psam in enumerate(self.motifs_from_R(k=k_seed, m_max=max_motifs, thresh=thresh, **kwargs)):
            params.append(ModelParametrization.from_PSAM(psam, n_samples=n_samples, **kwargs))

        param_set = ModelSetParams(params, sort=True)
        # print "len param_set in seeded_multi_params()", len(param_set.param_set)
        self.store_logos(param_set)
        return param_set

    def store_logos(self, params=None):
        self.logger.debug("generating sequence logos")
        from RBPamp.cmdline import ensure_path
        path = ensure_path(os.path.join(self.rbns.out_path,'seed/'))
        rbp_name = self.rbns.reads[0].rbp_name

        if not params is None:
            fname = os.path.join(path, 'seeded_{}.pdf'.format(rbp_name))
            params.save_logos(fname, title="{} seeded PSAMs".format(rbp_name))


if __name__ == "__main__":

    test_data = [
        (100, 'UGCAUGC'),
        (100, 'GCAUGCA'),
        (90, 'UGCAUGU'),
        (80, 'GCAUGCA'),
        (79, 'GCAUGCU'),
        (78, 'GCAUGCC'),
        (77, 'GCAUGCG'),
        (70, 'GCAUGUA'),
        (69, 'GCAUGUU'),
        (68, 'GCAUGUC'),
        (67, 'GCAUGUG'),
        (85, 'AGCAUGU'),
        (75, 'CGCAUGU'),
        (55, 'GGCAUGU'),
        (79, 'AGCAUGC'),
        (78, 'CGCAUGC'),
        (58, 'GGCAUGC'),
        (30, 'UGCACGC'),
        (25, 'UGCACGU'),
        (15, 'UGCACGA'),
        (15, 'UGCACGG'),
        (90, 'GCAAUGC'), # test, secondary motif
        (85, 'GCAAUGU'),
        (88, 'UGCAAUG'),
        (75, 'AGCAAUG'),
    ]

    test_data_msi = [
        (9, 'UAGUUAG'),
        (6, 'UAGAUAG'),
        (5.6, 'UUAGUUA'),
        (5.2, 'UAGUUUA'), # <- 3
        (5, 'AUAGUUA'),
        (4.7, 'UAGGUAG'),
        (4.3, 'UAGCUAG'),
        (4.5, 'AGUUAGU'),
        (4.1, 'UUAGUUU'), # <- 8
        (4.2, 'AGUUUAG'), # <- 9
        (4.0, 'UUUAGUU'),
    ]
    pb = PSAMBuilder(test_data_msi, init=False)
    print("done building tables")
    pb.align(pb.P[3], pb.P[8], debug=True)
    pb.align(pb.P[3], pb.P[0], debug=True)
    sys.exit(0)

    pb.aggregate()


    import logging
    logging.basicConfig(level=logging.DEBUG)
    A = Alignment()
    # A.add('GCAUG', 2.)
    # A.add('GCACG', .4)
    # A.add('UGCAU', 2.)

    # A.add('UGC-UG', 3.)

    # # print A.matrix
    # print A
    # A.save_logo("bla.eps")

    from RBPamp.analysis import RBNSAnalysis
    from RBPamp.reads import RBNSReads
    from RBPamp import auto_detect

    rbp_name, reads_files, rbp_concentrations = auto_detect('.')

    rbns = RBNSAnalysis(
        rbp_name = rbp_name,
        out_path = 'RBPamp',
        ska_runner = None,
    )
    
    for fname, rbp_conc in zip(reads_files, rbp_concentrations):
        reads = RBNSReads(
            fname, 
            rbp_conc=rbp_conc,
            rbp_name = rbp_name,
            n_max=0,
            pseudo_count=10, 
            rna_conc = 1000.,
            temp = 4,
            n_subsamples = 10,
            acc_storage_path = 'acc',
        )
        rbns.add_reads(reads)

    # DK = DependentKmerAnalysis(rbns, km=3)    
    # print "6mer R-value derived linear logo"
    # kmer_seed = DK.topR_PSAM_seed(6, n_max=9)
    # print kmer_seed
    # kmer_seed.save_logo('kmer_seed.eps')

    # DK.build_matrices()
    # print "assembled linear logo"
    # asm_seed = DK.linear_PSAM_seed(n_max=9)
    # asm_seed.save_logo("asm_seed.eps")
    # # DK.A.save_logo("A.eps")
    # # DK.B.save_logo("B.eps")
    # # print "linear alignment"
    # # print DK.linear
    # # print DK.linear.matrix
    # ls = DK.lin_score / DK.linear.wlen
    # ABs = (DK.A_score + DK.B_score) / (DK.A.wlen + DK.B.wlen)
    # print "A effective length", DK.A.wlen, "score", DK.A_score, "score-density", DK.A_score/DK.A.wlen
    # print "B effective length", DK.B.wlen, "score", DK.B_score, "score-density", DK.B_score/DK.B.wlen
    # print "combined", DK.A.wlen + DK.B.wlen, "score", DK.A_score + DK.B_score, "score-density", (DK.A_score + DK.B_score)/(DK.A.wlen + DK.B.wlen)

    # print "linear eff length", DK.linear.wlen, "score", DK.lin_score, "score-density", DK.lin_score/DK.linear.wlen
    # print "linear_motif score", DK.linear_motif_score

    # DK.interaction_plot()


    # print "A"
    # psam = DK.A.to_PSAM()
    # for mer, aff in zip(*psam.kmer_affinities):
    #     print mer, aff

    # print "B"
    # psam = DK.B.to_PSAM()
    # kmers, aff = psam.kmer_affinities
    # for mer, a in zip(kmers, aff):
    #     print mer, a



    # # TESTING mutual information
    # import matplotlib.pyplot as pp
    # from cyska import yield_kmers
    # km = 4
    # kmers = list(yield_kmers(km))
    # profs = []
    # joints = []
    # for reads in rbns.reads:
    #     joint = reads.joint_kmer_freq_distance_profile(km)
    #     joints.append(joint)
    #     prof = reads.kmer_mutual_information_profile(km)
    #     profs.append(prof)
    
    # j0 = joints[0]
    # for joint, reads in zip(joints[1:], rbns.reads):
    #     for d in range(18):
    #         print reads.name, d
            
    #         # print joint[:,:,d]
    #         # pp.figure()
    #         # pp.pcolormesh(joint[:,:,d])
    #         # pp.show()
    #         jR = np.log2(joint / j0)
    #         I = jR[:,:,d].flatten().argsort()[::-1]
    #         # print I
    #         for n in I[:10]:        data_colors = plt.get_cmap("YlOrBr")(np.linspace(.3, 1, len(labels)-1))

    #             i, j = np.unravel_index(n, joint.shape[:2])
    #             # print n, i, j
    #             print "most-co-enriched 3mers at d=", d, kmers[i], kmers[j], jR[i,j,d]

    # pp.figure()
    # pp.title(rbp_name)
    # for prof,reads in zip(profs[1:], rbns.reads[1:]):
    #     pp.plot(prof/profs[0], '.-', label=reads.name)
    
    # pp.legend()
    # pp.xlabel("{0}mer separation".format(km))
    # pp.ylabel("MI ratio to input")
    # pp.show()
    # sys.exit(0)
    
