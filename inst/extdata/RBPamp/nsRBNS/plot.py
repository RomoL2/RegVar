import os, sys
import numpy as np
import matplotlib.pyplot as pp
from scipy.stats import spearmanr, pearsonr
import scipy.stats
from RBPamp.report import density_scatter_plot
import RBPamp.cyska as cyska
from byo.io import fasta_chunks
import os, sys, logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.INFO)


class MutualInformationScore(object):
    def __init__(self, X, Y,n=10, n_permut = 100):
        self.X = X
        self.Y = Y
        self.n = n
        self.n_permut = n_permut
        self.xbins, self.Xd, self.Xf = self.make_eq_bins(X, n=n)
        self.xbins, self.Yd, self.Yf = self.make_eq_bins(Y, n=n)
        
        self.joint = MutualInformationScore.joint_freq(self.Xd, self.Yd, n)
        # print self.joint.shape
        self.indep = np.outer(self.Xf, self.Yf)
        # print self.indep.shape
        self.MI = MutualInformationScore.mutual_information(self.joint, self.indep)

        self.MI_permut = []
        N = len(self.Xd)
        for i in xrange(n_permut):
            perm = np.random.permutation(N)
            xd = self.Xd[perm]
            joint = MutualInformationScore.joint_freq(xd, self.Yd, n)
            self.MI_permut.append(MutualInformationScore.mutual_information(joint, self.indep))
        
        self.MI_permut = np.array(self.MI_permut)
        
        # ad hoc p-value
        s = np.std(self.MI_permut)
        m = np.mean(self.MI_permut)
        self.z = (self.MI - m)/s
        self.p_value = scipy.stats.norm.sf(self.z)

    
    @staticmethod
    def mutual_information(joint, indep):
        return (joint * np.log2(joint/indep)).sum()

    @staticmethod
    def joint_freq(xd, yd, n):
        joint = np.ones((n, n), dtype=np.float32)
        for x,y in zip(xd, yd):
            joint[x,y] += 1

        joint /= float(joint.sum())
        return joint
    
    def heatmap_plot(self, fname):
        pp.figure()
        pp.pcolor(self.joint, cmap='viridis')
        pp.colorbar(orientation='horizontal')
        pp.savefig(fname, dpi=150)
        pp.close()

    def dist_plot(self, fname):
        pp.figure()
        pp.hist(self.MI_permut, lw=2, histtype='step', bins=self.n_permut/self.n)
        pp.axvline(self.MI)
        pp.savefig(fname)
        pp.close()

    def make_eq_bins(self, x, n=10):
        I = x.argsort()
        N = len(I)

        bp_i = np.linspace(0,N-1, num=n)

        bins = [x[I[int(bp)]] for bp in bp_i]
        d = np.digitize(x, bins, right=True) # discretized version
        n = np.bincount(d) + 1
        f = n / float(n.sum())
        # print bins, f

        return bins, d, f




class nsRBNSOligos(object):

    def __init__(self, rna_conc=100., fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out'):
        self._seqs_raw = []
        self._fa_ids_raw = []
        self._index = {}
        self._seqs = {}

        self.nt_freqs = []
        def GC_content(seq):
            gc = 0
            for s in seq:
                if s == 'G' or s == 'C':
                    gc += 1
            return gc / float(len(seq))

        self.entropy = []
        def nt_entropy(seq):
            S = seq.upper().replace('U','T')
            counts = np.array([S.count('A'), S.count('C'), S.count('G'), S.count('T')])
            f = counts / float(counts.sum())
            self.nt_freqs.append(f)

            return - np.where(f > 0, f * np.log2(f), 0).sum()

        self.l5 = len(adap5)
        self.l3 = len(adap3)

        self.rna_conc = rna_conc
        self.adap5 = adap5
        self.adap3 = adap3
        self.fa_name = fa_name
        for i, (fa_id, seq) in enumerate(fasta_chunks(file(fa_name))):
            self._seqs_raw.append(seq)
            self._fa_ids_raw.append(fa_id)
            self._seqs[fa_id] = seq
            self._index[fa_id] = i
            self.entropy.append(nt_entropy(seq))

        self.N = len(self._seqs_raw)
        self.SEQ = [s[self.l5:-self.l3] for s in self._seqs_raw] 
        self.nt_freqs = np.array(self.nt_freqs)
        self.GC = self.nt_freqs[:,[1,2]].sum(axis=1)
        self.entropy = np.array(self.entropy)

        self.xtalk, self.xtalk_score = self.xtalk_from_blast(fname=xtalk_file)
        from RBPamp.reads import RBNSReads
        self.reads = RBNSReads.from_seqs(self.SEQ, fname=self.fa_name, rbp_name='nsRBNS', rna_conc=rna_conc, acc_storage_path='RBPamp/acc', adap5=self.adap5, adap3=self.adap3, temp=22.5)

    def xtalk_from_blast(self, fname='blast/results.out'):
        N = self.N
        xtalk = np.zeros( (N,N), dtype=np.float32)
        print "loading"

        for line in file(fname):
            parts = line.split('\t')
            query, subject, score = parts[:3]
            i = self._index[query]
            j = self._index[subject]

            xtalk[i,j] += float(score)

        rowsum = xtalk.sum(axis=1)
        ind = rowsum > 800
        xtalk = xtalk[ind,:][:,ind]
        # xtalk = xtalk[:100,:100]

        # outfile = file(fname+'.scoresums.tsv','w')
        # for i, r in enumerate(rowsum):
        #     if r < 1:
        #         continue
        #     outfile.write("{0}\t{1}\n".format(self._fa_ids_raw[i], r))

        return xtalk, rowsum

    def xtalk_plot(self, fname='xtalk.pdf'):
        xtalk = self.xtalk
        print xtalk.shape
        print xtalk.min(), xtalk.max(), xtalk.argmax()
        print "plotting"
        pp.figure()
        pp.imshow(np.log10(xtalk), cmap='viridis', interpolation='none')
        pp.colorbar()
        pp.savefig(fname)
        pp.close()

    def get_SPA_model(self, k=7, rbp_conc=[25.,125.,625.], seq_only=False):
        # Bridget used 250 nM of RNA oligos
        import RBPamp.spa

        path = os.path.dirname(self.fa_name)
        
        # reads.seqm
        mdl = RBPamp.spa.SPAModel(self.reads, k, rbp_conc, n_subsample=0, seq_only=seq_only)
        return mdl

    # def get_PSAM_model(self, params, rna_conc=100., rbp_conc=[25.,125.,625.], seq_only=False)
    #     from RBPamp.reads import RBNSReads
    #     from RBPamp.partfunc import PartFuncModel

    #     path = os.path.dirname(self.fa_name)
    #     reads = RBNSReads.from_seqs(self.SEQ, fname=self.fa_name, rbp_name='nsRBNS', rna_conc=rna_conc, acc_storage_path='RBPamp/acc', adap5=self.adap5, adap3=self.adap3)

    #     mdl = PartFuncModel(reads, params, None, rbp_conc=rbp_conc)
    #     return mdl

def phist(*argc, **kwargs):
    pp.hist(*argc, lw=2, bins=100, histtype='step', normed=True, cumulative=True, **kwargs)

from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def default_lm_data(lm):
    # print "Z1-shape", lm.Z1.shape, lm.Z1.min(), lm.Z1.max(), lm.Z1.argmin()

    data = np.array([
        lm.A, lm.C, lm.G, lm.T, 
        # lm.GC_score, 
        np.log2(lm.ns_exp.f0),
        # lm.entropy,
        lm.Z1,
        # lm.Zlog
        # acc,
            # data = np.array([
            #     A, C, G, T, GC_score, GC_score**2,
            #     np.log(self.f0),
            #     entropy,
            #     # np.log((P*self.state.Z1)**n/(1 + (P*self.state.Z1)**n)),
            #     Z1,
            #     Z1**2,
            #     Z1**3,
            #     acc,
            # ])
        
    ])
    # print "lm-data row-means before normalization", data.mean(axis=1)
    # df = pd.DataFrame(data=preprocessing.scale(data.T), columns = ['A','C','G','T','entropy','Z1'])
    # df = pd.DataFrame(data=preprocessing.scale(data.T), columns = ['A','C','G','T','GC_score', 'f0', 'entropy','Z1'])
    # df = pd.DataFrame(data=preprocessing.scale(data.T), columns = ['A','C','G','T','GC_score','f0', 'Z1'])
    df = pd.DataFrame(data=preprocessing.scale(data.T), columns = ['A','C','G','T','f0', 'Z1'])
    # print "lm-data row-means after normalization", df.values.shape, df.values.mean(axis=0)
    # print "lm-data row-std after normalization", df.values.shape, np.std(df.values, axis=0)

    # print "correlation coefficient matrix", df.corr()
    return df


class RBPBindModel(object):
    def __init__(self, ns_exp, name, param_file, seq_only=False, low_perc=10, z_cut=0, prepare_data=None, scale=1,k=7, mdl_type='SPA'):
        self.ns_exp = ns_exp
        self.ns = ns_exp.ns
        self.name = self.ns_exp.name + "_" + name
        self.param_file = param_file
        self.rbp_conc = ns_exp.rbp_conc
        self.z_cut = z_cut
        if not prepare_data:
            self.prepare_data = default_lm_data
        else:
            self.prepare_data = prepare_data

        self.state = self.evaluate_PSAM(param_file, seq_only=seq_only)

    def evaluate_PSAM(self, fname, seq_only = False):
        from RBPamp.params import ModelSetParams
        params = ModelSetParams.load(fname, 1)
        # seqm = self.ns.reads.get_padded_seqm(psam.n)
        # openen = self.ns.reads.acc_storage.get_raw(psam.n)
        # acc = openen.acc
        # print params
        if seq_only:
            params.acc_k = 0 
        # Z1 = cyska.PSAM_partition_function(seqm, acc, psam.psam, openen_ofs = openen.ofs - psam.n + 1)

        # params.acc_scale = 1.
        # import RBPamp.cyska as cyska
        seqm, accs_k, accs, accs_scaled, accs_ofs = self.ns.reads.get_data_for_PSAM(params, full_reads=True)
        # Z1 = self.ns.reads.PSAM_partition_function(params, full_reads=True)
        Z1 = self.ns.reads.evaluate_partition_function(params, seqm, accs_k, accs, accs_scaled, accs_ofs)
        Zm = self.ns.reads.evaluate_partition_function_split(params, seqm, accs_k, accs, accs_scaled, accs_ofs)

        Zm_read = [cyska.clipped_sum_and_max(zm, clip=1E6) for zm in Zm]
        Z1_read, Z1_read_max = cyska.clipped_sum_and_max(Z1, clip=1E6)

        params0 = params.copy()
        params0.acc_k = 0
        Z1_0 = self.ns.reads.PSAM_partition_function(params0, full_reads=True)
        class State(object):
            pass
        
        s = State()
        s.seqm = seqm
        s.accs_k = accs_k
        s.accs = accs
        s.accs_scaled = accs_scaled
        s.accs_ofs = accs_ofs
        s.Z1 = Z1_read[self.ns_exp.indices]
        s.Zm = Zm
        s.params = params
        s.invkd_bare = Z1_0  # ignore accessibility. primary sequence only
        s.invkd_SPA = Z1  # expected affinity if single-protein approx. were correct
        s.invkd_SPA_read = Z1_read 

        return s # return a fake "state" object. Only thing needed is per-read partition function

    def run_rbpbind(self, j):
        import rbpbind as rb
        params = self.state.params
        def bits_to_seq(v):
            return "".join(["ACGU"[i] for i in v])
    
        # print self.ns.reads.adap5, "...", self.ns.reads.adap3
        seqm = self.ns.reads.get_full_seqm()
        acc = self.ns.reads.acc_storage.get_raw(params.acc_k).acc

        # print "{0}mer PSAM, {1}mer footprint starting at {2}".format(params.k, params.acc_k, params.acc_shift)
        # print "full sequence length", seqm.shape[1]
        # print "accessibilities per row", acc.shape[1]

        # print seqm.shape
        shift = params.acc_shift
        sbits = seqm[j]
        seq = bits_to_seq(sbits)
        # ofs = self.ns.reads.l5 - params.k + params.acc_k + params.acc_shift
        ofs = params.acc_shift
        invkd = np.zeros(len(sbits), dtype=np.float32)
        indep = np.zeros(len(sbits), dtype=np.float32)
        buffer = np.zeros(len(sbits), dtype=np.float32)

        bare = self.state.invkd_bare[j]
        aff = self.state.invkd_SPA_read[j] * params.A0
        Kd = 1./aff
        # print bare.max(), "max rel. affinity", bare.argmax()
        # print "expected Kd_eff", Kd
        l = len(bare)
        # print len(sbits), l, ofs
        invkd[ofs:l + ofs] = self.state.invkd_bare[j] * params.A0
        indep[ofs:l + ofs] = self.state.invkd_SPA[j] * params.A0

        # print "aff_motif", self.state.invkd_bare[j].shape
        # print "7-mer accessibility", 
        # print (invkd[:147] * acc[j])[:30]
        # print indep[:30]
        
        # print "initializing RBPbind"
        buffer[params.acc_k-1:] = invkd[:-params.acc_k+1]
        # buffer[46+7] = 0.15594828
        rb.init(T=self.ns.reads.temp)
        rb.set_invkd_vector(buffer, params.acc_k)
        # rb.dump_invkd(seq)

        # a = acc[j]
        # print "accessibilities"
        # for i in range(len(seq)-7):
        #     print i, seq[i:i+7], a[i]

        # rb.dump_invkd(seq)
        # fold w/o protein
        pf0 = rb.compute_Z(seq, 0, update_seq=False)
        # print "no RBP", pf0


        p_bound = []
        rbp_conc = np.array([Kd/100, Kd/10, Kd/2, Kd, 2*Kd, 10*Kd, 100*Kd])
        k_i = 3
        rbp_conc = 10**np.linspace(-2, 2, 21) * Kd
        k_i = 11

        # rb.dump_invkd(seq)

        for conc in rbp_conc:
            pfc = rb.compute_Z(seq, conc, update_seq=False)
            p = 1. - np.exp(-(pf0-pfc))
            # print "with {0} nM {1} -> pbound ={2}".format(conc, pfc, p)
            p_bound.append( p )                

        p_bound = np.array(p_bound)
        p_exp = rbp_conc / (rbp_conc + 1./aff)
        print p_bound
        print p_exp

        RMSE = np.sqrt(((p_bound - p_exp)**2).mean())
        print "RMSE", RMSE, p_bound
        if RMSE > .0 and p_bound[k_i] > .0:
            print ">>>>>", j, Kd
            print seq

            pp.figure()
            pp.loglog(rbp_conc, p_bound, label="RBPbind")
            pp.loglog(rbp_conc, p_exp, label="SPA")
            pp.legend()
            pp.xlabel("RBP conc [nM]")
            pp.ylabel("p_bound")
            pp.savefig("p_bound_{}.pdf".format(j))
            pp.close()
            # fold at expected Kd
            z0 = rb.compute_Z(seq, Kd, update_seq=False)
            print "at Kd_expect", z0, "->p_bound=", 1. - np.exp(-(pf0-z0))
            # get occupancies from derivative

            pp.figure()
            occ = rb.occ_vector_z0(z0, len(seq) - params.acc_k + 1, seq, Kd)
            occ_naive = Kd / (Kd + 1/indep)
            pp.plot(occ, "x", label="RBPbind")
            pp.plot(occ_naive, ".", label="SPA")
            pp.ylabel("occupancy")
            pp.xlabel("footprint pos [nt]")
            pp.legend()
            pp.savefig('occ_{}.pdf'.format(j))
            pp.close()

            i = indep.argmax()
            a = acc[j, i]
            keff = 1./(invkd[i] * acc[j, i])
            oexp = Kd / (Kd + keff)
            print "at position {0} we have acc={1} aff={2} -> Keff={3} occ={4}".format(i, acc[j, i], invkd[i], keff, oexp)

            for i, o in enumerate(occ):
                print i, seq[i:i+params.acc_k], invkd[i], indep[i], "RBPbind_occ=",o, "SPA_occ=", Kd / (Kd + 1/indep[i])

class nsRBNSModel(object):
    def __init__(self, ns_exp, name, param_file, seq_only=False, low_perc=10, z_cut=0, prepare_data=None, scale=1,k=7, mdl_type='SPA'):
        self.ns_exp = ns_exp
        self.ns = ns_exp.ns
        self.name = self.ns_exp.name + "_" + name
        self.param_file = param_file
        self.rbp_conc = ns_exp.rbp_conc
        self.z_cut = z_cut
        if not prepare_data:
            self.prepare_data = default_lm_data
        else:
            self.prepare_data = prepare_data

        if mdl_type == 'SPA':
            self.state = self.evaluate_SPA(param_file, k=k, seq_only=seq_only, scale=scale)
        elif mdl_type == 'bg':
            self.state = self.evaluate_background()
        else:
            self.state = self.evaluate_PSAM(param_file, seq_only=seq_only)
        # self.betas = self.optimal_betas()

        self.A, self.C, self.G, self.T = self.ns.nt_freqs[self.ns_exp.indices].T
        self.GC = self.C+self.G
        self.AT = self.A+self.T
        self.res = []
        self.GC_score = self.GC/(1-self.GC)
        
        # self.Z1 = np.log(self.state.Z1)
        self.Z1 = self.state.Z1
        self.Zlog = np.log(self.state.Z1/ self.state.Z1 + 1.)
        self.mean_enr = self.ns_exp.enr.mean(axis=0)
        self.entropy = 2 - self.ns.entropy[self.ns_exp.indices]
        self.coeff_matrix = []

        self.thresh = np.percentile(self.state.Z1, low_perc)
        self.I_low = self.state.Z1 <= self.thresh

    # def optimal_betas(self):
    #     betas = []
    #     for p, obs in zip(self.state.p_bound, self.ns_exp.enr):
    #         betas.append(self.optimize_beta(p, obs))
        
    #     return np.array(betas)

    # def optimize_beta(self, p, obs):

    #     def to_opt(beta):
    #         expect = self.predict_R(p, beta)
    #         # R = np.corrcoef(np.log2(obs), np.log2(expect))[0][1]
    #         R, p_value = pearsonr(np.log2(obs), np.log2(expect))

    #         return (R - 1)**2
        
    #     from scipy.optimize import minimize_scalar

    #     opt = minimize_scalar(to_opt, bounds=[0,1.], method='bounded')
    #     # print opt
    #     return opt.x

    def evaluate_SPA(self, fname, k=7, seq_only=False, scale=1.):
        letters = 'ACGT'
        def convert(bits):
            return "".join([letters[i] for i in bits])
        mdl = self.ns_exp.ns.get_SPA_model(k=k, rbp_conc = self.ns_exp.rbp_conc, seq_only=seq_only)  

        # this selects directly form the index_matrix, NOT seqm. So seqm is invalid after subsample!?
        mdl.new_subsample(indices=self.ns_exp.indices)
        if fname.endswith('rnacompete'):
            mdl.parameters.load_rbpbind(fname)
        else:
            mdl.parameters.load(fname)

        mdl.params[:mdl.nA] *= scale
        if self.z_cut:
            z = np.array(mdl.params[:mdl.nA])
            z -= z.mean()
            z /= z.std()
            print "z-scores", z.min(), z.mean(), z.max()
            print "best kmer", cyska.index_to_seq(z.argmax(), 7)

            mask = z < self.z_cut
            mdl.params[:mdl.nA][mask] = 1e-6
            print "pulled", mask.sum(), 'kmer affinities to 0 because their z-score was <', self.z_cut
            print (mask ==0).sum(), "left"

        state = mdl.evaluate(mdl.params)

        return state        

    def evaluate_PSAM(self, fname, seq_only = False):

        from RBPamp.params import ModelSetParams
        params = ModelSetParams.load(fname, 1)
        # seqm = self.ns.reads.get_padded_seqm(psam.n)
        # openen = self.ns.reads.acc_storage.get_raw(psam.n)
        # acc = openen.acc
        # print params
        if seq_only:
            params.acc_k = 0 
        # Z1 = cyska.PSAM_partition_function(seqm, acc, psam.psam, openen_ofs = openen.ofs - psam.n + 1)


        seqm, accs_k, accs, accs_scaled, accs_ofs = self.ns.reads.get_data_for_PSAM(params, full_reads=True)
        # Z1 = self.ns.reads.PSAM_partition_function(params, full_reads=True)
        Z1 = self.ns.reads.evaluate_partition_function(params, seqm, accs_k, accs, accs_scaled, accs_ofs)
        Z1_read, Z1_read_max = cyska.clipped_sum_and_max(Z1, clip=1E6)

        Zm = self.ns.reads.evaluate_partition_function_split(params, seqm, accs_k, accs, accs_scaled, accs_ofs)
        Zm_read = [cyska.clipped_sum_and_max(zm, clip=1E6)[0][self.ns_exp.indices] for zm in Zm]

        # params0 = params.copy()
        # params0.acc_k = 0
        # Z1_0 = self.ns.reads.PSAM_partition_function(params0, full_reads=True)

        class State(object):
            pass
        
        s = State()
        s.seqm = seqm
        s.accs_k = accs_k
        s.accs = accs
        s.accs_scaled = accs_scaled
        s.accs_ofs = accs_ofs
        s.Z1 = Z1_read[self.ns_exp.indices]
        s.Zm = Zm_read

        s.params = params
        # s.invkd_bare = Z1_0  # ignore accessibility. primary sequence only
        s.invkd_SPA = Z1  # expected affinity if single-protein approx. were correct
        s.invkd_SPA_read = Z1_read 

        return s 


    def evaluate_background(self):

        class State(object):
            pass
        
        s = State()
        s.Z1 = np.ones(len(self.ns_exp.indices), dtype=float)
        return s

    def akira_fit(self, df, logR):
        
        f0 = self.ns_exp.f0
        Z = self.state.Z1
        # print Z.shape, Z.min(), Z.max()

        def scale(acc, a):
            acc_scaled = np.array(acc)
            cyska.pow_scale(acc_scaled, a)
            return acc_scaled

        def predict(params):
            conc = params[-1]
            beta = params[-2]
            # ascale = params[-3]

            # accs_scaled = [scale(acc, ascale) for acc in self.state.accs]

            # # non-linear model (fitting concentration) for binding
            # Z1 = self.ns.reads.evaluate_partition_function(
            #     self.state.params,
            #     self.state.seqm, 
            #     self.state.accs_k, 
            #     self.state.accs,
            #     accs_scaled,
            #     self.state.accs_ofs
            # )
            # Z, Zmax = cyska.clipped_sum_and_max(Z1, clip=1E6)
            # Z = Z[self.ns_exp.indices]

            p_bound = conc / (conc + 1./Z)
            q = f0 * (p_bound + beta)
            # print "q", q.shape, q.min(), q.max()

            # linear model for background enrichment
            coeff = params[:-2]
            # print df.iloc[:, :len(coeff)]


            w = q
            f = w / w.sum()

            lR_pred = np.log2(f / f0)
            # print "lR_bound prediction", lR_pred.min(), lR_pred.max()
            
            lR_lin = (coeff[np.newaxis, :] * df.iloc[:, :len(coeff)]).sum(axis=1)

            return lR_pred #+ lR_lin
        
        def error(params):
            lR_pred = predict(params)
            err = ((lR_pred - logR)**2).mean()
            
            # print params[-2:], '->', err
            return err
        
        from scipy.optimize import minimize

        p0 = np.ones(df.shape[1] + 1, dtype=float)
        # y0 = predict(p0)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(logR,y0, '.')
        # plt.savefig('y0_{self.name}.pdf'.format(self=self))
        # plt.close()

        bounds = [(None, None)] * len(p0)
        bounds[-1] = (1e-3, 1000.) # effective RBP concentration
        bounds[-2] = (1e-6, 10.) # beta background parameter
        # bounds[-3] = (0., 1.) # accessibility scale
        # print bounds
        res = minimize(error, p0, bounds=bounds, method='L-BFGS-B', options=dict(eps=1e-3))
        # print res
        lR_pred = predict(res.x)
        # plt.figure()
        # plt.plot(logR, lR_pred, '.')
        # plt.savefig('y1_{self.name}.pdf'.format(self=self))
        # plt.close()

        return res, lR_pred


    def lm_fit(self, akira=False):
        df = self.prepare_data(self)
        # from sklearn.ensemble import RandomForestRegressor

        fits = []
        for conc, enr in zip(self.ns_exp.rbp_conc, self.ns_exp.enr):
            # reg = linear_model.LinearRegression()
            # reg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
            # reg = linear_model.RidgeCV(alphas=[.1,.3,.5,.75,1.])
            reg = linear_model.Ridge(alpha=[.5])

            y = np.log2(enr)
            if akira:
                res, ya = self.akira_fit(df, y)
                # print "biophysics optimal parameters", res.x[-2:], res.success, res.fun
                # print "AKIRA fit alone explains {:.3f} % of variance".format(r2_score(y, ya) * 100.)
                # replace partition function with AKIRA-predicted 
                # log2(R-values) and only regress on the residuals
                df['lR_AKIRA'] = preprocessing.scale(ya)
                for par, zm in zip(self.state.params, self.state.Zm):
                    # print zm.shape
                    cons = par.as_PSAM().consensus
                    df['Z_{}'.format(cons)] = preprocessing.scale(zm)

            DF = df.copy()
            DF['log2R'] = y
            # print "cross-correlation matrix before fit", DF.corr()
                # print "lm-data row-means after akira fit", df.values.shape, df.values.mean(axis=0)
                # print "lm-data row-std after akira fit", df.values.shape, np.std(df.values, axis=0)

            reg.fit(df, y)

            # predict on full data
            y_pred = reg.predict(df)
            
            # attach data to regression object
            reg.df = df
            reg.conc = conc
            reg.y_obs = y
            reg.y_pred = y_pred
            reg.residuals = y-y_pred

            # and various metrics
            reg.spearman_R, reg.spearman_pval = spearmanr(y, y_pred)
            reg.pearson_R, reg.pearson_pval = pearsonr(y, y_pred)
            reg.r2_full = r2_score(y, y_pred) * 100.
            reg.r2_bg = r2_score(y[self.I_low], y_pred[self.I_low])* 100

            fits.append(reg)

            print ">>>>>", self.ns_exp.name, self.name, conc, reg.r2_full, '%'
            # print "coeff", reg.coef_
            d = df.values
            # print d.shape, d.mean(axis=0)
            # print "intercept", reg.intercept_
            # print "r2 on bg", reg.r2_bg, '%'
            # print "r2 on full ", reg.r2_full, '%'
            # print "alpha", reg.alpha_

        return fits

    def regression_analysis(self, scatter_plots=False, res_plots=True, akira=False):
        fits = self.lm_fit(akira=akira)
        if scatter_plots:
            for reg in fits:
                self.prediction_scatter_plot(reg)

        if res_plots:
            for reg in fits:
                self.residuals_plot(reg)
        
        self.coeff_matrix_plot(fits)
        return fits

    def prediction_scatter_plot(self,reg):
        pp.figure()
        # pp.plot(y, y_pred, '.', label='{conc:.1f}nM: rho={R:.3f} (P < {p_val:.2e})'.format(**locals()))
        density_scatter_plot(reg.y_obs, reg.y_pred,x_ref=False, label=r'{reg.conc:.1f}nM: $\rho$={reg.spearman_R:.3f} (P<{reg.spearman_pval:.2e}) $R^2$={reg.r2_full:.0f}%'.format(**locals()))
        pp.xlabel('log2 nsRBNS enrichment')
        pp.ylabel('linear model prediction')
        pp.legend(loc='upper center', facecolor='white')
        pp.savefig('{self.name}_{reg.conc}_scatter.pdf'.format(**locals()), dpi=150)
        pp.close()
            
    def coeff_matrix_plot(self, fits):
        coeff_matrix = np.array([reg.coef_ for reg in fits])
        pp.figure()
        pp.pcolor(coeff_matrix)
        pp.colorbar(orientation='horizontal', fraction=.05, label='weight')
        pp.xlabel("regression coefficient")
        pp.ylabel("experiment")
        df = fits[0].df
        
        # pp.xticks(np.arange(len(df.columns)+1)+.5, list(df.columns) + ['intercept'], rotation=45)
        pp.xticks(np.arange(len(df.columns))+.5, df.columns, rotation=45)
        pp.yticks(np.arange(len(self.ns_exp.rbp_conc))+.5, ["{0:.1f}nM".format(c) for c in self.ns_exp.rbp_conc])
        pp.tight_layout()
        pp.savefig('{self.name}_coeff_matrix.pdf'.format(**locals()))
        pp.close()

    def residuals_plot(self, reg):
            # pp.plot(df['A'], res, '.', label='A')
            # pp.plot(df['C'], res, '.', label='C')
            # pp.plot(df['G'], res, '.', label='G')
            # pp.plot(df['T'], res, '.', label='T')
            # pp.plot(df['GC'], res, '.', label='log(GC)')
            # pp.plot(df['f0'], res, '.', label='log(f0)')
            # pp.plot(df['entropy'], res, '.', label='entropy')
            # pp.plot(df['binding'], res, '.', label='log(Z1)')

        for col in reg.df.columns:
            pp.figure()
            pp.title('{0} residuals'.format(col))
            density_scatter_plot(reg.df[col], reg.residuals, x_ref=False, label=col)
            pp.legend(loc='upper center', facecolor='white')
            pp.savefig('{self.name}_{col}_{reg.conc}_residual.pdf'.format(**locals()))
            pp.close()

            # res.append() # keep the residuals

class nsRBNSExperiment(object):
    def __init__(self, nsrbns, fcount_matrix, name='', rbp_conc = [25.,125.,625.], pseudo=1, skip_xtalk=True, skip_low=True, seq_only=False):
        if not name:
            name = fcount_matrix.split('_')[0]
        self.name = os.path.basename(name)
        self.ns = nsrbns
        
        self.fcount_matrix = fcount_matrix
        self.indices, self.counts = self.load_counts(fcount_matrix, skip_xtalk=skip_xtalk, skip_low=skip_low)

        self.N = self.counts.sum(axis=1)
        self.scale = self.N[1:]/self.N[0]
        self.enr = (self.counts[1:,:] + pseudo) / (self.counts[0,:] + pseudo) * self.scale[:,np.newaxis]
        self.frac = self.counts / self.N[:,np.newaxis]
        self.f0 = self.frac[0]
        self.rbp_conc = rbp_conc

        
        # self.enr_expect, self.pearson, self.ppval, self.spearman, self.spval = self.prediction()

    def load_counts(self, fname, skip_xtalk=True, skip_low=True):
        counts = []
        indices = []

        for line in file(fname):
            parts = line.split('\t')
            name = parts[0]
            if skip_xtalk and (self.ns.xtalk_score[self.ns._index[name]] > 0):
                # skip cross-talking oligos!
                continue

            n0 = float(parts[1]) # freq in input
            if skip_low and n0 < 100:
                continue

            indices.append(self.ns._index[name])
            counts.append(parts[1:])
            
            s = self.ns._seqs[name]
            if not self.ns._seqs_raw[self.ns._index[name]] == s:
                print ">>>MISMATCH", name
                print "CM ",s
                print "RAW", self.ns._seqs_raw[self.ns._index[name]]

            # if (not 'GCATG' in s) and (not 'GCACG' in s):
            #     motif.append(0)
            #     # print s
            # elif 'TGCATGT' in s or 'TGCATGC' in s:
            #     motif.append(10)
            # elif 'AGCATG' in s or 'CGCATG' in s or 'TGCATG' in s:
            #     motif.append(5)
            # else:
            #     motif.append(1)
        counts = np.array(counts, dtype=float).T
        indices = np.array(indices)
        # print "COUNT MATRIX", counts.shape, counts.min(axis=0), counts.max(axis=0), "finite", np.isfinite(counts).all()

        return indices, counts

        # acc = np.log(self.state.mdl.reads.acc_storage.get_raw(11).acc[self.indices].mean(axis=1))
        # print acc.shape
            # P = conc
            # n = 1.
            # data = np.array([
            #     A, C, G, T, GC_score, GC_score**2,
            #     np.log(self.f0),
            #     entropy,
            #     # np.log((P*self.state.Z1)**n/(1 + (P*self.state.Z1)**n)),
            #     Z1,
            #     Z1**2,
            #     Z1**3,
            #     acc,
            # ])
            # data = preprocessing.scale(data.T)
            # df = pd.DataFrame(data=data, columns = ['A','C','G','T','GC', 'GC2', 'log_f0', 'entropy','binding','bind2', 'bind3', 'mean_acc'])





            # df = pd.DataFrame(data=data, columns = ['A','C','G','T','GC', 'f0', 'entropy','binding',])
            

        # pp.show()


    def affinity_selection_plot(self, lm, fname = '{lm.name}_affinity_selection.pdf'):
        ### Affinity selection plot
        pp.figure()
        Z = np.log10(lm.state.Z1)
        # Z -= Z.mean()
        # Z /= Z.std()
        phist(Z, color='gray', label='input')
        for i, f in enumerate(self.counts[1:]):
            phist(Z, weights = f, label="{0}nM".format(self.rbp_conc[i]))

        pp.xlabel("natural sequence affinity [1/nM] (log10)")
        pp.ylabel("rel. cumulative frequency of pull-down")
        pp.legend(loc='upper left')
        pp.savefig(fname.format(**locals()))
        pp.close()

    def GC_bias_plot(self, fname='GC_{self.name}.pdf', n_bin = 100):
        ### GC bias plot
        GC = self.ns.GC[self.indices]
        Y = np.log2(self.enr)
        I = GC.argsort()

        GC = GC[I]
        Y = Y[:,I]
        # x = []
        # y = []
        # i = 0
        pp.figure()
        for y in Y:
            pp.plot(GC, y, '.')
        

        # phist(GC, weights=self.counts[0], color='gray', label="input")
        # for i, c in enumerate(self.counts[1:]):
        #     phist(GC, weights=c, label="{0}nM".format(self.rbp_conc[i]))

        # pp.xlabel("G,C content of natural sequence")
        # pp.ylabel("rel. cumulative frequency")
        # pp.legend(loc='upper left')
        pp.savefig(fname.format(self=self))
        pp.close()

    def entropy_plot(self, fname='entropy.pdf'):
        ### nt entropy bias plot
        pp.figure()
        ent = self.ns.entropy[self.indices]
        phist(ent, weights=self.counts[0], color='gray', label="input")
        for i, c in enumerate(self.counts[1:]):
            phist(ent, weights=c, label="{0}nM".format(self.rbp_conc[i]))

        pp.xlabel("nt entropy of natural sequence [bits]")
        pp.ylabel("rel. cumulative frequency")
        pp.legend(loc='upper left')
        pp.savefig(fname)
        pp.close()

    # def predict_R(self, p, beta):
    #     expect = (p + beta) * self.f0 
    #     expect /= expect.sum()
    #     expect /= self.f0
    #     return expect

    # def prediction(self):
    #     pearson = []
    #     ppval = []
    #     spearman = []
    #     spval = []
    #     enr_expect = []

    #     for conc, p, obs, beta in zip(self.rbp_conc, self.state.p_bound, self.enr, self.betas):
    #         expect = self.predict_R(p, beta)
    #         enr_expect.append(expect)
            
    #         R, pp_value = pearsonr(np.log2(obs), np.log2(expect))
    #         rho, p_value = spearmanr(np.log2(obs), np.log2(expect))
    #         pearson.append(R)
    #         ppval.append(pp_value)
    #         spearman.append(rho)
    #         spval.append(p_value)

    #     return np.array(enr_expect), np.array(pearson), np.array(ppval), np.array(spearman), np.array(spval)

    # def scatter_plot(self, fname='{self.fcount_matrix}_{conc:.2f}.pdf'):
    #     for conc, expect, obs, R, pp_value, rho, p_value in zip(self.rbp_conc, self.enr_expect, self.enr, self.pearson, self.ppval, self.spearman, self.spval):
    #         pp.figure()
    #         pp.loglog(obs, expect, '.', label=r"conc={conc:.1f} R={R:.2f} (P={pp_value:.2e}) $\rho=${rho:.2f} (P={p_value:.2e})".format(**locals()), rasterized=True)
    #         pp.legend()
    #         pp.savefig(fname.format(**locals()))
    #         pp.close()

    def heatmap_plot(self, lm, fname='{lm.name}_heatmap.pdf'):
        Z = lm.state.Z1
        I = (-Z).argsort()
        N = len(Z)

        pp.figure(figsize=(5,7))
        pp.subplot(311)
        pp.semilogy(Z[I], color='k', label='RBNS model prediction')
        pp.xlim(0,N)
        pp.ylabel('predicted affinity [1/nM]')
        pp.xlabel('natural sequence index')
        pp.legend(loc='upper right')
        
        import matplotlib.colors as colors
        pp.subplot(312)
        data = self.enr[:,I]
        vmin, vmax = np.percentile(data, [5.,95.])
        pp.pcolor(data, cmap='viridis', norm=colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True )
        pp.yticks(np.arange(len(self.rbp_conc))+.5, ["{0:.1f} nM".format(c) for c in self.rbp_conc])
        pp.xticks([],[])
        pp.xlim(0,N)
        pp.colorbar(orientation='horizontal', fraction=.05, label='observed nsRBNS enrichment')

        pp.subplot(313)
        # pp.semilogy(Z[I])
        pp.ylabel('observed mean log2 enrichment')
        
        import scipy.ndimage.filters
        for conc, R in zip(self.rbp_conc, self.enr):
            lR = np.log2(R[I])
            # avg, var = moving_average(lR)
            avg = scipy.ndimage.filters.gaussian_filter1d(lR, 50.)

            pp.plot(avg, label="nsRBNS {0:.1f} nM".format(conc))
        
        pp.xlim(0,N)
        pp.legend(loc='upper right')
        pp.tight_layout()
        pp.savefig(fname.format(**locals()))
        pp.close()

    def error_analysis(self, fits, track, name='source', fname='error_{name}.pdf'):
        # sort oligos by prediction error
        N = len(self.indices)
        all_lfc = np.array([ reg.residuals for reg in fits])
        II = []

        pp.figure(figsize=(5,7))
        pp.subplot(311)
        for conc, lfc in zip(self.rbp_conc, all_lfc):
            I = (-lfc).argsort()
            pp.plot(lfc[I], label='prediction error @{0:.1f}nM'.format(conc))
            II.append(I)

        II = np.array(II)
        pp.xlim(0,N)
        pp.ylabel(r'$\log_2 \frac{expect}{observed}$')
        pp.xlabel('natural sequence index')
        pp.legend(loc='upper right')
        
        import matplotlib.colors as colors
        pp.subplot(312)
        track = track[self.indices]
        data = np.array([track[I] for I in II])
        vmin, vmax = np.percentile(data, [5.,95.])

        pp.pcolor(data, cmap='viridis', vmin=vmin, vmax=vmax )
        pp.yticks(np.arange(len(self.rbp_conc))+.5, ["{0:.1f} nM".format(c) for c in self.rbp_conc])
        pp.xticks([],[])
        pp.xlim(0,N)

        from matplotlib import ticker
        cb = pp.colorbar(orientation='horizontal', fraction=.05, label=name)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.update_ticks()

        pp.subplot(313)
        # pp.semilogy(Z[I])
        pp.ylabel('observed mean {name}'.format(name=name))
        
        import scipy.ndimage.filters
        for conc, R in zip(self.rbp_conc, data):
            avg = scipy.ndimage.filters.gaussian_filter1d(R, 50.)
            pp.plot(avg, label="nsRBNS {0:.1f} nM".format(conc))
        
        pp.xlim(0,N)
        pp.legend(loc='lower center')
        pp.tight_layout()
        pp.savefig(fname.format(**locals()))
        pp.close()


def get_r2(fits):
    r2s = [f.r2_full for f in fits]
    # r2s = [f.spearman_R for f in fits]
    return np.array(r2s)
    

def rbfox2_analysis(lp=0, z_cut=4):
    import seaborn as sns
    nsrbns = nsRBNSOligos(fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out')
    # nsrbns = nsRBNSOligos(fa_name = '3utrOligoPool_final_T7.fa', adap5='GGGAGTTCTACAGTCCGACGATC', adap3='TGGAATTCTCGGGTGCCAAG', xtalk_file='blast/bridget_results.out')
    # exp = nsRBNSExperiment(nsrbns, 'rbfox2_matrix.csv', skip_xtalk=True, seq_only=False)
    exp = nsRBNSExperiment(nsrbns, 'mapping/rbfox2_trial1.counts', skip_xtalk=False, seq_only=False)
    # exp.noaffinity_analysis(nsrbns.GC, 'GC content')
    # exp.noaffinity_analysis(nsrbns.entropy, 'entropy')
    # kw = dict(scatter_plots=True, res_plots=False)
    mbase = '/home/mjens/engaging/RBNS/RBFOX2/RBPamp/CI10M/'
    exp.GC_bias_plot()
    kw = dict(scatter_plots=False, res_plots=False, akira=False)

    lm = nsRBNSModel(exp, 'bg only', None, seq_only=False, low_perc=lp, mdl_type='bg')
    fits_bg = lm.regression_analysis(**kw)

    lm = nsRBNSModel(exp, 'RNAcompete_na', 'RBFOX1.rnacompete', seq_only=True, low_perc=lp)
    fits_rc_na = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=False)

    lm = nsRBNSModel(exp, 'RNAcompete', 'RBFOX1.rnacompete', seq_only=False, low_perc=lp)
    fits_rc = lm.regression_analysis(**kw)

    lm = nsRBNSModel(exp, 'R_values_na', 'rbfox3_R_value_7mers.tsv', seq_only=True, low_perc=lp, z_cut=z_cut)
    fits_r_na = lm.regression_analysis(**kw)

    lm = nsRBNSModel(exp, 'R_values', 'rbfox3_R_value_7mers.tsv', seq_only=False, low_perc=lp, z_cut=z_cut)
    fits_r = lm.regression_analysis(**kw)

    # lm = nsRBNSModel(exp, 'SPA_na', 'rbfox3_psam_spa_grad_7mers.tsv', seq_only=True, low_perc=10)
    # psam_model = '/scratch/data/RBNS/RBFOX3/RBPamp/test_mfa2/meanfield/7mer_affinities.tsv'
    psam_model = 'RBFOX3_nostruct.tsv'
    psam_model = '/home/mjens/engaging/RBNS/RBFOX2/RBPamp/CI10M/opt_nostruct/parameters.tsv'
    psam_model = mbase + 'opt_nostruct/parameters.tsv'
    lm = nsRBNSModel(exp, 'SPA_psam_na', psam_model, seq_only=True, low_perc=lp, mdl_type='PSAM')
    fits_psam_na = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=True)

    psam_model = "RBFOX2_struct_PSAM.tsv"  # 'RBFOX3_full.tsv'
    psam_model = '/home/mjens/engaging/RBNS/RBFOX2/RBPamp/CI10M/opt_full/parameters.tsv'
    psam_model = mbase + 'opt_full/parameters.tsv'
    lm = nsRBNSModel(exp, 'SPA_psam', psam_model, seq_only=False, low_perc=lp, mdl_type='PSAM')
    fits_psam = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=True)
    exp.heatmap_plot(lm)

    # kmer_model = '/scratch/data/RBNS/RBFOX3/RBPamp/test_mfa2/affinity/7mer_affinities.tsv'
    # # kmer_model = '/scratch/data/RBNS/RBFOX3/RBPamp/test_mfa2/affinity/7mer_affinities__UGCAUGC_Kd=3.368e-01_t=11.tsv'
    # # lm = nsRBNSModel(exp, 'SPA_kmer_na', '/scratch/data/RBNS/RBFOX3/RBPamp/test_mfa2/affinity/7mer_affinities__UGCAUGC_Kd=3.368e-01_t=11.tsv', seq_only=True, low_perc=10)
    # lm = nsRBNSModel(exp, 'SPA_kmer_na', kmer_model, seq_only=True, low_perc=lp)
    # fits_kmer_na = lm.regression_analysis(**kw)

    # # lm = nsRBNSModel(exp, 'SPA_kmer', '/scratch/data/RBNS/RBFOX3/RBPamp/test_mfa2/affinity/7mer_affinities__UGCAUGC_Kd=3.368e-01_t=11.tsv', seq_only=False, low_perc=10)
    # lm = nsRBNSModel(exp, 'SPA_kmer', kmer_model, seq_only=False, low_perc=lp)
    # fits_kmer = lm.regression_analysis(scatter_plots=True, res_plots=False)
    # exp.heatmap_plot(lm)

    def pos_control_data(lm):
        data = np.array([
            lm.A, lm.C, lm.G, lm.T, #GC_score,
            lm.entropy,
            np.log2(lm.mean_enr),
        ])
        df = pd.DataFrame(data=preprocessing.scale(data.T), columns = ['A','C','G','T','entropy','mean_enr'])
        return df

    # lm = nsRBNSModel(exp, 'mean_nsRBNS', 'RBFOX1.rnacompete', seq_only=False, low_perc=10, prepare_data=pos_control_data)
    # fits_m = lm.regression_analysis(**kw)

    x = np.arange(len(exp.rbp_conc))*.7
    pp.figure(figsize=(5,4))
    w = .09
    colors= ['#80FFC3','#98E86D','#FFF384','#E8B668','#FF8912','#FF886A','#D552FF','#B069FF']
    pp.bar(x + 0, get_r2(fits_bg), w, label='background', facecolor='.8')
    pp.bar(x + .1, get_r2(fits_r_na), w, label='top R-values', facecolor=colors[0])
    pp.bar(x + .2, get_r2(fits_r), w, label='top R-values (+footprint)', facecolor=colors[1])
    pp.bar(x + .3, get_r2(fits_psam_na), w, label='AKIRA (seq. only)', facecolor=colors[3])
    pp.bar(x + .4, get_r2(fits_psam), w, label='AKIRA', facecolor=colors[4])
    pp.bar(x + .5, get_r2(fits_rc_na), w, label='RNAcompete', facecolor=colors[2])
    # pp.bar(x + .5, get_r2(fits_kmer_na), w, label='our model (kmer w/o structure)', facecolor=colors[5])
    # pp.bar(x + .6, get_r2(fits_kmer), w, label='our model (kmer)', facecolor=colors[6])
    # pp.bar(x + .5, get_r2(fits_m), w, label='mean nsRBNS (pos. control)', facecolor=colors[5])

    pp.xticks(x+.25, ['{0:.0f}'.format(conc) for conc in exp.rbp_conc])
    pp.legend(
        bbox_to_anchor=(0., 1.02, 1., .202), 
        loc=3, ncol=2, mode="expand", borderaxespad=0.,
        frameon=False
    )

    pp.xlabel('RBFOX2 concentration [nM]')
    pp.ylabel(r'$R^2$ (variance explained)')
    pp.ylim(0,60)
    pp.tight_layout()
    sns.despine(trim=False)

    pp.savefig('performance_RBFOX2.pdf')
    pp.close()


def msi1_analysis(lp=0, z_cut=4):
    import seaborn as sns
    nsrbns = nsRBNSOligos(fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out')
    # nsrbns = nsRBNSOligos(fa_name = '3utrOligoPool_final_T7.fa', adap5='GGGAGTTCTACAGTCCGACGATC', adap3='TGGAATTCTCGGGTGCCAAG', xtalk_file='blast/bridget_results.out')
    # exp = nsRBNSExperiment(nsrbns, 'msi1_matrix.csv', skip_xtalk=True, seq_only=False)
    exp = nsRBNSExperiment(nsrbns, 'mapping/msi1_trial1.counts', skip_xtalk=True, seq_only=False)
    exp.GC_bias_plot()
    # exp.noaffinity_analysis(nsrbns.GC, 'GC content')
    # exp.noaffinity_analysis(nsrbns.entropy, 'entropy')
    # kw = dict(scatter_plots=True, res_plots=False)
    kw = dict(scatter_plots=False, res_plots=False)
    mbase = '/home/mjens/engaging/RBNS/MSI1/RBPamp/CI/'
    # mbase = '/home/mjens/engaging/RBNS/MSI1/RBPamp/CI10M/'

    lm = nsRBNSModel(exp, 'bg only', None, seq_only=False, low_perc=lp, mdl_type='bg')
    fits_bg = lm.regression_analysis(**kw)

    lm = nsRBNSModel(exp, 'RNAcompete', 'msi1.rnacompete', seq_only=True, low_perc=lp)
    fits_rc = lm.regression_analysis(**kw)

    lm = nsRBNSModel(exp, 'R_values_na', 'msi1_R_value_7mers.tsv', seq_only=True, low_perc=lp, z_cut=z_cut)
    fits_r_na = lm.regression_analysis(**kw)

    lm = nsRBNSModel(exp, 'R_values', 'msi1_R_value_7mers.tsv', seq_only=False, low_perc=lp, z_cut=z_cut)
    fits_r = lm.regression_analysis(**kw)

    # lm = nsRBNSModel(exp, 'SPA_psam_na', '/scratch/data/RBNS/MSI1/RBPamp/test_mfa2/meanfield/7mer_affinities.tsv', seq_only=True, low_perc=lp)
    # lm = nsRBNSModel(exp, 'SPA_psam_na', 'MSI1_nostruct_PSAM.tsv', seq_only=True, mdl_type='MSI1')
    psam_model = mbase + 'opt_nostruct/parameters.tsv'
    # psam_model = mbase + 'opt_nostruct/parameters.tsv'
    lm = nsRBNSModel(exp, 'SPA_psam_na', psam_model, seq_only=True, mdl_type='MSI1')
    # lm = nsRBNSModel(exp, 'SPA', 'msi1_mfa_spa_7mer.tsv', seq_only=False, low_perc=10)
    fits_psam_na = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=True)

    # lm = nsRBNSModel(exp, 'SPA_psam', '/scratch/data/RBNS/MSI1/RBPamp/test_mfa2/meanfield/7mer_affinities.tsv', seq_only=False, low_perc=lp)
    # lm = nsRBNSModel(exp, 'SPA_psam', 'MSI1_full_1M.tsv', seq_only=False, low_perc=lp, mdl_type="PSAM")
    psam_model = mbase + 'opt_full/parameters.tsv'
    # psam_model = '/home/mjens/engaging/RBNS/MSI1/RBPamp/CI/opt_full/parameters.tsv'

    lm = nsRBNSModel(exp, 'SPA_psam', psam_model, seq_only=False, mdl_type='MSI1')

    # lm = nsRBNSModel(exp, 'SPA_psam', 'msi1_mfa_spa_7mer.tsv', seq_only=False, low_perc=10)
    fits_psam = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=True)
    exp.heatmap_plot(lm)

    # kmer_model = '/scratch/data/RBNS/MSI1/combined_7mer_affinities.tsv'
    # # kmer_model = '/scratch/data/RBNS/MSI1/RBPamp/test_seeded3/affinity/8mer_affinities.tsv'
    # kmer_model = '/scratch/data/RBNS/MSI1/RBPamp/test_seeded2/affinity/7mer_affinities.tsv'
    # # lm = nsRBNSModel(exp, 'SPA_kmer_na', '/scratch/data/RBNS/MSI1/RBPamp/test_seeded2/affinity/7mer_affinities.tsv', seq_only=True, low_perc=10)
    # lm = nsRBNSModel(exp, 'SPA_kmer_na', kmer_model, seq_only=True, low_perc=10, k=7)
    # fits_kmer_na = lm.regression_analysis(**kw)

    # # lm = nsRBNSModel(exp, 'SPA_kmer', '/scratch/data/RBNS/MSI1/combined_7mer_affinities.tsv', seq_only=False, low_perc=10, scale=1e-5)
    # # lm = nsRBNSModel(exp, 'SPA_kmer', '/scratch/data/RBNS/MSI1/RBPamp/test_seeded2/affinity/7mer_affinities.tsv', seq_only=False, low_perc=10)
    # lm = nsRBNSModel(exp, 'SPA_kmer', kmer_model, seq_only=False, low_perc=10, k=7)
    # fits_kmer = lm.regression_analysis(scatter_plots=True, res_plots=False)
    # exp.heatmap_plot(lm)

    x = np.arange(len(exp.rbp_conc))*.9
    pp.figure(figsize=(5,4))
    w = .09
    colors= ['#80FFC3','#98E86D','#FFF384','#E8B668','#FF8912','#FF886A','#D552FF','#B069FF']
    pp.bar(x + 0, get_r2(fits_bg), w, label='background', facecolor='.8')
    pp.bar(x + .1, get_r2(fits_r_na), w, label='top R-values', facecolor=colors[0])
    pp.bar(x + .2, get_r2(fits_r), w, label='top R-values (+footprint)', facecolor=colors[1])
    pp.bar(x + .3, get_r2(fits_psam_na), w, label='AKIRA (seq. only)', facecolor=colors[3])
    pp.bar(x + .4, get_r2(fits_psam), w, label='AKIRA', facecolor=colors[4])
    pp.bar(x + .5, get_r2(fits_rc), w, label='RNAcompete', facecolor=colors[2])
    # pp.bar(x + .5, get_r2(fits_kmer_na), w, label='our model (kmer w/o structure)', facecolor=colors[5])
    # pp.bar(x + .6, get_r2(fits_kmer), w, label='our model (kmer)', facecolor=colors[6])
    # pp.bar(x + .5, get_r2(fits_m), w, label='mean nsRBNS (pos. control)', facecolor=colors[5])

    pp.xticks(x+.25, ['{0:.0f}'.format(conc) for conc in exp.rbp_conc])
    pp.legend(
        bbox_to_anchor=(0., 1.02, 1., .202), 
        loc=3, ncol=2, mode="expand", borderaxespad=0.,
        frameon=False
    )

    pp.xlabel('MSI1 concentration [nM]')
    pp.ylabel(r'$R^2$ (variance explained)')
    pp.ylim(0,80)
    pp.tight_layout()
    sns.despine(trim=False)

    pp.savefig('performance_MSI1.pdf')
    pp.close()


def mbnl1_analysis(lp=0, z_cut=4):
    import seaborn as sns
    nsrbns = nsRBNSOligos(fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out')
    # nsrbns = nsRBNSOligos(fa_name = '3utrOligoPool_final_T7.fa', adap5='GGGAGTTCTACAGTCCGACGATC', adap3='TGGAATTCTCGGGTGCCAAG', xtalk_file='blast/bridget_results.out')
    # exp = nsRBNSExperiment(nsrbns, 'mbnl1_matrix.csv', skip_xtalk=True, seq_only=False)
    exp = nsRBNSExperiment(nsrbns, 'mapping/mbnl1_trial1.counts', skip_xtalk=True, seq_only=False)
    exp.GC_bias_plot()
    # exp.noaffinity_analysis(nsrbns.GC, 'GC content')
    # exp.noaffinity_analysis(nsrbns.entropy, 'entropy')
    # kw = dict(scatter_plots=True, res_plots=False)
    kw = dict(scatter_plots=False, res_plots=False)
    mbase = '/home/mjens/engaging/RBNS/MBNL1/RBPamp/CI/'
    mbase10 = '/home/mjens/engaging/RBNS/MBNL1/RBPamp/CI10M/'
    mexp = '/scratch/data/RBNS/MBNL1/RBPamp/low_conc/'
    # mexp = mbase

    # lm = nsRBNSModel(exp, 'bg only', None, seq_only=False, low_perc=lp, mdl_type='bg')
    # fits_bg = lm.regression_analysis(**kw)

    # lm = nsRBNSModel(exp, 'RNAcompete', 'mbnl1.rnacompete', seq_only=True, low_perc=lp)
    # fits_rc = lm.regression_analysis(**kw)

    # lm = nsRBNSModel(exp, 'R_values_na', mbase10 + '../metrics/metrics/MBNL1.R_value.7mer.tsv', seq_only=True, low_perc=lp, z_cut=z_cut)
    # fits_r_na = lm.regression_analysis(**kw)

    # lm = nsRBNSModel(exp, 'R_values', 'MBNL1.R_value.7mer.tsv', seq_only=False, low_perc=lp, z_cut=z_cut)
    # fits_r = lm.regression_analysis(**kw)

    # # lm = nsRBNSModel(exp, 'SPA_psam_na', '/scratch/data/RBNS/MSI1/RBPamp/test_mfa2/meanfield/7mer_affinities.tsv', seq_only=True, low_perc=lp)
    # # lm = nsRBNSModel(exp, 'SPA_psam_na', 'MBNL1_nostruct_PSAM.tsv', seq_only=True, mdl_type='PSAM')
    # lm = nsRBNSModel(exp, 'SPA_psam_na', mbase + 'opt_nostruct/parameters.tsv', seq_only=True, mdl_type='PSAM')

    # # lm = nsRBNSModel(exp, 'SPA', 'msi1_mfa_spa_7mer.tsv', seq_only=False, low_perc=10)
    # fits_psam_na = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=True)
    # lm = nsRBNSModel(exp, 'SPA_psam_na', mbase + 'opt_nostruct/parameters.tsv', seq_only=True, mdl_type='PSAM')
    lm = nsRBNSModel(exp, 'SPA_psam_na', mexp + 'opt_nostruct/parameters.tsv', seq_only=True, mdl_type='PSAM')
    # lm = nsRBNSModel(exp, 'SPA_psam', 'msi1_mfa_spa_7mer.tsv', seq_only=False, low_perc=10)
    fits_psam_na = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=True)


    # lm = nsRBNSModel(exp, 'SPA_psam', '/scratch/data/RBNS/MSI1/RBPamp/test_mfa2/meanfield/7mer_affinities.tsv', seq_only=False, low_perc=lp)
    # lm = nsRBNSModel(exp, 'SPA_psam', 'MBNL1_nostruct_PSAM.tsv', seq_only=False, low_perc=lp, mdl_type="PSAM")
    # lm = nsRBNSModel(exp, 'SPA_psam', mbase + 'opt_full/parameters.tsv', seq_only=False, mdl_type='PSAM')
    # # lm = nsRBNSModel(exp, 'SPA_psam', 'msi1_mfa_spa_7mer.tsv', seq_only=False, low_perc=10)
    # fits_psam = lm.regression_analysis(scatter_plots=False, res_plots=False, akira=True)
    # exp.heatmap_plot(lm)

    # kmer_model = 'MBNL1.SKA_weight.7mer.tsv'
    # # kmer_model = '/scratch/data/RBNS/MSI1/RBPamp/test_seeded3/affinity/8mer_affinities.tsv'
    # # kmer_model = '/scratch/data/RBNS/MSI1/RBPamp/test_seeded2/affinity/7mer_affinities.tsv'
    # # lm = nsRBNSModel(exp, 'SPA_kmer_na', '/scratch/data/RBNS/MSI1/RBPamp/test_seeded2/affinity/7mer_affinities.tsv', seq_only=True, low_perc=10)
    # lm = nsRBNSModel(exp, 'SPA_ska_na', kmer_model, seq_only=True, low_perc=10, k=7)
    # fits_kmer_na = lm.regression_analysis(**kw)

    # # lm = nsRBNSModel(exp, 'SPA_kmer', '/scratch/data/RBNS/MSI1/combined_7mer_affinities.tsv', seq_only=False, low_perc=10, scale=1e-5)
    # # lm = nsRBNSModel(exp, 'SPA_kmer', '/scratch/data/RBNS/MSI1/RBPamp/test_seeded2/affinity/7mer_affinities.tsv', seq_only=False, low_perc=10)
    # lm = nsRBNSModel(exp, 'SPA_ska', kmer_model, seq_only=False, low_perc=10, k=7)
    # fits_kmer = lm.regression_analysis(scatter_plots=True, res_plots=False)

    x = np.arange(len(exp.rbp_conc))*.9
    pp.figure(figsize=(5,4))
    w = .09
    colors= ['#80FFC3','#98E86D','#FFF384','#E8B668','#FF8912','#FF886A','#D552FF','#B069FF']
    pp.bar(x + 0, get_r2(fits_bg), w, label='background', facecolor='.8')
    pp.bar(x + .1, get_r2(fits_r_na), w, label='top R-values', facecolor=colors[0])
    pp.bar(x + .2, get_r2(fits_r), w, label='top R-values (+footprint)', facecolor=colors[1])
    pp.bar(x + .3, get_r2(fits_psam_na), w, label='AKIRA (seq. only)', facecolor=colors[3])
    pp.bar(x + .4, get_r2(fits_psam), w, label='AKIRA', facecolor=colors[4])
    pp.bar(x + .5, get_r2(fits_rc), w, label='RNAcompete', facecolor=colors[2])
    # pp.bar(x + .5, get_r2(fits_kmer_na), w, label='SKA (w/o structure)', facecolor=colors[5])
    # pp.bar(x + .6, get_r2(fits_kmer), w, label='SKA', facecolor=colors[6])
    # pp.bar(x + .5, get_r2(fits_m), w, label='mean nsRBNS (pos. control)', facecolor=colors[5])

    pp.xticks(x+.25, ['{0:.0f}'.format(conc) for conc in exp.rbp_conc])
    pp.legend(
        bbox_to_anchor=(0., 1.02, 1., .202), 
        loc=3, ncol=2, mode="expand", borderaxespad=0.,
        frameon=False
    )
    pp.xlabel('MBNL1 concentration [nM]')
    pp.ylabel(r'$R^2$ (variance explained)')
    pp.ylim(0,40)
    pp.tight_layout()
    sns.despine(trim=False)

    pp.savefig('performance_MBNL1.pdf')
    pp.close()

def rbpbind_analysis(lp=0, z_cut=4):
    nsrbns = nsRBNSOligos(fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out')
    # nsrbns = nsRBNSOligos(fa_name = '3utrOligoPool_final_T7.fa', adap5='GGGAGTTCTACAGTCCGACGATC', adap3='TGGAATTCTCGGGTGCCAAG', xtalk_file='blast/bridget_results.out')
    exp = nsRBNSExperiment(nsrbns, 'rbfox2_matrix.csv', skip_xtalk=True, seq_only=False)
    # exp.noaffinity_analysis(nsrbns.GC, 'GC content')
    # exp.noaffinity_analysis(nsrbns.entropy, 'entropy')
    # kw = dict(scatter_plots=True, res_plots=False)
    kw = dict(scatter_plots=False, res_plots=False)

    lm = RBPBindModel(exp, 'RBPBind', 'RBFOX3_full.tsv', seq_only=False, low_perc=lp)
    I = lm.state.invkd_SPA_read.argsort()[::-1][:100]
    I = [7833,] # candidate for cooperativity?
    for j in I:
        # print j, lm.state.invkd_SPA_read[j]
        lm.run_rbpbind(j)

    # lm.run_rbpbind(884)
    # fits_rc = lm.regression_analysis(**kw)

# rbpbind_analysis()
# rbfox2_analysis()
# msi1_analysis()
mbnl1_analysis()
sys.exit(0)

nsrbns = nsRBNSOligos(fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out')
# nsrbns = nsRBNSOligos(fa_name = '3utrOligoPool_final_T7.fa', adap5='GGGAGTTCTACAGTCCGACGATC', adap3='TGGAATTCTCGGGTGCCAAG', xtalk_file='blast/bridget_results.out')
nsrbns.xtalk_plot()
exp = nsRBNSExperiment(nsrbns, sys.argv[1], skip_xtalk=False, seq_only=False)
# exp.noaffinity_analysis(nsrbns.GC, 'GC content')
# exp.noaffinity_analysis(nsrbns.entropy, 'entropy')
# lm = nsRBNSModel(exp, 'SPA', sys.argv[2], seq_only=False, low_perc=10)
# fits = lm.regression_analysis(scatter_plots=False, res_plots=False)

exp.GC_bias_plot()
exp.entropy_plot()

exp.affinity_selection_plot(lm)
exp.heatmap_plot(lm)

# for reg in fits:
#     mis = MutualInformationScore(reg.y_obs, reg.y_pred)
#     # mis.heatmap_plot('MI_obs_predicted_{0:0f}nM.pdf'.format(conc))
#     print reg.conc, "observed vs expected", mis.MI, 'bits'
#     print "null", np.mean(mis.MI_permut), np.std(mis.MI_permut)
#     mis.dist_plot('MI_obs_predicted_{0:0f}nM.pdf'.format(reg.conc))
#     print mis.z, mis.p_value


print "analysis"
exp.error_analysis(fits, nsrbns.entropy, name='entropy')
# print "residual log error vs. entropy"
# all_lfc = np.log2(exp.enr_expect/exp.enr)
# for conc, lfc in zip(exp.rbp_conc, all_lfc):
#     mis = MutualInformationScore(lfc, nsrbns.entropy[exp.indices])
#     # mis.dist_plot('MI_entropy_lfc_{0:0f}nM.pdf'.format(conc))
#     print conc,"nM", mis.MI, mis.z, mis.p_value

exp.error_analysis(fits, nsrbns.GC, name='GC_content')
# print "residual log error vs. GC content"
# all_lfc = np.log2(exp.enr_expect/exp.enr)
# for conc, lfc in zip(exp.rbp_conc, all_lfc):
#     mis = MutualInformationScore(lfc, nsrbns.GC[exp.indices])
#     # mis.dist_plot('MI_entropy_lfc_{0:0f}nM.pdf'.format(conc))
#     print conc,"nM", mis.MI, mis.z, mis.p_value

exp.error_analysis(fits, nsrbns.xtalk_score, name='xtalk')
# print "residual log error vs. xtalk"
# all_lfc = np.log2(exp.enr_expect/exp.enr)
# for conc, lfc in zip(exp.rbp_conc, all_lfc):
#     mis = MutualInformationScore(lfc, nsrbns.xtalk_score[exp.indices])
#     # mis.dist_plot('MI_entropy_lfc_{0:0f}nM.pdf'.format(conc))
#     print conc,"nM", mis.MI, mis.z, mis.p_value

# pp.show()



# pp.figure()
# pp.loglog(frac[0], frac[1],'x')
# pp.loglog(frac[0], frac[2],'.')
# pp.loglog(frac[0], frac[3],'^')

# pp.figure()
# for conc, R in zip(rbp_conc, enr):
#     pp.hist(np.log10(R), bins=100, normed=True, cumulative=True)



# pp.figure()
# pp.hist(GC, bins=100)

# pp.figure()
# pp.title("GC content")
# x = np.log2(enr[1])
# pp.hist(x[GC < .4], bins=100, alpha=.6, normed=True)
# pp.hist(x[GC > .5], bins=100, alpha=.6, normed=True)
# pp.hist(x[GC > .6], bins=100, alpha=.6, normed=True)

# pp.figure()
# pp.title("motif content")

# def plot_dist(cond, color, label, bins =100):
#     y = x[cond]
#     m = np.median(y)
#     pp.hist(y, color=color, bins=bins, alpha=.6, normed=True, label=label)
#     pp.axvline(m, color=color)
    
# plot_dist(motif == 0, 'gray', 'no GCAYG')
# plot_dist((0 < motif) & (motif < 5), 'blue', 'GCATG')
# plot_dist((5 <= motif) & (motif < 10), 'yellow', 'HGCATG')
# plot_dist(motif == 10, 'red', 'TGCATGY')
# pp.legend()





# sys.exit(0)
# reads = RBNSReads(fname, format='fasta', rbp_name='nsRBNS', rna_conc=100., acc_storage_path='RBPamp/acc')

def evaluate_RBPbind(seqs, rbp_conc, state):
    import copy
    import rbpbind
    rbpbind.init(T=22)
    rbpbind.set_kmer_invkd_lookup(state.A, k) # assign same affinities as used for SPA model
    p_bound = []
    Z1 = []

    for i,s in enumerate(np.array(SEQ)[indices]):
        z0 = rbpbind.compute_Z(s,0)
        z1 = rbpbind.compute_Z(s,1, update_seq=False)
        Z1.append(z1)
        
        pb = []
        for conc in rbp_conc:
            zc = rbpbind.compute_Z(s,conc, update_seq=False)
            pb.append(1. - np.exp(-(z0-zc)))
        
        p_bound.append(pb)
        print '\r {0}'.format(i),
    
    nstate = copy.copy(state)
    nstate.Z1 = np.exp(np.array(Z1))
    nstate.p_bound = np.array(p_bound).T

    return nstate

# state = evaluate_SPA(SEQ, rbp_conc)

# nstate = evaluate_RBPbind(SEQ, rbp_conc, state)

# pp.figure()

# for conc, prbp, pspa in zip(rbp_conc, nstate.p_bound, state.p_bound):
#     pp.loglog(prbp, pspa, '.', label='{0:.1f}'.format(conc))

# pp.legend(loc='upper left')
# pp.xlabel('RBPbind')
# pp.ylabel('SPA')
# pp.show()

# state = nstate


def detailed_analysis(i, flavor='detail'):
    j = indices[i]
    fa_id = _fa_ids_raw[j]
    s = adap5.lower() + np.array(SEQ)[indices][i] + adap3.lower()
    pp.figure()
    pp.title(fa_id)
    print i, fa_id
    print s

    # import rbpbind
    # rbpbind.init(T=22)
    # rbpbind.set_kmer_invkd_lookup(state.A, k)
    print state.mdl.subsample_acc.shape, len(adap5), len(adap3)
    start = len(adap5)-k+1
    # end = len(s) - len(adap3)+k-1
    acc = state.mdl.subsample_acc[i][start:-len(adap3)+k-1]
    ind = state.mdl.subsample_index_matrix[i]
    aff = state.A[ind]

    Z = aff * acc
    pos = Z.argmax()
    print s[start+pos:start+pos+k], "best hit", Z[pos], aff[pos], acc[pos]

    pp.semilogy(acc, color='gray', label=r'$\alpha$')
    pp.semilogy(aff, color='red', label=r'$\frac{1}{K_d}$ [1/nM]')


    # for conc in rbp_conc:
    #     # occ = rbpbind.occupancy_vector(s, conc) + 1e-6
    #     # pos = occ.argmax()
    #     # conc, occ.min(), occ.max(), pos, s[pos:pos+k]
    #     occ = (conc*Z) / ( 1 + conc*Z)
    #     pp.semilogy(occ,label=r'$\theta_i$ ({0:.1f}nM)'.format(conc))
    
    pp.legend()
    pp.xlabel('nt pos.')
    pp.ylabel('occupancy')
    pp.savefig('{0}_{1}.pdf'.format(flavor, i))
    pp.close()

    # sys.exit(0)





# pp.figure()
# a = enr[0]
# b = enr[1]
# pp.loglog(a[motif==0], b[motif==0],'.',color='gray')
# # pp.loglog(a[motif==1], b[motif==1],'r.')

# pp.loglog(a[GC < .4], b[GC < .4],'.',color='gray')
# pp.loglog(a[GC > .6], b[GC > .6],'r.')
