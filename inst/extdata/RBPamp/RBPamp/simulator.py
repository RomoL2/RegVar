from __future__ import print_function
import numpy as np
import logging
import scipy
import logging
import time
import sys
import os
from collections import defaultdict
from scipy.optimize import minimize, brentq, minimize_scalar
import RBPamp.cyska as cyska
from RBPamp.caching import CachedBase, cached, pickled
from RBPamp.crosstalk_matrix import CrosstalkMatrix
from RBPamp.reads import RBNSReads
from RBPamp.caching import CachedBase, cached, pickled

class RBNSGenerator(CachedBase):
    def __init__(self, k, l=20, min_E=-11., seed=None, temp=22, mode='ordered', **kwargs):
        
        CachedBase.__init__(self)

        self.k = k
        self.l = l
        self.min_E = min_E
        self.temp = temp
        self.RT = (temp + 273.15) * 8.314459848 / 4.184E3 # RT in kcal/mol

        raw_energies = RBNSGenerator.energy_distribution(min_E=min_E, N=4**k, **kwargs)
        if mode == 'ordered':
            self.kmer_energies = np.array(sorted(raw_energies)[::-1], dtype=np.float32) / self.RT
        else:
            self.kmer_energies = np.array(raw_energies, dtype=np.float32) / self.RT
            
        self.kmer_invkd = np.exp(-self.kmer_energies)*1e-9
        self.logger = logging.getLogger("RBNSGenerator")
        self.input_reads = None

    def energy_plot(self, store="kmer_energies.pdf"):
        import matplotlib.pyplot as pp
        pp.figure()
        y, bins = np.histogram(self.kmer_energies*self.RT,bins=50)
        print(bins.shape, y.shape)
        pp.semilogx(kcal_to_Kd(bins[:-1]), y, linestyle='steps', linewidth=2.)
        pp.xlabel(r'$K_d$ [nM]')
        #pp.xlabel(r'$\Delta G$ [kcal/mol]')
        pp.ylabel('frequency')
        pp.savefig(store)

        ##pp.figure()
        #pp.hist(kcal_to_Kd(self.kmer_energies*self.RT),bins=100)
        #pp.xlabel(r'$K_d$ [nM]')
        #pp.ylabel('frequency')
        

    def store_invKd(self, fname):
        """
        write flat file with kmer 1./Kd as neede by RBPbind
        """
            
        invKd = 1./(np.exp(self.kmer_energies)*1e9)
        with file(fname,'w') as f:
            f.write(" Motifs invKd\n===================\n")
            for kmer, ikd in zip(cyska.yield_kmers(self.k), invKd):
                f.write("{0} {1}\n".format(kmer, ikd))
        

    def __str__(self):
        top_kmers = []
        
        for i in self.kmer_energies.argsort()[:10]:
            kmer = cyska.index_to_seq(i, self.k)
            E = self.kmer_energies[i]* self.RT
            top_kmers.append( "{0}\t{1}\t{2}".format(kmer, E, kcal_to_Kd(E*self.RT, self.temp) ) )
            
        return "\n".join(top_kmers)

    @staticmethod
    def energy_distribution(mu=10., sigma=.2, min_E = -11., N=1024, temp=22.):
        E = - np.random.lognormal(10., sigma, N) 
        E -= E.mean()
        E *= min_E / E.min()
        return E

    @staticmethod
    def Kd_distribution(**kwargs):
        return kcal_to_Kd(RBNSGenerator.energy_distribution(**kwargs))

    @property
    def Kd(self):
        return np.exp(self.kmer_energies)*1e9
        
    def assign_experimental_input(self, real_input, pseudo_count=0):
        from RBPamp.rbns_reads import RBNSReads
        reads = RBNSReads(real_input, pseudo_count=pseudo_count)
        nt_freq = reads.kmer_frequencies(1) / 4.
        di_freq = reads.kmer_frequencies(2).reshape(4,4) / 16.
        di_freq /= di_freq.sum(axis=1)[:, np.newaxis] # normalize rows to one
        
        self.input_reads = reads
        self.input_nt_freq = nt_freq
        self.input_di_freq = di_freq
        
        return nt_freq, di_freq
        
    def generate_input_reads(self, N=20000000, store=""):
        if self.input_reads:
            self.logger.debug("generating random sequence matrix, mimicking '{0}'".format(self.input_reads.fname))
            t0 = time.time()
            seqm = cyska.generate_random_sequence_matrix_dinuc(self.l, N, self.input_nt_freq, self.input_di_freq)
            dt = time.time() - t0
            
            rps = N / dt
            self.logger.debug("took {0:.3f} seconds. {1:.1f} reads per second".format(dt, rps) )

        else:
            self.logger.debug("generating random sequence matrix")

            t0 = time.time()
            seqm = cyska.generate_random_sequence_matrix(self.l, N)
            dt = time.time() - t0
            
            rps = N / dt
            self.logger.debug("took {0:.3f} seconds. {1:.1f} reads per second".format(dt, rps) )

        if store:
            cyska.write_seqm(seqm, file(store, 'w') )

        return seqm
   
    def generate_bound_reads(self, N=20000000, P=320., p_ns=0.01, real_input="", store=""):
        self.logger.debug("simulating bound sequence matrix, mimicking input from '{0}'".format(self.input_reads.fname))
        
        t0 = time.time()
        seqm, bound_fraction = cyska.simulate_rbns_reads(self.l, N, self.k, self.input_nt_freq, self.input_di_freq, self.kmer_energies, P, p_ns)
        # Hacky wacky just to debug and troubleshoot!
        #seqm, self.input_kmer_counts, self.pd_kmer_weights, self.pd_kmer_counts, self.bound_fraction, self.Z_full = cyska.simulate_rbns_reads(self.l, N, self.k, self.input_nt_freq, self.input_di_freq, self.kmer_energies, P, p_ns)

        
        dt = time.time() - t0
        
        print("bound fraction", bound_fraction) #, "non-specific", n_ns
        rps = N / dt
        self.logger.debug("took {0:.3f} seconds. {1:.1f} reads per second".format(dt, rps) )
        
        if store:
            cyska.write_seqm(seqm, file(store, 'w') )

        reads = RBNSReads(store, seqm=seqm, rbp_conc=P, n_subsamples=0, pseudo_count=0)
        # simulation results should be temporary or we run into trouble!
        reads._do_not_unpickle = True
        reads._do_not_pickle = True
        return reads
        #return seqm

    @cached
    def boltzmann_weights(self, P=320.):
        # chemical potential in units of RT. 
        # $\mu = \log(c/c_0)$ c_0=1M, P is in nM, hence 1e-9
        mu = np.log(P*1e-9) 
        
        w = np.exp(- self.kmer_energies + mu)
        return w
        
    def predict_occupancies(self, P=320., store=""):
        Kd = np.exp(self.kmer_energies)*1e9
        occ = P / (P+Kd)
        
        if store:
            with file(store,'w') as f:
                for kmer, o in zip(cyska.yield_kmers(self.k), occ):
                    f.write("{0}\t{1}\n".format(kmer, o) )
            
        return occ
    
    def predict_r_values(self, P=320., store="", limit=True):
        occ = self.predict_occupancies(P)
        r, pi_sum, beta = self.predict_r_values_from_occ(occ, limit=limit)
        
        if store:
            with file(store,'w') as f:
                for kmer, o in zip(cyska.yield_kmers(self.k), r):
                    f.write("{0}\t{1}\n".format(kmer, o) )

        return r, pi_sum, beta
    
    def predict_r_values_from_occ(self, occ, store="", limit=True):
        k = self.k
        kappa = self.input_reads.kmer_frequencies(k) / 4**k
        omega = (occ * kappa).sum()
        #print "omega", omega
        l = self.l - self.k + 1
        beta = omega*(l-2.*(self.k) + 1)
        
        M, M_inv = self.crosstalk_matrix_and_inverse()
        
        
        vec = (np.dot(M, occ) + beta)
        print("lin approx max expected binding", vec.max())
        if limit:
            #vec = np.where(vec > 1, 1, vec)  # hard cap
            #vec = vec / (vec + 2.) # sigmoidal
            
            vec = np.arcsinh(vec*1.5 ) # log-like for high values
            print("lin approx max binding regularized", vec.max())
            
        pi = kappa * vec
            
        r = pi / pi.sum() / kappa
        return r, pi.sum(), beta
    

    def binding_constants_from_R_vec(self, r, k, P=320., limit=True):
        # WORK IN PROGRESS!
        print("correct Kd", self.Kd[-5:])

        M, M_inv = self.crosstalk_matrix_and_inverse()
        k = self.k
        kappa = self.input_reads.kmer_frequencies(k) / 4**k

        
        r_inv = np.dot(M_inv, r)
        ofs = np.dot(M_inv, np.ones(r.shape))

        def get_occ(x):
            a,b = x
            return a*(r_inv + b)
        
        def err(x):
            r_pred = self.predict_r_values_from_occ(get_occ(x))
            E = ((r - r_pred)**2).mean()
            print(x, E)
            return E
        
        from scipy.optimize import minimize
        res = minimize(err, (.01,.01))
        print(res.success, res.x)
        
        inv_occ = get_occ(res.x)
        
        #import matplotlib.pyplot as pp
        #pp.figure()
        #pp.plot(self.predict_occupancies(P=P), inv_occ, 'ob')
        #pp.show()
        
    
    @cached
    def crosstalk_matrix_and_inverse(self, store=""):
        k = self.k
        N = 4**k
        
        M = np.zeros((N,N), dtype=np.float32)
        
        # kmer overlap extension frequencies
        kfreqs = {}
        for x in range(1,k):
            kfreqs[x] = self.input_reads.kmer_frequencies(x) / 4**x
        
        t0 = time.time()        
        for i in np.arange(N):
            M[i,i] = 1
            for x in range(1,k):
                weights, shifts = cyska.weighted_kmer_shifts(i, k, self.l, x, kfreqs[x]) 
                for s,f in zip(shifts, weights):
                    M[i,s] += f

        t1 = time.time()
        M_inv = np.linalg.inv(M)
        t2 = time.time()
        self.logger.debug("computed crosstalk matrix for k={0} in {1:.3f}s, inverted in {2:.3f}s".format(self.k, t1-t0, t2-t1) )
        return M, M_inv
    
    
    def linear_fit(self, rbp_conc, r_matrix, n_top=20):

        k = self.k
        n = len(rbp_conc)

        def correct_parameters():
            corr_sum_pi = []
            corr_bg = []
            for P in rbp_conc:
                occ = self.predict_occupancies(P)
                kappa = self.input_reads.kmer_frequencies(k) / 4**k
                omega = (occ * kappa).sum()

                # adding baseline
                l = self.l - k + 1
                bg = omega*(l-2.*(k-1)) #* kappa 
                corr_bg.append(bg)

                # predict amount in pulldown
                pi = kappa * (np.dot(M, occ ) + bg )
                sum_pi = pi.sum()
                corr_sum_pi.append(sum_pi)
        
            return np.array(corr_sum_pi), np.array(corr_bg)
        
        # all the expensive matrix multiplications are 
        # done *once*
        
        M, M_inv = self.crosstalk_matrix_and_inverse()
        r_inv = np.array([np.dot(M_inv, r) for r in r_matrix])
        bg_vec = np.dot(M_inv,np.ones(4**k))
        
        #occ_recover = r_inv * sum_pi - bg_vec * bg 
        
        
        I = np.arange(4**k - n_top, 4**k)
        
        kmers = np.array(list(cyska.yield_kmers(k)))[I]
        
        r_inv = r_inv[:,I]
        bg_vec = bg_vec[I]
        

        def infer_lkd(sum_pi, bg):
            occ = r_inv * sum_pi[:,np.newaxis] - bg_vec[np.newaxis,:] * bg[:,np.newaxis]
            
            kd = rbp_conc[:,np.newaxis] * (1/occ - 1)
            return np.log(np.where(kd > 0, kd, 1e8))
        
        
        sp0, bg0 = correct_parameters()
        print("correct parameters are", sp0, bg0)
        #kd = infer_kd(sp0, bg0)


        def objective_function(k_est):
            #k_mean = k_est.mean(axis=0)
            k_mean = np.median(k_est, axis=0)
            
            #err_weight = 1./k_mean
            #err_weight /= err_weight.sum()
            
            S = ((k_est - k_mean[np.newaxis,:])**2).mean()
            
            #n = len(k_est)
            #S = (err_weight[np.newaxis,:] * np.log(k_est / k_mean[np.newaxis,:])**2).sum() / n
            #S = (np.log(k_est / k_mean[np.newaxis,:])**2).sum() / n
            return S


        def score(x):
            sum_pi = x[:n]
            bg = x[n:]
            
            lkd = infer_lkd(sum_pi, bg)
            
            #err = RBNSKmerModel.objective_function(kd)
            err = objective_function(lkd)
            
            print("score(",sum_pi,bg,")-> err=",err)
            return err

        
        def best_fit():
            from scipy.optimize import minimize, basinhopping
            x0 = np.ones(2*n, dtype=float)*0.02
            bounds = np.ones((2*n,2), dtype=float) * 0.1
            bounds[:,0] = 1e-3
            min_dict = dict(bounds = bounds, method='L-BFGS-B')
            x_best = np.concatenate( (sp0, bg0) )
            
            print(score(x_best), "<- best score")
            res = minimize(score, x_best, method='SLSQP', bounds = bounds)#, options=dict(eps=1e-3))
            #res = basinhopping(score, x0, minimizer_kwargs=min_dict)
            print(res)
            x = res.x
            print(score(x), "<- optimized score")
            sum_pi = x[:n]
            bg = x[n:]
            
            lkd = infer_lkd(sum_pi, bg)
            
            return lkd
        
        kd = np.exp(best_fit())
            
        #import matplotlib.pyplot as pp
        #pp.figure()
        #pp.loglog(occ, occ_recover,'xr')
        #pp.show()
        print("correct parameters are", sp0, bg0)
        print(kd)
        dG = self.RT * np.log(kd/1e9).mean(axis=0)
        
        return kmers, kd, dG

class RBNSSimulator(CachedBase):
    def __init__(self, reads, openen, k, temp=22.):
        CachedBase.__init__(self)
        
        self.logger = logging.getLogger("RBNSSimulator")
        self.temp = temp
        self.RT = (temp + 273.15) * 8.314459848/4.184E3
        self.reads = reads
        if openen.disc:
            self.openen = openen
        else:
            # we need to discretize first
            self.openen = openen.discretize()
            self.openen.store()

        self.openen_lookup = self.openen.disc.x/self.RT # open energies in units of RT for each discretization level
        self.acc_lookup = np.exp(-self.openen_lookup)
        
        self.k = k
        self.L = reads.L

    @property
    def cache_key(self):
        return "RBNSSimulator {self.reads.cache_key} {self.openen.cache_key} {self.k}".format(self=self)
    
    #@pickled
    #def expected_kmer_counts(self, kmer_energies, protein_conc, n_max=0, E_ns = 0, seqm=None):
        #from cyska import eval_energy_model_on_seqs
        #if seqm == None:
            #seqm = self.reads.seqm
        #p_bound, kmer_count_matrix, openen_kmer_bincount_matrix = eval_energy_model_on_seqs(seqm, self.openen.oem, self.openen_lookup, kmer_energies, np.array(protein_conc, dtype=np.float32), self.k, E_ns = E_ns, n_max=n_max)
        #return p_bound, kmer_count_matrix, openen_kmer_bincount_matrix

    @pickled
    def expected_kmer_counts(self, kmer_invkd, protein_conc, n_max=0, E_ns = 0, indices=None, seq_only=False):
        from .cyska import eval_energy_model_on_seqs
        if indices == None:
            seqm = self.reads.seqm
            oem = self.openen.oem
        else:
            seqm = self.reads.seqm[indices]
            oem = self.openen.oem[indices]

        if seq_only:
            acc_lookup = np.ones(self.acc_lookup.shape, dtype = np.float32)
        else:
            acc_lookup = self.acc_lookup
            
        t0 = time.time()
        p_bound, kmer_count_matrix, openen_kmer_bincount_matrix, jacobi = eval_energy_model_on_seqs(seqm, oem, acc_lookup, kmer_invkd, np.array(protein_conc, dtype=np.float32), self.k, E_ns = E_ns, n_max=n_max)
        t1 = time.time()
        if n_max:
            n = n_max
        else:
            n = len(seqm)
        self.logger.debug("evaluated energy model on {0} sequences in {1:.2f} ms".format(n, 1000*(t1-t0)) )
        return p_bound, kmer_count_matrix, openen_kmer_bincount_matrix
                             


if __name__ == "__main__":
    import matplotlib
    #matplotlib.use('pdf')
    import matplotlib.pyplot as pp
    logging.basicConfig(level=logging.DEBUG)

    from RBPamp.reads import RBNSReads
    from RBPamp.fol import RBNSOpenen, OpenenStorage
    import RBPamp.fold
    reads = RBNSReads('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads', n_max=10000000)
    storage = OpenenStorage(reads, '/scratch/data/RBNS/RBFOX2/ska_RBFOX2/openen/', disc_mode='gamma')
    openen = storage.get_discretized(5)
    #openen._do_not_unpickle = True
    disc = openen.disc

    sim = RBNSSimulator(reads, openen, 5)    
    
    print("joint frequencies")
    kmer_openen = openen.kmer_openen_counts()

    gen = RBNSGenerator(5,l=40, seed=47110815)
    gen.store_invKd("/home/mjens/git/RBPbind/server/proteinfiles/RBNSGenerator_rev.txt")
    #sys.exit(1)
    
    kmer_energies = np.random.permutation(gen.kmer_energies) #- 1.5 # non-specific binding
    kmer_energies = gen.kmer_energies #- 1.5 # non-specific binding


    print(kmer_energies)
    print("simulating binding")
    rbp_conc = [.5,1.,10.,120.,360.]
    #sim._do_not_unpickle=True
    counts, openen_bincounts = sim.expected_kmer_counts(kmer_energies, rbp_conc, E_ns=-2/gen.RT)
    print(counts.shape, openen_bincounts.shape)
    print(">>>openen-bins")
    
    best_i = kmer_energies.argmin()
    med_i = (kmer_energies == np.median(kmer_energies)).argmax()

    print("median kmer energy", kmer_energies[med_i]*gen.RT, "kcal/mol")
    colors = ['k','r','g','y','c']
    styles = ['x','*','^','.','o']
    
    pp.figure(figsize=(10,8) )
    for I,color in zip([best_i, med_i], colors):
        ref = kmer_openen[I]
        x, ref_y = disc.get_hist_xy(ref)
        
        for i,style in zip(list(range(len(openen_bincounts))), styles):
            bc = openen_bincounts[i,I,:]
            x, y = disc.get_hist_xy(bc)
            
            lratio = np.log(y, ref_y)
            seq = cyska.index_to_seq(I, 5)
            pp.plot(disc.x, lratio, color+style, label="{1} P={0}".format(rbp_conc[i], seq))

        
    #pp.show()
    pp.xlabel(r"$\Delta U$ [kcal/mol]")
    pp.ylabel(r"$\log(\frac{pd}{input})$")
    pp.legend()
    pp.savefig('simulated.pdf')
    
    
    freqs = counts * (4**5)/counts.sum(axis=1)[:,np.newaxis]
    print(freqs[:,-10:])
    
    R = freqs/reads.kmer_frequencies(5)
    print("R-values", R.min(), R.max())
    print(R[:,-10:])
    
    pp.show()    
    
    
    
        
    #test_fastrand()
    sys.exit(0)

    ### test partition function for overlap
    gen = RBNSGenerator(5,l=40, seed=47110815)
    #gen.assign_experimental_input("/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads")
    gen.assign_experimental_input("bla.reads")
    
    #P = 3200.
    #P = 320.
    #P = 50.
    P = 10.
    
    B = np.exp(- gen.kmer_energies + np.log(P*1e-9) )
    print("Boltzmann weights for top sites", B[-5:])
    
    k = 5
    N = 4**k
        
    # kmer overlap extension frequencies
    kfreqs = {}
    for x in range(1,k+1):
        kfreqs[x] = gen.input_reads.kmer_frequencies(x) / 4**x
        
    # test using simulated reads
    from RBPamp.rbns_analysis import RBNSComparison
    #reads = gen.generate_input_reads(N=1000000, store="6mer@0nM.reads")
    reads = gen.generate_bound_reads(P=P, p_ns=0.00, N=10000, store="7mer@{0}nM.reads".format(P))
    #print gen.Z_full
    print(">>> most abundant kmer-frequencies in pulldown simulation", reads.kmer_frequencies(k)[-5:])
    comp = RBNSComparison(gen.input_reads, reads)
    R, R_err = comp.R_values(k, _do_not_unpickle=True)
    print("simulation R-values", R[-5:])

    def P_bound_given_kmer_nn(x):
        P = 0
        fx = kfreqs[k][x]
        
        for j in range(4):
            fj = kfreqs[1][j]
            
            # overlapping kmers
            ij = (x << 2 | j) & (N-1)
            #print "kmer i={0:b} ij={1:b}".format(i,ij)
            ji = j << ((k-1)*2) | x >> 2
            #print "kmer i={0:b} ij={1:b} ji={2:b}".format(i,ij, ji)
            
            Zij = B[x] + B[ij]
            Zji = B[x] + B[ji]
                            
            Pij = Zij / (Zij + 1)
            Pji = Zji / (Zji + 1)
            
            P += fj * (Pij + Pji)

        return fx * P/2.


    def P_bound_given_kmer_nnn(x):
        P = 0
        fx = kfreqs[k][x]
        
        for l in range(4):
            fl = kfreqs[1][l]
            
            for r in range(4):
                fr = kfreqs[1][r]
                
                xr = (x << 2 | l) & (N-1)
                lx = l << ((k-1)*2) | x >> 2
                
                Zlxr = B[lx] + B[x] + B[xr]
                                
                Plxr = Zlxr / (Zlxr + 1)
                
                P += fl * fr * Plxr

        for l in range(16):
            fl = kfreqs[2][l]
            l1 = l & 3
            
            # overlapping kmers
            lx = l << ((k-2)*2) | x >> 4
            l1x = l1 << ((k-1)*2) | x >> 2
            
            Zlx = B[lx] + B[l1x] + B[x]
            Plx = Zlx/ (Zlx + 1)
            
            P += fl * Plx

        for r in range(16):
            fr = kfreqs[2][r]
            r1 = r >> 2
            
            # overlapping kmers
            xr = (x << 4 | r) & (N-1)
            xr1 = (x << 2 | r1) & (N-1)
            
            Zxr = B[x] + B[xr1] + B[xr]
            Pxr = Zxr/ (Zxr + 1)
            
            P += fr * Pxr

        return fx * P/3.


    
    def P_kmer_in_read():
        Z = np.zeros(N)
        
        for i in range(N):
            fi = kfreqs[k][i]
            
            #Z[i] += fi
            
            for j in range(4):
                fj = kfreqs[1][j]
                
                # overlapping kmers
                ij = (i << 2 | j) & (N-1)
                
                ji = j << ((k-1)*2) | i >> 2
                
                Z[ij] += fj * fi
                Z[ji] += fj * fi

        return Z / Z.sum()
        
       
    kmer_occ = B / (B + 1)
    naive_freq = kmer_occ * kfreqs[k]
    naive_r = naive_freq / naive_freq.sum() / kfreqs[k]
    print("naive r",naive_r[-5:])
    
    lin_approx_r, pisum, beta = gen.predict_r_values(P, limit=False)
    print("lin matrix approx.", lin_approx_r[-5:])
    print("factors pisum", pisum, "beta", beta)
    #p_in = kfreqs[k] * 2#P_kmer_in_read()
    #p_in = P_kmer_in_read()
    
    M, M_inv = gen.crosstalk_matrix_and_inverse()
    # inverting linear approximation and comparing to real occupancies
    occ_approx = np.dot(M_inv, lin_approx_r * pisum - beta)
    
    pp.figure()
    plot = pp.plot
    plot(kmer_occ, np.dot(M_inv, R * pisum - beta), 'or')
    plot(kmer_occ, np.dot(M_inv, R * 1.5*pisum - beta), 'oy')
    plot(kmer_occ, np.dot(M_inv, lin_approx_r * pisum - beta), 'ok' ,label="pos. control")
    
    pp.xlabel('kmer occ')
    pp.ylabel('lin approx occ')
    
    pp.show()
    
    nn_freq = np.array([P_bound_given_kmer_nn(x) for x in np.arange(N)])
    nn_r = nn_freq / nn_freq.sum() / kfreqs[k]
    print("nn r",nn_r[-5:])

    nnn_freq = np.array([P_bound_given_kmer_nnn(x) for x in np.arange(N)])
    nnn_r = nnn_freq / nnn_freq.sum() / kfreqs[k]
    print("nnn r",nnn_r[-5:])

    
    pp.figure()
    M = max(R.max(), nn_r.max(), nnn_r.max(), lin_approx_r.max(), naive_r.max() )
    M = max(R.max(), lin_approx_r.max())
    
    plot = pp.loglog
    plot = pp.plot
    
    plot(np.array([1.,M]),np.array([1., M]), '--', color='gray')

    #plot(R, naive_r, 'ok', alpha=.2, label="naive kmer soup R={0:.2f}".format(np.corrcoef(np.log(R), np.log(naive_r))[0][1]) )
    #plot(R, nn_r, 'oy', alpha=.5, label="nearest neighbor avg. R={0:.2f}".format(np.corrcoef(np.log(R), np.log(nn_r))[0][1]) )
    #plot(R, nnn_r, 'or', alpha=.5, label="next-nearest neighbor avg. R={0:.2f}".format(np.corrcoef(np.log(R), np.log(nnn_r))[0][1]) )
    plot(R, lin_approx_r, 'og', label="linear matrix R={0:.2f}".format(np.corrcoef(np.log(R), np.log(lin_approx_r))[0][1]) )
    pp.xlim(1,M)
    pp.ylim(1,M)
    pp.legend(loc='upper left')
    
    gen.binding_constants_from_R_vec(R, k, P)
    #pp.show()
        #t0 = time.time()        
        #for i in np.arange(N):
            #M[i,i] = 1
            #for x in range(1,k):
                #weights, shifts = cyska.weighted_kmer_shifts(i, k, self.l, x, kfreqs[x]) 
                #for s,f in zip(shifts, weights):
                    #M[i,s] += f

        #t1 = time.time()
        #M_inv = np.linalg.inv(M)
        #t2 = time.time()
        #self.logger.debug("computed crosstalk matrix for k={0} in {1:.3f}s, inverted in {2:.3f}s".format(self.k, t1-t0, t2-t1) )
        #return M, M_inv

    
    #sys.exit(0)
    
    
    
    #gen = RBNSGenerator(5,l=40, seed=47110815)
    #pp.figure()
    
    #colors = ['k','b','y','g','r']
    #for P,c in zip([1., 10., 100., 500.], colors):
        ##Q = PartitionFunction("GAATGGAGTTTTTTGTTTTC",P, gen.kmer_energies, 5)
        #Q = PartitionFunction("TTTTT",P, gen.kmer_energies, 5)
        #Pb_1 = Q.P_bound_single()
        #Pb_ex = Q.P_bound_indep_exact()
        ##print Q.P_bound_indep()
        #Pb_corr = Q.P_bound_rec()
        
        #print "single protein unit binding",Pb_1
        #P_bound_single = 1 - np.product( 1 - Pb_1)
        #P_bound_indep = 1 - np.product( 1 - Pb_ex)
        #P_bound_corr = 1 - np.product( 1- Pb_corr)
        #print "P", P, "indep",P_bound_indep, "corr",P_bound_corr, "ratio", P_bound_indep/P_bound_corr
        ##print Pb_ex
        #pp.semilogy(Pb_1, '^-', color=c, label="P={0}".format(P))
        #pp.semilogy(Pb_ex, 'x--', color=c, label="P={0}".format(P))
        #pp.semilogy(Pb_corr, 'o-', color=c, label=None)
    
    #pp.legend(loc='lower right')
    #pp.show()
    #import sys
    #sys.exit(0)

    gen.assign_experimental_input("/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads")
    print(gen)
    #gen.energy_plot()
    #gen.generate_input_reads(store="input.reads", N=100)
    # test different partition functions
    
    


    
    #gen.generate_bound_reads(store="bound_320.reads", P=320., p_ns=0.00, N=20000000)
    #occ = gen.predict_occupancies(P=320., store="occ_320.tsv")
    
    
    r_obs = []
    mers = []
    for line in file('ska_sim/RBP.R_value.5mer.tsv'):
        if line.startswith('#'): 
            continue
        parts = line.rstrip().split('\t')
        r_obs.append([float(parts[i]) for i in [1,3,5,7,9]])
        mers.append(parts[0])
            
    mers = np.array(mers)
    r_obs = np.array(r_obs).T[:,mers.argsort()]
    
    
    rbp_conc = np.array([1., 40, 80, 160, 320])
    #rbp_conc = np.array([1.,5.])
    rbp_conc = np.array([1., 80, 320])
    
    pp.figure()
    r_matrix = []
    occ_matrix = []
    for i,P in enumerate(rbp_conc):
        r = gen.predict_r_values(P=P, store="r_{0}.tsv".format(P))
        occ = gen.predict_occupancies(P=P, store="occ_{0}.tsv".format(P))
        r_matrix.append(r)
        occ_matrix.append(occ)
        #gen.generate_bound_reads(store="bound_{0}.reads".format(P), P=P, p_ns=0.00, N=1000000)
        
        pp.loglog(occ, r, 'o', alpha=.5, label="P={0:.0f}nM".format(P))
        
    pp.legend(loc='upper left')
    pp.xlabel('predicted occupancy')
    pp.ylabel('predicted R-value')
    pp.savefig("R_pred_vs_occ.pdf")
    pp.close()
    
    pp.figure()
    rmin = 1.
    rmax = 1.
    for P, ro, r in zip(rbp_conc, r_obs, r_matrix):
        pp.loglog(r, ro, 'o', alpha=.5, label="P={0:.0f}nM".format(P))

        rmin = min(r.min(), ro.min(), rmin)
        rmax = max(r.max(), ro.max(), rmax)
    
    rmin /= 2
    rmax *= 2
    pp.plot([rmin, rmax], [rmin, rmax], '--k')
    #print "minmax", rmin, rmax
    pp.legend(loc='upper left')
    pp.xlabel('predicted R-value')
    pp.ylabel('observed R-value')


    M, M_inv = gen.crosstalk_matrix_and_inverse()

    def inverse(r):
        r_inv = np.dot(M_inv, r) 
        ofs = np.dot(M_inv, np.ones(r.shape))

        delta = r_inv - ofs
        corr = max(- delta.min(), 0) # no negative terms allowed
        
        beta = corr / ofs
        print((r_inv + beta*ofs).min())
        
        pre_scaled = r_inv + beta * ofs
        scale = 1./pre_scaled.max()
        
        beta = beta * scale
        inv = pre_scaled * scale
        print("inv minmax", inv.min(), inv.max())
        return inv
        
    pp.figure()
    for P, ro, occ, r in zip(rbp_conc, r_obs, occ_matrix, r_matrix):
        Z = 1
        beta = 1
        ro_inv = inverse(ro)
        r_inv = inverse(r)

        print(r_inv[-10:])
        pp.loglog(r_inv, ro_inv, 'o', alpha=.5, label="P={0:.0f}nM".format(P))

    pp.legend(loc='upper left')
    #pp.xlabel('predicted occupancy')
    pp.xlabel(r'$M^{-1} \cdot r_{pred}$')
    pp.ylabel(r'$M^{-1} \cdot r_{obs}$')
    
    #pp.show()
        
        
    kmers, k_est, dG = gen.linear_fit(rbp_conc, np.array(r_matrix), n_top=20)
    

    import matplotlib.pyplot as pp
    ##pp.figure()
    ##pp.loglog(r, rm, 'ok')
    
    #pp.figure(figsize=(8,4))
    #pp.subplot(121)
    #pp.imshow(np.log10(M), cmap=pp.get_cmap('plasma'))
    #pp.colorbar(label=r'$\log_{10}(M)$',fraction=0.046, pad=0.04)

    #pp.subplot(122)
    #pp.imshow(np.log10(M_inv), cmap=pp.get_cmap('plasma'))
    #pp.colorbar(label=r'$\log_{10}(M^{-1})$',fraction=0.046, pad=0.04)
    #pp.tight_layout()
    #pp.savefig('crosstalk_matrix.pdf')
        
    pp.figure()
    pp.title("inferred binding energies")
    print(gen.kmer_energies[-20:].shape, dG.shape)
    pp.plot(gen.kmer_energies[-20:]*gen.RT, dG, 'ob')
    pp.xlabel("simulated energies [kcal/mol]")
    pp.ylabel("inferred energies [kcal/mol]")

    pp.show()
    #pp.savefig('predict.pdf')
    
