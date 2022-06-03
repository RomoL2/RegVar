from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np
import unittest
from RBPamp.reads import RBNSReads
from .cyska import *

test_reads = """GAGGTCACTCTCTTGCATGTATGCATGCAGTCTCAACGAA
CATTTTTTTTGAAACTACGTGCATGTACAATAGGCGACGA
ATACGACTGCTTATCCACTGCATGCTGCAGGAAGTGCATG
TTGTAGCATGTCCGTCAACAGAAACTGCATGTTTCTAATA
CATCCAATGCATGTTTCGGTTCTCTGACAAACCTTCCCTT
AATATATCTCGACGTTGCATGTAATTACCACATAAAAACG
ATTAACCGTAACGCCATAAGCGCGCACCTACTGCATGTTT
ATCCGATGAGCAATAGTGAGAAAAAACATACTGCATGTAT
ATTAAATGCAAGTCGTCGGAGCCTGCATGTACTAAATTAG
AATTAACCGTCTGCATGTGGACGTCCAATTAAACAAATGT
GACGGGTGTTACTGCATGTTAGGATCCCCGTGCATGACAG
TAAATGGCGGGTACCCTTGTGAACGGTACGGATGCATGTG
GGTGGCAGCTTGAATTCTGCGAGTGCATGTGAATACATAT
GGAGCAGATGCATGTGTCCCCGGAGTGGAAATAGGGTCCA
TTTCGTATCTCATGCATGTACACTATCAGCTGTGAAAAAT
GAATTAAACATTGCATGTATGGGTAGGATGGAAATCCCAC
TCATGCATGTTTGATTATAACTGGTAAGTCCTGTACACGT
TCCAGAATTAGTGCATGTAGGAGAAACACACGATATTGAT
GAGGAAAATAACGTGCATGTCCCACTTTAAATATATAGCA
GAATGCAGTCCGGCGCTTTAATGCATGTGCATCCTATACT
"""
minimal_reads = """
TTTTTGCATGTTTTTTTTTTTTTTT
AAAAAAAAAAAAATGCACGTAAAAA
TACGTACGTACGTACGTACGTACGT
TACGTACGTACGTACGTACGTACGT
TACGTACGTACGTACGTACGTACGT
TACGTACGTACGTACGTACGTACGT
TACGTACGTACGTACGTACGTACGT
"""

def get_test_reads(adap5='CCCCCCC', adap3='GGGGGGG', seqs=test_reads, pseudo_count = 1e-8, **kwargs):
    return RBNSReads.from_seqs(seqs.split('\n'), adap5=adap5, adap3=adap3, pseudo_count=pseudo_count, **kwargs)


real_reads = {}
def get_real_reads(N =100000):
    if not N in real_reads:
        real_reads[N] = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='RBPamp/acc', n_max=N, pseudo_count=1e-3)

    return real_reads[N]

class TestPhysModel(unittest.TestCase):

    def test_clipped_sum_and_max(self, N = 100000, L = 28, thresh=10000., rnd_seed=47110815, n = 100):
        from time import time
        np.random.seed(rnd_seed)
        Z = np.array(np.exp(4*np.random.random( (N,L))), dtype=np.float32)
        
        t0 = time()
        for i in range(n):
            Zsum = np.clip(Z.sum(axis=1), None, thresh)
            Zmax = Zsum.max()
        t1 = time()

        from RBPamp.cyska import clipped_sum_and_max
        for i in range(n):
            zsum, zmax = clipped_sum_and_max(Z, clip=thresh)

        t2 = time()

        # print "timing {0:.2f} ms vs {1:.2f} ms".format(1000./n* (t1-t0), 1000./n * (t2-t1))
        self.assertTrue(np.allclose(Zsum, zsum))
        self.assertTrue(np.allclose(zmax, Zmax))


    def test_weighted_kmer_counts(self, N = 100000, k=6, rnd_seed=47110815, n = 100):
        from time import time
        np.random.seed(rnd_seed)
        reads = get_real_reads(N)
        w = np.array(np.random.random( reads.N ), dtype=np.float32)
        
        im = reads.get_index_matrix(k)
        from RBPamp.cyska import weighted_kmer_counts
        t0 = time()
        prev = None
        for i in range(n):
            counts = weighted_kmer_counts(im, w, k)
            if not prev is None:
                self.assertTrue((counts == prev).all()) # test for consistency
            else:
                prev = counts
        t1 = time()
        # print "took {0:.2f} ms".format(1000./n * (t1-t0))


from RBPamp.pwm import *
from RBPamp.gradient import *
class TestGradientMethods(unittest.TestCase):


    @staticmethod
    def params_from_motif(motif, betas = [], a0=1e-6):
        psam = motif.psam + a0
        params = ModelParametrization(motif.n, max(1, len(betas)), psam=psam, A0=motif.A0, betas=betas)
        return params

    def from_kmers(self, *argc, **kwargs):
        return TestGradientMethods.params_from_motif(PSAM.from_kmer_variants(*argc), **kwargs)

    def setup_model(self, correct_params, k_monitor=5, rbp_conc=None, N=1000000):
        # generating reference state
        R0 = np.ones((correct_params.n_samples, 4**k_monitor), dtype=np.float32)
        from RBPamp.partfunc import PartFuncModel
        if rbp_conc is None:
            rbp_conc = [1.,] * correct_params.n_samples
        
        reads = get_real_reads(N)
        model = PartFuncModel(reads, correct_params, R0, rbp_conc=rbp_conc, Z_thresh=0)
        state0 = model.predict(correct_params, beta_fixed=True)
        R0 = state0.R
        model.set_R0(R0)
        state0 = model.predict(correct_params, beta_fixed=True)

        return model, state0

    def setUp(self):
        # params = self.from_kmers(['GCATG', 'GCACG', ], [5., .5, ], betas = [.001,.005,.009])
        # # params = self.from_kmers(['GCATG',], [5., ], betas = [.01,.05,.09])
        # self.model, self.state0 = self.setup_model(params, rbp_conc=[.2, 20., 2000.])
        kmers = ['TGCATG', 'TGCACG', 'AGCATG', 'CGCATG', 'GGCATG']
        aff = np.array([1.,.1,.5, .42, .4]) * 5
        params = self.from_kmers(kmers, aff, betas = [.001,.005,.025])
        self.default_params = params
        self.model, self.state0 = self.setup_model(params, rbp_conc=[1.,5.,25.])

    def get_default(self):
        return self.model, self.state0

    def test_lowlevel(self):
        from RBPamp.partfunc import PartFuncModel

        reads = get_test_reads(seqs=minimal_reads, rna_conc=1. )
        params = self.state0.params.copy()
        params.A0 = .1
        params.betas = [1e-5, 1e-2, 1e-3]
        R0 = np.zeros(self.state0.R.shape, dtype=np.float32)
        model = PartFuncModel(reads, params, R0, rbp_conc=[1.,5.,25.])
        for i,f in enumerate(model.F0):
            if f > 0:
                print(index_to_seq(i,5), f, model.f0[i])
        # sys.exit()
        state = model.predict(params, beta_fixed=True, tune=False)
        print(state)
        print(np.round(state.Z1_read,2))
        print(np.isnan(state.R_errors[0]).sum())
        print(np.isnan(state.w).sum())

        I = np.array([0,582,590,1023,seq_to_index('GTACG')]) # AAAAA GCACG GCATG TTTTT
        print("psi", state.psi)

        g = state.grad
        print(g)
        from RBPamp.gradient import emp_gradi
        eg = emp_gradi(state, eps=1e-4)

        out = params.copy()
        out.data[:] = 0
        l = len(out.psam_vec)

        for i in I:
            ana = state.gradi[j, i]
            emp = eg[j, i][:l]

            delta = np.fabs(ana - emp)
            print("largest abs. deviation between empirical and analytical gradient", delta.max(), "index", delta.argmax())
            if delta.max() > .5:
                out.psam_vec[:] = ana
                print("dR_dA", index_to_seq(i, 5), out)
                out.psam_vec[:] = emp
                print("emp. dR_dA", index_to_seq(i, 5), out)

        emp = emp_grad(state, eps=1e-4)
        print("analytical gradient")
        print(g)
        print("empirical")
        print(emp)

        rec = - 2 * (state.R_errors[:,:,np.newaxis] * eg).mean(axis=(0,1))
        print("emp. from individual")
        out.data[:] = rec
        print(out)

        rec = - 2 * (state.R_errors[:,:,np.newaxis] * state.gradi).mean(axis=(0,1))
        print("ana. from individual")
        out.psam_vec[:] = rec
        print(out)


    def run_descent(self, correct_params, initial_params, k_monitor=5, dec=.75, rbp_conc=None):
        model, state0 = self.setup_model(correct_params, k_monitor=k_monitor, rbp_conc=rbp_conc)
        from RBPamp import vector_stats
        print(">>> reference state")
        print(state0)

        state = model.tune(initial_params)
        print(">>> TUNED")
        print(state)
        # print ">>> ORACLE0"
        # print correct_params
        # print ">>> INITIAL"
        # print initial_params
        
        print(R0.min(), R0.max(), R0.mean(), cyska.index_to_seq(R0.argmax(), k_monitor))
        G = GradientDescent(model, initial_params, dec=dec)
        res = G.optimize(initial_params, maxiter=1, debug=True)
        print(">>> ORACLE")
        print(correct_params)
        print(">>> INITIAL")
        print(initial_params)
        print(">>> FINAL")
        print(G.params)
        print(">>> FINAL GRADIENT")
        print(G.last_state.grad)
        print(">>> FINAL EMP. GRADIENT")
        print(emp_grad(G.last_state))

        d = np.fabs(G.params.data - correct_params.data)
        print("maximal parameter deviation:", d.max(), d.argmax())
        self.assertTrue(G.status.startswith('CONVERGED'))
        # self.assertLess(G.t, 30)
        self.assertTrue(np.allclose(G.params.data, correct_params.data, rtol=1e-3, atol=1e-2))


    def noisy_variant(self, motif, noise=.01, A0=None, betas=None, seed=4711):
        if seed:
            np.random.seed(seed)

        delta = motif.copy()
        delta.data[:] = np.array(np.random.randn(motif.n) * noise, dtype=np.float32)[:]
        variant = GradientDescent.apply_delta(motif, delta)

        if not A0 is None:
            variant.A0 = A0

        if not betas is None:
            variant.betas[:] = betas[:]

        return variant

    # @unittest.skip("")
    def test_beta_opt(self):
        model, state0 = self.get_default()
        state = model.predict(state0.params, beta_fixed=False)
        print(state0.params.betas)
        print(state.params.betas)
        self.assertTrue( np.allclose(state.params.betas, state0.params.betas) )

    def test_gradient_descent(self):
        from RBPamp import vector_stats
        from RBPamp.gradient import GradientDescent
        from RBPamp.report import GradientDescentReport

        params = dict(self.default_params)
        params['debug'] = True
        model, state0 = self.setup_model(params, k_monitor=5, rbp_conc=[1.,5.,25.], N=1000000)
        params1 = state0.params.copy()
        params1.A0 *= .9
        params1.betas[:] = [.2,.3,.5]
        params1.psam_matrix[1,0] = .1
        params1.psam_matrix[2,0] = .01
        params1.psam_matrix[4,1] = .01

        descent = GradientDescent(model, params1, ref_state=state0, predict_kwargs=dict(beta_fixed=False, tune=True))
        descent.optimize(params1, debug=True)

        report = GradientDescentReport(descent, path='.')
        report.plot_param_ref_comparison()
        report.plot_report()
        report.plot_scatter()


    def test_fit_A0(self):
        from RBPamp import vector_stats
        model, state0 = self.setup_model(self.default_params, rbp_conc=[1.,5.,25.], N=1000000)

        subopt_params = state0.params.copy()
        subopt_params.psam_matrix[4,1] = .01 # set C4 too low
        subopt_params.A0 = 1
        # subopt_params.betas[:] = [.1,.5,.1]
        state1 = model.predict(subopt_params, beta_fixed=True)
        print("PERTURBED STATE")
        print(state1)

        state = model.tune(state1)
        print("TUNED STATE")
        print(state)
        print("GRADIENT")
        print(state.grad)

        import matplotlib.pyplot as pp
        a0s = state.a0s
        beta0s = state.beta0s

        # R-value scatter plot
        pp.figure()
        pp.loglog(state.mdl.R0.T, state.R.T, 'x')
        m = min(state.mdl.R0.min(), state.R.min())
        M = max(state.mdl.R0.max(), state.R.max())
        print(m,M)
        pp.loglog([m,m],[M,M], 'k', linestyle='dashed')

        # (A0, beta) tune iterations
        pp.figure()
        pp.loglog(state.a0s, state.beta0s, '-b')
        pp.loglog(state.a0s, state.beta0s, '.k')
        pp.plot([state.a0s[0],], [state.beta0s[0],],'^k') # start
        pp.plot([state.a0s[-1],], [state.beta0s[-1],],'xk') # end
        pp.plot([state0.params.A0,], [state0.params.betas[0],],'.r')

        # beta estimator scatter plot
        est = state.beta_estimators
        pp.figure()
        pp.title('beta estimators')
        for i, per_sample in enumerate(est):
            pp.semilogx(model.R0[i], per_sample, '.')
            pp.semilogx(model.R0[i][model.top_Ri], per_sample[model.top_Ri], 'xr')
            pp.axhline(state0.params.betas[i], color='black')
            pp.axhline(state.params.betas[i], color='red')

        # pp.ylim(0, .03)
        pp.ylabel('beta estimator')

        # A0 estimator scatter plot
        est = state1.A0_estimators
        est_A0 = np.median(est[:,model.top_Ri],axis=1).mean()
        pp.figure()
        pp.title('A0 estimators')
        for i, per_sample in enumerate(est):
            pp.semilogx(model.R0[i], per_sample, '.')
            pp.semilogx(model.R0[i][model.top_Ri], per_sample[model.top_Ri], 'xr')
            pp.axhline(state0.params.A0, color='black')
            pp.axhline(state.params.A0, color='red')

        # pp.ylim(0, 10)
        pp.ylabel('A0 estimator [nM]')

        pp.show()
        return


    def test_acc_shift(self):
        params = self.default_params.copy()
        params.acc_k = 7
        params.acc_ofs = -1

        model0, state0 = self.setup_model(params, rbp_conc=[1.,5.,25.])
        params = params.copy()

        params.acc_k = 7
        params.acc_ofs = -1
        model1, state1 = self.setup_model(params, rbp_conc=[1.,5.,25.])
        model0.R0 = state1.R
        print("performing gradient descent optimization")
        G = GradientDescent(model0, params)
        from RBPamp.report import GradientDescentReport
        rep = GradientDescentReport(G, path='.')
        def callback(descent):
            state = descent.last_state
            rep.plot_report()
            rep.plot_param_hist()
            rep.plot_line_search()
            rep.plot_A0_fit()

            # descent.print_state(state)
            # import matplotlib.pylab as pp
            # pp.figure()
            # print state0.R.shape, state.R.shape
            # pp.loglog(state0.R[0], state.R[0], 'x', label="A0={0:.2f} beta={1:.2e}".format(state.params.A0, state.params.betas[0]))
            # pp.legend(loc='lower right')
            # pp.savefig('dA0_{}.pdf'.format(descent.t))
            # pp.show()
            # pp.close()

        res = G.optimize(params, maxiter=100, debug=True, tune=True, callback=callback)
        # res = G.optimize(subopt_params, maxiter=50, debug=True, tune=True, callback=callback)
        print(res.last_state.params)


    def test_grad_subopt(self):
        from RBPamp import vector_stats
        model, state0 = self.get_default()

        subopt_params = state0.params.copy()
        subopt_params.A0 *= .1
        subopt_params.psam_matrix[0,0] += .0 # set A1 too high
        # subopt_params.betas[2] += 1e-4
        subopt_params.psam_matrix[4,1] = .01 # set C4 too low
        print(subopt_params)

        # state = model.predict(subopt_params, beta_fixed=True)
        # print "ANALYTICAL GRADIENT AT SUB-OPTIMUM"
        # from time import time

        # t0 = time()
        # print state.grad
        # dt = 1000. * (time() - t0)
        # print "# gradient computation took {0:.2f} ms".format(dt)
        
        # from RBPamp.gradient import emp_grad
        # print "EMPIRICAL GRADIENT AT SUB-OPTIMUM"
        # t0 = time()
        # print emp_grad(state)
        # dt = 1000. * (time() - t0)
        # print "# gradient computation took {0:.2f} ms".format(dt)
        # # sys.exit()

        print("performing gradient descent optimization")
        G = GradientDescent(model, subopt_params)
        from RBPamp.report import GradientDescentReport
        rep = GradientDescentReport(G, path='.')
        def callback(descent):
            state = descent.last_state
            rep.plot_report()
            rep.plot_param_hist()
            rep.plot_line_search()
            rep.plot_A0_fit()

            # descent.print_state(state)
            # import matplotlib.pylab as pp
            # pp.figure()
            # print state0.R.shape, state.R.shape
            # pp.loglog(state0.R[0], state.R[0], 'x', label="A0={0:.2f} beta={1:.2e}".format(state.params.A0, state.params.betas[0]))
            # pp.legend(loc='lower right')
            # pp.savefig('dA0_{}.pdf'.format(descent.t))
            # pp.show()
            # pp.close()

        res = G.optimize(subopt_params, maxiter=100, debug=True, tune=True, callback=callback)
        # res = G.optimize(subopt_params, maxiter=50, debug=True, tune=True, callback=callback)
        print(res.last_state.params)

    def test_dA0(self):
        model, state0 = self.get_default()

        print("correct reference state")
        print(state0)
        subopt_params = state0.params.copy()
        subopt_params.A0 *= .1 # set A0 too low

        state = model.predict(subopt_params, beta_fixed=False)
        print("perturbed state")
        print(state)
        import matplotlib.pylab as pp
        pp.loglog(state0.R, state.R, 'x')
        pp.savefig('dA0.pdf')
        pp.close()

        # print "ERROR", state.error
        print("ANALYTICAL GRADIENT")
        from time import time

        t0 = time()
        print(state.grad)
        dt = 1000. * (time() - t0)
        print("# gradient computation took {0:.2f} ms".format(dt))
        
        from RBPamp.gradient import emp_grad
        print("EMPIRICAL GRADIENT")
        t0 = time()
        print(emp_grad(state, eps=1e-4))
        dt = 1000. * (time() - t0)
        print("# gradient computation took {0:.2f} ms".format(dt))

        print("performing gradient descent optimization")
        G = GradientDescent(model, subopt_params)
        res = G.optimize(subopt_params, maxiter=10, debug=True)
        print(res.last_state.params)


    # @unittest.skip("")
    def test_consistency(self):
        """
        Subsequent evaluations of the same model should yield the exact same results.
        """
        # from RBPamp import vector_stats
        model, state0 = self.get_default()
        states = [model.predict(state0.params) for i in range(3)]

        def check(getter):
            for i in range(1,len(states)):
                self.assertTrue( (getter(states[0]) == getter(states[i])).all() )

        check(lambda x : x.psi)
        check(lambda x : x.w )
        check(lambda x : x.rbp_free )
        check(lambda x : x.params.betas )
        check(lambda x : x.Q )
        check(lambda x : x.W )
        check(lambda x : x.R )
        check(lambda x : x.mdl.R0 )
        check(lambda x : x.R_errors )
        check(lambda x : x.grad.data )

    # @unittest.skip("")
    def test_grad_optimum(self):
        """
        At the optimal parameters, the gradient should be 0 in every element.
        """
        model, state0 = self.get_default()
        grad = state0.grad
        from RBPamp.gradient import emp_grad
        egrad = emp_grad(state0, eps=1e-6)
        
        print("\nANALYTICAL GRADIENT AT OPTIMUM")
        print(grad)
        print("EMP. GRADIENT AT OPTIMUM")
        print(egrad)
        self.assertTrue(np.allclose(grad.data,0))

    # @unittest.skip("")
    def test_grad_matrix(self, d=2e-2):
        """
        Small perturbations of the parameter matrix should result in gradients pointing
        in the opposite direction.
        """
        from RBPamp.gradient import GradientDescent, emp_grad

        model, state0 = self.get_default()
        # grad0 = state0.grad
        # print "unperturbed model's gradient"
        # print grad0
        n = len(state0.params.psam_vec)

        ratios = np.ones(n, dtype=np.float32)
        for i in range(1, n):
        # for i in [1,2,3,4, 14,]:
            if i and state0.params.data[i] >= 1:
                continue # only non-cognate

            delta = state0.params.copy()
            delta.data[:] = 0
            delta.data[i] = d

            params = GradientDescent.apply_delta(state0.params, delta)
            # print "perturbation", i, "params[i] =", state0.params.data[i], "->", params.data[i]
            pert = params.copy()
            pert.data[:] = params.data - state0.params.data
            # print pert

            state = model.predict(params, beta_fixed=False, tune=True)
            # state = model.tune(params)
            grad = state.grad
            print("gradient")
            print(grad) #.unity()
            # egrad = emp_grad(state, eps=1e-5)
            # print "emp. gradient"
            # print egrad #.unity()

            i_pert = pert.data.argmax()
            assert i_pert == i
            # I = grad.data.argsort()[::-1]
            I = grad.psam_vec.argsort()[::-1]
            i_grad = I[0]
            i_next = I[1]
            # self.assertTrue(i_grad == i_pert)
            # ratio of highest value in gradient to second-highest.
            
            if i_grad == i_pert:
                ratios[i] = grad.data[i_pert] / grad.data[i_next]
                print("SUCCESS")
            else:
                ratios[i] = grad.data[i_pert] / grad.data[i_grad]
                print("FAILED")

            print("i_pert", i_pert, "i_grad", i_grad, "i_next", i_next, "ratio", ratios[i])

        print("summary", ratios)
        self.assertTrue((ratios >= 1.).all())

    @unittest.skip("")
    def test_5mer_noise(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        self.run_descent(motif, self.noisy_variant(motif, noise=.05), k_monitor=5, rbp_conc=[.1,.5,2.])

    @unittest.skip("")
    def test_5mer_1off(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        off = motif.copy()
        off.A0 = .5
        self.run_descent(motif, off, k_monitor=5, rbp_conc=[.1,.5,2.])

    @unittest.skip("")
    def test_5mer_betas(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        variant = self.noisy_variant(motif, noise=.05)
        variant.psam_vec[:] = motif.psam_vec[:]
        self.run_descent(motif, variant, k_monitor=5, rbp_conc=[.5,50.,200.])

    @unittest.skip("")
    def test_5mer_optimum(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        print(motif)
        init = motif.copy()
        init.A0 = .5
        self.run_descent(motif, init, k_monitor=5, rbp_conc=[.5,50.,200.])
        # self.run_descent(motif, motif, k_monitor=5, rbp_conc=[15., 50.,200.])

    # def test_5mer_A0(self):
    #     motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [5., 3.0, .1,])
    #     self.run_descent(motif, self.noisy_variant(motif, noise=0, A0=6.), k_monitor=5)

    # def test_5mer_high_noise(self):
    #     motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,])
    #     self.run_descent(motif, self.noisy_variant(motif, noise=.1), k_monitor=5)

    # def test_7mer_noise(self):
    #     motif = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU', 'UGCAGGC', ], [1., .5, .02])
    #     self.run_descent(motif, self.noisy_variant(motif))

    # def test_7mer_high_noise(self):
    #     motif = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU', 'UGCAGGC', ], [1., .5, .02])
    #     self.run_descent(motif, self.noisy_variant(motif, noise=.2), k_monitor=5)

    # def test_5mer_off_matrix(self):
    #     motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GCATG'], [1., 1., 1.])
    #     motif_variant = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG','GGGGG'], [1., .2, .8,.1])
    #     self.run_descent(motif_correct, motif_variant)


if __name__ == '__main__':
    # import gzip
    # path = os.path.join(os.path.dirname(__file__), '../tests/reads_20.txt.gz')
    # TODO: include small amount of raw data in git repo for testing!
    # import logging
    # logging.basicConfig(level=logging.WARNING)
    # reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='RBPamp/acc', n_max=1000000)
    unittest.main(verbosity=2)



# i0 = seq_to_index('TGCATGT')
# i1 = seq_to_index('TGCATGC')
# i2 = seq_to_index('TGCATGA')
# i3 = seq_to_index('TGCATGG')
# i4 = seq_to_index('ATGCATG')
# i5 = seq_to_index('CTGCATG')
# i6 = seq_to_index('GTGCATG')
# i7 = seq_to_index('TTGCATG')

# R = RBNSReads(reads, adap5='CCCCCCC', adap3='TTTTTTTT')
# im = R.get_index_matrix(5)
# print reads[0]
# print im.shape, len(reads[0])
# for x in im[0]:
#     print index_to_seq(x, 5),
    
# sys.exit(0)
# counts = R.kmer_counts(7)
# print counts[i0]

# F = R.reads_with_kmers(7)
# print F[i0]

# k = 7

# candidates = np.zeros(4**k, dtype=np.uint32)
# candidates[i0] = 1
# candidates[i1] = 2
# candidates[i2] = 3
# candidates[i3] = 4
# candidates[i4] = 5
# candidates[i5] = 6
# candidates[i6] = 7
# candidates[i7] = 8
# R.write_pure_reads_fasta(sys.stdout, k, candidates, n_sample=100000)
