# coding=future_fstrings
from __future__ import print_function
import os
import unittest
import time
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class Tracked(object):
    """
    Used as a container to hold data on line-search intermediate results, A0 fitting, beta fitting, etc.
    """
    def __init__(self, **kwargs):
        self._kw = kwargs
        for k,v in list(kwargs.items()):
            setattr(self, k,v)


def emp_grad(state, eps=1e-4):
    v0 = state.params.get_data()
    v = np.array(v0)
    var = state.params.copy()
    grad = np.array(v0)
    state0 = state
    # print "err0", err0
    kw = dict()#.predict_kwargs)
    kw['beta_fixed'] = False #True
    kw['tune'] = False
    kw['rbp_free'] = state0.rbp_free

    for i in range(len(v0)):
        # print ">>> EMP GRAD", state.params.names[i]
        # d = max(v0[i] * eps,1e-6)
        d = eps
        v[i] = v0[i] + d
        var.set_data(v)
        state = state.mdl.predict(var, **kw)
        derr = state.error - state0.error
        grad[i] = derr/d
        # print "derr", derr
        v[i] = v0[i]

        # print ">>>GRAD ELEMENT", grad.data[i]
    
    var.set_data(grad)
    return var


def emp_gradi(state, eps=1e-4):
    v0 = state.params.as_vector()
    var = state.params.copy()
    Nk = state.mdl.nA
    n_data = len(state.params.data)
    n_samples = state.params.n_samples
    gradi = np.zeros((n_samples, Nk, n_data), dtype=np.float32)

    R0 = state.R
    state0 = state

    kw = dict()
    kw['beta_fixed'] = True
    kw['rbp_free'] = state0.rbp_free

    for i in range(state.params.n):
        d = eps
        var.data[i] = v0[i] + d
        state = state.mdl.predict(var, **kw)
        dR = state.R - R0
        gradi[:,:,i] = dR/d

        var.data[i] = v0[i]
    
    return gradi


def emp_grad_A0(state, eps=1e-4, _dE=True, _dR=False, _dq=False, _dPsi=False, _dZ=False):
    "useful for debugging"
    ret = []
    for i, params in enumerate(state.params.copy()):
        err0 = state.error

        Z0 = np.array(state.Z1_read_motif[i])
        params.A0 += eps
        kw = dict()#.predict_kwargs)
        kw['beta_fixed'] = False
        kw['tune'] = False
        kw['rbp_free'] = state.rbp_free

        paramset = state.params.copy()
        paramset[i] = params
        new = state.mdl.predict(paramset, **kw)

        dE = (new.error - err0)/eps
        dZ = new.Z1_read_motif[i] - Z0
        dPsi = (new.psi - state.psi)/eps
        dq = (new.q - state.q)/eps
        # dQ = (new.Q - state.Q)/eps
        dR = (new.R - state.R)/eps

        res = {}
        if _dE: res['dE'] = dE
        if _dR: res['dR'] = dR
        if _dq: res['dq'] = dq
        if _dPsi: res['dPsi'] = dPsi
        if _dZ: res['dZ'] = dZ
        
        ret.append(res)

    return ret


def ana_grad_A0(state, eps=1e-3, _dE=True, _dR=False, _dq=False, _dPsi=False, _dZ=False):
    "useful for debugging"
    ret = []
    for i, params in enumerate(state.params):
        norm = (params.A0/state.params.A0)
        print("norm", norm)
        Z1m = state.Z1_read_motif[i] 
        dZ = Z1m / state.Z1_read
        # print "dZ", dZ.shape, dZ
        dPsi_dA0 = (state.psi - state.psi**2) * dZ
        dPsi_dA0 /= state.params.A0
        dq = state.mdl.PD_kmer_weights(dPsi_dA0)
        dR = state.R / state.q * (dq - state.mdl.f0[np.newaxis,:] * state.R * dq.sum(axis=1)[:, np.newaxis])
        dE = 2 * ((state.R - state.mdl.R0) * dR).mean()

        res = {}
        if _dE: res['dE'] = dE
        if _dR: res['dR'] = dR
        if _dq: res['dq'] = dq
        if _dPsi: res['dPsi'] = dPsi
        if _dZ: res['dZ'] = dZ
        
        ret.append(res)

    return ret



def minimize_logspaced(func, bounds=[], n_samples=7, debug=False, nested=2, options=None, **kwargs):
    """
    first evaluate at log-spaced sampling points along parameter range
    then select at most 3 orders of magnitude around the lowest observed value
    for Brent optimization. Requires pos. valued bounds!
    """
    from scipy.optimize import minimize_scalar, OptimizeResult
    import time
    t0 = time.time()
    bmin = bounds.min()
    bmax = bounds.max()

    known = {}

    def func_or_lookup(x):
        if not x in known:
            known[x] = func(x)
        return known[x]

    def logsearch(bmin, bmax):
        
        lmin = np.log10(bmin)
        lmax = np.log10(bmax)
        
        sample_x = 10**np.linspace(lmin, lmax, n_samples)
        samples = np.array([func_or_lookup(x) for x in sample_x])
            
        if debug:
            print("logspaced sample", list(zip(sample_x, samples)))

        i = samples.argmin()
        li = max(0, i -1)
        ri = min(n_samples-1, i+1)
        
        brent_min = sample_x[li]
        brent_max = sample_x[ri]
        if debug:
            print("search optimum between", brent_min, brent_max)
    
        return brent_min, brent_max

    for i in range(nested):
        bmin, bmax = logsearch(bmin, bmax)

    if debug:
        print("minimize_scalar(bounds=[{bmin}, {bmax}])".format(**locals()))

    try:
        res = minimize_scalar(func_or_lookup, bounds = np.array([bmin, bmax]), method='Bounded', options=options) #, **kwargs)
    except UnboundLocalError:
        # known issue in scipy triggered if the interval 
        # is too small for minimize_scaler to actually consider
        # evaluation of the function. Make up a decent guesstimate.
        x = (bmax + bmin)/2. # half-point should do
        res = OptimizeResult(
            fun=func_or_lookup(x), status=0, success=True,
            message='Solution found, despite UnboundLocalError',
            x=x, nfev=1
        )

    t1 = time.time()
    x = sorted(known.keys())
    y = [known[i] for i in x]
    data = Tracked(x=x, y=y)
    res.opt_data = data
    # self.logger.debug("minimize_logspaced took {dt:.2f}ms".format(dt= 1000. * (t1-t0)) )
    return res


class GradientDescent(object):
    def __init__(self, model, params0, dec=.5, ref_state=None, maxiter=1000, maxtime=11.5*3600, eps=1e-6, tau=13, predict_kwargs=dict(beta_fixed=False, tune=True), debug_grad=False, fix_A0=False, errors=[]):
        self.logger = logging.getLogger('opt.GradientDescent')
        self.model = model
        self.params = params0
        self.ref_state = ref_state # used for simulations, where true values are known.
        self.predict_kwargs = predict_kwargs
        self.fix_A0 = fix_A0
        if fix_A0:
            self.predict_kwargs['tune'] = False

        self.debug_grad = debug_grad
        self.model.opt = self # link model to this optimizer instance so it can find out R0 etc.

        # momentum smoothing of the gradient
        self.past_grad = None
        self.past_sqg = 1
        self.past_var = 1
        self.dec = dec

        # records
        self.errors = list(errors)
        # self.history = []
        self.ls_nfev = [0,]
        self.ls_step = [0,]
        self.t = 0
        self.last_subsample_t = 0
        self.last_quantile = 0

        # optimization result/status
        self.status = None
        self.maxiter = maxiter
        self.maxtime = maxtime
        self.eps = eps
        self.tau = tau

    def line_search(self, state, vec, debug=True, min_step = 1e-6, max_step = 5., maxiter=10, xatol=1e-1):
        from scipy.optimize import minimize_scalar

        e0 = state.error
        # assert e0 == state.mdl.predict(state.params, **self.predict_kwargs).error

        params0 = state.params
        t0 = time.time()
        N = {'fev' : 0}
        kw = dict(self.predict_kwargs)
        kw['beta_fixed'] = False
        kw['tune'] = False
        kw['keep_Z1_motif'] = False

        scales = []
        errors = []

        # self.model.set_mask( state.Z1_read > self.model.Z_thresh * state.Z1_read.max())
        # throw away large buffers of reference state before 
        # actual line-search bc we don't need them anymore
        state.flush()

        def err(s):
            # s = np.exp(x)
            m = params0.apply_delta(vec * s)
            new = self.model.predict(m, **kw)

            N['fev'] += 1
            scales.append(s)
            new_err = new.error
            errors.append(new_err)
            if debug:
                print(s,"->", new_err - e0)
            return new_err - e0

        # assert np.fabs(err(0)) < 1e-6

        res = minimize_logspaced(err, bounds = np.array([min_step, max_step]), options=dict(maxiter=maxiter) )
        # res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([min_step, max_step])), options=dict(maxiter=maxiter, xatol=xatol))
        # self.logger.debug("minimize_logspaced took {dt:.2f}ms".format(dt= 1000. * (t1-t0)) )

        # self.model.set_mask()
        self.logger.debug("line_search took {0:.3f} seconds for {1} iterations".format(time.time() - t0, N['fev']))
        # print res.success, res.fun, res
        if res.fun > 0:
            self.logger.warning("line_search could not decrease error!")
            s = 0
        else:
            s = res.x
        
        scales = np.array(scales)
        errors = np.array(errors)
        I = scales.argsort()
        return s, Tracked(scales = scales[I], errors=errors[I], res=res, s_opt=s, err0=e0)

    def Adam(self, local_grad, delta=.00001):
        if self.past_grad is None:
            self.past_grad = local_grad.get_data()

        m = self.dec * self.past_grad + (1 - self.dec) * local_grad.get_data()
        s = self.dec * self.past_sqg + (1 - self.dec) * local_grad.get_data()**2

        self.past_grad = m
        self.past_sqg = s

        upd = local_grad.copy()
        upd.set_data(m / (np.sqrt(s) + delta))

        return upd

    def new_subsample(self):
        self.logger.debug("drawing new sub-sample at t={}".format(self.t))
        self.model.init_subsample()
        self.last_subsample_t = self.t
        state = self.model.predict(self.params, **self.predict_kwargs)

        return state

    def converged(self, atol=1e-7):
        if len(self.errors):
            if self.errors[-1] < atol:
                return 'CONVERGED_ERR_MINIMAL'

        if self.past_grad is None:
            mag = np.inf
        else:
            mag = np.sqrt((self.past_grad**2).sum())
        if mag < atol:
            return 'CONVERGED_GRAD_NULL'

        if len(self.errors) < self.tau:
            self.logger.debug("not enough data to estimate convergence (t={})".format(self.t))
            return self.status
        
        last_errs = np.array(self.errors[-self.tau:])
        rel_error = self.errors[-1] / self.errors[0]
        rel_err_dec = self.errors[-2] / self.errors[0] - rel_error

        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(self.tau), last_errs)
        rel_decrease = - slope / self.errors[0]

        rel_dec_eps = rel_decrease / self.eps
        self.logger.debug("rel_decrease={rel_dec_eps:.1f} x eps (P < {p_value:.3e}) |gradient|={mag:.2e}".format(**locals()))
        if rel_decrease < self.eps:
            return 'CONVERGED_NO_MORE_DECREASE'
        else:
            return self.status

    def reached_maxtime(self, dt):
        return self.maxtime and (dt >= self.maxtime)
    
    def reached_maxiter(self, t):
        return self.maxiter and (t >= self.maxiter)

    @property
    def error_reduction(self):
        if len(self.errors) < 2:
            return 0.
        
        return self.errors[-1] / self.errors[0]

    def print_state(self, state):
        print(">>>>>>>>>PARAMS")
        print(state.params)
        s = 0
        if len(self.ls_step):
            s = self.ls_step[-1]

        print("=" * 50)
        last_err = self.errors[-1]
        print("step={self.t} error={last_err:.5e} n_fev={self.model.n_fev} n_grad={self.model.n_grad} scale={s} corr={state.correlations[0]}".format(**locals()))


    def optimize(self, params, debug=False, callback=None):
        # if debug:
        #     print "INITIAL PARAMETERS"
        #     print params
        #     from RBPamp.caching import _dump_cache_sizes
        #     _dump_cache_sizes()

        state = self.model.predict(self.params, **self.predict_kwargs)

        self.errors.append(state.error)
        # self.history.append(state.archive())
        self.last_state = state

        if debug:
            print("INITIAL STATE AFTER FIRST EVAL kwargs=", self.predict_kwargs)
            self.print_state(state)

        if callback:
            callback(self, state)

        t0 = time.time()
        dt = 0

        try:
            while not self.converged() and not self.reached_maxiter(self.t) and not self.reached_maxtime(dt):
                # print "computing gradient"
                local_grad = state.grad #.unity()
                # local_grad.A0 = 0.0046

                if debug:
                    print("LOCAL GRAD, EMP. GRAD")
                    if self.debug_grad:
                        # for lcl, emp_A0_res, ana_A0_res, emp in zip(local_grad, emp_grad_A0(state), ana_grad_A0(state), emp_grad(state)):
                        for lcl, emp, ana_A0_res in zip(local_grad, emp_grad(state), ana_grad_A0(state)):

                            print("ana_dA0", ana_A0_res['dE'])
                            print("LCL")
                            print(lcl)
                            # print "emp_dA0", emp_A0_res['dE']
                            print("EMP")
                            print(emp)
                    else:
                        print(local_grad)
                
                local_grad.betas[:] = 0. # model.predict automatically finds optimal beta values!!!
                if self.fix_A0:
                    local_grad.A0s[:] = 0.

                descent = self.Adam( - local_grad ) #.unity()
                # descent = self.momentum_grad( - local_grad).unity()

                # print "ERRORS", self.errors
                stuck = False
                s, ls_data = self.line_search(state, descent, debug=debug)
                if s == 0:
                    self.logger.warning("line_search could not decrease error! Resetting search direction to local gradient ...")
                    # take local gradient instead
                    descent = - local_grad.unity()
                    s, data = self.line_search(state, descent, debug=debug)
                    
                    # and reset Adam
                    self.past_grad = None
                    self.past_sqg = 1
                    if s == 0:
                        self.logger.warning("line_search unable to reduce error using local gradient")
                        stuck = True ## signal via callback to PSAMGradientDescent that we need a re-sample

                self.ls_step.append(s)
                self.ls_nfev.append(ls_data.res.nfev)

                upd = descent * s
                self.params = state.params.apply_delta(upd)
                self.model.params = self.params

                state = self.model.predict(self.params, **self.predict_kwargs)
                state._ls_data = ls_data
                self.t += 1
                # self.history.append(state.archive())
                self.last_state = state
                self.errors.append(state.error)

                if debug:
                    print(">>>>>>>>>UPDATE, scale=",s)
                    # print descent
                    self.print_state(state)

                if callback:
                    if stuck:
                        state.stuck = True
                    state = callback(self, state)

                # if state != self.last_state:
                #     self.logger.debug("callback changed state:")
                #     print "AFTER CALLBACK"
                #     self.print_state(state)

                self.logger.debug("n_fev={self.model.n_fev} t_aff={t_aff:.3f} t_fev={t_fev:.3f}ms n_grad={self.model.n_grad} t_grad={t_grad:.3f}ms".format(
                    self=self,
                    t_aff = 1000. * self.model.t_aff/self.model.n_fev,
                    t_fev = 1000. * self.model.t_fev/self.model.n_fev,
                    t_grad = 1000. * self.model.t_grad/self.model.n_grad,
                ))
                dt = time.time() - t0
                # if debug:
                #     from RBPamp.caching import _dump_cache_sizes
                #     print "caches at the end of loop"
                #     _dump_cache_sizes()

        # except ValueError: #KeyboardInterrupt
        except KeyboardInterrupt:
            self.status = "KEYBOARD_INTERRUPT"
        else:
            if self.reached_maxtime(dt):
                self.status = "MAX_TIME"

            elif self.reached_maxiter(self.t):
                self.status = "MAX_ITER"

            else:
                self.status = self.converged()

        self.logger.info("optimization ended with status {self.status} after {self.t} iterations".format(self=self))
        # print "last gradient"
        # print self.past_grad
        # print "squared"
        # print self.past_sqg
        
        return self




if __name__ == '__main__':
    # import gzip
    # path = os.path.join(os.path.dirname(__file__), '../tests/reads_20.txt.gz')
    # TODO: include small amount of raw data in git repo for testing!
    from RBPamp.reads import RBNSReads
    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='RBPamp/acc', n_max=1000000)
    unittest.main(verbosity=2)
