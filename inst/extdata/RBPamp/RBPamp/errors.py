import numpy as np
import shelve
import os
import logging

class PSAMErrorEstimator(object):
    def __init__(self, descentpath, q=[25., 50., 75.], tol=.05, use_shelve=None, max_t=None):
        self.q = np.array(q)
        self.descentpath = descentpath
        self.tol = tol
        self.logger = logging.getLogger("opt.PSAMErrorEstimator")
        if use_shelve:
            self.shelve = use_shelve
        else:
            try:
                spath = os.path.join(descentpath, 'history')
                self.shelve = shelve.open(spath, flag='r')
            except IOError:
                self.logger.error("no history to estimate error from in '{}'".format(descentpath))
                self.shelve = None

        if max_t is None:
            self.max_t = self.find_max_t()
        else:
            self.max_t = max_t

        self.logger.debug(f"{self.descentpath} max_t={self.max_t}")

    def find_max_t(self):
        if self.shelve is None:
            return 0

        t = -1
        while "params_t{}".format(t+1) in self.shelve:
            t += 1
        return t

    def get(self, name, t):
        if t == -1:
            t = self.max_t

        key = "{0}_t{1}".format(name, t)
        if key in self.shelve:
            return self.shelve[key]

    def load_data(self):
        params = []
        stats = []
        t = -1
        par = self.get("params", t+1)
        while not par is None:
            t += 1
            params.append(self.get("params", t))
            stats.append(self.get("stats", t))
            par = self.get("params", t+1)
        
        # for t, par in enumerate(params):
        #     print t, len(par.param_set)

        return np.array(params), np.array(stats)
    
    def estimate(self, save=True, t_ref=-1):
        if self.shelve is None:
            return None

        if t_ref == -1:
            t_ref = self.max_t

        params0 = self.get("params", 0)
        pd0 = params0.get_data()
        params, stats = self.load_data()
        mdl_errors = np.array([s.error for s in stats])
        
        cut = mdl_errors[t_ref] * (1+self.tol)

        I = (mdl_errors < cut)[:t_ref].nonzero()[0]
        if len(I) < 10:
            self.logger.warning("can not estimate errors for t_ref={0}. Insufficient data n={1}.".format(t_ref, len(I)))
            return

        self.logger.debug("esimating errors from n={0} data points for reference t={1}".format(len(I), t_ref))
        param_data = [par.get_data() for par in params[I]]
        param_data = np.array([pd for pd in param_data if pd.shape == pd0.shape])

        # print(param_data.shape, param_data.dtype)
        params_q = np.percentile(param_data, self.q, axis=0)

        p_q = [params0.copy().set_data(perc) for perc in params_q]
        if save:
            for p, q in zip(p_q, self.q):
                p.save(os.path.join(self.descentpath, 'parameters_q{}.tsv'.format(q)))

        p_lo, p_mid, p_hi = p_q
        p_mid.lo = p_lo
        p_mid.hi = p_hi

        return p_mid
    

