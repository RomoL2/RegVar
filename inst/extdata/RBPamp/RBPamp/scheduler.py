import numpy as np
import scipy
import logging
import time
import sys
import os
from collections import defaultdict
from scipy.optimize import minimize, brentq, minimize_scalar

import RBPamp.cyska 

class ParamUpdateScheduler(object):
    #def __init__(self, opt, n_blocked=5, ttl=10, n_max=100, n_avg=5, beta_burn_in=True):
    def __init__(self, opt, n_blocked=5, ttl=5, n_max=100, n_avg=5, beta_burn_in=True, monitor_params=[]):
        # iterative kmer selection 
        self.opt = opt
        self.logger = logging.getLogger("ParamUpdateScheduler")
        self.n_blocked = n_blocked
        self.n_max = n_max
        self.n_avg = n_avg
        self.ttl = ttl
        self.N = len(self.opt.current.params)
        self.monitor_params = [self.opt.mdl.param_index[p] for p in monitor_params]
        if not self.monitor_params:
            # top 10 R-value k-mers
            self.monitor_params = list(self.opt.R_obs.max(axis=0).argsort()[::-1][:10])
            # betas
            self.monitor_params += range(self.opt.nA, self.N)
        
        if beta_burn_in:
            self.blocked_params = range(self.opt.nA, len(self.opt.current.params))
        else:
            self.blocked_params = []

        self.param_last_improvement = {}
        self.param_last_error = {}
        self.param_last_updated = defaultdict(int)
        self.last_param_update = None
        self.R_max = self.opt.R_obs.max(axis=1)
        
    def update(self, pick):
        """
        param selection data structure maintenance
        """

        # param blocked list
        self.blocked_params.append(pick)
        while len(self.blocked_params) > self.n_blocked:
            self.blocked_params.pop(0)

        for i,rel in self.param_last_improvement.items():
            if self.opt.t - self.param_last_updated[i] > self.ttl:
                del self.param_last_improvement[i]

    def param_changed(self, param_i, t, better):
        self.last_param_update = param_i
        self.param_last_updated[param_i] = t
        self.param_last_improvement[param_i] = better
        
    @property
    def kmer_residuals(self):
        ## max mismatch between predicted and observed R-values
        return (self.opt.kmer_errors(self.opt.current.R)**2).sum(axis=0)

    @property
    def beta_residuals(self):
        residual_beta = []
        for i in range(self.opt.n_conc):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self.opt.current.R[i,:], self.opt.R_obs[i,:])
            #print "LINREGRESS", self.rbp_conc[i], slope, intercept, r_value, p_value, std_err
            self.logger.debug("beta{i} slope={slope:.3e} intercept={intercept:.3e} r_value={r_value:.3e} p_value={p_value:.3e} std_err={std_err:.3e}".format(**locals()) )
            #residual_beta.append(0)
            residual_beta.append( self.R_max[i] * (np.fabs(1 - slope) + np.fabs(intercept) ) ) 

        self.logger.debug("beta residuals: {0}".format(residual_beta) )
        return np.array(residual_beta)
  
    @property
    def susceptibility(self):
        # high affinity kmers are more susceptible to changes, unless we look at under-estimation
        error = self.opt.kmer_errors(self.opt.current.R)
        affinities = self.opt.current.params[:self.opt.nA]
        a = affinities / affinities.max()
        suscept = (error > 0).all(axis=0) * a + (error < 0).sum(axis=0)
        suscept = np.concatenate( (suscept, np.ones(self.opt.n_conc)) )
        #caccc = cyska.seq_to_index('CACCC')
        #print "suscept of GCATG", suscept[590]
        #print "suscept of CACCC", suscept[caccc], a[caccc]

        return suscept

    @property
    def expectation(self):
        ## expected improvement upon parameter optimization
        if len(self.opt.rel_improvements):
            # default = average over past improvements + 10% benefit
            exp0 = np.array(self.opt.rel_improvements[-self.n_avg:]).mean()
        else:
            # fallback
            exp0 = 1
            
        exp0 = max(.01, exp0)

        expect = np.ones(self.N) * exp0
        for i in self.param_last_improvement.keys():
            expect[i] = self.param_last_improvement[i]

        return expect

    @property
    def time_passed(self):
        ## time since last update
        dt = np.ones(self.N, dtype=int) * (self.opt.t + 1)
        for i in self.param_last_updated.keys():
            dt[i] -= self.param_last_updated[i]
        
        dt[self.opt.nA:] *= 10 # make beta updates 10 times more often
        return dt
    
    def debug_monitor(self, param_list=[], title="monitored parameters"):
        self.logger.info(">>>>> {0} <<<<<".format(title))
        self.logger.info("kmer\tblocked\tscore\tresidual\tdt\texpect\tsuscept\tcurrent\tknown\tdR")

        if not len(param_list):
            param_list = self.monitor_params

        for i in param_list:
            self.logger.info(self.debug_str_from_param(i))

    def debug_str_from_param(self, i):
        if i in self.blocked_params:
            state = 'B'
        else:
            state = ' '

        if i < self.opt.nA:
            R_str = ",".join(["{0:.2f}".format(x) for x in self.opt.current.R[:,i] - self.opt.R_obs[:,i]])
        else:
            R_str = 'n/a'
        
        #print "DUMP"
        #print self.opt.k
        #print len(self.opt.mdl.param_name)
        #print len(self.opt.known_params)
        return "{0:10s} {1}\t{2:.2e}\t{3:.2e}\t{4}\t{5:.2e}\t{6:.3e}\t{7:.3e}\t{8:.3e}\t{9}".format( 
            self.opt.mdl.param_name[i], 
            state, 
            self._score[i], 
            self._residual[i], 
            self._dt[i], 
            self._expect[i], 
            self._suscept[i],
            self.opt.current.params[i], 
            self.opt.known_params[i], 
            R_str 
        )
        
    def find_worst_param(self):
        """
        ideas: 
            * take into account if over-under-representation is systematic or only seen at some concentrations (contradicted at others?)
            * start preferring kmers whose error peaks at low concentrations (high affinity), then move to kmers whose error peaks at intermediate or high concentrations (lower affinity)
            * perhaps overall R-value correlation can serve as guide? (> .9 do a round of low affinity optimizations in fixed energy background?)
        """

        self._residual = np.concatenate( (self.kmer_residuals, 0*self.beta_residuals) ) # HACK: de-activate beta updates from inside same framework
        self._dt = self.time_passed
        self._expect = self.expectation
        self._suscept = self.susceptibility
        #score = residual * dt * expect * self.susceptibility
        self._score = self._residual #* self._dt * self._expect * self._suscept
        #self.logger.debug("beta expect {0}".format(expect[self.opt.nA:]))
        #self.logger.debug("beta scores {0}".format(score[self.opt.nA:]))
        
        #self.debug_monitor()
        ranked = self._score.argsort()[::-1]
        #self.debug_monitor(param_list = ranked[:10], title="candidate search")
        pick = None        
        for i in ranked:
            if not i in self.blocked_params:
                pick = i
                break
        
        self.logger.info("selected {0}".format(self.opt.mdl.param_name[pick]) )
        return pick
    
    def pwm_set(self, seed, k):
        """
        generate all single base substitution variants of a 
        seed motif by bit-operations on the corresponding kmer index.
        """
        variants = [seed]
        for j in range(k):
            nt = (seed >> j*2) & 3
            #print "j,nt",j,nt
            mask = seed ^ nt << (j*2)
            #print "mask", cyska.index_to_seq(mask,k)
            
            for l in range(4):
                var = mask | (l << j*2)
                #print "variant", cyska.index_to_seq(var,k)

                if var != seed:
                    variants.append(var)

        scores = [self._score[i] for i in variants]
        to_sort = zip(scores, variants)
        ordered = [i for s,i in sorted(to_sort, reverse=True)]
        
        return ordered
        
