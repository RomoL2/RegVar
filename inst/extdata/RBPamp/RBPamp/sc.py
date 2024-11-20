from __future__ import print_function
import numpy as np
import logging
import time
from scipy.optimize import minimize_scalar


class SelfConsistency(object):
    def __init__(self, Z1, rna_conc, bins=None):
        self.logger = logging.getLogger("model.SelfConsistency")
        self.Z1 = Z1[Z1 > 0]
        self.last_error = -1
        self.all_free = False
        
        if not len(self.Z1):
            self.all_free = True
            self.logger.warning("returning all RBP as free because Z1 was not usable!")
            return

        # # HACK! Just to mask annoying issues that arent important right now
        # self.free_rbp = self.all_free
        # return

        # print self.Z1.shape, self.Z1.min(), self.Z1.max()
        self.N = len(self.Z1)
        self.rna_conc = rna_conc
        self.logger.debug('rna_conc={0:.3e}'.format(self.rna_conc))
        
        self.bins = bins
        if bins:
            # logarithmic binning
            lZ = np.log(self.Z1)
            self.counts, bins = np.histogram(lZ, bins=bins)
            self.bins = np.exp(bins)
            # midpoint integration
            self.x = (self.bins[1:] + self.bins[:-1])/2.
            
    # def all_free(self, rbp_total, Z_scale=1.):
    #     return rbp_total

    def occupancies(self, rbp_free, Z_scale=1.):

        Z = self.x * Z_scale 
        # p = Z / (Z + 1.)
        p = rbp_free / (rbp_free + 1./Z)
        print("occ", self.x.max(), p.max())
        return Z, p

    def fast_free_rbp(self, rbp_total, Z_scale=1.):
        if self.all_free:
            return rbp_total

        if self.bins is None:
            raise ValueError("fast_free_rbp called without binning!")

        # TODO: use the binned version. Compare accuracy!
        t0 = time.time()
        y = self.x * Z_scale
        # self.logger.debug("y={}".format(y[:10]))

        def to_optimize(p_free):
            Z = p_free * y
            p = Z / (Z + 1.)
            
            rbp_bound = ((p * self.rna_conc)*self.counts).sum() / self.N
            return ((rbp_total - rbp_bound) - p_free)**2
            
        res = minimize_scalar(to_optimize, bounds=(0, rbp_total), method='Bounded')
        t1 = time.time()
        perc = 100. * res.x / rbp_total
        complex = rbp_total - res.x
        if not np.allclose(res.fun, 0, atol=1e-3):
            self.logger.debug("could not satisfy RBP conservation. error={}".format(res.fun))

        self.logger.debug("total RBP={0:.1f} free={1:.1f} ({2:.2f}%) complex={4:.2e} nM in {3:.2f} ms".format(rbp_total, res.x, perc, 1000*(t1-t0), complex))
        self.last_error = res.fun
        return res.x
        
    def free_rbp(self, rbp_total, Z_scale=1.):
        t0 = time.time()

        def to_optimize(p_free):
            Z = p_free * self.Z1 * Z_scale
            p = Z / (Z + 1.)
            
            rbp_bound = (p * self.rna_conc).sum() / self.N
            
            return ((rbp_total - rbp_bound) - p_free)**2
            
        res = minimize_scalar(to_optimize, bounds=(0, rbp_total), method='Bounded')
        t1 = time.time()
        perc = 100. * res.x / rbp_total
        self.logger.debug("total={0:.1f} free={1:.1f} ({2:.2f}%) in {3:.2f} ms".format(rbp_total, res.x, perc, 1000*(t1-t0)))
        
        return res.x
        
    def free_rbp_vector(self, rbp_total, Z_scale=1.):
        free = np.array([self.fast_free_rbp(rbp, Z_scale=Z_scale) for rbp in rbp_total], dtype=np.float32)
        return free


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger('model').setLevel(logging.DEBUG)
    Z1 = np.load('/scratch/data/RBNS/RBFOX3/new/opt/5mers/5mer_Z1_.npy')
    sc = SelfConsistency(Z1, 1000., bins=10000)
    print(5, "->", sc.free_rbp(5.))
    print(5, "->", sc.fast_free_rbp(5.))
