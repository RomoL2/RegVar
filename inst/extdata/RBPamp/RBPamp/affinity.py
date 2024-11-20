import numpy as np
import scipy
import logging
import time
import sys
import os
import RBPamp.cyska as cyska
from RBPamp.caching import CachedBase, cached, pickled

def Kd_to_kcal(K,temp=22):
    RT = (temp + 273.15) * 8.314459848# RT in Joules/mol
    kcal = 4.184E3 # kcal in Joules
    # Kd in nM to E in kcal/mol
    return np.log(K/1E9)*RT/kcal


def kcal_to_Kd(E,temp=22):
    RT = (temp + 273.15) * 8.314459848# RT in Joules/mol
    kcal = 4.184E3 # kcal in Joules
    #print "1./RT",kcal/RT
    # kcal/mol to Kd in nM
    return np.exp(E*kcal/RT)*1e9

#class AffinityDistribution(object):
    #def __init__(self, invkd, T=22):
        #self.RT = (T + 273.15) * 8.314459848 / 4.184E3 # RT in kcal/mol
        #self.invkd = invkd
        
    #@classmethod
    #def singleton(cls, best = 'GCATG', best_kd=1., bg_kd=1e6, T = 22):
        #k = len(best)
        #invkd = np.ones(4**k, dtype=np.float32) / bg_kd

        #best_i = cyska.seq_to_index(best)
        #invkd[best_i] = 1./best_kd
        
        #return cls(invkd, T=T)
    
    #@classmethod
    #def doublet(cls, a = 'GCATG', b='GCACG', a_kd=1., b_kd=20., bg_kd=1e6, T=22):
        #k = len(best)
        #invkd = np.ones(4**k, dtype=np.float32) / bg_kd

        #invkd[cyska.seq_to_index(a)] = 1./a_kd
        #invkd[cyska.seq_to_index(b)] = 1./b_kd
        
        #return cls(invkd, T=T)
        
        
            
class AffinityDistribution(object):
    def __init__(self, mdl, reads, openen):
        self.mdl = mdl
        self.reads = reads
        self.openen = openen
        self.bins = np.arange(-15,4,1.)
        self.logger = logging.getLogger("model.AffinityDistribution")
    
    def get_affinities(self):
        im = cyska.seq_matrix_to_index_matrix(self.reads.seqm, self.mdl.k)
        oem = self.openen.oem
        kmer_invkd = self.mdl.params[:self.mdl.nA]

        Z1 = cyska.SPA_partition_function(
            im, oem, self.mdl.acc_lookup, 
            kmer_invkd, self.mdl.k, 
            openen_ofs = self.mdl.openen.ofs
        )
        return Z1

    def get_affinity_distribution(self):
        self.logger.debug("computing affinity distribution for {0}".format(self.reads.name) )

        Z1 = self.get_affinities()
        self.logger.info("affinities {3} min={0}, max={1}, median={2}".format( Z1.min(), Z1.max(), np.median(Z1), self.reads.name ) )
        
        y, x = np.histogram(np.log10(Z1), bins = self.bins, density=False )
        return y / float(y.sum()), x

    def predict_affinity_distribution(self, rbp_conc, beta=0, bg=None):
        self.logger.debug("predicting affinity distribution @RBP_conc={0}nM from {1}".format(rbp_conc, self.reads.name) )
        Z1 = self.get_affinities()
        psi = (rbp_conc * Z1) / (rbp_conc * Z1 + 1.)

        y, x = np.histogram(np.log10(Z1), bins = self.bins, weights = psi )
        y /= y.sum()
        
        y += beta * bg
        return y / y.sum(), x
