__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import sys
import itertools
import numpy as np
import copy
import time
import os
import logging
import collections
import RBPamp.cyska
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as pp

from RBPamp.caching import cached, pickled, CachedBase
from RBPamp.rbns_reads import RBNSReads
from RBPamp.rbns_analysis import RBNSAnalysis
from RBPamp.ska_runner import SKARunner


class PairInteractionScreen(object):
    def __init__(self, res, core, k_int, pseudo=10.):
        self.res = res
        self.core = core
        self.k_int = k_int
        self.k_core = len(core)
        
        #print "scanning flanking {0}-mers".format(k_int)
        pd_matrix, pd_mask = self.res.pd_reads.kmer_flank_profiles(core, k_int)
        in_matrix, in_mask = self.res.in_reads.kmer_flank_profiles(core, k_int)

        self.core_density_pd = pd_mask.sum(axis=0)
        self.core_density_bg = in_mask.sum(axis=0)
        
        pd_matrix = np.array(pd_matrix, dtype=np.float32) + pseudo
        in_matrix = np.array(in_matrix, dtype=np.float32) + pseudo

        obsv = pd_matrix / pd_matrix.sum(axis=0)[np.newaxis,:]
        bgnd = in_matrix / in_matrix.sum(axis=0)[np.newaxis,:]

        # Kullback-Leibler (KL) divergence terms
        self._KL = (obsv * np.log2(obsv / bgnd) )
        # and per-position KL
        self.KL = self._KL.sum(axis=0)
        self.log_ratios = np.log2(obsv / bgnd)

        # mask positions overlapping with the core motif
        self.l = self.res.pd_reads.L - self.k_core
        self.KL[self.l-self.k_int+1:self.l+self.k_core] = 0
        self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core] = 0
        #print "mask",self.l-self.k_int+1,self.l+self.k_core 
        #print "log_ratios after masking", self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core]

    def top_interactors(self, n_top=10):
        #TODO: cook a set of candidate interacting kmers from significance for now let's just take top 10
        top_i = self._KL.max(axis=1).argsort()[::-1][:n_top]
        # and sort alphabetically to ensure reproducibility across successive runs
        top_i = sorted(top_i)
        top_kmers = [cyska.index_to_seq(i, self.k_int) for i in top_i]
        #print "top interacting kmer candidate list", top_kmers
        return top_i, top_kmers
    
    def make_plot(self, fname, n_top=10):
        import matplotlib as mp
        mp.rcParams['font.family'] = 'Arial'
        mp.rcParams['font.size'] = 8
        mp.rcParams['font.sans-serif'] = 'Arial'
        mp.rcParams['legend.fontsize'] = 'small'
        mp.rcParams['legend.frameon'] = False
        #mp.rcParams['axes.labelsize'] = 8
        
        import matplotlib.pyplot as pp
        fig = pp.figure()
        fig.subplots_adjust(hspace=0.5)
        
        pp.title("{self.res.pd_reads.rbp_name}@{self.res.pd_reads.rbp_conc}nM {self.core} interacting with {self.k_int}-mers".format(self=self))
        pp.subplot(311)
        pp.gca().set_title("density of {0} core".format(self.core.upper()))
        pp.plot(self.core_density_pd, drawstyle='steps-mid', label="pd")
        pp.plot(self.core_density_bg, drawstyle='steps-mid', label="in")
        pp.gca().locator_params(axis='y',nbins=3)
        pp.gca().locator_params(axis='x',nbins=10)

        pp.legend(loc='upper left')
        pp.xlabel("read start pos [nt]")
        pp.ylabel("frequency")
        
        pp.subplot(312)
        pp.title("Kullback-Leibler divergence of flanking kmer composition")
        x = np.arange(len(self.KL)) - len(self.KL)/2 +1.
        pp.plot(x,self.KL, drawstyle='steps-mid', label="{0}mers around {1}".format(self.k_int, self.k_core) )
        pp.xlim(-self.l-.5,self.l+.5)
        pp.xlabel("rel. {0}-mer start pos [nt]".format(self.k_int))
        pp.ylabel("KL [bits]")
     
        pp.subplot(313)
        pp.gca().set_title("enriched {0}-mers".format(self.k_int))
        top_i, top_kmers = self.top_interactors(n_top = n_top)
        z = self.log_ratios[top_i[::-1],:]
        y, x = np.mgrid[slice(0, len(top_i)+1),slice(-(self.l+.5), +self.l+1)]
        
        # symmetric, dynamic range of colorbar
        dr = np.fabs(z).max()
        pp.pcolor(x,y,z, cmap=pp.get_cmap('seismic'), vmin=-dr, vmax=dr)

        pp.yticks(np.arange(len(top_i))+.5, top_kmers[::-1]) # reverse order of kmers so pcolor is not upside down
        pp.xlim(-self.l-.5,self.l+.5)
        cbar = pp.colorbar(orientation="horizontal", fraction=0.1, shrink=0.75, label=r"$\log_2( \frac{pd}{in} )$")
        cbar.ax.tick_params(labelsize=8)
        pp.savefig(fname)
        pp.close()
        
