__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens", "Alex Robertson"]
__email__ = "mjens@mit.edu"

import numpy as np
import copy
import time
import logging
import collections
import RBPamp.cyska as cyska

class SKARunner(object):
    def __init__(self, max_iterations=10, convergence=0.5, subsamples=10):
        self.logger = logging.getLogger('SKARunner')
        self.max_iterations = max_iterations
        self.convergence = convergence
        self.n_subsamples = subsamples
        
    def stream_counts(self, k, pd_reads, in_reads):
        self.logger.info("streaming {k} mers in {pd_reads.name}".format(**locals()))
        t0 = time.time()
        kmers = list(cyska.yield_kmers(k))
        #background, bg_source = self.try_load_background_freqs(k)

        pd_freqs = pd_reads.kmer_frequencies(k)
        in_freqs = in_reads.kmer_frequencies(k)
        R_values = pd_freqs / in_freqs # kmer-frequency 'R'-atios

        current_weights = copy.copy(R_values)
        
        weight_history = []
        for iteration_i in range(self.max_iterations):
            new_weights = cyska.seq_set_SKA(pd_reads.seqm, current_weights, in_freqs, k)
                
            weight_history.append(new_weights)
            current_weights = copy.copy(new_weights)
            
            if len(weight_history) > 1:
                delta = weight_history[-1] - weight_history[-2]
                
                change = np.fabs(delta)
                mean_change = change.mean()
                max_i = change.argmax()
                max_change = change[max_i]
                
                self.logger.debug("iteration {0}: mean_change={1} max_change={2} for '{3}' ({4} -> {5})".format(iteration_i, mean_change, max_change, kmers[max_i], weight_history[-2][max_i], weight_history[-1][max_i]))
                if max_change < self.convergence:
                    self.logger.info("reached convergence after {0} iterations".format(iteration_i))
                    break
        
        if iteration_i >= self.max_iterations:
            self.logger.warning("reached max_iterations without convergence!")

        t = time.time() - t0
        self.logger.debug("streaming of {0:.2f}M reads took {1:.2f}ms".format(pd_reads.N / 1e6,  1000. * t) )
        return current_weights

