from __future__ import absolute_import
import numpy as np
import logging
# from RBPamp import timed

class SubSampler(object):
    def __init__(self, n_total, n_samples, replace=True):
        self.n_samples = n_samples
        self.n_total = n_total
        self.replace = True
        self.logger = logging.getLogger("reads.SubSampler({self.n_samples}/{self.n_total} replace={self.replace})".format(self=self))
        self.n = 0
        self.ind = None

    # @timed
    def new_indices(self):
        from .cyska import fast_randint
        self.n += 1
        self.logger.debug('sample {self.n}: subsampling {self.n_samples} out of {self.n_total} sequences. replacement={self.replace}'.format(self=self) )
        if self.replace:
            ind = fast_randint(self.n_samples, self.n_total)
        else:
            ind = np.random.choice(self.n_total, size=self.n_samples, replace=False)
        
        self.ind = ind
        return ind

    def draw(self, data, ind=None):
        if ind is None:
            if self.ind is None:
                self.ind = self.new_indices()
            sample_name = str(self)
            ind = self.ind
        else:
            sample_name = "custom sample"

        self.logger.debug("drawing {0} from {1}".format(sample_name, data))
        return data[ind]

    def redraw(self, data):
        self.ind = self.new_indices()
        return self.draw(data)

    def __str__(self):
        return "sample{:02d}".format(self.n)
