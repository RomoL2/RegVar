# coding=future_fstrings
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import logging


class Proxy(object):
    def __init__(self, data, start, end, shape=None, unpack=True):
        self.data = data
        self.start = start
        self.end = end
        self.shape = shape
        self.unpack = unpack
    
    def get_values(self):
        # print "get_values", self.start, self.end
        d = self.data[self.start:self.end]
        if not self.shape is None:
            d = np.reshape(d, self.shape)
        
        if len(d) == 1 and self.unpack:
            return d[0]
        else:
            return d

    def set_values(self, d):
        # print "set_values", self.start, self.end, d
        
        if hasattr(d, '__len__'):
            if not isinstance(d, np.ndarray):
                d = np.array(d)
            d = d.flatten()
            assert len(d) == self.end - self.start
            self.data[self.start:self.end] = d[:]
        else:
            assert self.end - self.start == 1
            self.data[self.start] = d

        return d

class ModelSetParams(object):
    def __init__(self, param_set, forward = ['k', 'n_samples', 'k_mdl', 'acc_shift', 'acc_scale'], sort=False):
        self._forward = set(forward)
        if sort:
            self.param_set = sorted(param_set, key = lambda params: params.A0, reverse=True)
        else:
            self.param_set = param_set

    def __getattr__(self, attr):
        fw = object.__getattribute__(self, '_forward')
        p0 = object.__getattribute__(self, 'param_set')[0]
        # print p0, attr
        if attr in fw:
            return getattr(p0, attr)
        else:
            return object.__getattribute__(self, attr)

    # @property
    # def n_samples(self):
    #     return self.param_set[0].n_samples

    # @property
    # def n_samples(self):
    #     return self.param_set[0].n_samples

    @property
    def A0(self):
        return self.param_set[0].A0

    @A0.setter
    def A0(self, value):
        # change all motif A0s in proportion
        ratio = value / self.param_set[0].A0
        for par in self.param_set:
            par.A0 *= ratio

    @property
    def A0s(self):
        return np.array([par.A0 for par in self.param_set])
    
    @A0s.setter
    def A0s(self, values):
        for par, a in zip(self.param_set, values):
            par.A0 = a

    @property
    def acc_k(self):
        return np.array([p.acc_k for p in self.param_set])

    @acc_k.setter
    def acc_k(self, value):
        # change all motif acc_k's (esp. for acc_k=0)
        for par in self.param_set:
            par.acc_k = value

    @property
    def betas(self):
        return self.param_set[0].betas

    @betas.setter
    def betas(self, value):
        self.param_set[0].betas = value

    def copy(self, sort=False):
        return ModelSetParams([p.copy() for p in self.param_set], sort=sort)

    def get_data(self):
        "return one np.ndarray containing all model parameters"
        all_data = [p.data for p in self.param_set]
        # print all_data
        return np.concatenate(all_data)

    def set_data(self, data):
        "broadcast raw data write across all model parameters"
        i = 0
        for p in self.param_set:
            l = len(p.data)
            # print "p.data", l
            p.data[:] = data[i:i+l]
            i += l

        # print i, len(data)
        assert i == len(data)
        return self
    
    def unity(self):
        p = self.copy()
        n = np.linalg.norm(self.get_data())
        if n > 0:
            p /= n
        return p

    @classmethod
    def load(cls, fname, n_samples, max_motifs=None, sort=False):
        param_set = list(ModelParametrization.load(fname, n_samples))
        if sort:
            param_set = sorted(param_set, key=lambda par :  - par.A0)

        if (not max_motifs is None):
            max_motifs = len(param_set)
        
        param_set = param_set[:max_motifs]
        return cls(param_set)

    def save(self, fname, comment=""):
        with open(fname, 'w') as f:
            if comment:
                f.write(f"# {comment}\n")

            f.write(str(self))

    def __str__(self):
        buf = ["# ModelSetParams with {} PSAMs".format(len(self.param_set))]
        for i, params in enumerate(self.param_set):
            buf.append("# PSAM {}".format(i))
            buf.append(str(params))
            buf.append('')
        
        return "\n".join(buf)

    def __iter__(self):
        for params in self.param_set:
            yield params   

    def __getitem__(self, i):
        return self.param_set[i]

    def __setitem__(self, i, params):
        self.param_set[i] = params

    def __add__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() + x.get_data() )
        else:
            c.set_data(self.get_data() + x)
        return c
    
    def __sub__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() - x.get_data() )
        else:
            c.set_data(self.get_data() - x)
        return c

    def __mul__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() * x.get_data() )
        else:
            c.set_data(self.get_data() * x)
        return c

    def __truediv__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() / x.get_data() )
        else:
            c.set_data(self.get_data() / x)
        return c

    def __neg__(self):
        c = self.copy()
        c.set_data( - self.get_data())
        return c

    def apply_delta(self, delta_set, min_rel_A0=1e-4):
        new = []
        for i, (params, delta) in enumerate(zip(self.copy(), delta_set)):
            p = params.psam_matrix + delta.psam_matrix
            # print i, "after applying update of magnitude", np.fabs(delta.data).max(), "min/max", p.min(), p.max()
            # m = p.min(axis=1) # find out if we dropped below zero
            # m = np.where(m < 0, -m + 1e-6, 0)
            # # print "raise", m
            # p += m[:, np.newaxis] # and raise the level in these columns accordingly
            p = np.clip(p, 1e-6, None)
            M = p.max(axis=1) # increases above 1 on cognate should increase A0
            p /= M[:,np.newaxis]
            params.psam_matrix = np.clip(p, 1e-6, 1)

            params.A0 *= M.prod() # keep matrix elements <= 1 and absorb excess into A0
            # print i, "increasing A0 by", M.prod()
            params.A0 = max(1e-6, params.A0 + delta.A0) # prevent underflow

            params.betas = np.clip(params.betas + delta.betas, 1e-9, None)
            new.append(params)

        # ensure max and min A0 don't drift more than a factor 
        # of min_rel_A0 apart. Otherwise motifs can get stuck at A0 ~ 0
        # and never come back bc grad -> 0 as A0 -> 0
        a0s = np.array([params.A0 for params in new])
        min_a0 = a0s.max() * min_rel_A0
        a0s = np.where(a0s > min_a0, a0s, min_a0)
        for a0, params in zip(a0s, new):
            params.A0 = a0

        return ModelSetParams(new)

    def save_logos(self, fname, lo=None, hi=None, title="", minimal=False, align=False, logo_height=1.4, savefig=None):
        # TODO: add error estimates to Kd 
        import matplotlib.pyplot as plt
        from RBPamp.affinitylogo import plot_afflogo, nice_conc

        if savefig is None:
            savefig = plt.savefig

        n = len(self.param_set)
        print("param_set size", n)
        fig, (lax, rax) = plt.subplots(1, 2, figsize=(3, n*.4), sharey=False, gridspec_kw=dict(wspace=0.02, left=.02, right=.98, top=.98, bottom=0.02))
        if title:
            plt.suptitle(title)

        # lax.set_aspect(3)
        # rax.set_aspect(3)
        # lax.tick_params(axis='x', which='both', bottom=False, top=False)
        # lax.tick_params(axis='y', which='both', left=False, right=False)
        # rax.tick_params(axis='x', which='both', bottom=False, top=False)
        rax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        #     ax.tick_params(
            #         axis='x',          # changes apply to the x-axis
            #         which='both',      # both major and minor ticks are affected
            #         bottom=False,      # ticks along the bottom edge are off
            #         top=False,         # ticks along the top edge are off
            #         labelbottom=False
            #     )

        psams = [params.as_PSAM() for params in self.param_set]
        offsets = [0]
        p0 = psams[0]
        xmin = 0
        xmax = len(p0.psam)
        for p1 in psams[1:]:
            if align:
                ofs, score = p0.align(p1.consensus)
            else:
                ofs = 0
            offsets.append(ofs)

            xmin = min(xmin, ofs)
            xmax = max(xmax, ofs + len(p1.psam))

        for i, (params, psam, x0) in enumerate(zip(self.param_set, psams, offsets)):
            kd = 1. / params.A0
            if (lo is None) or (hi is None):
                kdstr = "$K_d$ = {}".format(nice_conc(kd))
            else:
                kdstr = "$K_d$ = {}".format(nice_conc(kd, lo = 1. / hi[i].A0, hi = 1. / lo[i].A0))

            y0 = (n-i-1) * logo_height
            # print(i, y0, x0)
            plot_afflogo(
                lax, 
                psam.psam,
                minimal=minimal,
                y0 = y0,
                x0 = x0,
            )
            rax.text(0, y0 + .5, kdstr)
        
        lax.set_ylim(0,n*logo_height)
        rax.set_ylim(0,n*logo_height)
        # print("ylim",0, n*logo_height)
        # print("xlim",xmin, xmax)
        lax.set_xlim(xmin, xmax)
        rax.set_xlim(0, 1)
        rax.axis('off')
        try:
            plt.tight_layout()
        except ValueError:
            logging.warning("ModelSetParams.save_logos() tight_layout error silenced")

        savefig(fname)
        plt.close()


class ModelParametrization(object):
    def __init__(self, k, n_samples, nt=1, psam=[], A0=1., betas = [], data = [], acc_shift=0, acc_k=None, acc_scale=1., **kwargs):
        self.k = k
        self.nt = nt
        self.depth = 4**nt
        self.n_samples = n_samples
        self.n = self.depth * k + 1 + n_samples
        self.n_psam = self.depth * k+1
        self.Nk = 4**k
        self.psam_start = 0
        self.psam_end = self.depth * k + 1
        self.betas_start = self.psam_end
        self.betas_end = self.n
        self.dtype = np.float32
        
        # accessibility might be selected in a shifted region of size != k
        self.acc_shift = acc_shift
        self.acc_scale = acc_scale
        if acc_k is None: 
            self.acc_k = self.k
        else:
            self.acc_k = acc_k

        self.data = np.zeros(self.n, dtype=np.float32)

        # self.psam_vec = Proxy(self.data, self.psam_start, self.psam_end)
        # self.psam_matrix = Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k,4))
        self.attrs = {
            'psam_vec' : Proxy(self.data, self.psam_start, self.psam_end),
            'psam_matrix' : Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k, self.depth)),
            'A0' : Proxy(self.data, 0, 1),
            'beta' : Proxy(self.data, self.betas_start, self.betas_start + 1),
            'betas' : Proxy(self.data, self.betas_start, self.betas_end, unpack=False)
        }

        if len(psam):
            self.psam_matrix = psam
        if A0:
            self.A0 = A0
        if len(betas):
            self.betas = betas

        if len(data):
            self.set_vector(data)

        self.names =['A0']
        for i in range(self.k):
            self.names.extend(['{0}{1}'.format(nt, i+1) for nt in 'ACGU'])
        for i in range(self.n_samples):
            self.names.append('beta{0}'.format(i))

    @classmethod
    def from_vector(cls, vec, k, n_samples=1):
        return cls(k, n_samples, data=vec)

    @classmethod
    def from_PSAM(cls, psam, n_samples=1, **kwargs):
        params = cls(psam.n, n_samples, psam=psam.psam, **kwargs)
        params.A0 = psam.A0
        return params

    @classmethod
    def load(cls, fname, n_samples, beta0=1e-6, mina=1e-6):

        aff = []
        attrs = {}
        rbp_name = ''
        def make_params():
            psam = np.array(aff, dtype=np.float32)
            psam = np.where(psam > 0, psam, mina)
            params = cls(len(psam), n_samples, psam=psam, A0=attrs.get('A0', 1))
            for k, v in attrs.items():
                setattr(params, k, v)
            params.acc_k = int(attrs.get('acc_k', len(psam)))
            params.acc_shift = int(attrs.get('acc_shift', 0))
            params.acc_scale = attrs.get('acc_scale', 1)
            params.betas[:] = beta0
            if rbp_name:
                params.rbp_name = rbp_name
            return params

        with open(fname) as f:
            for line in f:
                if line.startswith('PSAM'):
                    if aff:
                        yield make_params()
                        aff = []
                        attrs = {}
                    # parse attributes
                    for kw in line.split()[1:]:
                        if not kw.strip():
                            continue

                        if not '=' in kw:
                            rbp_name = kw
                            continue

                        k,v = kw.split('=')
                        attrs[k] = float(v)

                elif line.startswith('seeded'):
                    continue

                elif line.startswith('#'):
                    continue

                elif not line.strip():
                    continue

                else:
                    parts = line.split('\t')
                    aff.append(parts[:4])

        if aff:
            yield make_params()

    def save(self, fname, append=False):
        if append:
            mode = 'a'
        else:
            mode = 'w'
        open(fname, mode).write(str(self) + '\n')

    def as_vector(self, dtype=np.float32):
        return self.data
    
    def as_PSAM(self):
        from RBPamp.pwm import PSAM
        return PSAM(self.psam_matrix, A0=self.A0)

    def copy(self):
        new = ModelParametrization(self.k, self.n_samples, data=self.data, acc_k=self.acc_k, acc_shift=self.acc_shift, acc_scale=self.acc_scale, nt=self.nt)
        if not np.allclose(new.data, self.data):
            d = np.fabs(new.data - self.data)
            i = d.argmax()
            print("OFFENDING PARAMETER:", i, new.data[i], self.data[i])
            print(self.data)
            1/0
        return new

    def set_vector(self, vec):
        self.data[:len(vec)] = vec[:]
        return self

    def unity_bounded(self):
        p = self.copy()
        
        v = p.psam_vec
        i = np.fabs(v).argmax()
        x = v[i]
        if x > 0:
            p.psam_vec = v / x
        elif x < 0:
            p.psam_vec = - v / x
        
        # print 'unity_bounded', i, x
        return p

    def unity(self):
        p = self.copy()
        n = np.linalg.norm(self.data)
        if n > 0:
            p.data /= n
        return p

    def __getattr__(self, a):
        try:
            attrs = object.__getattribute__(self, 'attrs') 
        except AttributeError:
            attrs = {}
        # print 'getattr', a
        if a in attrs:
            return attrs[a].get_values()
        else:
            return object.__getattribute__(self, a)
        # return super(ModelParametrization, self).__getattr__(a)

    def __setattr__(self, a, v):
        # attrs = super(ModelParametrization, self).__getattr__('attrs') 
        try:
            attrs = object.__getattribute__(self, 'attrs') 
        except AttributeError:
            attrs = {}
        # print "setattr", a, v
        if a in attrs:
            return attrs[a].set_values(v)
        else:
            return object.__setattr__(self, a, v)
        # return super(ModelParametrization, self).__setattr__(a, v)

    def __str__(self):
        from RBPamp.pwm import project_column
        import RBPamp.cyska as cyska
        buf = []
        rbp_name = getattr(self, 'rbp_name', '')
        buf.append(f"PSAM {rbp_name} A0={self.A0} n={self.k} acc_k={self.acc_k} acc_shift={self.acc_shift} acc_scale={self.acc_scale}")
        rel_err = ''
        if hasattr(self, 'rel_err'):
            buf[0] += f' rel_err={self.rel_err}'

        buf.append("#\t{}\tconsensus".format("\t".join(cyska.yield_kmers(self.nt))))
        
        for row in self.psam_matrix:
            buf.append("\t".join(["{0:>10.5f}".format(x) for x in row] + [project_column(row)]))

        # buf.append("BACKGROUND")
        # for i, beta in enumerate(self.betas):
        #     buf.append('# beta{0}={1:.3e}'.format(i, beta))
        
        return '\n'.join(buf)

    def __add__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data += x.data
        else:
            c.data += x
        return c
    
    def __sub__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data -= x.data
        else:
            c.data -= x
        return c

    def __mul__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data *= x.data
        else:
            c.data *= x
        return c

    def __truediv__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data /= x.data
        else:
            c.data /= x
        return c

    def __neg__(self):
        return ModelParametrization.from_vector(- self.data, self.k, self.n_samples)


def test_logo():
    from RBPamp.pwm import PSAM
    A = PSAM.from_kmer('TATTTTATT')
    A.psam = np.where(A.psam < 1., 1e-6, 1.)
    A.A0 = .5

    B = PSAM.from_kmer('TTAATTAAA')
    B.psam = np.where(B.psam < 1., 1e-6, 1.)
    B.A0 = 3.28

    C = PSAM.from_kmer('TTGAGTTTT')
    C.psam = np.where(C.psam < 1., 1e-6, 1.)
    C.A0 = 1.8

    param_set = [ModelParametrization.from_PSAM(A), ModelParametrization.from_PSAM(B), ModelParametrization.from_PSAM(C)]
    for p in param_set:
        print(p)
    params0 = ModelSetParams(param_set)
    print(params0)
    params0.save_logos('motifs.svg')

def test_save_load():
    psam = np.identity(4)
    print(psam)

    params = ModelParametrization(4, 3, psam=psam, A0=2.)
    params.save('bla.tsv')

    params = ModelParametrization.load('bla.tsv', 3)
    print(params)

if __name__ == "__main__":
    params = ModelSetParams.load('RBPamp/z4t75p01k99fix/footprint/calibrated.tsv', 1)
    print(params)

    # for rbp in ['MSI1', 'UNK', 'HNRNPA0', 'NOVA1', 'IGF2BP1']:
    #     params = ModelSetParams.load(f"/home/mjens/engaging/RBNS/{rbp}/RBPamp/z4t75p01k99fix/seed/initial.tsv", 1)
    #     params.save_logos(f'{rbp}.pdf', minimal=True, align=True)
    # test_logo()
    #test_save_load()
