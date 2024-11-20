import sys
import numpy as np
import RBPamp
import RBPamp.cyska as cyska
import logging
logger = logging.getLogger(f"RBPamp.{__name__}")

bases = np.array(list('ACGU'))
base_idx = { 
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'T' : 3,
    'U' : 3 
}

ambig_codes = "ACGUT-NMRWSYKVHDB"
ambig_index = dict([(code, n) for n,code in enumerate(ambig_codes)])
ambig_vectors = np.array([
    # A    C    G    U
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [.25, .25, .25, .25],
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0, 0.5],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.5, 0.5],
    [.33, .33, .33, 0.0],
    [.33, .33, 0.0, .33],
    [.33, 0.0, .33, .33],
    [0.0, .33, .33, .33],
])
N = np.linalg.norm(ambig_vectors, axis=1, ord=1)[:, np.newaxis]
ambig_normed = ambig_vectors / np.where(N > 0, N, 1.)

import itertools
ambig_codes16 = ["{0}{1}".format(*p) for p in itertools.product(ambig_codes, ambig_codes)]
ambig_vectors16 = []
ambig_index16 = {}
for i,a in enumerate(ambig_vectors):
    for j,b in enumerate(ambig_vectors):
        # print a,b,i,j
        v = np.zeros(16, dtype=float)
        v[0:4] = a[0] * b
        v[4:8] = a[1] * b
        v[8:12] = a[2] * b
        v[12:] = a[3] * b
        ambig_index16[ambig_codes[i] + ambig_codes[j]] = len(ambig_vectors16)
        ambig_vectors16.append(v)

ambig_vectors16 = np.array(ambig_vectors16)
N = np.linalg.norm(ambig_vectors16, axis=1, ord=1)[:, np.newaxis]
ambig_normed16 = ambig_vectors16 / np.where(N > 0, N, 1.)

def project_column(col):
    if len(col) == 4:
        ambig = ambig_normed
        codes = ambig_codes
    else:
        ambig = ambig_normed16
        codes = ambig_codes16

    n = np.linalg.norm(col, ord=1)
    if n:
        col = col / n

    # project onto ambiguity codes as vector
    scores = (col[np.newaxis,:] * ambig).sum(axis=1)
    i = scores.argmax()
    c = codes[i]
    if scores[i] < .9:
        c = c.lower()
    # print scores.shape, scores, c, scores[i]
    return c

def project_column_old(col):
    if len(col) == 4:
        ambig = ambig_vectors
        codes = ambig_codes
    else:
        ambig = ambig_vectors16
        codes = ambig_codes16

    n = col.sum()
    if n:
        col = col / n

    # project onto ambiguity codes as vector
    scores = (col[np.newaxis,:] * ambig).sum(axis=1)
    i = scores.argmax()
    c = codes[i]
    # if scores[i] < .9:
    #     c = c.lower()
    # print scores.shape, scores, c, scores[i]
    return c

def hull(kmer):
    """
    generate all single base substitution variants of a 
    seed motif by bit-operations on the corresponding kmer index.
    """
    k = len(kmer)
    subst = [kmer]
    seed = cyska.seq_to_index(kmer)
    for j in range(k):
        nt = (seed >> j*2) & 3
        mask = seed^ nt << (j*2)
        for l in range(4):
            var = mask | (l << j*2)
            if var != seed:
                subst.append(cyska.index_to_seq(var,k))

    return subst

def expand(kmer_set, left=True):
    for nt in 'acgt':
        for kmer in kmer_set:
            if left:
                yield nt + kmer
            else:
                yield kmer + nt
    
def weblogo_save(counts, fname="pwm.eps", title="", scale_width=True):
    import weblogolib as wl
    from corebio.seq import unambiguous_rna_alphabet
    #data = LogoData(alphabet=unambiguous_rna_alphabet, length=5, counts=counts, entropy=np.ones(5), weight=np.ones(5))
    data = wl.LogoData.from_counts(unambiguous_rna_alphabet, counts)
    #import sys
    #sys.stderr.write(str( data))
    options = wl.LogoOptions(color_scheme=wl.classic, fineprint="", logo_title=title, yaxis_label='A.U.', scale_width=scale_width, resolution=300)
    # options.title = "A Logo Title"
    fmt = wl.LogoFormat(data, options)
    dump = wl.eps_formatter( data, fmt)
    
    if fname:
        with file(fname,'wb') as f:
            f.write(dump)
    
    return fname

def afflogo_save(psam, fname="psam.pdf", title="", scale_width=True, **kwargs):
    import matplotlib.pyplot as plt
    from RBPamp.affinitylogo import plot_afflogo
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    plot_afflogo(ax, psam, title=title)
    plt.savefig(fname)
    plt.close()

    return fname


class PSAM(object):
    def __init__(self, psam, A0 = 1e-6):
        self.psam = np.array(psam, dtype=np.float32)
        M = self.psam.max()
        d = M - self.psam.max(axis=1)
        self.psam += d[:, np.newaxis] # now, every column should have at least one position with the max value of M
        self.psam /= M
        cond = (self.psam.max(axis=1) == 1).all()
        if not cond:
            logger.warning(f"invalid matrix elementes in PSAM {self.psam}")
        
        self.A0 = A0
        self.n = len(psam)

    def expand16(self):
        newm = np.zeros( (self.n - 1, 16) , dtype=np.float32)
        for i, a in enumerate(self.psam[:-1]):
            b = self.psam[i+1]
            print("a",i,a, "b",b)
            newm[i][0:4] = a[0] * b
            newm[i][4:8] = a[1] * b
            newm[i][8:12] = a[2] * b
            newm[i][12:] = a[3] * b
            print("newm[i]", newm[i], project_column(newm[i]))
    
        return PSAM(newm, A0=self.A0)

    @property
    def Kd(self):
        return 1./self.A0

    @property
    def matrix(self):
        return self.psam * self.A0

    @property
    def fraction_GC(self):
        return (self.psam[:,1] + self.psam[:,2]).sum() / self.psam.sum()

    @classmethod
    def from_kmer(cls, kmer, **kwargs):
        kmer = kmer.upper()
        cols = []
        for nt in kmer:
            cols.append(ambig_vectors[ambig_index[nt]])
        
        m = np.array(cols, dtype=np.float32)
        psam = cls(m, **kwargs)
        psam.kmer_seed = kmer
        psam.kmer_set = hull(kmer)
        return psam
    
    @classmethod
    def from_kmer_variants(cls, kmers, aff, **kwargs):
        """
        Input HAS TO BE SORTED by descending affinity
        or you will get assertion errors.
        """
        #print ">>> THIS BETTER BE SORTED"
        #for mer, a in zip(kmers, aff):
            #print mer, a
            
        psam = cls.from_kmer(kmers[0], A0=aff[0])
        mask_sum = np.zeros(psam.psam.shape, dtype=float)
        for kmer, a in zip(kmers[1:], aff[1:]):
            new = cls.from_kmer(kmer, A0=a)
            mask = (new.psam == 1) & (psam.psam != 1)
            rel_A = new.A0/psam.A0
            #if rel_A > 1:
                #print "fuckup adding",kmer,a
                #print psam
                #print new
                #print "rel_A", rel_A
                
            assert rel_A <= 1. # should be enforced by prior sorting
            psam.psam += mask * rel_A
            mask_sum += mask
            #print "->merged", kmers[i], rel_A

        #if not (psam.psam <= 1.).all():
            #print mask_sum
            #print psam.psam
        assert (psam.psam <= 1.).all()
        #print psam
        psam.kmer_set = kmers
        return psam

    def kmer_affinity_table(self, aff0=1e-6):
        params, indices = cyska.params_from_pwm(self.psam, A0=self.A0, aff0=aff0)
        return params

    @property
    def affinities(self):
        return self.kmer_affinity_table()

    @property
    def consensus(self):
        return "".join([project_column_old(col) for col in self.psam])
        # return "".join([project_column(col) for col in self.psam])
        
    # def __add__(self, mdl):
    #     assert self.n == mdl.n
        
    #     A0 = self.A0 + mdl.A0
    #     psam = self.A0 * self.psam + mdl.A0 * mdl.psam
    #     amax = psam.max(axis=1)
    #     psam /= amax[:, np.newaxis]
        
    #     return PSAMState(psam, max(self.A0, mdl.A0))

    @property
    def consensus_ul(self):
        return "".join([project_column(col) for col in self.psam])

    def highest_scoring_kmers(self, k=7, n_max=10):
        """ slide kmer over matrix and classify best, gapless alignment"""
        from RBPamp.seed import Alignment
        from RBPamp.cyska import yield_kmers

        A = Alignment()
        A.matrix = self.psam
        
        matches = []
        for kmer in yield_kmers(k):
            ofs, score = A.align(kmer, multiply=True)
            matches.append( (score, kmer) )
        
        return sorted(matches, reverse=True)[:n_max]


    def align(self, kmer, **kw):
        """ slide kmer over matrix and classify best, gapless alignment"""
        from RBPamp.seed import Alignment
        if np.allclose(self.psam, 0) or not np.isfinite(self.psam).all():
            logger.warning(f"invalid values in PSAM detected during call to align() {self.psam}")
            return 0, -np.inf

        A = Alignment()
        A.matrix = self.psam
        
        ofs, score = A.align(kmer, multiply=True, **kw)
        # frac = score / (self.n - abs(ofs) )

        # if ofs == 0:
        #     cat = ("match", kmer, frac)
        # elif ofs < 0:
        #     cat = ("left-shift", kmer[:-ofs], frac )
        # elif ofs > 0:
        #     cat = ("right-shift", kmer[-ofs:], frac )
        
        return ofs, score
            
    def score(self, kmer):
        p = PSAM.from_kmer(kmer)
        score = (self.psam * p.psam).sum() * self.A0
        return score

    @property
    def discrimination(self):
        return (self.psam.max(axis=1) / self.psam.sum(axis=1) - .25 ) / .75

    def pad_to_size(self, w):
        n,d = self.psam.shape
        for i in range(w-n):
            disc = self.discrimination
            if disc[0] > disc[-1]:
                # pad left
                left = True
            elif disc[0] < disc[-1]:
                # pad right
                left = False
            else:
                left = i % 2 # alternate

            # print "padding",i,'/',w-n, "left=", left
            if left:
                self.psam = np.concatenate( (np.ones((1,d)), self.psam), axis=0 )
            else:
                self.psam = np.concatenate( (self.psam, np.ones((1,d))), axis=0 )

            self.n = len(self.psam)


    def __str__(self):
        buf = ["PSAM A0={0} n={1}".format(self.A0, self.n)]
        for col,d  in zip(self.psam, self.discrimination):
            buf.append("\t".join(["{0:.6e}".format(s) for s in col] + [project_column(col), str(d)]) )

        kmer_seed = getattr(self, "kmer_seed","")
        kmer_set =  getattr(self, "kmer_set","")
        if kmer_seed:
            buf.append("seeded from '{0}'".format(kmer_seed))
        if len(kmer_set):
            buf.append("built from {0} kmers '{1}'".format(len(kmer_set), ",".join(sorted(kmer_set)) ) )
        return "\n".join(buf)

    def store_params(self, fname):
        file(fname, 'w').write(str(self) + '\n')

    @classmethod
    def load(cls, fname):
        A0 = 1.
        aff = []
        with file(fname) as f:
            for line in f:
                if line.startswith('PSAM'):
                    A0 = float(line.split()[1].split('=')[1])
                elif line.startswith('seeded'):
                    break
                else:
                    parts = line.split('\t')
                    aff.append(parts[:4])

        psam = np.array(aff, dtype=np.float32)
        return cls(psam, A0=A0)


    def save_logo(self, fname='pwm.svg', title=""):
        counts = self.psam
        if not title:
            from RBPamp.affinitylogo import nice_conc
            title = "$K_d$ = {}".format(nice_conc(self.Kd))

        afflogo_save(self.psam, fname=fname, title=title, scale_width=False)

    def shrink(self, thresh = .75, n=0):
        disc = self.discrimination
        D = disc.sum()

        best = {self.n : (1,0,self.n)}
        for i in range(self.n):
            for j in range(i, self.n+1):
                
                d = disc[i:j].sum()/D
                if d >= thresh:
                    best[j-i] = (d, i,j)

        bylength = sorted(best)        
        for l in bylength:
            d,i,j = best[l]
            print(l, d, i,j, self.consensus[i:j])

        if n:
            d, i, j = best[n]
        else:
            d, i, j = best[bylength[0]]    

        psam = self.psam[i:j,:]
        A0 = self.A0 
        if i > 0:
            A0 *= self.psam[:i,:].mean()
        if j < self.n:
            A0 *= self.psam[j:,:].mean()

        return PSAM(psam, A0 = A0)

   
if __name__ == "__main__":
    # def is_shifted(kmer, s_max=2):
    #     k = len(kmer)
    #     kmer_pwm_map = {'TGCATG':1}
    #     #print "mapped", sorted(self.pwm_by_kmer.keys())
    #     for x in range(1,s_max+1):
    #         for pad in list(cyska.yield_kmers(x)):
    #             rshifted = (pad + kmer[:k-x]).lower()
    #             lshifted = (kmer[x:]+pad).lower()
    #             print "l", x, kmer, lshifted
    #             print "r", x, kmer, rshifted
    #             if lshifted in kmer_pwm_map:
    #                 seed = kmer_pwm_map[lshifted].kmer_seed
    #                 return -x, seed, kmer[:x] + seed
    #             elif rshifted in kmer_pwm_map:
    #                 seed = kmer_pwm_map[rshifted].kmer_seed
    #                 return x, seed, seed + kmer[-x:]

    #     return 0, kmer, kmer

    # "gggcat is 1-shift of TGCATG"
    # "ttgggc is 2-shift of GTGCAT"
    # print is_shifted('gggcat')
    
    import RBPamp.params
    psam = RBPamp.params.ModelSetParams.load('/scratch2/RBNS/RBFOX3/RBPamp/std/opt_nostruct/parameters.tsv', 1)[0].as_PSAM()
    for score, kmer in psam.highest_scoring_kmers():
        print(kmer, score)
    sys.exit(0)
    #psam = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU', 'AGCAUGU', 'CGCAUGU', 'GGCAUGU'], [1., 1., 1., 1., 1.])
    psam = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU',], [1., .1, ])
    psam = PSAM.from_kmer_variants(['AGCAUGU', 'CGCAUGU', 'GGCAUGU', 'UGCAUGU'], [.99, .8, .8, .8,])
    
    
    #.expand16()
    print(psam.consensus)
    sys.exit(1)
    
    import sys
    psam = PSAM.load(sys.argv[1])

    print(psam)
    # psam.kmer_affinity_table()
    params = psam.kmer_affinity_table(aff0=1e-6)
    # print len(kmers)
    # I = aff.argsort()[::-1]
    # for mer, a in zip(kmers[I], aff[I]):
    #     print mer, a

    import RBPamp.cyska as cyska
    params_new = cyska.params_from_pwm(psam.psam, A0=psam.A0, aff0=1e-6)

    import matplotlib.pyplot as pp
    pp.figure()
    pp.loglog(params, params_new)
    print(np.fabs(params - params_new).sum())
    pp.show()