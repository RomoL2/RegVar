# coding=future_fstrings
from __future__ import print_function
from __future__ import unicode_literals
import os
import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import time
from collections import defaultdict
import matplotlib
# matplotlib.use('agg')
# matplotlib.rc('xtick.major', size = .5)
# matplotlib.rc('ytick.major', size = .5)

sns_style = { 
    'axes.linewidth': .5, 
    'axes.grid' : False, 
    # 'ticks.xtick.major.linewidth' : .5,
    # 'xtick.major.linewidth' : .5, # Whut is the right one? Seaborn docs, where are u?
    'legend.frameon' : False,
    'legend.fancybox' : False,
    'font.size' : 8,
}

import matplotlib
font = {
    'family' : 'Arial',
    'weight' : 'normal',
    'size'   : 8
}
matplotlib.rcParams['axes.unicode_minus'] = False	

matplotlib.rc('axes.spines',
    top=False,
    right=False,
    left=True,
    bottom=True
)
matplotlib.rc('boxplot.flierprops', marker='.')
matplotlib.rc('font', **font)
matplotlib.rc('lines', markersize=3)
matplotlib.rc('figure', dpi=300)
matplotlib.rc('figure', figsize=(3,3))
matplotlib.rc('legend', 
    handlelength=.5,
    labelspacing=.3,
    handletextpad=.4,
    frameon=True,
    fancybox=False,
    framealpha=1,
    borderpad=.3,
    borderaxespad=.5,
    columnspacing=.5,
    edgecolor='k'
)
bpkw = dict(
    medianprops=dict(color='red'),
    boxprops=dict(linewidth=.5,),
    whiskerprops=dict(linewidth=.5,),
    capprops=dict(linewidth=.5,),
    flierprops=dict(marker='.', markerfacecolor='k', markersize=3),
    notch=False,  # notch shape
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
)

import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import RBPamp

def sane_colorbar(cb, nbins=4):
    from matplotlib import ticker

    # tick_locator = ticker.MaxNLocator(nbins=nbins)
    # cb.locator = ticker.LinearLocator(numticks=nbins)
    cb.locator = ticker.MaxNLocator(nbins=nbins, min_n_ticks=3, prune='both')
    # cb.ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=nbins))
    cb.update_ticks()
    return cb

def pval_str(p):
    if p > 0:
        return "P < {0:.3e}".format(p)
    else:
        return "P ≈ 0"

def pval_stars(p, levels=[0.000001, 0.00001, .0001, .001, .01, .05,]):
    if (p < 0) or (p > 1):
        raise ValueError(f'P-value {p} outside [0, 1] range')

    import bisect
    x = bisect.bisect_right(levels, p)
    n = len(levels)

    if x == n:
        return 'n.s.'
    else:
        return '*'*(n-x)

def roundmax(x, m):
    i = int(x)
    remain = x - i
    if remain:
        return np.round(x, m)
    else:
        return i

def repel_labels(x, y, labels, **kwargs):
    from adjustText import adjust_text
    texts = []
    for x_, y_, label in zip(x, y, labels):
        texts.append(pp.text(x_, y_, label, horizontalalignment='center', color='k'))
        
    adjust_text(
        texts, 
        #add_objects=lines, 
        #autoalign='y', 
        #expand_objects=(0.1, 1),
        #only_move={'points':'', 'text':'y', 'objects':'y'}, force_text=0.75, force_objects=0.1,
        arrowprops=dict(arrowstyle="-", color='k', lw=0.5)
    )


def sparse_y(ax, nth=2):
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % nth != 0:
            label.set_visible(False)

def repel_labels_nx(x, y, labels, k=0.15, ax=None):
    import networkx as nx
    if ax == None:
        ax = plt.gca()
    G = nx.DiGraph()
    # print(x.shape, y.shape, len(labels), ax)
    
    data_nodes = []
    init_pos = {}
    for xi, yi, label in zip(x, y, labels):
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k, iterations=200)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in list(pos.items()):
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    shrinkA=0, shrinkB=0,
                                    #connectionstyle="arc3", 
                                    color='k'), )
    # expand limits
    all_pos = np.vstack(list(pos.values()))
    x_span, y_span = np.ptp(all_pos, axis=0)
    mins = np.min(all_pos-x_span*0.15, 0)
    maxs = np.max(all_pos+y_span*0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])


def density_scatter_plot(
    x,y, 
    outlier_percentile=10, 
    density_kw = dict(cmap=pp.cm.YlGn, nbins=100), 
    plot_kw = dict(style='.', color='#4495c3'), 
    contour=False, 
    plot_outliers=True,
    label=None, data_labels=[],
    dens_thresh=1000,
    x_ref=True,
    tick_exp=0,
    margin=.05,
    lim_max=None,
    lim_min=None,
    sym=True,
    ax=None,
    ):
    from scipy.stats import kde
    import seaborn as sns
    if ax is None:
        ax = plt.gca()
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    t0 = time.time()
    N = len(x)
    
    w = x.max() - x.min()
    h = y.max() - y.min()
    _xmin = x.min() - w * margin
    _xmax = x.max() + w * margin
    _ymin = y.min() - h * margin
    _ymax = y.max() + h * margin

    xmin = _xmin if lim_min is None else lim_min
    xmax = _xmax if lim_max is None else lim_max
    ymin = _ymin if lim_min is None else lim_min
    ymax = _ymax if lim_max is None else lim_max
    
    if N > dens_thresh and x_ref:
        # use experiment as reference
        m = xmin  
        M = xmax
    else:
        # show full range
        m = min(xmin, ymin)
        M = max(xmax, ymax)
    
    # add margin in log-space
    m += np.log10(3./4.)
    M += np.log(4./3.)

    if sym:
        xmin, xmax = m,M
        ymin, ymax = m,M

    nbins = density_kw['nbins']
    t1 = time.time()
    #Z = zi.reshape((len(yi), len(xi)))
    #print Z.shape
    #pp.imshow(Z, interpolation='none', cmap=density_kw['cmap'], origin='lower', extent=[xmin,xmax,ymin,ymax])
    import matplotlib
    with sns.axes_style("ticks", sns_style):
        if N > dens_thresh:
            from scipy.linalg import LinAlgError
            try:
                k = kde.gaussian_kde([x,y])
            except LinAlgError:
                logging.error(LinAlgError)
                # KDE failed. Fall back to scatter plot with a trick!
                plot_outliers=True
                outlier_percentile = 1e-9
                dens_thresh = N + 1
            else:
                xi, yi = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                zi[zi < 1e-3] = np.nan

                z_min = np.nanmin(zi)
                z_max = np.nanmax(zi)
                # print "zmin/max", z_min, z_max
                # pca().set_facecolor('w')
                pm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=density_kw['cmap'], edgecolors='None', linewidth=0, rasterized=True, vmin=0, vmax=z_max)
                pm.set_rasterized(True)
                cb = sane_colorbar(plt.colorbar(pm, ax=ax, shrink=.3, aspect=10)) #orientation='horizontal', fraction=.05)
                cb.set_label('density')
                zt = np.array([z_min, (z_max + z_min)/2., z_max])
                ztr = np.round(zt, 2)
                cb.outline.set_linewidth(.5)
                # cb.ax.yaxis.set_ticks_position('right')
                cb.set_ticks(zt)
                cb.ax.set_yticklabels([str(z) for z in ztr])
                cb.ax.tick_params(axis='y', direction='out', length=3, width=.5, )
                # cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
                # cb.locator = matplotlib.ticker.MaxNLocator(nbins=4)
                # cb.update_ticks()

                if contour:
                    ax.contour(xi, yi, zi.reshape(xi.shape))
            
            ax.grid(False)

        t2 = time.time()

        if plot_outliers and outlier_percentile > 0:
            data = np.vstack([x,y])
            if N <= dens_thresh:
                # print("plotting all data points")
                out = np.arange(N)
            else:
                dens_at_points = k(data)
                lower = np.percentile(dens_at_points, outlier_percentile)
                out = dens_at_points < lower

            out_x = x[out]
            out_y = y[out]
            ax.plot(out_x, out_y, plot_kw['style'], color=plot_kw['color'], markersize=3, label=label, rasterized=True)


        t3 = time.time()
        
        if len(data_labels):
            if N < dens_thresh*.1:
                # print("just add the damn labels")
                # print(x,y, data_labels)
                repel_labels_nx(x, y, data_labels)
            else:
                # annotate the most enriched and most off-diagonal k-mers
                top = x.argsort()[::-1][:5]
                # print("top", top)
                ax.plot(x[top], y[top], 'o', markersize=6, markerfacecolor='none', markeredgecolor='red', label="most enriched", alpha=.75 )

                repel_labels_nx(x[top], y[top], data_labels[top])
                # for _x, _y, mer in zip(x[top], y[top], data_labels[top]):
                #     mer = mer.upper().replace('T','U')
                #     #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
                #     pp.annotate(mer, xy=(_x, _y), xytext=(_x-.05*xmax, _y), arrowprops=dict(facecolor='red', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)
                    
                off = np.fabs(x-y).argsort()[::-1][:10]
                ax.plot(x[off], y[off], 'o', markersize=6, markerfacecolor='none', markeredgecolor='blue', alpha=.75, label="highest error" )

                repel_labels_nx(x[off], y[off], data_labels[off])
                # for _x, _y, mer in zip(x[off], y[off], data_labels[off]):
                #     #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
                #     mer = mer.upper().replace('T','U')
                #     pp.annotate(mer, xy=(_x, _y), xytext=(_x-.05*xmax, _y), arrowprops=dict(facecolor='blue', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)

        # pp.plot([0,0],mM, '-k', linewidth=.5, alpha=.3)
        # pp.plot(mM,[0,0], '-k', linewidth=.5, alpha=.3)
        
        if sym:
            # draw guides through zero and the diagonal
            mM = np.array([m,M])
            mM = (mM - mM.mean()) *.95 + mM.mean()
            ax.plot(mM, mM, '-k', linewidth=.5, alpha=.3)

            ax.set_xlim(m,M)
            ax.set_ylim(m,M)

        if tick_exp:
            xlocs = ax.get_xticks()
            xlocs = xlocs[1:-1]
            ax.set_xticks(xlocs)
            ax.set_xticklabels([roundmax(tick_exp**l, 1) for l in xlocs])

            # ylocs, labels = pp.yticks()
            # ylocs = ylocs[1:-1]
            ylocs = xlocs
            ax.set_yticks(ylocs)
            ax.set_yticklabels([roundmax(tick_exp**l, 1) for l in ylocs])

        t4 = time.time()
        logger = logging.getLogger('timing.density_plot')
        t_kde = 1000. * (t1-t0)
        t_mesh = 1000. * (t2-t1)
        t_out = 1000. * (t3-t2)
        t_label = 1000. * (t4-t3)
        logger.debug('KDE={t_kde:.2f}ms pcolormesh={t_mesh:.2f}ms outliers={t_out:.2f}ms labels={t_label:.2f}ms'.format(**locals()) )
        
        # sns.despine(trim=True)




class TrackedValues(object):
    def __init__(self):
        self.d0 = None
        self.last = None
        self.updates = []
        self.times = []
        self.N = 0
        self.logger = logging.getLogger("reports.TrackedValues")
        
    def store(self, t, d):
        
        if not self.times:
            self.d0 = d
            self.last = d
            self.times.append(t)
            return
        
        delta = self.last - d
        self.last = d

        if (delta == 0).all():
            return
        
        ind = delta.nonzero()[0]
        #self.logger.error(delta)
        #self.logger.error(ind)
        self.updates.append( (ind, d[ind]) )
        self.times.append(t)
        
    def read(self):
        if self.d0 is not None:
            data = [self.d0,]
        else:
            data = []
        last = self.d0
        for ind, vals in self.updates:
            d = np.array(last)
            d[ind] = vals
            data.append(d)
            last = d
        
        data = np.array(data)
        #print len(self.times), len(self.updates), data.shape

        assert len(self.times) == len(data)
        return self.times, data
        


class Container(object):
    pass


class RunReport(object):
    def __init__(self, path):
        self.path = path

    def load_descent(self, fname):
        data = []
        for line in open(os.path.join(self.path, fname)):
            if line.startswith("#"):
                continue
            row = line.split('\t')
            data.append([float(col) for col in row])

        data = np.array(data).T
        n_samples = (data.shape[0] - 5) / 2

        A0 = data[0]
        errors = data[3:3+n_samples]
        correlations = data[3+n_samples:3+2*n_samples]
        nfev, step = data[-2:]

        # print data.shape, errors.shape
        descent = Container()
        # descent.history = history
        descent.ls_nfev = nfev
        descent.ls_step = step
        descent.history = []
        for err, corr in zip(errors.T, correlations.T):
            state = Container()
            # print err, corr
            state.sample_errors = err
            state.correlations = (corr, 0)
            descent.history.append(state)
        

        fparams = os.path.join(self.path, os.path.dirname(fname), 'parameters.tsv')
        from RBPamp.params import ModelParametrization
        descent.params = ModelParametrization.load(fparams, n_samples)

        return descent 



class ReportBase(object):
    def __init__(self, path='.', rbns=None, fmts=['svg', 'pdf'], dpi=300, **kw):
        self.path = path
        self.fmts = fmts
        self.dpi = dpi
        from RBPamp.cmdline import ensure_path
        for fmt in self.fmts:
            ensure_path(os.path.join(self.path, fmt+'/'))

    def savefig(self, name, fig=None):
        if not fig:
            fig = plt
        
        for fmt in self.fmts:
            fname = os.path.join(self.path, fmt, f'{name}.{fmt}')
            if hasattr(self, 'logger'):
                self.logger.debug(f'saving plot "{fname}"')

            fig.savefig(fname, dpi=self.dpi)


class Vignette(object):
    def __init__(self, path='.', grad_report=None, fp_report=None, rbns=None):
        self.path = path
        self.rbns = rbns
        self.grad_report = grad_report
        self.fp_report = fp_report
        from jinja2 import Template
        ftemplate = os.path.join(os.path.dirname(__file__), 'vignette.html')
        self.dst = os.path.join(path, f'{rbns.rbp_name}_vignette.html')
        self.template = Template(open(ftemplate, 'r').read())

    def render(self, **kw):
        context = dict(
            rbns=self.rbns,
            t_nostruct=self.grad_report.epochs[0][1],
            t_struct=self.grad_report.t[-1],
            rbp_conc=self.grad_report.rbp_conc,
            fp_motifs=self.fp_report.motifs,
        )
        context.update(kw)
        print("context", context)

        open(self.dst, 'w').write(self.template.render(**context))
    
    def make_pdf(self, **kw):
        self.render()
        import pdfkit
        # kw['options'] = dict(orientation='Landscape')ls
        pdfkit.from_file(self.dst, self.dst.replace('html', 'pdf'), **kw) 


class SeedReport(ReportBase):
    def __init__(self, path='.', **kw):
        super(SeedReport, self).__init__(path=path, **kw)

        import shelve
        self.logger = logging.getLogger("plot.SeedReport")
        spath = os.path.join(self.path, '../seed/history')
        # print spath
        try:
            self.shelf = shelve.open(spath, 'r')
        except:
            self.logger.error("could not open database {}".format(spath))

    def kmer_PSAM_assignment(self, kmer):
        # from RBPamp.seed import Alignment
        scores = np.array([psam.align(kmer) for psam in self.shelf['psams']])
        print(kmer)
        print(scores)
        return scores.argmax()


    def plot_R_dist(self, n_top=5):
        plt.figure(figsize=(2,1.8))
        R = self.shelf['R0']
        Rw = (R.min() / R.max()) **.15
        Nk = len(R)
        # I = self.shelf['I']
        i_cut = self.shelf['i_cut']
        i_ns = self.shelf['i_ns']
        # print i_cut, i_ns
        kmer_set = self.shelf['kmer_set']
        if 'assignments' in self.shelf:
            assign = self.shelf['assignments']
            # print(assign)
        else:
            assign = None

        kmers = np.array([kmer for kmer, r in kmer_set])
        I = R.argsort()
        k = len(kmers[0])
        # pp.fill_between(np.arange(len(R)), R[I], color='k')
        plt.fill_between(np.arange(Nk-i_cut), R[I][:Nk-i_cut], color='.75')
        plt.fill_between(np.arange(Nk-i_cut, Nk), R[I][Nk-i_cut:], color='#c83737')

        # top = np.linspace(0, len(kmers)-1, num=n_top, dtype=int)
        # # print top, len(kmers), n_top
        # kmer_ann = kmers[top]
        from RBPamp.cyska import seq_to_index
        kmer_ann = self.shelf['founders']

        print(f"kmer_ann={kmer_ann}")
        for i, kmer in enumerate(kmer_ann):
            # print i, kmer
            Kmer = kmer.upper()
            # pp.text(Nk/2, R.max() * (Rw ** i), kmer.upper(), fontdict=dict(family='fixed'))
            # j = top[i]
            if assign:
                m = assign[kmer]
            else:
                m = self.kmer_PSAM_assignment(Kmer)

            r = R[seq_to_index(kmer)]
            plt.annotate(
                "{} {}".format(kmer.upper(), m + 1),
                (Nk-i*10, r),
                # (Nk-j, R[I][Nk-j-1]),
                xytext=(Nk/4.-i*4**(k-2), R.max() * (Rw ** i)),
                arrowprops=dict(arrowstyle='-'),
                # fontfamily='monospace',
            )
        plt.text(Nk/2-5000, R.max() * (Rw ** n_top), "...")

        plt.legend(loc='best', frameon=False)
        plt.ylabel(f"{k}-mer enrichment")
        plt.xlabel("rank")
        plt.tight_layout()
        plt.gca().set_yscale('log')
        sns.despine()
        self.savefig("seed_report")
        plt.close()


class GradientDescentReport(ReportBase):
    def __init__(self, path='.', comp=None, rbns=None, **kw):
        self.logger = logging.getLogger('plot.GradientDescentReport')
        
        super(GradientDescentReport, self).__init__(path=path, **kw)

        self.comp = comp
        self.rbns = rbns
        self.shelves = []
        self.shelf_map = [0]
        self.t = None
        self.t_ofs = 0
        self.epoch_names = []
        self.epochs = []
        self.error_estimators = []

    def load(self, fname, epoch_name):
        import shelve
        try:
            self.shelves.append(shelve.open(fname, flag='r'))
            self.logger.info("reading from shelve '{}'.".format(fname))
        except:
            self.logger.error("could not open '{}'. No data to plot!".format(fname))
            return

        descent_file = os.path.join(os.path.dirname(fname), "descent.tsv")
        lines = list(open(descent_file))
        try:
            max_t = int(lines[-1].split('\t')[0])
        except ValueError:
            self.logger.error("Empty or malformed file '{}'. No data to plot!".format(fname))
            # empty file or only header
            return

        # print "max_t found in", descent_file, max_t

        t = np.arange(max_t) + self.t_ofs
        if not len(t):
            self.logger.error("'{}' contained no data!".format(fname))
            return

        if self.t is None:
            self.rbp_conc = self.shelves[0]['rbp_conc']
            if (self.rbp_conc == np.round(self.rbp_conc)).all():
                self.rbp_conc = np.array(self.rbp_conc, dtype=int)

            self.R_exp = self.shelves[0]['R_exp']
            self.logR0 = np.log2(self.R_exp)
            self.n_samples = self.R_exp.shape[0]
            self.k_mer = int(np.log2(self.R_exp.shape[1])/2.)
            self.t = t
        else:
            self.t = np.concatenate((self.t, t))

        self.epochs.append( (self.t_ofs, self.t_ofs + len(t) - 1) )
        from RBPamp.errors import PSAMErrorEstimator
        est = PSAMErrorEstimator(os.path.dirname(fname)+'/', use_shelve=self.shelves[-1], max_t=max_t)
        self.error_estimators.append(est)
        self.t_ofs += len(t)
        self.shelf_map.append(self.t_ofs)
        self.epoch_names.append(epoch_name)
        # print "shelfmap", self.shelf_map

    def plot_scatter_all(self):
        if self.t is None:
            return

        for i, name in enumerate(self.epoch_names):
            t0, t = self.epochs[i]
            # print("epoch", t0, t, name)
            self.plot_scatter(t0, title="before {}".format(name))
            self.plot_scatter(t, title="after {}".format(name))

    def plot_logos(self):
        if self.t is None:
            return

        for i, name in enumerate(self.epoch_names):
            t0, t = self.epochs[i]
            self.plot_motifs(t0, title="before {}".format(name))
            self.plot_motifs(t, title="after {}".format(name))

    def report(self):
        if self.t is None:
            return
        
        self.plot_literature()
        self.plot_report()

        self.plot_scatter_all()
        self.plot_logos()
    
    def find_max_t(self, shelf):
        t = -1
        n_samples = None
        while "stats_t{}".format(t+1) in shelf:
            stats = shelf["stats_t{}".format(t+1)]
            n = len(stats.rbp_free)
            if n_samples is None:
                n_samples = n

            if n == n_samples:
                t += 1
            else:
                break
        return t

    def map_t_shelf(self, t):
        if t == -1:
            t = self.t[-1]

        from bisect import bisect
        shelf_i = bisect(self.shelf_map, t) - 1
        t_shelf = t - self.shelf_map[shelf_i]

        return shelf_i, t_shelf

    def get(self, name, t):
        shelf_i, t_shelf = self.map_t_shelf(t)
        # print t, "->", shelf_i, t_shelf
        val = self.shelves[shelf_i]["{0}_t{1}".format(name, t_shelf)]
        return val

    def read_sample_errors(self):
        errors = [self.get('stats', t).errors for t in self.t]
        print([e.shape for e in errors])
        return np.array(errors, dtype=float)

    def read_correlations(self):
        pearsonR = np.array([self.get('stats', t).pearsonR for t in self.t])
        pearsonP = np.array([self.get('stats', t).pearsonP for t in self.t])
        return pearsonR, pearsonP

    def read_linesearch(self):
        return np.array([self.get('linesearch', t) for t in self.t]).T

    def read_grad(self):
        res = [self.get('grad', t) for t in self.t[:-1]]
        return res

    def plot_report(self):
        pp.figure(figsize=(2, 4))

        artists = []
        labels = []
        pp.subplot(312)
        errors = (self.read_sample_errors()**2).mean(axis=2)
        m_err = errors.mean(axis=1)
        # pp.semilogy(m_err, 'k-', label='sample mean')
        data_colors = plt.get_cmap("YlOrBr")(np.linspace(.3, 1, len(errors.T)))

        for i, err in enumerate(errors.T):
            lerr = np.log10(err)
            a = pp.plot(lerr, color=data_colors[i])
            artists.append(a[0])
            labels.append('{0} nM'.format(self.rbp_conc[i]))

        if len(self.epoch_names) > 1:
            for i, name in enumerate(self.epoch_names):
                pp.axvline(self.shelf_map[i+1], color='k', linewidth=.5 )#, linestyle='dashed')

        # pp.legend(loc='upper right', frameon=False)
        pp.ylabel("model error (log10)")
        # pp.xlabel('gradient descent step')
        # pp.gca().get_xaxis().set_visible(False)
        sns.despine()

        pp.subplot(313)
        corr, pval = self.read_correlations()
        for i, c in enumerate(corr.T):
            pp.plot(c, color=data_colors[i], label='{0} nM'.format(self.rbp_conc[i]))

        if len(self.epoch_names) > 1:
            for i, name in enumerate(self.epoch_names):
                print(name)
                pp.axvline(self.shelf_map[i+1], color='k', linewidth=.5, ) #label='accessibility\nfootprint' )

        # pp.legend(loc='lower right', frameon=False)
        pp.ylim(0.75, 1)
        pp.ylabel("k-mer correlation")
        plt.xlabel('gradient descent step')
        sns.despine()

        pp.subplot(311)
        pp.legend(artists, labels, ncol=3, loc='lower center')
        pp.axis('off')
        plt.tight_layout()
        self.savefig("descent_report")
        pp.close()



        nfev, step = self.read_linesearch()

        pp.figure(figsize=(2, 4))
        pp.subplot(211)
        pp.semilogy(nfev, label='no. function evaluations during line-search')
        pp.semilogy(step, label='step size')

        if len(self.epoch_names) > 1:
            for i, name in enumerate(self.epoch_names):
                pp.axvline(self.shelf_map[i+1], color='k', linewidth=.5 , linestyle='dashed')
                # TODO: add annotation 'no structure' 'full model'

        pp.xlabel('iteration #')
        pp.legend(loc='upper right', frameon=False)
        sns.despine()

        params0 = self.get('params', 0).copy()
        n_motifs = len(params0.param_set)
        # print "number of motifs", n_motifs
        def mag(grad_data):
            if grad_data is None:
                return [np.nan,] * n_motifs

            grad = params0.copy().set_data(grad_data)
            # print "number of gradients", len(grad.param_set)

            for par in grad:
                par.betas[:] = 0

            return [np.sqrt((g.data**2).sum()) for g in grad] 

        mags = np.array([mag(grad) for grad in self.read_grad()]).T
        # print "mags.shape", mags.shape
        # with sns.axes_style("ticks", sns_style)
        plt.subplot(212)
        # plt.semilogy(mags.mean(axis=0), '-k', label="motif mean")
        for i, mag in enumerate(mags):
            plt.semilogy(mag, label="motif {}".format(i))

        plt.legend(loc='upper right', ncol=2, frameon=False)
        plt.ylabel("magnitude of gradient (log-scale)")
        plt.xlabel("iteration #")
        plt.tight_layout()
        sns.despine()
        self.savefig("descent_linesearch")
        pp.close()

    def plot_scatter(self, t=-1, title=""):
        if t == -1:
            t = self.t[-1]

        stats = self.get("stats", t)
        logRt = np.log2(self.get("R", t))

        maxR = max(self.logR0.max(), logRt.max())
        for i in range(self.n_samples):
            pp.figure(figsize=(4, 4))
            # pp.title("{0}mer R-value scatter plot {1}".format(self.k_mer, title))

            x = self.logR0[i]
            y = logRt[i]
            label = "{0} nM R={1:.3f} ({2})".format(self.rbp_conc[i], stats.pearsonR[i], pval_str(stats.pearsonP[i]))
            # data_labels = self.opt.mdl.parameters.param_name
            density_scatter_plot(x, y, label=label, tick_exp=2, lim_max=maxR)
            pp.legend(loc='upper left', frameon=False)
            pp.xlabel("observed {}-mer enrichment".format(self.k_mer))
            pp.ylabel("predicted {}-mer enrichment".format(self.k_mer))
            pp.gca().set(aspect="equal")
            pp.tight_layout()
            name = "scatter_{0}mers_{1}nM_t{2}".format(self.k_mer, self.rbp_conc[i], t)
            try:
                self.savefig(name)
            except ValueError as err:
                self.logger.warning("caught '{}' while trying to save {}".format(err, fname))

            pp.close()

    def plot_param_error_scatter(self, param_i=0):
        plt.figure(figsize=(3,3))

        for err_est in self.error_estimators[:1]:
            params, stats = err_est.load_data()
            mdl_errors = np.array([s.error for s in stats])
            par = np.array([p.get_data()[param_i] for p in params])

            sc = plt.scatter(
                mdl_errors,
                par,
                alpha=.75,
            )
            plt.gca().set_xscale("log")

        # pp.colorbar(sc, shrink=.05, label="time step")

        pp.xlabel("model error")
        pp.ylabel("affinity [1/nM]")
        # pp.gca().set(aspect="equal")
        sns.despine()
        pp.tight_layout()

        self.savefig(f"param_{param_i}_error_scatter")
        plt.close()

    def plot_motifs(self, t=42, title=""):
        from RBPamp.params import ModelSetParams

        shelf_i, shelf_t = self.map_t_shelf(t)
        est = self.error_estimators[shelf_i]
        p_mid = est.estimate(save=False, t_ref=shelf_t)
        if p_mid is None:
            params = ModelSetParams(self.get('params', t), sort=True) # re-initialize in case it's not sorted
            lo = None 
            hi = None
        else:
            params = ModelSetParams(p_mid.param_set, sort=True)
            lo = ModelSetParams(p_mid.lo.param_set, sort=True)
            hi = ModelSetParams(p_mid.hi.param_set, sort=True)
        
        def sf(fname, **kw):
            self.savefig(fname, **kw)

        params.save_logos(f"motifs_t{t}", lo=lo, hi=hi, title=title, savefig=sf)
        params.save_logos(f"minimal_motifs_t{t}", lo=lo, hi=hi, title=title, minimal=True, align=True, savefig=sf)

    def make_affinity_dist_plots(self):
        from RBPamp.params import ModelSetParams

        for i, name in enumerate(self.epoch_names):
            t = self.shelf_map[i+1] - 1
            shelf_i, shelf_t = self.map_t_shelf(t)
            print(f"shelf_i={shelf_i} shelf_t={shelf_t}")
            est = self.error_estimators[shelf_i]
            p_mid = est.estimate(save=False, t_ref=shelf_t)
            if p_mid is None:
                params = ModelSetParams(self.get('params', t), sort=True) # re-initialize in case it's not sorted
            else:
                params = ModelSetParams(p_mid.param_set, sort=True)

            self.plot_affinity_dists(params, name=name)

    def plot_affinity_dists(self, params, k_fit=6, name=""):
        xticks = [1, 10, 100, 1000, 10000]
        xtick_labels = ["1", "", "$10^2$", "", "$10^4$"]

        def set_x_rel_Kd(ax, xmin=1, xmax=10000):
            ax.set_xlabel(f'{self.rbns.rbp_name} rel. $K_d$')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlim(xmin, xmax)

        from RBPamp.partfunc import PartFuncModel
        gradient = np.linspace(.3, 1., len(self.rbns.reads) -1)
        data_colors = plt.get_cmap("YlOrBr")(gradient) # highest conc == darkest color
        print("epoch name", name)
        name = name.replace(' ','_')
        inr = self.rbns.reads[0]
        # evaluate partition function on all samples
        states = []
        for reads in self.rbns.reads:
            mdl = PartFuncModel(
                reads,
                params,
                self.rbns.R_value_matrix(k_fit)[0],
                rbp_conc = self.rbns.rbp_conc
            )
            state = mdl.predict(params, keep_Z1_motif=False)
            states.append(state)
                
        max_A = max([s.Z1_read.max() for s in states]) * params.A0
        min_A = min([s.Z1_read.min() for s in states]) * params.A0
        bins = 10 ** np.linspace(np.log10(min_A), np.log10(1.1 * max_A), 100)
        x = (bins[1:] + bins[:-1])/2
        x_ = 1./(x/params.A0)  # rel. Kd

        ref = states[0]
        w = np.ones(len(ref.Z1_read)) * (inr.rna_conc / inr.N)
        # print(w)
        ## plot affinity distribution of binding sites in RNA pool
        
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5,1.5), sharex=True, gridspec_kw=dict(wspace=0.6, hspace=0.3, bottom=.3, top=.95))
        ax1.set_xscale('log')
        # ax2.set_xscale('log')
        # ax3.set_xscale('log')

        A_binned = np.histogram(
            ref.Z1_read * params.A0, 
            bins=bins, 
            weights=w, 
        )[0]
        ax1.step(x_, A_binned) # bin by affinity but plot by rel. Kd
        # ax.legend(loc='upper right')
        ax1.set_ylabel('estimated RNA\nin pool [nM]')


        A_binned = np.array(A_binned)

        print("RBP free", ref.rbp_free)
        for free, color, reads in zip(ref.rbp_free, data_colors, self.rbns.reads[1:]):
            pb = 1 / (1 + 1 / (x * free))
            ax2.plot(x_, pb, color=color, label=reads.name, solid_capstyle='round')
        
        ax2.set_ylabel('predicted\noccupancy')
        # ax.legend(loc='lower right')
        # plt.locator_params(axis='x', numticks=3)
        complex_binned = []
        ## plot predicted affinity distribution of bound sequences
        # ax = plt.subplot(133)
        complex_conc, non_specific = ref.concentrations

        bins_rel_Kd = 10**np.linspace(-3, 6, 100)
        # print(f"bins_Kd: {bins_rel_Kd}")
        x_Kd = (bins_rel_Kd[1:] + bins_rel_Kd[:-1])/2

        for reads, psi, cc, color in zip(self.rbns.reads[1:], ref.psi, complex_conc, data_colors):
            w = psi * (cc / inr.N)
            b_binned = np.histogram(
                ref.Z1_read * params.A0,
                bins=bins,
                weights=w
            )[0]
            complex_binned.append(np.array(b_binned))

            b_density = np.histogram(
                1./ref.Z1_read,
                bins=bins_rel_Kd,
                weights=psi / psi.sum(),
                density=True,
            )[0]
            # Zt = np.trapz(b_density, x_Kd)
            # print(f"density integral: {Zt}")
            ax3.step(x_Kd, b_density,  # b_density,
                color=color,
                label=reads.name,
            )
            # print(f"b_density, {b_density}")
        # ax.legend(loc='upper left')
        # ax.set_ylabel('pred. bound [nM]')
        ax3.set_ylabel(f'prob. density\n{self.rbns.rbp_name} binding')
        # plt.xticks(xticks, xtick_labels)
        # plt.locator_params(axis='x', numticks=3)
        plt.tight_layout()
        set_x_rel_Kd(ax1)
        # set_x_rel_Kd(ax2)
        # set_x_rel_Kd(ax3)
        self.savefig(f'affdist_{self.rbns.rbp_name}_{name}')
        plt.close()

        n = len(complex_binned)
        fig = plt.figure(figsize=(2, n*1.7))
        ## plot decomposition of observed affinity distribution into directly bound and non-specific
        zs_binned = [np.histogram(state.Z1_read * params.A0, bins=bins)[0] for state in states[1:]]
        
        from scipy.optimize import minimize, minimize_scalar
        HANDLES = []
        LABELS = []
        spbase = n*100+11
        print(f"complex_binned before plotting affmatch: {complex_binned}")
        for i,(b_binned, zs, reads, color) in enumerate(zip(complex_binned, zs_binned, self.rbns.reads[1:], data_colors)):
            ax = plt.subplot(spbase+i)
            ax.set_xscale('log')
            # print(b_binned.shape, zs.shape, A_binned.shape)
            zs_rel = zs / zs.sum()
            def err(bg):
                # scale, bg = args
                pred = b_binned + A_binned * bg
                # E = ((scale * zs - pred)**2).sum()
                pred_rel = pred / pred.sum()
                E_rel = ((zs_rel - pred_rel)**2).sum()
                # print(f"scale={scale:.3e} bg={bg:.3e} => E={E_rel}")
                print(f"bg={bg:.3e} => E={E_rel}")
                return E_rel

            res = minimize_scalar(err, bounds=(1e-8, 100.), method='bounded')
            print(res)
            bg = res.x
            total = (bg * A_binned + b_binned).sum()
            bg_perc = 100. * (bg * A_binned).sum() / total
            print(f"inferred total conc. of pull-down RNA: {total:.3f} nM. background={bg_perc} %")

            ax.plot(x_, zs_rel * total, '-', color='k', label="observed")
            # plt.plot(x, bg * A_binned + b_binned, '^', color=color, label='bound + background')
            # ax.fill_between(x_, bg * A_binned, bg * A_binned + b_binned, color=color, label='specific')
            # ax.fill_between(x_, bg * A_binned, color='gainsboro', label='non-specific')
            ax.fill_between(x_, bg * A_binned + b_binned, color=color, label=f'predicted {int(reads.rbp_conc)} nM')
            ax.plot(x_, bg * A_binned, '--', color='darkgray', label='est. non-specific')

            # ax.legend(loc='upper center', ncol=3)
            ax.set_ylabel('est. concentration [nM]')
            set_x_rel_Kd(ax)
            # plt.xticks(xticks, xtick_labels)
            ymax = zs_rel.max() * total
            plt.ylim(0, ymax*1.2)
    
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in LABELS:
                    LABELS.append(l)
                    HANDLES.append(h)
        
        fig.legend(HANDLES, LABELS, ncol=2)
        plt.tight_layout()
        self.savefig(f'affmatch_{self.rbns.rbp_name}_{name}')
        plt.close()
    
        # x = np.sort(ref.Z1_read) * params.A0
        # y = (1 - np.arange(1,len(x)+1)/float(len(x))) * inr.rna_conc

        # 
    
        #     # plt.plot(x, y, label=par.as_PSAM().consensus_ul)
        #     # plt.plot(x, y, label=reads.name)
        #     a = plt.hist(
        #         state.Z1_read, 
        #         bins=bins, 
        #         histtype='step', 
        #         weights=np.ones(len(state.Z1_read)) * (inr.rna_conc / inr.N), 
        #         alpha=.75,
        #         label=reads.name
        #     )
        #     # print(a)
        #     # print(a.color)
        #     if reads.rbp_conc == 0:
        #         # this is the input sample
        #         N_reads = len(state.Z1_read)
        #         print(state.params.betas)
        #         cplx, nsc = state.concentrations
        #         for cc, ns, rbpc, psi, color in zip(cplx, nsc, self.rbns.rbp_conc, state.psi, data_colors)[:]:
        #             plt.hist(state.Z1_read,
        #                 bins=bins,
        #                 histtype='step',
        #                 weights=psi * cc / psi.sum(),
        #                 label=f"pred. complex {rbpc} nM RBP",
        #                 alpha=.75,
        #                 color=color
        #             )
        #             # plt.hist(state.Z1_read,
        #             #     bins=bins,
        #             #     histtype='step',
        #             #     weights=np.ones(N_reads) * ns / float(N_reads),
        #             #     label=f"n.s @{rbpc} nM RBP",
        #             #     alpha=.75,
        #             #     color=color
        #             # )

                    

       
        # plt.legend(loc='upper right')
        # plt.ylabel('conc in RNA pool [nM]')
        # plt.xlabel(f'{self.rbns.rbp_name} affinity > x [1/nM]')
        # plt.tight_layout()
        # plt.savefig(f"{self.rbns.rbp_name}_affdist.pdf")
        # plt.close()

        # plt.figure()
        # plt.gca().set_xscale('log')
        # # plt.gca().set_yscale('log')

        # for zb, reads in zip(zs_binned[1:], self.rbns.reads[1:]):
        #     plt.plot((bins[1:] + bins[:-1])/2, (zb+1)/(zs_binned[0]+1), label=reads.name)

        # plt.legend(loc='upper left')
        # plt.ylabel('enrichment')
        # plt.xlabel(f'{self.rbns.rbp_name} affinity > x [1/nM]')
        # plt.tight_layout()
        # plt.savefig(f"{self.rbns.rbp_name}_aff_R.pdf")
        # plt.close()
    
        
        # for reads in self.rbns.reads:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     Z1m = reads.PSAM_partition_function(params, split=True)
        #     for z in Z1m:
        #         # print z.shape
        #         plt.hist(z.sum(axis=1), bins=bins, histtype='step', cumulative=True)
        #     plt.gca().set_xscale('log')
        #     plt.savefig("{reads.name}_affdist.pdf".format(reads=reads))
        #     plt.close()

        




    def _get_lit_data(self, t, debug=False):
        if t == -1:
            t = self.t[-1]

        x = self.comp.observed_Kd
        if not len(x):
            return None

        # get the PSAM error estimator for this epoch
        shelf_i, shelf_t = self.map_t_shelf(t)
        est = self.error_estimators[shelf_i]
        p_mid = est.estimate(save=False, t_ref=shelf_t)
        # print "t=",t
        # print p_mid
        if not p_mid is None:
            y = 1/self.comp.predict_affinities_from_paramset(p_mid)
        else:
            y = 1/self.comp.predict_affinities_from_paramset(self.get('params', t))

        lfc = np.log10(y/x)
        I = lfc.argsort()

        from scipy.stats import pearsonr, spearmanr
        rho, p_spearman = spearmanr(np.log(x), np.log(y))
        try:
            R, p_pearson = pearsonr(np.log(x), np.log(y))
        except ValueError:
            R = np.NaN
            p_pearson = np.NaN

        psstr = pval_str(p_spearman)
        ppstr = pval_str(p_pearson)
        label="$R={R:.2f}$ ($P < {p_pearson:.2e}$)\n$\\rho={rho:.2f}$ ($P < {p_spearman:.2e}$)".format(**locals())

        # self.results.info("R={R:.3f} {ppstr} rho={rho:.3f} {psstr}".format(**locals()))
        if debug:
            print(">>> R={R} {ppstr}".format(**locals()))
            print(">>> rho={rho} {psstr}".format(**locals()))
            print("seq\tknown\tpredict\tlog-ratio")
            for _x, _y, seq in zip(x[I], y[I], self.comp.seqs[I]):
                print(seq, '\t', _x, '\t', _y, '\t', np.log2(_y/_x))

        class lcomp(object):
            pass

        res = lcomp()
        res.x = x
        res.x_err = self.comp.observed_Kd_err
        res.y = y
        res.rho = rho
        res.p_spearman = p_spearman
        res.R = R
        res.p_pearson = p_pearson
        res.label = label
        res.lfc = lfc
        res.min = min(x.min(), y.min())
        res.max = max(x.max(), y.max())

        if not p_mid is None:
            y_hi = 1/self.comp.predict_affinities_from_paramset(p_mid.lo)
            y_lo = 1/self.comp.predict_affinities_from_paramset(p_mid.hi)

            res.y_err = np.array((y - y_lo, y_hi - y))
            # print "y_err", res.y_err
        else:
            res.y_err = None

        return res

    def plot_literature(self, debug=True):
        if self.comp is None:
            return

        t0, t = self.epochs[0]
        times = [t0, t]
        titles = ["initial", "optimized", "+AFP"]
        if self.t[-1] > t:
            times.append(self.t[-1])
        
        # times = [145,] # DEBUG HACK!!!
        data = [self._get_lit_data(t, debug=debug) for t in times]
        _data = [res for res in data if res is not None]
        if not _data:
            return

        m = np.min(np.array([res.min for res in _data]))
        M = np.max(np.array([res.max for res in _data]))
        

        import seaborn as sns
            # matplotlib.rc('xtick.major', width = .1)
            # matplotlib.rc('ytick.major', width = .1)

        pp.figure(figsize=(4, 2.5))
        ax = plt.subplot(121)
        ax.set_aspect(1)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        
        # pp.title("comparison to {0} literature affinities".format(self.comp.n))

        errs = []
        err_cols = []
        err_titles = []
        artists = []
        labels = []
        for res, title, color in zip(data, titles, ['gray', '#3b8bc2', '#c83737']):
            if res == None:
                continue

            # HACK to only plot final result
            # if color == "#c83737":
            if color:
                a = pp.errorbar(
                    res.x,
                    res.y,
                    xerr=res.x_err,
                    yerr=res.y_err,
                    fmt='.',
                    ecolor='k',
                    mfc=color,
                    mec=color,
                    elinewidth=.5,
                    capsize=3,
                    markersize=4,
                    capthick=.5
                )
                artists.append(a)
                labels.append(title + "\n" + res.label)

            errs.append(res.lfc)
            err_cols.append(color)
            err_titles.append(title)
        
        pp.loglog([m,M],[m,M], 'k-', linewidth=.5)

        pp.ylabel(r"predicted {} $K_d$ [nM]".format(self.comp.rbp_name))
        pp.xlabel(r"measured {} $K_d$ [nM]".format(self.comp.rbp_data))
        sns.despine(trim=False)
        import matplotlib.ticker
        locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
        ax.xaxis.set_major_locator(locmaj)
        ax.yaxis.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        plt.subplot(122)
        plt.axis('off')
        pp.legend(tuple(artists), tuple(labels), loc='lower right')
        pp.tight_layout()

        self.savefig("literature_comparison")
        pp.close()

        # plot the error distributions in reverse order

        errs = errs[::-1]
        err_titles = err_titles[::-1]
        err_cols = err_cols[::-1]

        import scipy.stats
        pp.figure(figsize=(2,2))
        for ei, ej in zip(errs[:-1], errs[1:]):
            stat, pval = scipy.stats.mannwhitneyu(np.fabs(ei), np.fabs(ej))
            print("error lower than previous? ", stat, pval)

        e0 = np.fabs(errs[0])
        for e, lbl in zip(errs[1:], err_titles[1:]):
            stat, pval = scipy.stats.mannwhitneyu(e0, np.fabs(e))
            stat_t, pval_t = scipy.stats.ttest_ind(e0, np.fabs(e), equal_var=False)
            print(f"comparing {lbl} to initial: MWU: {pval:.2e} Welch's t-test: {pval_t:.2e}")

        _bpkw = dict(bpkw)
        _bpkw['vert'] = False
        bplot = pp.boxplot([np.fabs(e) for e in errs], **_bpkw)
        for patch, color in zip(bplot['boxes'], err_cols):
            patch.set_facecolor(color)

        pp.yticks(1 + np.arange(len(errs)), err_titles)#, rotation=90)
        pp.xlabel(r"$|\log_{10} \frac{K_d\; pred}{K_d \; measured}$|")

        # pp.xlim(0, 1.1 * np.array(errs).max())
        # pp.axhline(0, linestyle='dashed', color='k', linewidth=.5)
        sns.despine(trim=False)
        pp.tight_layout()
        self.savefig("literature_errors")
        pp.close()

    # def plot_A0_fit(self, t=-1):
    #     state = self.descent.history[t]
    #     if t == -1:
    #         t = self.descent.t

    #     if not hasattr(state, "_A0_data"):
    #         return

    #     a0 = state._A0_data.a0
    #     rerr = state._A0_data.rerr
    #     asem = getattr(state._A0_data, "asem", None)
    #     rcorr = state._A0_data.rcorr

    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plot = plt.loglog
    #     plt.subplot(311)

    #     plot(a0, rerr, '.b', label='MSE')
    #     plot(a0, rerr, '-b')
    #     plt.legend(loc='upper left')

    #     plt.subplot(312)
    #     if not asem is None:
    #         plot(a0, asem, '.r', label='SEM')
    #         plot(a0, asem, '-r')
    #         plt.legend(loc='upper left')

    #     plt.subplot(313)
    #     plt.plot(a0, rcorr, '.k', label='best correlation')
    #     plt.plot(a0, rcorr, '-k')
    #     plt.legend(loc='upper left')

    #     a_opt = a0[rerr.argmin()]

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.path,"A0_fit_{0}mers_t{1}.pdf".format(self.descent.model.k, t)))
    #     plt.close()

    # def plot_psam(self, psam, title):
    #     pp.pcolor(psam.T, cmap='bwr', vmin=-1, vmax=+1)
    #     pp.xlabel(title)
    #     pp.ylabel("base")
    #     pp.yticks(np.arange(0.5,4.5,1), ['A','C','G','U'])
    #     pp.colorbar(label='weight', shrink=.5, orientation='horizontal')
        
    # def plot_gradients(self, psam_correct, local_grad, grad):
    #     print descent
    #     pp.figure()
    #     pp.subplot(131)
    #     self.plot_psam(unity_matrix(psam_correct - self.psam),'actual delta')
    #     pp.subplot(132)
    #     self.plot_psam(unity_matrix(local_grad),'local gradient')
    #     pp.subplot(133)
    #     self.plot_psam(unity_matrix(grad),'RMSprop')

    # def plot_param_hist(self):
    #     from matplotlib.colors import LogNorm
    #     A0 = np.array([state.params.A0 for state in self.descent.history])
    #     a = np.array([state.params.psam_vec[1:] for state in self.descent.history])
    #     betas = np.array([state.params.betas for state in self.descent.history])

    #     pp.figure(figsize=(6,10))
    #     pp.subplot(311)
    #     pp.plot(A0, label=r'$A_0$')
    #     pp.legend(loc='upper right')
    #     pp.ylabel("affinity [1/nM]")

    #     pp.subplot(312)
    #     pp.imshow(a.T, interpolation='nearest', cmap='inferno', norm=LogNorm(vmin=1e-6, vmax=1), aspect='auto')
        
    #     t = []
    #     for i in range(self.descent.params.k):
    #         for n in 'ACGU':
    #             t.append('{0}{1}'.format(n,i+1))
    #     l = len(t)
    #     pp.yticks(np.arange(l), t)

    #     t = [1e-5, 1e-3, 1e-1]
    #     pp.colorbar(orientation='horizontal', shrink=.5, ticks=t, label="PSAM values")
    #     pp.ylabel("matrix elements")

    #     pp.subplot(313)
    #     for i,b in enumerate(betas.T):
    #         pp.semilogy(b, label='beta{0}'.format(i))

    #     pp.legend(loc='upper right')
    #     pp.ylabel("background estimate")
    #     pp.xlabel("optimization step")
    #     pp.tight_layout()

    #     pp.savefig(os.path.join(self.path,"descent_params_{0}mer.pdf".format(self.descent.params.k)))
    #     pp.close()


class FootprintCalibrationReport(ReportBase):
    def __init__(self, fparams, out_path='.', rbns=None, **kw):
        """
        fparams is path to calibrated.tsv params file
        expects database 'history' in same folder to retrieve
        intermediate results
        """ 
        from RBPamp.params import ModelSetParams
        import shelve
        super(FootprintCalibrationReport, self).__init__(path=out_path, **kw)

        self.out_path = out_path
        self.rbns = rbns
        self.logger = logging.getLogger('plot.FootprintCalibrationReport')
        self.shelve = {}
        try:
            self.params = ModelSetParams.load(fparams, 1) # len(self.rbp_conc)
            for par in self.params:
                cons = par.as_PSAM().consensus_ul
                par.motif = cons
                # print(f"{cons} acc_k={par.acc_k} s={par.acc_shift}")
                dbfile = os.path.join(os.path.dirname(fparams), 'history_{}'.format((cons)))
                self.shelve[cons] = shelve.open(dbfile, flag='r')
            
            self.motifs = sorted(self.shelve.keys())
            self.any_shelve = self.shelve[self.motifs[0]]
            self.rbp_conc = self.any_shelve['rbp_conc']

        except (OSError, IOError): # IOError:
            self.logger.error("could not open db. No data to plot!")
            self.rbp_conc = []
            self.params = []
            self.motifs = []
        
        if (self.rbp_conc == np.round(self.rbp_conc)).all():
            self.rbp_conc = np.array(self.rbp_conc, dtype=int)

    def baseline_error(self, motif):
        punp_input = self.shelve[motif]["{motif}_punp_profiles".format(**locals())]
        punp_naive = self.shelve[motif]["{motif}_naive_profiles".format(**locals())]        
        err0 = np.sum((punp_naive - punp_input[1:])**2)
        
        return err0

    def get_profile_data(self, motif, k, s):
        key = "{motif}_{k}_{s}".format(**locals())
        if not key in self.shelve[motif]:
            return None

        opt = self.shelve[motif][key]
        punp_input = self.shelve[motif]["{motif}_punp_profiles".format(**locals())]
        punp_naive = self.shelve[motif]["{motif}_naive_profiles".format(**locals())]

        S = self.shelve[motif]["{motif}_opt_profile_{k}_{s}".format(**locals())]
        if len(S) == 2:
            punp_predict, res = S
            onekey = f'{motif}_opt_profile_a_one_{k}_{s}'
            onekey2 = f'opt_profile_a_one_{k}_{s}'
            if onekey in self.shelve[motif]:
                punp_a_one, res_a_one = self.shelve[motif][onekey]
            elif onekey2 in self.shelve[motif]:
                punp_a_one, res_a_one = self.shelve[motif][onekey2]
            else:
                from pprint import pprint
                # print(f'looking for a_one profile for {k} {s}')
                # pprint(sorted(self.shelve[motif].keys()))
                punp_a_one = None
                res_a_one = None
        else:
            punp_predict, punp_a_one, res, res_a_one = S


        return res, res_a_one, opt, punp_input, punp_naive, punp_predict, punp_a_one

    def plot_profile(self, motif, acc_k, acc_shift, lw=1):
        data = self.get_profile_data(motif, acc_k, acc_shift)
        if data is None:
            return

        res, res_a_one, opt, punp_input, punp_naive, punp_expect, punp_a_one = data
        print(f" input: {len(punp_input)} naive: {len(punp_naive)} expect: {len(punp_expect)} a_one: {len(punp_a_one)}")

        err0 = self.baseline_error(motif)
        pad = int((punp_input.shape[1] - len(motif)) / 2)
        x = np.arange(-pad, len(motif) + pad )

        gradient = np.linspace(.3, 1., len(punp_naive))
        data_colors = plt.get_cmap("YlOrBr")(np.linspace(.3, 1, len(punp_expect)))
        print(f"data_colors {data_colors}")

        naive_colors = plt.get_cmap("Greens")(gradient)
        print(f"naive_colors {naive_colors}")
        vienna_colors = plt.get_cmap("Blues")(gradient)
        fit_colors = plt.get_cmap("Reds")(gradient)

        print(f"{len(punp_naive)}")
        print(f"data_colors {data_colors}")

        last = len(gradient) - 1
        def make_rect(ax=None, ofs=0, top=False):
            import matplotlib.patches as patches
            if ax is None:
                ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            height = ymax - ymin
            h = height * .02
            if top:
                y = ymax,
            else:
                y = ymin

            rect = patches.Rectangle(
                (acc_shift-.5 + ofs, y + h), 
                acc_k, h,
                linewidth=1,
                edgecolor='r',
                facecolor='r',
                # label='footprint'
            )
            ax.add_patch(rect)

        def finalize_plot(fp=True):
            if fp:
                make_rect()
            cons = motif
            xlabels = ["" for p in range(-pad,0)] + list(cons) + [str(p) for p in range(1, pad+1)]
            xlabels[0] = f"-{pad}"
            xlabels[-1] = f"+{pad}"
            plt.xticks(x, xlabels)
            plt.axvline( - .5, color='k', linewidth=lw, linestyle='dashed', zorder=-1000)
            plt.axvline(len(motif) - .5, color='k', linewidth=lw, linestyle='dashed', zorder=-1000)

            plt.legend(
                bbox_to_anchor=(0., 1.02, 1., .202), 
                loc=3, ncol=5, mode="expand", borderaxespad=0.,
                frameon=False
            )

            # plt.ylabel(r"$P_{unpaired}$ (motif-weighted)")
            plt.ylabel(r"$P_{unpaired}$")
            plt.xlabel('position [nt]')
            sns.despine()


        def plot_exp(with_label=False, with_input=True, colors=data_colors, sym='x', n_max_conc=3):
            if with_input:
                # if sym:
                #     plt.plot(x, punp_input[0], sym, color='k', label='input' if with_label else None)
                plt.plot(x, punp_input[0], '-', color='dimgray', linewidth=2*lw, solid_capstyle='round')

            print(f"{len(punp_input)} {len(self.rbp_conc)} {len(colors)}")
            for i, (obs, conc, color) in enumerate(zip(punp_input[1:], self.rbp_conc[:n_max_conc], colors)):
                print(f"conc = {conc}")
                if sym:
                    plt.plot(x, obs, sym, color=color, label="{} nM".format(conc) if with_label else None)

                plt.plot(x, obs, '-', color=color, linewidth=lw)
            print(f"done with plot_exp n_max_conc={n_max_conc}")
        # mats = np.array([
        #     punp_input[1:], 
        #     punp_naive,
        #     punp_expect,
        #     punp_a_one,
        # ])
        data = [
            punp_input, 
            punp_naive,
            punp_expect,
            punp_a_one,
        ]
        # print(data)
        # n_samples = np.array([d.shape for d in data])
        # print(n_samples)

        mats = np.concatenate([d for d in data if not d is None])

        labels = \
            ["EXP input",] + ["EXP {}nM".format(conc) for conc in self.rbp_conc] + \
            ["seq only {}nM".format(conc) for conc in self.rbp_conc] + \
            ["opt {}nM".format(conc) for conc in self.rbp_conc] + \
            ["RNAplfold {}nM".format(conc) for conc in self.rbp_conc]
        
        rcolors = \
            ['dimgray'] + \
            list(data_colors) + \
            ['gray'] * len(punp_naive) + \
            ['scarlet'] * len(punp_expect) + \
            ['g'] * len(punp_a_one)

        # rcolors = \
        #     ['k'] + list(data_colors) + \
        #     list(naive_colors) + \
        #     list(fit_colors) + \
        #     list(vienna_colors)


        ## heatmap currently broken bc can't understand the row_colors ????
        # print mats.shape
        # # m = mats[:, 0, :]
        # m = mats
        # print "-> heatmap shape", m.shape

        # plt.figure(figsize=(3,3))
        # hm = sns.clustermap(
        #     m, 
        #     figsize=(4,4), 
        #     col_cluster=False, 
        #     cmap='Spectral_r', 
        #     xticklabels=[str(p) for p in range(-pad,0)] + list(motif) + [str(p) for p in range(1, pad+1)],
        #     yticklabels=labels,
        #     row_colors=rcolors,
        #     method='centroid',
        #     cbar_kws=dict(label=r"$P_{unpaired}$", aspect=10)
        #     # metric='cosine'
        # )
        # make_rect(ax=hm.ax_heatmap, ofs=pad + .5, top=False)
        # # X = np.arange(len(punp_input[1])+1)
        # # Z = np.arange(len(mats)+1)
        # # plt.pcolor([X, Y], mats.T[0])
        # fname = os.path.join(self.out_path, '{motif}_{acc_k}_{acc_shift}_heatmap.pdf'.format(**locals()))
        # self.logger.debug("saving plot: '{}'".format(fname))
        # hm.savefig(fname)
        # plt.close()

        plt.figure(figsize=(6, 4))
        # fig1, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

        plt.subplot(221)
        plot_exp(with_label=True, with_input=True)
        finalize_plot(fp=False)

        plt.subplot(222)
        plot_exp(with_input=True)
        for i, (naive, conc, color) in enumerate(zip(punp_naive, self.rbp_conc, naive_colors)):
            lbl = "no footprint (a=0): err=100%"
            plt.plot(x, naive, '-', color=color, label=lbl if i == last else None)
        finalize_plot(fp=False)

        # from itertools import zip_longest
        plt.subplot(223)
        plot_exp()
        # print("punp_a_one", punp_a_one)
        if not punp_a_one is None:
            for i, (one, color) in enumerate(zip(punp_a_one, vienna_colors)):
                lbl = "RNAfold (a=1): err={rerr:.1f}%".format(rerr = 100. * res_a_one.fun/err0)
                plt.plot(x, one, '-', color=color, label=lbl if i == last else None)
        finalize_plot()

        plt.subplot(224)
        plot_exp()

        for i, (pred, color) in enumerate(zip(punp_expect, fit_colors)):
            lbl = 'optimized (a={res.x[0]:.2f}):  err={rerr:.1f}%'.format(res=res, rerr = 100. * res.fun/err0)
            plt.plot(x, pred, '-', color=color, label=lbl if i == last else None)
        finalize_plot()

        plt.tight_layout()
        # sns.despine(trim=True)
        # sparse_y(plt.gca())
        self.savefig(f'{motif}_profiles')
        plt.close()


        plt.figure(figsize=(2, 2))
        ax = plt.gca()
        # for i, (obs, naive, one, pred) in enumerate(zip(punp_input[1:], punp_naive, punp_a_one, punp_expect)):
        #     plt.plot(obs, naive, 'v', color=naive_colors[i], label="no structure" if i==last else None, alpha=.75)
        #     if not one is None:
        #         plt.plot(obs, one, '^', color=vienna_colors[i], label="RNAfold (a=1)" if i==last else None, alpha=.75)
        #     plt.plot(obs, pred, 'o', color=fit_colors[i], label="optimized" if i==last else None, alpha=.75)



        # select the lowest concentration sample, because that should show the most
        # pronounced effect
        obs = punp_input[1]
        vienna = punp_a_one[0]
        opt = punp_expect[0]

        plt.plot(obs, punp_naive[0], 'v', color="#00bc80", label="no footprint", alpha=1, mew=0, markersize=3)
        plt.plot(obs, vienna, '^', color="#0070b0", label="RNAfold", alpha=1, mew=0, markersize=3)
        plt.plot(obs, opt, 'o', color="#ff0000", label="optimized", alpha=1, mew=0, markersize=3)

        ymin, ymax = plt.gca().get_ylim()
        plt.legend(loc='upper left', frameon=False)
        plt.plot([ymin, ymax], [ymin, ymax], color='k', linestyle='dashed', linewidth=.5, zorder=-30000)
        plt.xlabel(r"observed $P_{unpaired}$")
        plt.ylabel(r"expected $P_{unpaired}$")
        ax.set_xlim(ymin*.9, ymax*1.1)
        ax.set_ylim(ymin*.9, ymax*1.1)
        ax.set_xticks([0.4, 0.6, 0.8])
        ax.set_yticks([0.4, 0.6, 0.8])
        sns.despine()
        plt.tight_layout()
        self.savefig(f'{motif}_scatter')
        plt.close()


    def report(self):
        for params in self.params:
            # self.kmer_acc_profiles(params.motif, params)
            self.matrix_plots(params.motif, highlight=(params.acc_k, params.acc_shift))
            # self.matrix_plot_3d(params.motif, highlight=(params.acc_k, params.acc_shift))
            self.plot_profile(params.motif, params.acc_k, params.acc_shift)
        
    def get_matrix_data(self, motif, k_range=(1, 21), s_range=(-10, 20)):
        kmin, kmax = k_range
        smin, smax = s_range

        scales = np.zeros((smax - smin + 1, kmax - kmin + 1), dtype=float)
        errors = np.zeros((smax - smin + 1, kmax - kmin + 1), dtype=float)

        k_found = set()
        s_found = set()
        err0 = self.baseline_error(motif)
        for k in range(kmin, kmax + 1):
            for s in range(smin, smax + 1):
                key = "{motif}_{k}_{s}".format(**locals())
                if key in self.shelve[motif]:
                    err, k, s, a, A0 = self.shelve[motif][key]
                    # print(k,s, '->', err/err0, a)
                    k_found.add(k)
                    s_found.add(s)
                else:
                    err = np.nan
                    a = np.nan
                
                scales[s - smin, k - kmin] = a
                errors[s - smin, k - kmin] = err
        
        ks = min(k_found) - kmin
        ke = kmax - max(k_found)
        ss = min(s_found) - smin
        se = smax - max(s_found)

        return scales[ss:-se, ks:-ke], (errors/err0)[ss:-se, ks:-ke], (min(k_found), max(k_found)), (min(s_found), max(s_found))

    def matrix_plot_3d(self, motif, highlight=None):
        self.logger.debug("matrix plot")
        import seaborn as sns
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        mat_a, mat_err, k_range, s_range = self.get_matrix_data(motif)
        kmin, kmax = k_range
        smin, smax = s_range

        # print("s_range", s_range)
        # print("k_range", k_range)
        n_shift = smax - smin + 1
        n_k = kmax - kmin + 1

        X = np.arange(n_k) + kmin
        Y = np.arange(n_shift) + smin
        X, Y = np.meshgrid(X, Y)
        Z = np.nan_to_num(1./mat_err)

        np.save('x.npy', X)
        np.save('y.npy', Y)
        np.save('z.npy', Z)
        # fig = plt.figure(figsize=(2.5,2.5))
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.gca(projection='3d')
        # ax.view_init(-75, 0)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
            linewidth=0, antialiased=False, edgecolor='gray',
            vmin=np.nanmin(Z), vmax=np.nanmax(Z), shade=True, alpha=0.5)

        # fig.colorbar(surf, label='fold error reduction', shrink=0.5, aspect=20)

        plt.xlabel("footprint size [nt]")
        plt.ylabel("footprint shift [nt]")

        self.savefig(f'{motif}_3d_error_mat')
        plt.close()


    def matrix_plots(self, motif, highlight=None):
        self.logger.debug("matrix plot")
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        mat_a, mat_err, k_range, s_range = self.get_matrix_data(motif)
        kmin, kmax = k_range
        smin, smax = s_range

        # print("s_range", s_range)
        # print("k_range", k_range)
        n_shift = smax - smin + 1
        n_k = kmax - kmin + 1

        # for i, row in enumerate(1./mat_err.T):
        #     print(i, row)

        def make_rect(k,s):
            import matplotlib.patches as patches
            rect = patches.Rectangle(
                (s-.5-smin, k-.5-kmin), 
                1, 1,
                linewidth=1.5,
                edgecolor='r',
                facecolor='none',
                label='optimum'
            )
            return rect

        fig = plt.figure(figsize=(4,4))
        if highlight:
            k, s = highlight
        # fig.suptitle("accessibility footprint analysis")
        def doticks(stride=5):
            sr = np.arange(smin, smax + 1, stride)
            kr = np.arange(kmin, kmax + 1, stride)
            plt.xticks(sr - smin, [str(s) for s in sr])
            plt.yticks(kr - kmin, [str(k) for k in kr])

        plt.subplot(211)
        plt.imshow(1./mat_err.T, interpolation='none', cmap="viridis", origin='lower')
        if highlight:
            plt.gca().add_patch(make_rect(*highlight))

        sane_colorbar(plt.colorbar(label=r'fold error reduction', fraction=.05, shrink=.75, aspect=20))
        plt.ylabel("footprint size [nt]")
        plt.xlabel("footprint shift [nt]")
        doticks()
        # plt.ylim(kmin, kmax + 1)

        plt.subplot(212)
        plt.imshow(mat_a.T, interpolation='none', cmap="inferno", origin='lower')
        if highlight:
            plt.gca().add_patch(make_rect(*highlight))

        sane_colorbar(plt.colorbar(label=r'accessibility scaling', fraction=.05, shrink=.5, aspect=20))
        plt.ylabel("footprint size [nt]")
        plt.xlabel("footprint shift [nt]")
        doticks()
        # plt.ylim(kmin, kmax + 1)

        plt.tight_layout()
        self.savefig(f'{motif}_footprint')
        plt.close()


    def kmer_acc_profiles(self, motif, params, maxU=15, n_bins=30):
        import seaborn as sns
        import matplotlib.pyplot as plt
        labels = self.rbns.sample_labels
        print(f"data labels {labels}")
        data_colors = plt.get_cmap("YlOrBr")(np.linspace(.3, 1, len(labels)-1))
        print(f"data_colors {data_colors}")

        key = '{}_high_affinity_kmers'.format(motif)
        if not key in self.shelve[motif]:
            return

        for score, kmer in self.shelve[motif][key]:
            print("plotting kmer accessibility profiles for", kmer, score)
            all_data = self.shelve[motif]['{}_acc_data'.format(kmer)]
            plt.figure(figsize=(2, 2))
            bins = np.linspace(0, maxU, num=30)
            counts = plt.hist(all_data, bins=bins, histtype='step', label=labels)[0]
            total = np.array([len(d) for d in all_data])
            R_avg = total[1:] / total[0]
            sns.despine()
            plt.tight_layout()
            self.savefig(f"{kmer}_energy_hist")
            plt.close()

            # print counts
            R = counts[1:] / counts[0]
            # print "R-value as function of bin", R
            acc = np.exp(- params.acc_scale * bins/self.rbns.reads[0].RT)
            acc_raw = np.exp(-bins/self.rbns.reads[0].RT)
            am = (acc[1:] + acc[:-1])/2.
            amr = (acc_raw[1:] + acc_raw[:-1])/2.
            
            plt.figure(figsize=(2,2))
            for r, ra, lbl, color in zip(R, R_avg, labels[1:], data_colors):
                plt.semilogx(am, r, color=color, label=lbl)
                plt.axhline(ra, color=color, linewidth=1, linestyle='dashed')

            plt.legend(loc='best')
            plt.xlabel("{} accessibility".format(kmer))
            plt.ylabel("enrichment over input".format())
            sns.despine()
            plt.tight_layout()
            self.savefig(f"{kmer}_R_vs_acc_scaled")
            plt.close()

            plt.figure(figsize=(2,2))
            for r, ra, lbl, color in zip(R, R_avg, labels[1:], data_colors):
                plt.semilogx(amr, r, color=color, label=lbl)
                plt.axhline(ra, color=color, linewidth=1, linestyle='dashed')

            plt.legend(loc='best')
            plt.xlabel("{} accessibility".format(kmer))
            plt.ylabel("enrichment over input".format())
            sns.despine()
            plt.tight_layout()
            self.savefig(f"{kmer}_R_vs_acc_raw")
            plt.close()



if __name__ == "__main__":
    pass
    # print(pval_stars(.8))
    # print(pval_stars(.04999))
    # print(pval_stars(.00999))
    # print(pval_stars(.000999))
    # print(pval_stars(.0000999))
    # print(pval_stars(.00000999))
    # print(pval_stars(0))
    # print(pval_stars(-1))
    # rep = RunReport('/scratch/data/RBNS/MBNL1/RBPamp/1M')
    # descent = rep.load_descent('opt_nostruct/descent.tsv')
    # print descent.params
    # # print descent.history
    # grep = GradientDescentReport('RBPamp/recent/opt_nostruct/history')
    # grep.plot_report()
    # grep.plot_scatter(t=0)
    # grep.plot_scatter(t=-1)

    # N = 4**6
    # x = np.array(np.random.random(N))
    # y = np.array(x + np.random.random(N) * .1)
    # mers = np.array([str(i) for i in y])
    # logging.basicConfig(level=logging.DEBUG)
    # logging.getLogger('matplotlib').setLevel(logging.INFO)
    # density_scatter_plot(x,y, data_labels=mers)
    # pp.show()
