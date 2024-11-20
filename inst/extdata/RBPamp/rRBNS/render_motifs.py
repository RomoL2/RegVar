import os
import numpy as np
import matplotlib.pyplot as plt
import RBPamp.report
import RBPamp.util
import RBPamp.pwm
import RBPamp
import logging
from RBPamp.affinitylogo import plot_afflogo, nice_conc
logging.basicConfig(level=logging.DEBUG)


def align(psam1, psam2):
    ofs, score = psam1.align(psam2.consensus)
    return ofs


def plot_psams(psams, axes, left_labels=[], right_labels=[]):
    offsets = np.array([align(psams[0], psam) for psam in psams])
    widths = np.array([psam.n for psam in psams])
    right = (offsets + widths).max() + 5
    left = offsets.min() - 2

    print(f"offsets={offsets} widths={widths} left={left} right={right}")
    if not left_labels:
        left_labels = [str(i+1) for i in range(n)]

    if not right_labels:
        right_labels = " " * len(psams)

    # print("into zip", psams, offsets, axes, left_labels, right_labels)
    for psam, ofs, ax, ll, rl in zip(psams, offsets, axes, left_labels, right_labels):
        plot_afflogo(ax, psam.psam, minimal=True, x0=ofs)
        ax.text(
            right + 3, 0.5, 
            rl, 
            verticalalignment='center',
            horizontalalignment='right',
        )

        ax.text(
            left, 0.5, 
            ll,
            verticalalignment='center',
            horizontalalignment='right'
        )

        ax.axis('off')
        ax.set_xlim(left-1, right+1)
        ax.set_ylim(0, 1)


def load_psams(fname, est_error=False):
    print(f"loading {fname}")
    try:
        mdl = RBPamp.util.load_model(fname, sort=False)
    except FileNotFoundError:
        mdl = RBPamp.util.NA_model
        est_error = False

    if est_error:
        from RBPamp.errors import PSAMErrorEstimator
        est = PSAMErrorEstimator(os.path.dirname(fname) + '/')
        # params, stats = err_est.load_data()
        p_mid = est.estimate()
        A0_ranges = [(plo.A0, phi.A0) for plo, phi in zip(p_mid.lo, p_mid.hi)]
        print(A0_ranges)
    else:
        A0_ranges = [(p.A0, p.A0) for p in mdl.param_set]

    return mdl, A0_ranges


def nice_conc_range(lo, hi, digits=None, u=None):
    """
    determines the appropriate unit to represent the concentration.
    takes into account low and high confidence interval to compute 
    error and print appropriate number of significant digits.
    example.

    >>> nice_conc(kd=1.555556, lo=1.2341434, hi=1.812314324) 
    1.56 (+0.25 -0.27) nM 
    """
    units = {
        -3: "pM",
        0: "nM",
        3: r"$\mu$M",
        6: "mM",
    }
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NA"

    def leading_digit(x, space=3, umin=-3, umax=6):
        if x <= 0:
            return umin

        dec = np.log10(x)
        u = int(np.floor(dec / space)) * space
        u = max(umin, u)
        u = min(umax, u)
        return u

    def round_sig(f, p, mode='round'):
        from math import floor, ceil
        if mode == 'ceil':
            f = ceil(10**p * f) / 10**p
        if mode == 'floor':
            f = floor(10**p * f) / 10**p

        if p == 0:
            r = int(round(f))
        else:
            r = float(('%.' + str(p) + 'f') % f)
        # print(f,p, "->", r)
        return r

    ul = leading_digit(lo)
    uh = leading_digit(hi)
    um = min(ul, uh)
    if u is None:
        u = max(ul, uh)
    if digits is None:
        digits = max(0, u-um)
    unit = units.get(u, "undefined")
    lo_str = round_sig(lo/10**u, digits, mode='floor')
    hi_str = round_sig(hi/10**u, digits, mode='ceil')
    res = f"({lo_str} - {hi_str}) {unit}"
    # print(res)
    return res


def load_set(pattern, rbp, runs=["py3",], est_errors=True, rbns_path = "/home/mjens/engaging/RBNS/"):
    param_set = []
    errs = []
    for run in runs:
        params, errs = load_psams(pattern.format(**locals()), est_error=est_errors)
        if not est_errors:
            for p in params.param_set:
                p.Kd_range = (None, None)
        else:
            for p, (lo, hi) in zip(params.param_set, errs):
                p.Kd_range = (1./hi, 1./lo)

        param_set.extend(params.param_set)

    return param_set


def render_RBP_PSAMs(rbp, with_single=True, dst_path="{rbp}_logos.pdf",
                     initial_pattern="{rbns_path}/{rbp}/RBPamp/{run}/seed/initial.tsv",
                     nostruct_pattern="{rbns_path}/{rbp}/RBPamp/{run}/opt_nostruct/optimized.tsv",
                     struct_pattern="{rbns_path}/{rbp}/RBPamp/{run}/opt_struct/optimized.tsv"):

    if with_single:
        runs = ["py3.1", "py3"]
    else:
        runs = ["py3"]

    params0 = load_set(initial_pattern, rbp, runs=runs, est_errors=False)
    params1 = load_set(nostruct_pattern, rbp, runs=runs, est_errors=True)
    params2 = load_set(struct_pattern, rbp, runs=runs, est_errors=True)
    n = len(params0)

    if with_single:
        psam_labels = ["single"] + [str(i+1) for i in range(n-1)]
    else:
        psam_labels = [str(i+1) for i in range(n)]

    fig, all_axes = plt.subplots(n, 3, figsize=(7, n*0.4 + 0.65),
                                 sharex=True, sharey=True,
                                 gridspec_kw=dict(
                                    wspace=0.1, hspace=0.04, left=.05,
                                    right=.95, top=.7, bottom=0.02))

    plt.suptitle(rbp)
    for params, axes, label_type, stage in zip(
            [params0, params1, params2],
            all_axes.T,
            ["rel", "abs", "abs"],
            ["initial", "optimized", "+AFP"]):

        if not hasattr(axes, "__iter__"):
            # fix inconsistent plt.subplots behaviour if only one 
            # axis per column is requested
            axes = [axes,]

        psams = [p.as_PSAM() for p in params]
        if label_type == "rel":
            a = [1, ] + [int(np.round(1. / (psam.A0 / psams[0].A0))) for psam in psams]
            aff = [f"{100 * x:.0f} %" for x in a]
            left = psam_labels
        else:
            Kd_ranges = [param.Kd_range for param in params]
            # print(f"Kd_ranges: {Kd_ranges}")
            aff = [nice_conc_range(lo, hi, digits=0) for lo, hi in Kd_ranges]
            left = " " * len(psams)

        plot_psams(psams, axes, left_labels=left, right_labels=aff)
        axes[0].title.set_text(stage)

    plt.tight_layout()
    plt.savefig(dst_path.format(**locals()))
    plt.close()


# for rbp in ['MSI1',]: # 'NOVA1', 'HNRNPA0', 'HNRNPA1', 'UNK', 'MBNL1', 'IGF2BP1', 'RBFOX2', 'PUM1']:
# for rbp in ['UNK', 'HNRNPA0', 'HNRNPA0', 'NOVA1', 'IGF2BP1', 'HNRNPA1', 'MBNL1', 'PUM1']:
# TODO: turn into snakemake run
for rbp in RBPamp.dominguez_rbps:
# for rbp in ['RBFOX3', 'ZFP36']:
    render_RBP_PSAMs(rbp, with_single=False, dst_path="gallery_pdf/{rbp}_motifs.pdf")
    render_RBP_PSAMs(rbp, with_single=False, dst_path="gallery_png/{rbp}_motifs.png")
    render_RBP_PSAMs(rbp, with_single=True, dst_path="gallery_wsingle_pdf/{rbp}_motifs.pdf")
    render_RBP_PSAMs(rbp, with_single=True, dst_path="gallery_wsingle_png/{rbp}_motifs.png")
