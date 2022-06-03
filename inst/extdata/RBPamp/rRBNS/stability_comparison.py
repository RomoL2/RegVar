import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from RBPamp.errors import PSAMErrorEstimator

compare = [
    'RBPamp/std/opt_nostruct/',
    'RBPamp/oneconc/opt_nostruct/',
]

labels = ["combined", "best alone"]
colors = ['blue', 'orange']

def plot_param_error_scatter(rbp, param_i=0, skip=5):

    error_estimators = [PSAMErrorEstimator(os.path.join(rbp, path)) for path in compare]
    multi_conc, one_conc = [err.shelve['rbp_conc'] for err in error_estimators]
    sample_i = (multi_conc == one_conc).argmax()

    sample_indices = [sample_i, 0]
    # print sample_i

    final_errors = []
    plt.figure(figsize=(3,3))

    end_x = []
    end_y = []
    lfcs = []
    for err_est, lab, col, j in zip(error_estimators, labels, colors, sample_indices):
        params, stats = err_est.load_data()
        if not len(stats):
            print "no data?", rbp, lab

        mdl_errors = np.array([(s.errors[j] **2).mean() for s in stats])
        par = np.array([p.get_data()[param_i] for p in params])
        lfcs.append(np.fabs(np.log10(par[1:]/par[:-1])))

        final_errors.append(mdl_errors[-1])
        # x = mdl_errors
        x = np.linspace(0, 1, len(par))
        y = 1./par

        sc = plt.scatter(
            x,
            y,
            alpha=.3,
            label=lab,
            color=col,
        )
        plt.plot(x, y, '-', linewidth=.5, color=col)
        end_x.append(x[-1])
        end_y.append(y[-1])

        # plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")

    # plt.plot(end_x, end_y, 'pr', label="end-point")
    # pp.colorbar(sc, shrink=.05, label="time step")

    plt.legend(loc='lower right', handletextpad=.1)
    plt.xlabel("optimization progress")
    plt.ylabel(r"est. $K_d^{opt}$ [nM]")
    # pp.gca().set(aspect="equal")
    sns.despine()
    plt.tight_layout()

    fname = os.path.join(rbp, "param_{0}_progress.pdf".format(param_i))
    plt.savefig(fname)
    plt.close()

    plt.figure(figsize=(3,3))
    plt.hist(lfcs, label=labels, color=colors)
    
    eratio = final_errors[1] / final_errors[0]
    uratio = lfcs[1].mean() / lfcs[0].mean()

    # for lfc in lfcs:
    #     print "mean abs. log-change", lfc.mean()
    # print "ratio of mean ratios", uratio

    # from scipy.stats import mannwhitneyu
    # print mannwhitneyu(lfcs[0], lfcs[1])
    
    plt.xlabel("rel. changes")
    plt.ylabel('frequency')
    sns.despine()
    plt.tight_layout()

    fname = os.path.join(rbp, "param_{0}_updates.pdf".format(param_i))
    plt.savefig(fname)
    plt.close()
    return eratio, uratio

from RBPamp import dominguez_rbps as rbps
# rbps = ['RBFOX3', 'NOVA1', 'MBNL1']
er, ur = np.log2(np.array([plot_param_error_scatter(rbp, 0) for rbp in rbps]).T)

print "mean error log ratio", er.mean()
print "mean update log ratio", ur.mean()

from scipy.stats import ttest_1samp, wilcoxon
print "t-test error log ratio is 0", ttest_1samp(er, 0)
print "t-test update log ratio is 0", ttest_1samp(ur, 0)

print "Wilcoxon test error log ratio is symmetric about 0", wilcoxon(er)
print "Wilcoxon test update log ratio is symmetric about 0", wilcoxon(ur)


plt.figure(figsize=(1.5,3))
bplot = plt.boxplot([er, ur],
    notch=False,  # notch shape
    widths=.5,
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
    labels=["model\nerror", "parameter\nchanges"],  # will be used to label x-ticks
)
plt.ylabel(r"best-alone / combined, $\log_2$") 
plt.axhline(0, color='k', linewidth=.5, linestyle='dashed')
# plt.yticks([-3, -2, -1,0,1,2], ["0.125","0.25", "0.5", "1", "2", "4"])
# plt.xticks(rotation=90)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'orange']

for patch, color in zip(bplot['boxes'], ['lightgray', 'gray']):
    patch.set_facecolor(color)

def lstr(x, base=2):
    if x >= 0:
        return "{}".format(int(base**x))
    else:
        return "1/{}".format(int(base**(-x)))

yt = plt.yticks()[0][1:-1]
plt.yticks(yt, [lstr(y) for y in yt])
plt.tight_layout()
plt.savefig('stability.pdf')
plt.close()
