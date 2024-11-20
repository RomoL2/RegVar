# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import RBPamp.report
import matplotlib.pyplot as plt
import seaborn as sns

# data = pd.read_csv('rbns_data.csv', header=None, sep='\t', index_col=None).values.T
# data = pd.read_csv('hisalt_data.csv', header=None, sep='\t', index_col=None).values.T

def process(fname):
    data = pd.read_csv(fname, header=None, sep='\t', index_col=None).values.T

    conc = data[0]
    unfolded = data[1:4]
    folded = data[4:7]
    sensor = data[7:10]


    relu = sensor/unfolded
    relf = sensor/folded

    Y = (sensor - folded) / (unfolded - folded)

    y = Y.mean(axis=0)
    lo = Y.min(axis=0)
    hi = Y.max(axis=0)

    y_err = (hi - y, y - lo)

    return conc, y, y_err

plt.figure(figsize=(2,2))
# plt.semilogx(conc, sensor.mean(axis=0))
# plt.semilogx(conc, unfolded.mean(axis=0))
# plt.semilogx(conc, folded.mean(axis=0))

conc, y, y_err = process('rbns_data.csv')
plt.errorbar(conc, y, yerr=y_err, ecolor='k', capsize=3, capthick=1, elinewidth=1, marker='o', color='blue', label='RBNS buffer')
conc, y, y_err = process('hisalt_data.csv')
plt.errorbar(conc, y, yerr=y_err, ecolor='k', capsize=3, capthick=1, elinewidth=1, marker='o', color='orange', label='1M Na+')

plt.gca().set_xscale('log')
plt.legend(loc='best')
plt.ylabel("rel. fluoresence")
plt.xlabel(u"RBFOX2 concentration [μM]")
sns.despine()
plt.tight_layout()
plt.savefig('beacon.pdf')
