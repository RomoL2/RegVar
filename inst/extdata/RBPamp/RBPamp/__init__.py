__name__ = "RBPamp"
__version__ = "0.9.20"
__license__ = "MIT"
__authors__ = ["Marvin Jens", ]
__email__ = "mjens@mit.edu"

def git_commit():
    import subprocess
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    gc_name = os.path.join(path, "../git_commit")
    if os.path.exists(gc_name):
         return open(gc_name, 'r').read().rstrip()
    else:
        try:
            git = subprocess.Popen(["git","describe","--always"], cwd=path, stdout=subprocess.PIPE).communicate()[0].rstrip()
        except:
            git = "unknown"
        return git

def load_dominguez_motifs(fname="dominguez_table_S3.6mers.csv"):
    import os
    from collections import defaultdict
    
    base = os.path.dirname(__file__)
    path = os.path.join(base, '..', 'rRBNS', fname)

    motifs = defaultdict(list)
    weights = defaultdict(list)
    rbps = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 1:
                continue
            if i == 1:
                rbps = [col.replace('RALYL', 'RALY') for col in line.rstrip().split('\t') if col.strip()]
            else:
                parts = line.rstrip().split('\t')
                for j, m in enumerate(parts):
                    if m.strip():
                        name = rbps[int(j / 2)]
                        if j % 2:
                            weights[name].append(float(m))
                        else:
                            motifs[name].append(m)
    
    res = {}
    for rbp in motifs.keys():
        res[rbp] = (motifs[rbp], weights[rbp])

    return res

dominguez_rbps = "BOLL,CELF1,CNOT4,CPEB1,DAZ3,DAZAP1,EIF4G2,ELAVL4,ESRP1,EWSR1,FUBP1,FUBP3,FUS,A1CF,HNRNPA0,HNRNPA1,HNRNPA2B1,HNRNPC,HNRNPCL1,HNRNPD,HNRNPDL,HNRNPF,HNRNPH2,HNRNPK,HNRNPL,IGF2BP1,IGF2BP2,ILF2,KHDRBS2,KHDRBS3,KHSRP,MBNL1,MSI1,NOVA1,NUPL2,PABPN1L,PCBP1,PCBP2,PCBP4,PRR3,PTBP3,PUF60,PUM1,RALY,RBFOX2,RBFOX3,RBM15B,RBM22,RBM23,RBM24,RBM25,RBM4,RBM41,RBM45,RBM47,RBM4B,RBM6,RBMS2,RBMS3,RC3H1,SF1,SFPQ,SNRPA,SRSF10,SRSF11,SRSF2,SRSF4,SRSF5,SRSF8,SRSF9,TAF15,TARDBP,TIA1,TRA2A,TRNAU1AP,UNK,ZCRB1,ZFP36,ZNF326".split(',')
dom_T21 = "CELF1,ESRP1,EWSR1,FUS,HNRNPC,HNRNPD,HNRNPF,HNRNPH2,HNRNPK,IGF2BP1,IGF2BP2,MBNL1,MSI1,PTBP3,PUF60,RBFOX2,RBM23,RBM47,SRSF10,SRSF2,SRSF4,SRSF5,SRSF8,TAF15,TRA2A".split(',')

import os
import pandas as pd
import numpy as np

path_template = "/home/mjens/engaging/RBNS/{rbp}/cska/metrics/metrics/{rbp}.R_value.{k}mer.tsv"

def get_significant_R_values(rbp, k=6, z_cut=4):
    df = pd.read_csv(path_template.format(rbp=rbp, k=k), header=0, sep='\t', index_col=0)
    cols = [c for c in df.columns if c.endswith('nM')]
    R = df[cols].to_numpy()

    R_max = R.max(axis=0)
    best_sample = R_max.argmax()
    R = R[:, best_sample]

    z = (R - R.mean())/ np.std(R)
    sig = z > z_cut
    if not sig.any():
        # report at least ONE motif per RBP
        sig = z >= z.min()

    motifs = [m.replace('U', 'T') for m in df.index.values[sig]]
    weights = R[sig]

    return motifs, weights

def get_all_significant_R_values(k):
    sig_R = {}
    for rbp in dominguez_rbps:
        sig_R[rbp] = get_significant_R_values(rbp, k=k)

    return sig_R

