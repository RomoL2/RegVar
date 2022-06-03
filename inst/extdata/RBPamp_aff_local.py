#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import RBPamp
import os
import pandas as pd
import sys
import re
import csv

user_11_mer_file = 'vars.k_11_seq.txt' # path/to/11_mer_file.txt
user_10_mer_file = 'vars.k_10_seq.txt' # path/to/10_mer_file.txt
user_full_seq_file = 'vars.full_seq.txt' # path/to/full_seq_file.txt
user_motif_location = './motifs2/' #path to folder with RBP motifs

# make empty versions of output files
os.system('> ' + re.sub('.txt', '', user_full_seq_file) + ".RBPamp_affs." + 'motifs.' + 'tsv')
os.system('> ' + re.sub('.txt', '', user_11_mer_file) + ".RBPamp_affs." + 'motifs.' + 'tsv')
os.system('> ' + re.sub('.txt', '', user_10_mer_file) + ".RBPamp_affs." + 'motifs.' + 'tsv')

# set up
def load_model(fname):
    from RBPamp.params import ModelSetParams
    params = ModelSetParams.load(fname, 1)
    return params

def eval_model(params, seq):
    if len(seq) < params.k:
        return np.zeros(len(seq), dtype=np.float32)

    from RBPamp.cyska import seq_to_bits, PSAM_partition_function
    seqm = seq_to_bits(seq)
    seqm = seqm.reshape((1, len(seq)) )
    accm = np.ones(seqm.shape, dtype=np.float32)
    Z = np.array([PSAM_partition_function(seqm, accm, par.psam_matrix, single_thread=True) * par.A0/params.A0 for par in params])
    return Z.sum(axis=0)[0, :]  # one sequence, sum over all sub-motifs
    
# get motif file list for version of motifs specified (skip hidden files)
motif_files = [f for f in os.listdir(user_motif_location) if not f.startswith('.')]

# loop through all motif files / RBPs (different for "old" and "new" motifs)
for curr_RBP in motif_files:
    print("processing " + curr_RBP)
    params = load_model(user_motif_location + curr_RBP)

    ## 1) full seqs:
    data_in = pd.read_csv(user_full_seq_file)
    seqs = data_in['seq']
    
    # set up
    affs = []
    rbps = []
    ks = []
    
    # evaluate
    for index in range(0, len(seqs)):
        curr_aff = eval_model(params, seqs[index])
        affs.append(curr_aff.sum())
        rbps.append(re.sub('.txt', '', curr_RBP))
        ks.append(params.k)
        

    # save (append for each RBP)
    zip(seqs, rbps, ks, affs)
    with open(re.sub('.txt', '', user_full_seq_file) + ".RBPamp_affs." + 'motifs.' + 'tsv', 'a') as f:
        writer = csv.writer(f, delimiter = '\t')
        writer.writerows(zip(seqs, rbps, ks, affs))
    
    # tidy
    data_in = None
    seqs = None
    affs = None
    rbps = None
    ks = None

    ## 2) k_mer seqs:
    if params.k == 11:
        data_in = pd.read_csv(user_11_mer_file)
        seqs = data_in['seq']
    elif params.k == 10:
        data_in = pd.read_csv(user_10_mer_file)
        seqs = data_in['seq']
    
    # set up
    affs = []
    rbps = []
    
    # evaluate
    for index in range(0, len(seqs)):
        curr_aff = eval_model(params, seqs[index])
        affs.append(curr_aff.sum())
        rbps.append(re.sub('.txt', '', curr_RBP))

    # save (append for each RBP)
    zip(seqs, rbps, affs)
    if params.k == 11:
        with open(re.sub('.txt', '', user_11_mer_file) + ".RBPamp_affs." + 'motifs.' + 'tsv', 'a') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerows(zip(seqs, rbps, affs))
    elif params.k == 10:
        with open(re.sub('.txt', '', user_10_mer_file) + ".RBPamp_affs." + 'motifs.' + 'tsv', 'a') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerows(zip(seqs, rbps, affs))
    
    # tidy
    data_in = None
    seqs = None
    affs = None
    rbps = None
    