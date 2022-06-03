#!/usr/bin/env python
# coding=future_fstrings
from __future__ import print_function

import sys
import os
import numpy as np
import RBPamp.cyska as cyska
import logging

class RefComparison(object):
    def __init__(self, rbp_name, ref_file=""):
        self.rbp_name = rbp_name
        self.rbp_data = rbp_name
        self.sequences = []
        self.kmer_sets = []
        self.names = []
        self.seqs = []
        self.affinities = []
        self.Kd = []
        self.Kd_err = []
        self.affinity_errs = []
        
        self.logger = logging.getLogger("report.ReferenceComparison")
        import RBPamp.cyska
        if not ref_file:
            ref_file = os.path.join(os.path.dirname(__file__),"../known_kds.csv")

        for line in open(ref_file, encoding='utf-8'):
            if line.startswith("# alias"):
                name, alias = line.rstrip().split(" ")[2:]
                if name == rbp_name:
                    rbp_name = alias
                    self.rbp_data = alias
                continue

            if line.startswith("#"):
                continue

            if not line.strip():
                continue
            
            parts = line.rstrip().split('\t')
            if len(parts) < 5:
                continue

            rbp, name, seq, kd, kd_err = parts[:5]
            if not rbp == rbp_name:
                continue
            
            if self.noncanonical(seq):
                # can not predict affinity for sequence with non-canonical bases
                continue

            self.seqs.append(seq)           
            self.logger.debug("{seq}".format(**locals()) )

            self.Kd.append(float(kd))
            self.Kd_err.append(float(kd_err))
            a = 1./float(kd)
            self.affinities.append(a)
            self.affinity_errs.append(a**2 * float(kd_err))
            self.names.append(name)
            
        self.observed_Kd = np.array(self.Kd, dtype=float)
        self.observed_Kd_err = np.array(self.Kd_err, dtype=float)
        self.observed_affinities = np.array(self.affinities)
        self.observed_affinity_errors = np.array(self.affinity_errs)
        self.seqs = np.array(self.seqs)
        self.n = len(self.seqs)
        self.logger.info("found {0} reference affinities for {1}".format(len(self.seqs), rbp_name) )

    def noncanonical(self, seq):
        S = seq.upper()
        return S.count('A') + S.count('C') + S.count('G') + S.count('T') + S.count('U') < len(S)

    def split_kmers(self, seq, k):
        kmers = []
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer.count('A') + kmer.count('C') + kmer.count('G') + kmer.count('U') < k:
                # discovered non-canonical nucleotide
                continue
            kmers.append(cyska.seq_to_index(kmer))
            
        return kmers

    def __len__(self):
        return len(self.seqs)
    
    def predict_affinities_from_paramset(self, paramset):
        aff = np.zeros(len(self.seqs), dtype=np.float32)
        for i, params in enumerate(paramset):
            a = self.predict_affinities(params)
            # # if i > 0:
            # a *= paramset.A0
            # print i, a
            aff += a
        
        # print "combined Kd", 1./aff
        return aff

    def predict_affinities(self, params):
        import RBPamp.cyska as cyska
        a = []
        # if hasattr(mdl, "parameters"):
        #     aff = mdl.parameters.affinities
        # else:
        #     aff = mdl.affinities

        # from RBPamp.seed import Alignment
        # A = Alignment()
        psam = params.as_PSAM()
        matrix = params.psam_matrix
        # print matrix
        missing = []
        core_start = max(0, params.acc_shift)
        core_end = min(core_start + params.acc_k, len(matrix))
        if core_end == core_start:
            # handle nostruct runs with acc_k=0
            core_end = len(matrix)

        # core_start = 0
        # core_end = len(matrix)
        # print "core", core_start, core_end
        disc = psam.discrimination
        for i in range(len(matrix)):

            # if i < core_start:
            #     missing.append( matrix[i].max() )
            # elif i < core_end:
            #     missing.append( matrix[i].min() )
            # else:
            #     missing.append( matrix[i].max() )
            d = disc[i]
            m = matrix[i].min()
            missing.append( (d*m) + (1-d) ) # if d ~ 1, take the minimum. If d ~ 0, don't care about the base.

        def align(seq, min_overlap=2):
            l = len(seq)
            n = len(matrix)
            bits = cyska.seq_to_bits(seq)

            ofs_range = list(range(-l + min_overlap, n + 1 - min_overlap))
            alignments = []
            for ofs in ofs_range:
                m_start = max(0, ofs)
                m_end = min(n,ofs+l)
                
                n_cols = m_end - m_start

                s_start = max(-ofs, 0)
                s_end = s_start + n_cols
                
                score = 1.
                for i in range(m_start):
                    score *= missing[i]
                
                for i in range(m_end, n):
                    score *= missing[i]

                for i in range(n_cols):
                    if bits[i+s_start] > 3:
                        continue # skip gaps
                    
                    S = matrix[i+m_start, bits[i+s_start]]
                    score = score * S
                
                alignments.append( (score, ofs) )

            Z = np.array([aln[0] for aln in alignments])
            best = sorted(alignments)[-1]
            # return best
            return Z.sum(), best[1]

        for seq in self.seqs:
            l = len(seq)
            # if l >= mdl.k_mdl:
            #     # the seq is longer than our motifs/model
            #     I = np.array(self.split_kmers(seq, mdl.k_mdl))
            #     a.append(aff[I].sum())
            # else:
            #     # the seq is shorter than our motifs/model
            score, ofs = align(str(seq))
            # print seq, ofs, score
            a.append(params.A0 * score)

        a = np.array(a)
        # print a.min(), a.max(), a.mean()
        return a

if __name__ == "__main__":
    from RBPamp.pwm import PSAM
    from RBPamp.reads import RBNSReads
    from RBPamp.partfunc import PartFuncModel
    from RBPamp.gradient import ModelParametrization, GradientDescent
    from RBPamp import auto_detect
    from RBPamp.analysis import read_kmer_matrix

    run_folder = "/scratch/data/RBNS/RBFOX3/RBPamp/recent/"
    k_R = 6
    rbp_name, read_files, rbp_conc = auto_detect(os.path.join(run_folder,"../../"))
    rbp_conc2, R0, R0_err = read_kmer_matrix(os.path.join(run_folder,"metrics/{rbp_name}.R_value.{k_R}mer.tsv".format(**locals())))

    print(rbp_conc, rbp_conc2)
    print(R0.shape)

    reads = RBNSReads(read_files[0], temp=4, rbp_conc=0, rbp_name=rbp_name, n_max=10000)
    from glob import glob
    pwm_file = list(glob(os.path.join(run_folder, 'meanfield/mean_field_*mer_PSAM.tsv')))[0]
    motif = PSAM.load(pwm_file)
    print(motif)
    params = ModelParametrization(motif.n, len(rbp_conc2), psam=motif.psam, A0= motif.A0)
    # need: params
    # TODO: alias support in known_kds.csv
    print(params)
    mdl = PartFuncModel(reads, params, R0, rbp_conc=rbp_conc)
    descent = GradientDescent(mdl, params)

    from RBPamp.comparison import RefComparison
    from RBPamp.report import GradientDescentReport, LiteratureComparisonReport
    ref = RefComparison(rbp_name, ref_file="")
    lrep = LiteratureComparisonReport(descent, ref, path='.')

    lrep.plot_scatter()
    