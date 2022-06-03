# coding=future_fstrings
from __future__ import print_function
from __future__ import absolute_import

import RBPamp
from RBPamp import __version__, __license__, __authors__, __email__

import sys
import os
import copy
import time
import itertools
import traceback
import logging
import collections
import numpy as np
import matplotlib
matplotlib.use('agg')

def parse_cmdline():
    import datetime
    datestr = datetime.datetime.now().strftime("%b-%d-%Y_%H:%M:%S")

    from optparse import OptionParser
    usage = "usage: %prog [options] [path_to_RBNS_splitreads]"

    parser = OptionParser(usage=usage)
    # basic options
    parser.add_option("","--version", dest="version", default=False, action="store_true", help="show version information and quit")
    parser.add_option("","--name", dest="name", default="RBP", help="name of the protein assayed (default=RBP)")
    parser.add_option("-o","--output", dest="output", default="RBPamp", help="path where results are to be stored (default='RBPamp')")
    parser.add_option("","--run-path", dest="run", default="run_{datestr}".format(datestr=datestr), help="pattern for run-folder name (default='run_{datestr}')")
    # parser.add_option("-a","--auto", dest="auto", default=False, action="store_true", help="SWITCH: attempt to automatically guess RPB name, reads files and concentrations from file names (default=specify manually)")

    parser.add_option("","--ranks", dest="ranks", default="1,2,3,4", type=str, help="analyze these ranks out of the best n samples (by top R-value) default=1,2,3,4")
    parser.add_option("","--min-R", dest="min_R", default=1.2, type=float, help="minimal enrichment value of the (overall) most enriched 7-mer required to consider a sample at all (QC pass) default=1.1")
    parser.add_option("","--resume", dest="resume", default=False, action="store_true", help="re-use previous results")
    parser.add_option("","--continue", dest="cont", default=False, action="store_true", help="add more iterations of optimization even if already completed")
    parser.add_option("","--redo", dest="redo", default=False, action="store_true", help="do not re-use previous results at all")
    
    parser.add_option("-R","--rna-concentration", dest="rna_conc", default=1000., type=float, help="concentration of random RNA used in the experiment in nano molars (default=1000 nM)")
    parser.add_option("-P","--rbp-concentration", dest="rbp_conc", default="0,320", help="(comma separated list of) protein concentration used in the experiment(s) in nano molars (default=0,300)")
    parser.add_option("-T","--temperature", dest="temp", default=4., type=float, help="temperature of the experiment in degrees Celsius (default=4.0)")
    parser.add_option("","--format", dest="format", default='raw', help="read file format [raw,fasta,fastq] (default=raw)")
    parser.add_option("","--adap5", dest="adap5", default="gggaguucuacaguccgacgauc", help="5'RNA adapter sequence to add to read sequence")
    parser.add_option("","--adap3", dest="adap3", default="uggaauucucgggugucaagg", help="3'RNA adapter sequence to add to read sequence")
    parser.add_option("-N","--n-max", dest="n_max", default=15000000, type=int, help="read at most N reads (preserves RAM for very deep sequencing libraries. default=15M, 0=off)")
    parser.add_option("","--no-replace", dest="replace", default=True, action="store_true", help="TESTING: disable drawing with replacement")

    # RNA folding
    parser.add_option("","--fold-input", dest="fold_input",default="", help="instead of a normal run, fold all reads in the INPUT sample and record accessibilities/open-energies for k in the given range. example --fold=1-12 (default=off)")
    parser.add_option("","--fold-samples", dest="fold_samples",default="", help="instead of a normal run, fold all reads from PULL-DOWN samples and record accessibilities/open-energies for k in the given range. example --fold=1-12 (default=off)")
    parser.add_option("","--acc-scan", dest="acc_scan", default=False, action="store_true",help="SWITCH: scan for high accessibility selection in bound libraries")
    # parser.add_option("","--acc-scale",dest="acc_scale",default=1.,type=float,help="[EXPERIMENTAL] scale unfolding energies")
    parser.add_option("","--openen-discretize", dest="openen_discretize", default="0", choices=["0","8","16"], help="discretize open-energies using <n> bits [8,16] set to 0 to disable (default)")
    parser.add_option("","--parallel", dest="parallel", default=8,type=int,help="number of parallel threads (currently only used for folding. default=8)")
    
    # RBNS metrics
    parser.add_option("","--metrics",dest="results",default="",help="list of RBNS metrics to compute and store (options='R_value,SKA_weight,F_ratio' default='')")
    parser.add_option("","--sample-correlations", dest="sample_corr", default=False, action="store_true", help="compute sample vs sample correlation matrices")
    parser.add_option("","--metrics-k", dest="metrics_k", default="3-8", help="range of kmer sizes for which to compute the desired metrics (default: --metrics-k=3-8)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("","--ska-max-passes",dest="n_passes",default=10,type=int,help="max number of passes (default=10)")
    parser.add_option("","--ska-convergence",dest="convergence",default=0.5,type=float,help="convergence is reached when max. change in absolute weight is below this value (default=0.5)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimation (default=10)")

    # seed motif analysis
   
    parser.add_option("", "--opt-seed", dest="opt_seed", default=False, action="store_true", help="perform initial motif construction (STAGE0: seed-stage)")
    parser.add_option("","--seed-k",dest="k_seed",default=8, type=int, help="kmer size used for seeding PSAM(s) (default=8)")
    parser.add_option("", "--z-cut", dest="z_cut", default=4., type=float, help="Z-score cutoff for R-values of kmers that go into motif building (default=4)")
    parser.add_option("", "--max-motifs", dest="max_motifs", default=5, type=int, help="maximal number of individual PSAMs (variant motifs) being fitted (default=5)")
    parser.add_option("", "--seed-thresh", dest="seed_thresh", default=.75, type=float, help="score threshold for k-mer:PSAM alignment to trigger a new PSAM (default=.75)")
    parser.add_option("-w","--max-width",dest="max_width",default=11, type=int, help="maximum number of nucleotides in PSAM motif (number of columns) default=11)")
    parser.add_option("", "--seed-pseudo", dest="seed_pseudo", default=.01, type=float, help="'pseudo' affinity for non-cognate bases. Non-zero allows gradient descent to act on all bases (default=.01)")
    parser.add_option("", "--seed-keep-weight", dest="seed_keep_weight", default=.99, type=float, help="how much total PSAM weight to keep when building final --max-width nt wide matrix (default=.99)")

    # accessibility footprint analysis
    parser.add_option("","--footprint-k", dest="footprint", default="5-20", help="size range [nt] to search for ideal accessibility footprint (default: --footprint-k=5-20)")
    parser.add_option("","--footprint-motif", dest="fp_num", default=0, type=int, help="which motif number to compute the footprint on (default=0 [all])")
    parser.add_option("","--footprint-min-overlap", dest="fp_min_ov", default=.5, type=float, help="minimum overlap between footprint and PSAM as fraction of footprint (default=.2)")
    # parser.add_option("","--footprint-min-scale", dest="fp_min_a", default=.05, type=float, help="minimum footprint scaling factor to keep footprint. 0=accessibility has no effect, 1=RNAfold  (default=.05)")
    parser.add_option("","--footprint-max-error", dest="fp_max_err", default=.90, type=float, help="maximum rel. footprint error acceptable to keep footprint")
    parser.add_option("","--top-kmer-acc", dest="top_kmer_acc", default=0, type=int, help="generate enrichment vs. accessibility data for top kmers (default=0 [off])")
    
    # affinity model optimization 
    # parser.add_option("","--seed-motif",dest="seed_motif",default="", help="DEBUGGING: override motif from seed analysis with this exact sequence.")
    parser.add_option("","--grad-k",dest="grad_k",default=6, type=int, help="k for gradient descent kmer R-value mean squared error objective function (default=6)")
    parser.add_option("","--grad-mdl",dest="grad_mdl",default="", choices=['partfunc', 'meanfield', 'invmeanfield', ''], help="method for gradient descent refinement of PSAM [partfunc, meanfield, invmeanfield, ''=off] default=partfunc")
    parser.add_option("","--grad-maxiter",dest="grad_maxiter",default=500, type=int, help="maximal number of gradient descent iterations (default=500)")
    parser.add_option("","--grad-maxtime",dest="grad_maxtime",default=11.5*3600, type=float, help="maximal time to spend for optimization in seconds (default=12 hours)")
    parser.add_option("","--excess-rbp",dest="excess_rbp",default=False, action="store_true",help="MODEL: pretend total RBP == free RBP")
    parser.add_option("","--linear-occ",dest="linear_occ",default=False, action="store_true",help="MODEL: pretend no saturation: occ = P/Kd")
    parser.add_option("-n","--n-samples", dest="n_samples", default=5000000, type=int, help="bootstrap sample n reads at regular intervals during gradient descent or when stuck (default=5M)")
    parser.add_option("-r","--resample-interval", dest="resample_int", default=1, type=int, help="re-sample/bootstrap reads every -r iterations of descent (default=1, 0 to disable)")


    parser.add_option("", "--opt-nostruct", dest="opt_nostruct", default=False, action="store_true", help="perform no-struct gradient descent (STAGE1: nostruct stage)")
    parser.add_option("", "--opt-footprint", dest="opt_footprint", default=False, action="store_true", help="perform footprint calibration (STAGE2: footprint stage)")
    parser.add_option("", "--opt-struct", dest="opt_struct", default=False, action="store_true", help="perform structure-aware gradient descent (STAGE3: struct stage)")
    parser.add_option("", "--opt-full", dest="opt_full", default=False, action="store_true", help="perform all stages of optimization (STAGE0 - STAGE3")
    parser.add_option("", "--est-errors", dest="est_errors", default=False, action="store_true", help="perform PSAM error estimation")

    parser.add_option("", "--plot", dest="plot", default="", help="(re-) generate plots. Comma-separated items from seed,descent,scatter,fp,lit,logos or 'all' ")

    parser.add_option("","--Z-threshold",dest="Z_thresh",default=0, type=float, help="drop reads that have Boltzmann weight of a factor of Z_thresh below the max weight (default=0/off)")
    # parser.add_option("-m","--model",dest="model",default=False, action="store_true",help="SWITCH: thermodynamic model parameter fit")
    parser.add_option("","--no-structure",dest="no_structure",default=False, action="store_true",help="ignore secondary structure folding information (default=False)")
    parser.add_option("","--load-psam",dest="mdl_psam_init",default=None,help="start with affinity parameters from this PSAM file")
    parser.add_option("","--fix-A0",dest="fix_A0",default=False, action="store_true",help="do not attempt to optimize A0 at all")
    parser.add_option("","--eps",dest="mdl_epsilon",default=1e-4, type=float, help="convergence threshold for relative error reduction (default=1e-4)")
    parser.add_option("","--tau",dest="mdl_tau",default=23, type=int, help="convergence estimation interval (default=13) [Note, this should be larger than the re-sampling interval -r]")

    # TODO: update
    # parser.add_option("","--sensors",dest="mdl_report_sensors",default="correlation,betas,errors,R_values", help="list of sensors to keep track of optimization progress. default='correlation,betas,errors,R_values'")
    # parser.add_option("","--report-interval",dest="mdl_report_interval",default=50, type=int, help="generate diagnostic/report PDFs every x iterations of the model fit (default=50)")
    # parser.add_option("","--report-skip",dest="mdl_report_trigger",default="", help="comma separated list of events that should *not* trigger new plots")

    parser.add_option("","--reference",dest="ref_file",default="", help="tab-separated file with measured (reference) Kd values (default=use builtin known_kds.csv)")
    parser.add_option("","--compare",dest="compare",default="", help="compare to literature values for this protein")

    # infrastructure and logging/debugging control
    parser.add_option("","--disable-caching",dest="disable_caching",default=False, action="store_true",help="DEBUG: disable transparent caching (SLOW!)")
    parser.add_option("","--disable-unpickle",dest="disable_unpickle",default=False, action="store_true",help="DEBUG: disable unpickling. Will recompute and overwrite existing pickled data")
    parser.add_option("","--disable-pickle",dest="disable_pickle",default=False, action="store_true",help="DEBUG: disable pickling. Will not create or overwrite any pickled data")
    
    parser.add_option("","--debug",dest="debug",default="",help="activate debug output for comma-separated subsystems [root, fold, cache, rbns, opt, model, report]")
    parser.add_option("","--debug-grad",dest="debug_grad",default=False, action="store_true", help="compute empirical gradient alongside analytical (for debugging only)")
    parser.add_option("","--info",dest="info",default="",help="activate info level output for comma-separated subsystems [root, fold, cache, rbns, opt, model, report]")
    parser.add_option("","--log-remote",dest="log_remote", default="", help="replicate all logging output to this remote server (useful to collect output from multiple runs in parallel)")

    # parser.add_option("","--track-kmers",dest="track_kmers",default="", help="comma separated list of kmers to track during optimization.")

    # read simulation (currently broken)
    # parser.add_option("","--simulate",dest="simulate",choices=["","reads","comparison"],default="",help="simulate RBNS instead of analysis, choices are ['reads','comparison']")    
    parser.add_option("","--rnd-seed",dest="seed",default=47110815,type=int,help="seed for fast pseudo-random number generator (for RBNS simulation)")
    # parser.add_option("","--sim-best-Kd",dest="sim_best_Kd",default=10.,type=float,help="best binding dissociation constant for simulation in nM (default=10 nM)")
    # parser.add_option("","--sim-var",dest="sim_var",default=10.,type=float,help="variance for simulated binding energy log-normal distribution (default=)")
    # parser.add_option("","--sim-mean",dest="sim_mean",default=10.,type=float,help="mean for simulated binding energy log-normal distribution (default=)")
    # parser.add_option("","--sim-N-reads",dest="sim_N_reads",default=1000000,type=int,help="number of reads to simulate (default=1,000,000)")
    
    options, args = parser.parse_args()
    
    if options.version:
        print(RBPamp.__name__, __version__, "git", RBPamp.git_commit())
        print(__license__, "license")
        print("by", ", ".join(__authors__))
        sys.exit(0)

    return options, args


def ensure_path(full):
    path = os.path.dirname(full)
    try:
        os.makedirs(path)
    except OSError:
        pass

    return full


def auto_detect(path='.', exts=["reads","txt"]):
    """
    auto-detect RBP name, reads files and concentrations from files in directory
    """
    from glob import glob
    from collections import defaultdict
    files = []
    for ext in exts:
        pattern = os.path.join(path,'*.{0}'.format(ext))
        hits = list(glob(pattern))
        files.extend(hits)
    
    rbp_names = defaultdict(list)
    rbp_conc = []
    
    for f in files:
        try:
            name, conc = os.path.basename(f).rsplit("_", 1)
            conc = conc.rsplit('.',1)[0]
            conc = float(conc.replace('input','0'))
        except ValueError:
            continue

        rbp_names[name].append( (conc, f) )
    
    hits = sorted([(len(rbp_names[name]), name) for name in rbp_names.keys()])[::-1]
    # print("RBP name auto-detect", hits)
    if not rbp_names:
        raise ValueError("unable to find any reads files of the form <RBP_name>_(<concentration>|input).(reads|txt) - bailing out.")

    rbp_name = hits[0][1]

    results = sorted(rbp_names[hits[0][1]])
    files = np.array([f for conc, f in results])
    rbp_conc = np.array([conc for conc, f in results], dtype=np.float32)

    return rbp_name, files, rbp_conc


def vector_stats(v):
    print(getattr(v,"__name__", "no name"), type(v))
    print("shape",v.shape)
    print("pos. values", (v > 0).sum())
    print("0 values", (v == 0).sum())
    print("neg. values", (v < 0).sum())
    print("nan values", np.isnan(v).sum())
    print("non-finite values", (~np.isfinite(v)).sum())
    print("min max", v.min(), v.max())
    print("mean median", np.mean(v), np.median(v))


def touch(fname, times=None):
    # print "touching", fname
    with open(fname, 'a'):
        os.utime(fname, times)


class Run(object):
    def __init__(self, options, args):
        self.options = options
        self.args = args
        if args:
            os.chdir(args[0])
        
        self.rbp_name, self.reads_files, self.rbp_concentrations = auto_detect('.')

        # control caching framework behavior
        from RBPamp.caching import CachedBase
        CachedBase._do_not_cache = options.disable_caching
        CachedBase._do_not_pickle = options.disable_pickle
        CachedBase._do_not_unpickle= options.disable_unpickle

        self._init_paths()
        self._init_invocation()
        self._init_logging()
        self._init_RNG()
        self._init_signal_handler()

        self.state_trackers = {}
        self.last_tracker = None

        from RBPamp.comparison import RefComparison
        if self.options.compare:
            self.ref = RefComparison(self.options.compare, ref_file=self.options.ref_file)
        else:
            self.ref = RefComparison(self.rbp_name, ref_file=self.options.ref_file)

    def _init_paths(self):
        """prepare and initialize outout paths"""
        self.run_folder = self.options.run+"/"
        self.run_path = ensure_path(os.path.join(self.options.output, self.run_folder))
        
        if self.options.run != "recent":
            # keep a symlink named "recent" always pointing to last run folder
            recent_path = os.path.join(self.options.output, "recent")
            try:
                os.remove(recent_path)
                os.symlink(self.run_folder, recent_path)
            except OSError:
                pass
        
        # where to put/find transparent pickle/unpickle objects
        from RBPamp.caching import CachedBase
        CachedBase.pkl_path = os.path.join(self.options.output, ".pkl")

        # accessibility prediction from folding
        self.fold_path = os.path.join(self.options.output, "acc")
        if self.options.no_structure:
            self.fold_path = "NOSTRUCTURE"

    def get_state_tracker(self, stage, **kwargs):
        from RBPamp.status import StateTracker
        if not stage in self.state_trackers:
            self.state_trackers[stage] = StateTracker(self, stage, **kwargs)
        
        self.last_tracker = self.state_trackers[stage]
        return self.last_tracker

    def _init_invocation(self):
        import socket
        self.hostname = socket.gethostname()
        self.git_commit = RBPamp.git_commit()
        self.cmdline = " ".join(sys.argv)
        self.version = __version__

    @property
    def mini_run_info(self):
        return f"runinfo: version={self.version} git={self.git_commit} cmdline={self.cmdline}"

    def _init_logging(self):
        # set up logging
        self.log_path = os.path.join(self.run_path,"run.log")

        FORMAT = '%(asctime)-20s\t%(levelname)s\t{self.hostname}\tgit {self.git_commit}\t{self.rbp_name}\t{self.options.run}\t%(name)s\t%(message)s'.format(**locals())
        self.log_format = FORMAT
        formatter = logging.Formatter(FORMAT)
        logging.basicConfig(level=logging.INFO, format=FORMAT)    
        root = logging.getLogger('')

        fh = logging.FileHandler(filename=self.log_path, mode='a')
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)
        

        if self.options.log_remote:
            # replicate all log-output to the remote log-server
            from . import zmq_logging
            rh = zmq_logging.make_handler(address=self.options.log_remote, formatter=formatter)
            root.addHandler(rh)

        self.logger = logging.getLogger('CSKA')
        self.logger.setLevel(logging.INFO)
        self.logger.info("version {}".format(self.version))
        self.logger.info("invoked as '{}'".format(self.cmdline) )
        slurmid = os.getenv('SLURM_JOB_ID')
        if slurmid:
            self.logger.info("SLURM_JOB_ID={}".format(slurmid))

        # set info level for specific sub-systems
        for sub in self.options.info.split(','):
            if not sub:
                continue
            sub = sub.replace('root',"")
            logging.getLogger(sub).setLevel(logging.INFO)

        # set debug log level for specific sub-systems
        for sub in self.options.debug.split(','):
            if not sub:
                continue
            sub = sub.replace('root',"")
            logging.getLogger(sub).setLevel(logging.DEBUG)
            if sub == 'cache':
                from RBPamp.caching import CachedBase
                CachedBase.debug_caching = True

    def _init_RNG(self):
        if self.options.seed:
            self.logger.info("seeding RNG with {}".format(self.options.seed))
            np.random.seed(self.options.seed)
            import RBPamp.cyska
            RBPamp.cyska.rand_seed(self.options.seed)

    def _init_signal_handler(self):
        import signal
        import inspect

        def sigterm_handler(signal, frame):
            msg = "Received signal {} while executing {}.".format(signal, inspect.getframeinfo(frame))
            print(msg)
            self.logger.error(msg)
            
            if self.last_tracker:
                self.last_tracker.set(msg)

            import RBPamp.fold
            RBPamp.fold.interrupt()
            sys.exit(0)

        signal.signal(signal.SIGTERM, sigterm_handler)


    def make_SKA(self):
        """parametrize SKA algorithm"""
        from RBPamp.ska_runner import SKARunner
        ska = SKARunner(
            max_iterations = self.options.n_passes,
            convergence = self.options.convergence, 
        )
        return ska


    def get_storage_kwargs(self):
        storage_kw = dict(
            # overwrite = self.options.overwrite, 
            T=self.options.temp, 
            disc_mode='linear', 
            dummy=self.options.no_structure, 
            # acc_scale=self.options.acc_scale
        )
        if int(self.options.openen_discretize):
            dtype = getattr(np, "uint{0}".format(self.options.openen_discretize))
            storage_kw.update(dict(discretize=True, disc_dtype=dtype))
        else:
            storage_kw.update(dict(discretize=False, raw_dtype=np.float32))

        return storage_kw


    def select_reads(self):
        # start a new analysis
        from RBPamp.analysis import RBNSAnalysis
        from RBPamp.reads import RBNSReads

        storage_kw = self.get_storage_kwargs()

        rbns = RBNSAnalysis(
            rbp_name = self.rbp_name,
            out_path = self.run_path,
            ska_runner = self.make_SKA(),
        )
        self.logger.info("populating RBNS analysis with reads")
        
        for fname, rbp_conc in zip(self.reads_files, self.rbp_concentrations):
            reads = RBNSReads(
                fname, 
                format = self.options.format,
                rbp_conc=rbp_conc,
                rbp_name = self.rbp_name,
                n_max=self.options.n_max, 
                pseudo_count=self.options.pseudo, 
                rna_conc = self.options.rna_conc,
                temp = self.options.temp,
                n_subsamples = self.options.subsamples,
                adap3=self.options.adap3,
                acc_storage_path = self.fold_path,
                storage_kw=storage_kw,
                n_samples=self.options.n_samples,
                replace=self.options.replace
            )
            
            rbns.add_reads(reads)
        
        self.rbns = rbns
        return rbns


    def compute_metrics(self, metrics):
        tracker = self.get_state_tracker('metrics')
        self.logger.info("computing RBNS metrics '{0}'".format(metrics))
        kmin, kmax = self.options.metrics_k.split('-')


        for k in range(int(kmin), int(kmax) + 1):
            tracker.set(k)
            for values, errors in self.rbns.compute_results(k, self.options, results=metrics):
                pass

        tracker.set("COMPLETED")

    def keep_best(self):
        ranks = np.array(self.options.ranks.split(','), dtype=int)
        ranked_indices, self.rbns = self.rbns.keep_best_samples(ranks=ranks, k=7, min_R=self.options.min_R)
        return ranked_indices, self.rbns

    def fold_reads(self):
        self.logger.info("folding reads with '{0}' threads".format(self.options.parallel))
        tracker = self.get_state_tracker('fold')

        from RBPamp.fold import parallel_fold
        # prepare outout path
        if not os.path.exists(self.fold_path):
            os.makedirs(self.fold_path)

        def fold_sample(reads, kmin, kmax):
            kmin = int(kmin)
            kmax = int(kmax)

            tracker.set(f'folding {reads.name}')
            n_complete = 0
            n_left = reads.N
            if self.options.resume:
                n_complete = reads.acc_storage.count_complete_records_range(kmin, kmax)

            if n_complete < reads.N:
                perc = 100. * n_complete / reads.N
                n_left = reads.N - n_complete
                self.logger.info("{n_complete}/{reads.N} reads already folded ({perc:.2f}%). Folding remaining {n_left} reads".format(**locals()))

                self.logger.info("folding {reads.name} ({reads.fname})".format(reads=reads) )
                parallel_fold(
                    reads,
                    n_complete = n_complete,
                    k_min = int(kmin),
                    k_max = int(kmax),
                    n_parallel= self.options.parallel,
                    log_address = self.options.log_remote,
                    log_format = self.log_format, 
                )

            else:
                self.logger.info("skipped {} because accessibilities from k={}..{} have already been computed and stored.".format(reads.name, kmin, kmax))

            tracker.set(f'{reads.name} done')
            
        if self.options.fold_input:
            kmin, kmax = self.options.fold_input.split('-')        
            fold_sample(self.rbns.reads[0], kmin, kmax)

        if self.options.fold_samples:
            # fold the reads
            kmin, kmax = self.options.fold_samples.split('-')
            for reads in self.rbns.reads[1:]:
                fold_sample(reads, kmin, kmax)

        tracker.set('COMPLETED')

    def simulate(self):
        pass
        # # TODO: properly integrate simulation
        # if options.simulate == "reads":
        #     from RBPamp.optimize import RBNSGenerator
        #     for k in range(options.min_k, options.max_k + 1):
        #         gen = RBNSGenerator(k,l=40, seed=options.seed)
        #         gen.assign_experimental_input(reads_files[0])
        #         gen.energy_plot()

        #         r_matrix = []
        #         for P in np.array(rbp_concentrations[1:]):
        #             r = gen.predict_r_values(P=P, store="r_{0}.tsv".format(P))
        #             r_matrix.append(r)
                    
        #             read_path = os.path.join(self.out_path,"sim_bound_{0}.reads".format(P) )
        #             gen.generate_bound_reads(read_path, P=P, p_ns=0.00, N=20000000)
        #             occ = gen.predict_occupancies(P=P, store="occ_{0}.tsv".format(P))

    
    def probe_params(self, *locations):
        from RBPamp.params import ModelParametrization, ModelSetParams
        for path in locations:
            if not path:
                continue
            path = os.path.abspath(os.path.join(self.run_path, path))
            try:
                self.logger.info("attempting to resume parameters from '{}'".format(path))
                self.params = ModelSetParams.load(path, self.rbns.n_samples, max_motifs=self.options.max_motifs, sort=False)
            except IOError:
                self.logger.info("not found")
                self.params = None
            else:
                self.logger.info("success")
                break
        
        if not self.params:
            self.logger.error("unable to initiate model parameters. Did you skip a stage?")
            raise ValueError("previous stage incomplete or missing/malformed model parameters")

        return self.params
        

    def completed(self, stage, strict=False):
        stage_output = {
            'seed' : os.path.join(self.rbns.out_path, 'seed/initial.tsv'),
            'opt_nostruct' : os.path.join(self.rbns.out_path, 'opt_nostruct/optimized.tsv'),
            'footprint' : os.path.join(self.rbns.out_path, 'footprint/calibrated.tsv'),
            'opt_struct' : os.path.join(self.rbns.out_path, 'opt_struct/optimized.tsv'),
        }
        import datetime

        fout = stage_output[stage]
        out_flag = os.path.exists(fout)
        
        fcompleted = os.path.join(self.run_path,'completed.{}'.format(stage))
        comp_flag = os.path.exists(fcompleted)
        
        tracker = self.get_state_tracker(stage, startup=False)
        state_flag = tracker.is_completed(strict=strict)

        if not out_flag:
            self.logger.debug("output of stage {} has not been created yet".format(stage))
            return False
        
        if not (comp_flag or state_flag):
            self.logger.debug("some output of stage {} was created, but completed flag not set.".format(stage))
            return False

        mtime = os.path.getmtime(fout)
        dmtime = datetime.datetime.fromtimestamp(mtime)
        comp_uptodate = comp_flag and (os.path.getmtime(fcompleted) >= mtime)
        state_uptodate = state_flag and (tracker.completed_ts >= dmtime)
        # print(comp_uptodate, state_uptodate)
        if not (comp_uptodate or state_uptodate):
            self.logger.warning("a completed flag from a previous run was found and ignored for stage {}!".format(stage))
            return False
        
        self.logger.info("stage {} is completed".format(stage))
        if self.options.redo:
            self.logger.warning("but ignored because --redo was specified!")
            return False

        return True

    def seed_stage(self):
        tracker = self.get_state_tracker('seed')

        from RBPamp.seed import PSAMSeeding
        ps = PSAMSeeding(self.rbns)
        # print "enriched MOTIFs in this library"
        # for m in SR.analysis.motifs_from_R(7):
        #     print m
        #     m.save_logo(fname=m.consensus + '.svg')

        # self.params = SR.seeded_params(self.rbns.n_samples)
        # TODO!
        # SR.primer_analysis()
        # sys.exit(0)
        a5 = self.rbns.reads[0].adap5
        a3 = self.rbns.reads[0].adap3 
        # contaminants = [a5, a3, rev_comp(a5), rev_comp(a3)]
        contaminants = []
        self.params = ps.seeded_multi_params(
            self.rbns.n_samples,
            max_motifs = self.options.max_motifs,
            n_max = self.options.max_width,
            k_seed = self.options.k_seed,
            thresh = self.options.seed_thresh,
            z_cut = self.options.z_cut,
            contaminants = contaminants,
            pseudo = self.options.seed_pseudo,
            keep_weight = self.options.seed_keep_weight
        )
        # print "len params in seed_stage", len(self.params.param_set)
        # print self.params
        self.params.save(os.path.join(self.run_path, 'seed/initial.tsv'), comment=self.mini_run_info)
        n = len(self.params.param_set)
        tracker.set(f"COMPLETED seeding {n} PSAMs")
        return self.params

    def flush_reads(self):
        # clean up memory usage
        self.rbns.flush(all=True)
        import gc
        gc.collect()
        # for reads in self.rbns.reads:
        #     reads.cache_flush()


    def calibrate_footprint(self):
        tracker = self.get_state_tracker('footprint')
        from RBPamp.footprint import FootprintCalibration
        from RBPamp.params import ModelParametrization, ModelSetParams

        calibrated_set = []
        params = self.params.copy(sort=True)

        def calibrate(par):
            cal = FootprintCalibration(self.rbns, par, thresh=1e-2)
            if self.options.top_kmer_acc:
                cal.compute_kmer_acc_profiles()

            kmin, kmax = self.options.footprint.split('-')
            res = cal.calibrate(k_core_range = [int(kmin), int(kmax)], from_scratch=self.options.redo)
            if res:
                return res
            else:
                return par
            
            cal.close()

        def try_load(par):
            # load from file
            consensus = par.as_PSAM().consensus_ul
            fname = os.path.join(
                self.rbns.out_path, 
                'footprint/calibrated_{}.tsv'.format(consensus)
            )
            # print "trying to load", fname
            if os.path.exists(fname):
                return list(ModelParametrization.load(fname, self.rbns.n_samples, sort=False))[0]
            else:
                # print "not found"
                return None

        if self.options.fp_num:
            # calibrate only ONE motif
            self.logger.info("calibrating only motif number {}".format(self.options.fp_num))
            if self.options.fp_num <= len(self.params.param_set):
                tracker.set('calibrating motif {}'.format(self.options.fp_num))
                res = calibrate(self.params.param_set[self.options.fp_num-1])
                tracker.set('optimum k={res.acc_k} s={res.acc_shift} rel_err={res.rel_err}'.format(res=res))
            return None
        
        else:
            for i, par in enumerate(params):
                # res = try_load(par)
                # res = None
                # if not res:
                tracker.set('calibrating motif {}'.format(i+1))
                res = calibrate(par)
                tracker.set('optimum k={res.acc_k} s={res.acc_shift} rel_err={res.rel_err}'.format(res=res))
                calibrated_set.append(res)
                

        path = os.path.join(self.rbns.out_path, 'footprint', 'parameters.tsv')
        self.logger.info("storing footprint optimized model in '{}'".format(path))
        cal_params = ModelSetParams(calibrated_set)
        cal_params.save(path, comment=self.mini_run_info)

        def overlap(par):
            ov_nt = min(par.acc_k + min(par.acc_shift, 0), min((par.k - par.acc_shift), par.acc_k))
            return ov_nt / float(par.acc_k)

        # filter for minimum sanity checks to decide if we want to keep a footprint
        par_keep = []

        n = len(calibrated_set)
        n_fp = 0
        for par_fp, par in zip(calibrated_set, params):
            if overlap(par_fp) > self.options.fp_min_ov and par_fp.rel_err < self.options.fp_max_err:
                par_keep.append(par_fp)
                n_fp += 1
            else:
                par_keep.append(par)

        self.params = ModelSetParams(par_keep)
        path = os.path.join(self.rbns.out_path, 'footprint', 'calibrated.tsv')
        self.logger.info("storing filtered footprint optimized model in '{}'".format(path))
        self.params.save(path, comment=self.mini_run_info)

        tracker.set(f'COMPLETED with {n_fp}/{n} footprints kept.')

        return self.params

    def make_plots(self, plots):
        tracker = self.get_state_tracker('plots')

        if plots == ["all",] : 
            plots = ['seed', 'descent', 'logos', 'lit', 'scatter', 'fp', 'vig']  #, 'aff'

        import RBPamp.report as report
        plot_path = ensure_path(os.path.join(self.run_path, 'plots/'))

        srep = report.SeedReport(path=plot_path, rbns=self.rbns)

        fprep = report.FootprintCalibrationReport(
            os.path.join(self.run_path, 'footprint/parameters.tsv'),
            out_path=plot_path,
            rbns=self.rbns
        )

        grep = report.GradientDescentReport(path=plot_path, comp=self.ref, rbns=self.rbns)
        grep.load(os.path.join(self.run_path, 'opt_nostruct/history'), "no structure")
        grep.load(os.path.join(self.run_path, 'opt_struct/history'), "full model")

        vignette = report.Vignette(path=plot_path, rbns=self.rbns, grad_report=grep, fp_report=fprep)
        funcs = {
            'seed' : srep.plot_R_dist,
            'descent' : grep.plot_report,
            'scatter' : grep.plot_scatter_all,
            'fp' : fprep.report,
            'lit' : grep.plot_literature,
            'logos' : grep.plot_logos,
            'aff' : grep.make_affinity_dist_plots,
            'afit' : grep.plot_param_error_scatter, # EXPERIMENTAL
            'vig' : vignette.render,
            'vigpdf' : vignette.make_pdf,
        }

        for plt in plots:
            tracker.set(plt)
            funcs[plt]()

        tracker.set('COMPLETED')


    def PSAM_gradient_descent(self, name="opt"):
        tracker = self.get_state_tracker(name)

        from RBPamp.psamgrad import PSAMGradientDescent
        PGD = PSAMGradientDescent(
            self.rbns, 
            self.params, 
            ref = self.ref, 
            k_fit = self.options.grad_k, 
            mdl_name = self.options.grad_mdl, 
            Z_thresh = self.options.Z_thresh, 
            run_name = name, 
            maxiter = self.options.grad_maxiter, 
            maxtime = self.options.grad_maxtime,
            fix_A0 = self.options.fix_A0,
            eps = self.options.mdl_epsilon, 
            tau = self.options.mdl_tau,
            redo = self.options.redo,
            debug_grad = self.options.debug_grad,
            resample_int = self.options.resample_int,
            excess_rbp = self.options.excess_rbp,
            linear_occ = self.options.linear_occ,
            continuation = self.options.cont,
            tracker = tracker,
        )
        PGD.optimize(debug=self.options.debug_grad)
        self.params = PGD.descent.params

        done = PGD.descent.status.startswith('CONVERGED')
        if done:
            tracker.set(f"COMPLETED with status {PGD.descent.status} {PGD.metrics}")
            path = os.path.join(self.rbns.out_path, name, 'optimized.tsv')
            self.logger.info(f"storing gradient-descent optimized model in '{path}'")
            self.params.save(path, comment=self.mini_run_info)
        else:
            tracker.set(PGD.descent.status)

    def estimate_errors(self):
        from RBPamp.errors import PSAMErrorEstimator
        est = PSAMErrorEstimator(os.path.join(self.run_path, 'opt_nostruct/'))
        est.estimate()

def main():
    options, args = parse_cmdline()
    run = Run(options, args)

    # import RBPamp.track_allocations
    # track = RBPamp.track_allocations.AllocationTracker(1000000)
    # with track:
    try:
        rbns = run.select_reads()
        # first, compute RBNS metrics
        metrics = [m.strip() for m in options.results.strip().split(',') if m.strip()]
        if metrics:
            run.compute_metrics(metrics)

        ranked_indices, rbns = run.keep_best() # unless --best is non-zero this does nothing
        if options.sample_corr:
            R, R_err = run.rbns.R_value_matrix(6)
            print(np.corrcoef(R)[ranked_indices.argmin()])


        run.flush_reads()
        run.rbns.flush()
        from RBPamp.caching import _dump_cache_sizes
        _dump_cache_sizes()
        # npw.report_sizes('main startup')
        if options.fold_input or options.fold_samples:
            run.fold_reads()
            run.logger.info("folding completed.")

        if (options.opt_full or options.opt_seed) and not run.completed("seed"):
            # run.flush_reads()
            run.logger.info("STAGE0: initialize PSAM")
            run.seed_stage()

        param_sources = [run.options.mdl_psam_init, 'seed/initial.tsv']
        if (options.opt_full or options.opt_nostruct) and (not run.completed('opt_nostruct') or options.cont):
            run.logger.info("STAGE1: PSAM optimization without secondary structure accessibility")

            if options.resume:
                param_sources.insert(1, 'opt_nostruct/parameters.tsv')
            run.probe_params(*param_sources)

            run.params.acc_k = 0  # disable accessibility
            run.PSAM_gradient_descent('opt_nostruct')

        if (options.opt_full or options.opt_footprint) and (not run.completed('footprint') or options.cont):
            run.logger.info("STAGE2: footprint parameter estimation")

            run.probe_params(run.options.mdl_psam_init, 'opt_nostruct/optimized.tsv')
            run.calibrate_footprint()
            run.flush_reads()

        if (options.opt_full or options.opt_struct) and (not run.completed('opt_struct') or options.cont):
            run.logger.info("STAGE3: PSAM optimization with accessibility footprint")

            param_sources = [run.options.mdl_psam_init, 'footprint/calibrated.tsv']
            if options.resume:
                param_sources.insert(1, 'opt_struct/parameters.tsv')
            
            run.probe_params(*param_sources)
            run.PSAM_gradient_descent('opt_struct')

        if options.plot:
            run.make_plots(options.plot.split(','))

        if options.est_errors:
            run.estimate_errors()

    except SystemExit:
        # This is alright
        pass

    except:
        run.logger.error("Caught exception. Gathering traceback")
        exc = traceback.format_exc()
        run.logger.error(exc)
        sys.stderr.write(exc)
        
        if run.last_tracker:
            run.last_tracker.set(exc)

        ex_type, ex_val, ex_tb = sys.exc_info()
        if ex_type == MemoryError:
            import RBPamp.caching
            RBPamp.caching._dump_cache_sizes()
        
        # in case we have child processes, try to end them gracefully
        import RBPamp.fold
        RBPamp.fold.interrupt()
    else:
        run.logger.info("run completed.")
    
    # track.write_html("allocations.html")

if __name__ == '__main__':
    main()
