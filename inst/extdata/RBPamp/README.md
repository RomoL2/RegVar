# CSKA
A fast Cython implementation of the "Streaming K-mer Assignment" algorithm initially described in Lambert et al. 2014 (PMID: 24837674).

## What is CSKA?
CSKA is a tool for exploratory data analysis of RBNS experiments. It is based on the original streaming k-mer analysis (Lambert et al. 2014) which allowed to select k-mers that are most strongly associated with protein binding. However, CSKA is almost two orders of magnitude faster and features various extensions that allow to study the data in much more depth. Here is what CSKA can do for you:


   * identify most strongly associated kmers (original SKA function)
   * compare experiments at different concentrations
   * compute f-values for kmer lists (fraction of pull-down reads "expained")
   * [TODO: rough estimate of binding constants based on f-values]
   * [WIP: study of additional preferences/constraints on flanking sequence around a 'core'-motif]
   * [WIP: detection of multi-part motifs (combinations of kmers)]
   * [TODO: build compact models for binding based on few core motifs and weight matrices for context]

## Installation

It is in PyPI, so `pip install RBPamp` should work. You can also pull the latest development version using `git clone https://marjens@bitbucket.org/marjens/RBPamp.git`. `RBPamp` uses `setuptools` so

```
   python setup.py build
   python setup.py install
```
will do what you expect. To uninstall run `pip uninstall RBPamp` .

## Usage

`RBPamp --help`:


    Usage: RBPamp [options] <input_reads_file> <pulldown_reads_file1> [<pulldown_reads_file2] [...]

    Options:
    -h, --help            show this help message and exit
    -k MIN_K, --min-k=MIN_K
                            min kmer size (default=3)
    -K MAX_K, --max-k=MAX_K
                            max kmer size (default=8)
    -n N_PASSES, --n-passes=N_PASSES
                            max number of passes (default=10)
    -R RNA_CONC, --rna-concentration=RNA_CONC
                            concentration of random RNA used in the experiment in
                            micro molars (default=100uM)
    -p PROT_CONC, --rbp-concentration=PROT_CONC
                            (comma separated list of) protein concentration used
                            in the pulldown experiment(s) in nano molars
                            (default=300nM)
    --name=NAME           name of the protein assayed (default=RBP)
    --subsamples=SUBSAMPLES
                            number of subsamples for error estimateion (default=5)
    --pseudo=PSEUDO       pseudo count to add to kmer counts in order to avoid
                            div by zero for large k (default=10)
    -c CONVERGENCE, --convergence=CONVERGENCE
                            convergence is reached when max. change in absolute
                            weight is below this value (default=0.5)
    -B BACKGROUND, --background=BACKGROUND
                            path to file with background (input) kmer abundances
                            in the library
    -o OUTPUT, --output=OUTPUT
                            path where results are to be stored
    --debug               SWITCH: activate debug output
    --resume              SWITCH: load results from previous run, to resume with
                            any second stage analyses
    --interactions        SWITCH: activate combinatorial search
    --n-max=N_MAX         TESTING: read at most N reads
    --version             SWITCH: show version information and quit
    