
# RegVar

<!-- badges: start -->
<!-- badges: end -->

The goal of RegVar is to characterize 3'UTR single nucleotide variants by their potential regulatory elements.

## Installation
RegVar can be installed on a local computer, or on a computing cluster. Make sure you have updated base R as well as installed devtools to install RegVar.

First: You will need a command line to install RegVar; then it can be used in R. With a mac, use the terminal application. With a PC, you will need to download a terminal: https://learn.microsoft.com/en-us/windows/terminal/install.

Then: Make sure you have miniconda, pip, and bedtools installed locally (https://docs.conda.io/en/latest/miniconda.html, https://pip.pypa.io/en/stable/installation/, https://bedtools.readthedocs.io/en/latest/content/installation.html). You can then install the development version of RegVar like so in R:

``` r
#first, install R package: 
devtools::install_github("RomoL2/RegVar") #note, does not install large files
```

Then, in the RegVar directory in command line install the necessary annotation files, like so:
``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar
mkdir tmp
git clone https://github.com/RomoL2/RegVar ./tmp #clones repository into temporary directory
rm -r extdata #remove directory with symbolic links
mv tmp/inst/extdata ./ #replace with directory with real annotation files
rm -r tmp #ok to override, type "y" #remove temporary directory
```

## Example

Use these functions in R to run RegVar. First use install_reqs (this will install the hg38 fasta, the required conda environment for RBPamp, and RBPamp). The argument must be the location of the R package either on your local computer or on the cluster (for instance, if in a conda environment, '~/.conda/envs/myR/lib/R/library/RegVar'). 
Of note, install_reqs will install the g-zipped hg38 fasta, which is a large file (almost 1 GB zipped)

Then characterize variants in a standard vcf file with CharacterizeVariants or a single user-input variant with CharacterizeVariants_single_input:

``` r
library(RegVar)
install_reqs('/Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar')
CharacterizeVariants('file.vcf', '~/', '~/')
CharacterizeVariants_single_input('~/')

```
For CharacterizeVariants_single_input, the arguments are the output folder
For CharacterizeVariants, the arguments are:
1. The name of the input vcf file; must be tab delim with 8 columns:
chrom, pos, id, ref, alt, qual, filter, info
Chrom column must be either a number (1) or chr# (chr1)
2. The path to the directory where the vcf file is located
3. The path to the directory where you would like the output written

For both functions, the output columns are:
1. phastcons_100 score
2. phylop_100 score
3. CADD score (for gnomAD variants only)
4. variant ID: chr_position_ref_alt
5. RBPamp motif RBPs, separated by underscore
6. category of RBPamp motifs (based on whether variant is predicted to disrupt or preserve the motif)
7. variant microRNA info: miR name, seed type, Pct, strand, context percentile, family conservation, site conservation, separated by two underscores (not all info is available for all variants, listed as NULL if not)
8. variant eclip peaks in the following format: RBP_cell separated by two underscores 
9. variant eQTL info, if applicable (within 5NT): chromStart_chromEnd_ref_alt_signalid@tissue_name= PIP[SPIP:size_of_cluster]
10. variant GWAS info, if applicable (within 5NT), separated by underscore: chromStart, chromEnd, rsID, minor_allele, ref, alt, fine_map, pheno, maf, effect_size, pip
11. variant ClinVar info, if applicable (within 5NT); see ClinVar documentation for column description
12. whether the variant is predicted to be an eQTL (1 or 0 corresponds to yes or no)
13. whether the variant is predited to be a GWAS variant (1 or 0 corresponds to yes or no)
14. APA info: ensemble gene, strand, APA isoform number the variant falls in, total APA isoforms, and what region the variant falls in (common, single, partially-shared, or unique) separated by underscore
15. poly A site info (is NA if not within 50NT of a poly A site)
