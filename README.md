
# RegVar

<!-- badges: start -->
<!-- badges: end -->

The goal of RegVar is to characterize 3'UTR single nucleotide variants by their potential regulatory elements.

## Local installation
RegVar can be installed on a local computer, or on a computing cluster. Make sure you have updated base R as well as installed devtools to install RegVar.

First: You will need a command line to install RegVar; then it can be used in R. With a mac, use the terminal application. With a PC, you will need to download a terminal: https://learn.microsoft.com/en-us/windows/terminal/install.

Then: Make sure you have miniconda, pip, gcc, and bedtools installed locally (https://docs.conda.io/en/latest/miniconda.html, https://pip.pypa.io/en/stable/installation/, https://www.geeksforgeeks.org/how-to-install-gcc-compiler-on-linux/#, https://bedtools.readthedocs.io/en/latest/content/installation.html). You can then install the development version of RegVar like so in R:

``` r
#first, install R package: 
devtools::install_github("RomoL2/RegVar") #note, does not install large files
```

Then, in the RegVar directory in command line install the necessary annotation files, like so:
``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar
rm -r extdata #remove directory with symbolic links
wget https://zenodo.org/records/13738622/files/extdata.tar.gz 
tar -xf extdata.tar.gz #unzip files, makes directory
```

Pull the latest version of RBPamp into the newly created directory
``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar/extdata
! rm -r RBPamp #remove old version
! git clone https://marjens@bitbucket.org/marjens/RBPamp.git #pull new version
```

Then, again in the RBPamp directory of RegVar in command line create the conda environment for RBPamp (this solve may take quite a while, up to a few hours):
(source for RBPamp is: https://bitbucket.org/marjens/rbpamp/src/master/)
Note that RBPamp is the only part of the script that needs to run in a conda environment; bedtools should be installed outside of the environment (base). 
``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar/extdata/RBPamp
conda create --name RBPamp --file requirements.txt -c conda-forge
conda activate RBPamp #need to be in the same directory as above
export GCC=gcc #if you are getting an error here, it is likely because you don't have gcc installed; see above
python setup.py build #there will be a lot of warnings in the window; don't worry unless build fails
python setup.py install
```

Then, again in the RBPamp diretory of RegVar, download the hg38 fasta (large file, zipped is ~1GB)
``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar/extdata
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
```

Then, if VCF is in hg19, again in the RBPamp diretory of RegVar, download the liftover files and R package
``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar/extdata
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
```
R package (install in R):
``` r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("rtracklayer")
```

## Installation with docker
RegVar can be installed in a docker image to improve portability.

First: You will need to download docker (https://www.docker.com/products/docker-desktop/).

Then: in the terminal in the directory with the dockerfile (with docker running), use the following commands to build the docker image:

``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar
docker build -t regvar_docker .
```

Or: you can pull my docker image.
``` r
docker pull lromo/regvar_docker
```


Run the docker container:

``` r
docker run -it regvar_docker
```

Then you can use the R functions in the docker container terminal as below (will likely require increasing the memory allocation to docker).

## Example

Use these functions in R to run RegVar. 
Characterize 3'UTR variants in a standard vcf file with CharacterizeVariants (either hg19 or hg38), or a single user-input hg38 variant with CharacterizeVariants_single_input:

``` r
library(RegVar)
CharacterizeVariants('file.vcf', '~/', '~/', 'hg38')
CharacterizeVariants_single_input('~/')

```
For CharacterizeVariants_single_input, the arguments are the output folder

For CharacterizeVariants, the arguments are:
1. The name of the input vcf file; must be tab delim with 8 columns:
chrom, pos, id, ref, alt, qual, filter, info
Chrom column must be either a number (1) or chr# (chr1)
2. The path to the directory where the vcf file is located
3. The path to the directory where you would like the output written
4. The base of the input vcf file (options are 'hg19' or 'hg38')

For both functions, the output columns are:
1. variant ID: chr_position_ref_alt
2. CADD score (for gnomAD variants only)
3. phastcons_100 score
4. phylop_100 score
5. RBPamp motif RBPs, separated by underscore
6. category of RBPamp motifs (based on whether variant is predicted to disrupt or preserve the motif)
7. variant eclip peaks in the following format: RBP_cell separated by two underscores 
8. variant eQTL info, if applicable (within 5NT): chromStart_chromEnd_ref_alt_signalid@tissue_name= PIP[SPIP:size_of_cluster]
9. whether the variant is predicted to be an eQTL (1 or 0 corresponds to yes or no)
10. variant GWAS info, if applicable (within 5NT), separated by underscore: chromStart, chromEnd, rsID, minor_allele, ref, alt, fine_map, pheno, maf, effect_size, pip
11. whether the variant is predited to be a GWAS variant (1 or 0 corresponds to yes or no)
12. APA info: ensemble gene, strand, APA isoform number the variant falls in, total APA isoforms, and what region the variant falls in (common, single, partially-shared, or unique) separated by underscore
13. poly A site info (is empty if not within 50NT of a poly A site)
14. variant microRNA info: miR name, seed type, Pct, strand, context percentile, family conservation, site conservation, separated by two underscores (not all info is available for all variants, listed as NULL if not)
15. variant ClinVar info, if applicable (within 5NT); see ClinVar documentation for column description
16. variant ReP sites (RBP and strand)

