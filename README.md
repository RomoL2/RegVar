
# CharVar

<!-- badges: start -->
<!-- badges: end -->

The goal of CharVar is to characterize 3'UTR variants into their potential regulatory elements.

## Installation

First, make sure you have miniconda as well as bedtools installed locally (https://docs.conda.io/en/latest/miniconda.html, https://bedtools.readthedocs.io/en/latest/content/installation.html). You can then install the development version of CharVar like so:

``` r
#first, install R package: 
devtools::install_github("RomoL2/CharVar") #note, does not install large files
```

Then, in the CharVar directory in command line install the necessary annotation files, like so:
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/CharVar
pip install github-clone 
rm -r extdata 
ghclone https://github.com/RomoL2/CharVar/tree/main/inst/extdata 

## Example

First use install_reqs (this will install the hg38 fasta, the required conda environment for RBPamp, and RBPamp). Then characterize variants in a standard vcf file with CharacterizeVariants:

``` r
library(CharVar)
install_reqs('/Library/Frameworks/R.framework/Versions/4.0/Resources/library/CharVar')
CharacterizeVariants('file.vcf', '~/', '/Library/Frameworks/R.framework/Versions/4.0/Resources/library/CharVar', '~/')
CharacterizeVariants_single_input("/Library/Frameworks/R.framework/Versions/4.0/Resources/library/CharVar", "~/")

```

