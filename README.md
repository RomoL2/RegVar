
# CharVar

<!-- badges: start -->
<!-- badges: end -->

The goal of CharVar is to characterize 3'UTR variants into their potential regulatory elements.

## Installation

First, make sure you have miniconda installed locally (https://docs.conda.io/en/latest/miniconda.html). You can then install the development version of CharVar like so:

``` r
library(devtools) # Make sure that the devtools library is loaded
install_github("RomoL2/CharVar")
```

## Example

First use install_reqs to install the hg38 fasta, the required conda environment for RBPamp, and RBPamp, then characterized variants in a standard vcf file:

``` r
library(CharVar)
install_reqs('"~/CharVar")
CharacterizeVariants('file.vcf', "~/", "~/data_files/")
```

